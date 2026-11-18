"""Measure the engine end to end and (optionally) choose cascade thresholds and rules to disable.

Evaluation sets (whatever exists is used):
  silver_val   real posts where the local teacher and the stored DB label agree. Biased towards the teacher,
               so use it to compare configurations, not as an absolute accuracy figure.
  tweeteval    official TweetEval sentiment test split (held out from all training)
  hard_1/2     scripts/benchmark_hard_50.py and scripts/benchmark_hard_set_2.py
  extra        Python files passed with --extra, each defining CASES = [(category, text, label, alt)]
  gold         data/gold/gold_sample.csv rows where both human annotators agree (once labelled)

With --tune, thresholds and rule ablation use the silver *calib* split only, never the sets above.
--write-config saves the result to models/cascade_config.json.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/evaluate_cascade.py --tune --write-config \
        --extra path/to/holdout_set.py
"""
import argparse
import ast
import collections
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.models import PureSentimentClassifier  # noqa: E402
from src.models.calibration import expected_calibration_error, fit_temperature  # noqa: E402
from src.models.student_onnx import StudentClassifier  # noqa: E402
from src.preprocessing.context_analyzer import ContextAnalyzer  # noqa: E402
from src.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from src.serving.cascade import DEFAULT_CONFIG, Cascade, classify_batch, load_config  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
SHORT = {"p": "positive", "n": "negative", "u": "neutral"}
CONFIG_PATH = os.path.join(project_root, "models", "cascade_config.json")


# ---------------------------------------------------------------- data
def bucket(key):
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100


def load_silver():
    """Returns (val, calib, all_agree). val/calib follow scripts/train_student.py's split, so the student
    never trained on them. all_agree is every agreed post; fine for judging rules, which are not trained."""
    path = os.path.join(data_root, "data", "silver", "teacher_labels.jsonl")
    val, calib, everything = [], [], []
    if not os.path.exists(path):
        return val, calib, everything
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["original"] != r["teacher"] or r["teacher"] not in LABELS:
                continue
            b = bucket(r["id"])
            item = {"text": r["text"], "ok": {r["teacher"]}, "cat": r["source"] or "unknown"}
            everything.append(item)
            if b < 4:
                val.append(item)
            elif b < 8:
                calib.append(item)
    return val, calib, everything


def load_tweeteval():
    d = os.path.join(data_root, "data", "raw", "tweeteval")
    if not os.path.exists(os.path.join(d, "test_text.txt")):
        return []
    with open(os.path.join(d, "test_text.txt"), encoding="utf-8") as ft, \
            open(os.path.join(d, "test_labels.txt"), encoding="utf-8") as fl:
        return [{"text": t.strip(), "ok": {LABELS[int(l)]}, "cat": "tweet"} for t, l in zip(ft, fl)]


def load_hard(name):
    path = os.path.join(project_root, "scripts", name)
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "").endswith("EXAMPLES"):
            items = ast.literal_eval(node.value)
            return [{"text": x["text"], "ok": set(x["acceptable"]), "cat": "hard"} for x in items]
    return []


def load_extra(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [{"text": t, "ok": {SHORT[l]} | {SHORT[a] for a in alt}, "strict": SHORT[l], "cat": c}
            for c, t, l, alt in mod.CASES]


def load_gold():
    path = os.path.join(data_root, "data", "gold", "gold_sample.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a, b = r["label_annotator_1"].strip().lower(), r["label_annotator_2"].strip().lower()
            if a in LABELS and a == b:
                out.append({"text": r["text"], "ok": {a}, "cat": r["source"]})
    return out


# ---------------------------------------------------------------- engine
class MemoStudent:
    """The student is deterministic, so cache its answers: tuning re-runs the same posts many times."""

    def __init__(self, student):
        self.student = student
        self.cache = {}

    def predict_batch(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for t, r in zip(todo, self.student.predict_batch(todo) if todo else []):
            self.cache[t] = r
        return [self.cache[t] for t in texts]


class Engine:
    def __init__(self, use_student=True):
        self.model = PureSentimentClassifier()
        self.model.load(os.path.join(project_root, "models", "sentiment_model.json.gz"))
        self.pipeline = PreprocessingPipeline()
        self.student = None
        sdir = os.path.join(project_root, "models", "student")
        if use_student and StudentClassifier.available(sdir):
            self.student = MemoStudent(StudentClassifier(sdir, threads=1))

    def run(self, items, cfg, analyzer, use_student=True, bs=64):
        cascade = Cascade(student=self.student if use_student else None, config=cfg)
        out = []
        for i in range(0, len(items), bs):
            chunk = items[i: i + bs]
            out += classify_batch([x["text"] for x in chunk], [None] * len(chunk), self.model,
                                  self.pipeline, analyzer, cascade)
        return out


def score(items, results, strict=False):
    def hit(x, r):
        if strict and "strict" in x:
            return r["label"] == x["strict"]
        return r["label"] in x["ok"] or (r["label"] == "mixed" and len(x["ok"]) > 1)
    n = len(items)
    acc = sum(hit(x, r) for x, r in zip(items, results)) / n if n else 0.0
    esc = sum(r["tier"] == "transformer" for r in results) / n if n else 0.0
    ans = [(x, r) for x, r in zip(items, results) if not r["uncertain"]]
    sel = sum(hit(x, r) for x, r in ans) / len(ans) if ans else 0.0
    # what the API reports as fallback_required with its default confidence_threshold of 0.60
    fb = sum(r["uncertain"] or r["ctx_fallback"] or r["confidence"] < 0.60 or r["meta"].get("sarcasm_detected", False)
             for r in results) / n if n else 0.0
    return {"n": n, "acc": acc, "escalated": esc, "coverage": len(ans) / n if n else 0.0, "answered_acc": sel,
            "api_fallback_rate": fb}


# ---------------------------------------------------------------- tuning
def tfidf_logits(engine, items):
    preds = engine.model.predict_batch([engine.pipeline.process(x["text"])[0] for x in items])
    return np.log(np.array([[max(p["probabilities"].get(l, 0), 1e-9) for l in LABELS] for p in preds]))


def ablate_rules(engine, items, cfg):
    """Per rule: how often it fixes vs breaks the model's own answer (tier 1 only)."""
    analyzer = ContextAnalyzer(disabled_rules=[])
    with_rules = engine.run(items, cfg, analyzer, use_student=False)
    stats = collections.defaultdict(lambda: [0, 0, 0])  # fired, fixed, broke
    preds = engine.model.predict_batch([engine.pipeline.process(x["text"])[0] for x in items])
    for x, r, p in zip(items, with_rules, preds):
        if r["reason"] == "model_inference" or r["label"] == "mixed":
            continue  # 3-class silver labels can't say whether a "mixed" answer is right
        s = stats[r["reason"]]
        s[0] += 1
        base_ok = p["label"] in x["ok"]
        rule_ok = r["label"] in x["ok"] or (r["label"] == "mixed" and len(x["ok"]) > 1)
        s[1] += int(rule_ok and not base_ok)
        s[2] += int(base_ok and not rule_ok)
    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--max-escalation", type=float, default=0.0,
                    help="cap on share of posts sent to the transformer (0 = derive from the CPU budget)")
    ap.add_argument("--posts-per-day", type=float, default=1_000_000)
    ap.add_argument("--vcpu", type=float, default=0.1)
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--min-rule-firings", type=int, default=20)
    ap.add_argument("--short-text-chars", type=int, default=60,
                    help="Latin-script posts shorter than this stay on tier 1 (slang/emoji dictionaries)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    engine = Engine()
    silver_val, silver_calib, silver_all = load_silver()
    cfg = load_config() if not args.tune else dict(DEFAULT_CONFIG)
    report = {"config": None, "sets": {}, "latency_ms": {}, "rules": {}}

    # ---- latency (single thread; tier 1 includes the news detector) ----
    sample = (silver_val or load_tweeteval())[:400]
    if sample:
        from src.serving.cascade import news_probability
        t0 = time.perf_counter()
        engine.run(sample, dict(cfg, escalate_below=0.0, escalate_news=False, escalate_sarcasm_cues=False,
                                escalate_non_latin=False), ContextAnalyzer(), use_student=False)
        news_probability([x["text"] for x in sample])
        t_fast = (time.perf_counter() - t0) * 1000 / len(sample)
        report["latency_ms"]["tfidf_per_post"] = round(t_fast, 3)
        if engine.student:
            t0 = time.perf_counter()
            engine.student.student.predict_batch([x["text"] for x in sample])
            t_student = (time.perf_counter() - t0) * 1000 / len(sample)
            report["latency_ms"]["transformer_per_post"] = round(t_student, 3)
            budget = args.vcpu * 1000 * 86400 / args.posts_per_day  # CPU-ms available per post
            cap = max(0.0, min(1.0, (budget - t_fast) / t_student))
            report["latency_ms"]["cpu_budget_per_post"] = round(budget, 3)
            report["latency_ms"]["max_escalation_for_budget"] = round(cap, 3)
            print(f"latency: tfidf {t_fast:.2f} ms, transformer {t_student:.2f} ms/post; "
                  f"budget {budget:.2f} ms/post -> escalate at most {cap:.0%}", flush=True)

    if args.tune and silver_calib:
        # step 5: TF-IDF temperature
        y = np.array([LABELS.index(next(iter(x["ok"]))) for x in silver_calib])
        lg = tfidf_logits(engine, silver_calib)
        t = fit_temperature(lg, y)
        from src.models.calibration import softmax
        print(f"TF-IDF temperature {t:.3f}; ECE before {expected_calibration_error(softmax(lg), y):.3f} "
              f"after {expected_calibration_error(softmax(lg, t), y):.3f}", flush=True)
        cfg["tfidf_temperature"] = round(t, 4)

        # step 7: rule ablation. Rules are not trained, so every agreed silver post is fair to judge them on.
        # Disable a rule only on clear evidence of net harm, not on a handful of firings.
        stats = ablate_rules(engine, silver_all, cfg)
        disabled = sorted(k for k, (n, fix, broke) in stats.items()
                          if n >= args.min_rule_firings and broke - fix >= max(3, 0.1 * n))
        report["rules"] = {k: {"fired": v[0], "fixed": v[1], "broke": v[2]} for k, v in sorted(stats.items())}
        print(f"rule ablation on {len(silver_all):,} real posts (fired / fixed / broke):")
        for k, (n, fix, broke) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            note = "DISABLE" if k in disabled else ("too few firings to judge" if n < args.min_rule_firings else "")
            print(f"  {k:38} {n:5} {fix:5} {broke:5} {note}")
        cfg["disabled_rules"] = disabled
        analyzer = ContextAnalyzer(disabled_rules=disabled)

        # step 4: which escalation signals, which threshold, and who wins when a rule fired.
        # Pick the most accurate policy that fits the CPU budget; if none fits, the cheapest one.
        if engine.student:
            cap = args.max_escalation or report["latency_ms"].get("max_escalation_for_budget", 1.0)
            policies = []
            for news in (True, False):
                for sarc in (True, False):
                    for nonlatin in (True, False):
                        for override in (False, True):
                            for thr in [0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]:
                                c = dict(cfg, escalate_news=news, escalate_sarcasm_cues=sarc,
                                         escalate_non_latin=nonlatin, escalate_below=thr,
                                         rules_override_student=override, uncertain_below=0.0)
                                s = score(silver_calib, engine.run(silver_calib, c, analyzer))
                                policies.append((s["acc"], s["escalated"], c))
            fitting = [p for p in policies if p[1] <= cap]
            chosen = max(fitting, key=lambda p: (p[0], -p[1])) if fitting else min(policies, key=lambda p: p[1])
            if not fitting:
                print(f"WARNING: no policy fits the CPU budget (cap {cap:.0%}); using the cheapest", flush=True)
            print("best policies by accuracy (acc, escalated, news/sarcasm/non-latin/override/threshold):")
            for acc, esc, c in sorted(policies, key=lambda p: -p[0])[:8]:
                print(f"  {acc:.2%} {esc:5.0%}  {c['escalate_news']!s:5} {c['escalate_sarcasm_cues']!s:5} "
                      f"{c['escalate_non_latin']!s:5} {c['rules_override_student']!s:5} {c['escalate_below']}")
            report["policy_frontier"] = [{"acc": round(a, 4), "escalated": round(e, 4)} for a, e, _ in policies]
            cfg.update({k: chosen[2][k] for k in ("escalate_news", "escalate_sarcasm_cues", "escalate_non_latin",
                                                  "escalate_below", "rules_override_student")})
            # uncertainty threshold: best answered accuracy while still answering >= min_coverage of posts
            best_u = (0.0, 0.0)
            for u in [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                s = score(silver_calib, engine.run(silver_calib, dict(cfg, uncertain_below=u), analyzer))
                print(f"  uncertain_below={u:.2f}: coverage {s['coverage']:.0%} answered acc {s['answered_acc']:.2%}")
                if s["coverage"] >= args.min_coverage and s["answered_acc"] >= best_u[1]:
                    best_u = (u, s["answered_acc"])
            cfg["uncertain_below"] = best_u[0]
            print(f"chosen policy: acc {chosen[0]:.2%} escalated {chosen[1]:.0%} | "
                  f"{ {k: cfg[k] for k in DEFAULT_CONFIG if k != 'disabled_rules'} }", flush=True)
    if args.tune:
        cfg["short_text_chars"] = args.short_text_chars
        cfg["keep_romanized_indic_on_tier1"] = True
        # hand-written rules are high-precision where they fire; silver data can't break the tie (rules rarely fire)
        cfg["rules_override_student"] = True
    report["config"] = cfg
    if args.write_config:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote {CONFIG_PATH}")

    # ---- evaluation ----
    analyzer = ContextAnalyzer(disabled_rules=cfg.get("disabled_rules", []))
    sets = {"silver_val (teacher-agreement)": silver_val, "tweeteval_test": load_tweeteval(),
            "hard_1": load_hard("benchmark_hard_50.py"), "hard_2": load_hard("benchmark_hard_set_2.py"),
            "gold": load_gold()}
    for p in args.extra:
        sets[os.path.basename(p)[:-3]] = load_extra(p)
    print(f"\n{'set':22} {'n':>6} | {'tfidf+rules':>11} | {'transformer':>11} | {'cascade':>8} "
          f"{'escal.':>6} | {'coverage':>8} {'answered':>8}")
    for name, items in sets.items():
        if not items:
            continue
        old = engine.run(items, dict(DEFAULT_CONFIG, tfidf_temperature=1.0), ContextAnalyzer(disabled_rules=[]),
                         use_student=False)
        row = {"tfidf_rules_before": score(items, old, strict=True)}
        if engine.student:
            st = engine.run(items, dict(cfg, escalate_below=1.01, rules_override_student=False, short_text_chars=0,
                                        keep_romanized_indic_on_tier1=False), analyzer)
            row["transformer_only"] = score(items, st, strict=True)
            cas = engine.run(items, cfg, analyzer)
            row["cascade"] = score(items, cas, strict=True)
            if "strict" in items[0]:
                row["cascade_lenient"] = score(items, cas)
                cats = collections.defaultdict(list)
                for x, r in zip(items, cas):
                    cats[x["cat"]].append((x, r))
                row["cascade_by_category"] = {c: round(score([a for a, _ in v], [b for _, b in v],
                                                             strict=True)["acc"], 4) for c, v in cats.items()}
        report["sets"][name] = row
        tr = row.get("transformer_only", {}).get("acc", float("nan"))
        c = row.get("cascade", {})
        print(f"{name:22} {len(items):6} | {row['tfidf_rules_before']['acc']:11.2%} | {tr:11.2%} | "
              f"{c.get('acc', float('nan')):8.2%} {c.get('escalated', 0):6.0%} | "
              f"{c.get('coverage', 1):8.0%} {c.get('answered_acc', float('nan')):8.2%} | "
              f"{c.get('api_fallback_rate', 0):6.0%}", flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
