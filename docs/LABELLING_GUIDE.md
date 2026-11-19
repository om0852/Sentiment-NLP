# Sentiment labelling guide (gold set)

The gold set is the only trustworthy measure of how accurate the engine is on **our** posts. Every
accuracy number we publish should come from it.

- File: `data/gold/gold_sample.csv` (created by `scripts/build_silver_dataset.py`; gitignored because it holds user posts)
- 1,500 posts picked at random from the 300k MongoDB export, after cleaning and de-duplication
- These posts are never used for training, teacher labelling, calibration or threshold tuning

## Process

1. Two people label every row independently. Fill in `label_annotator_1` and `label_annotator_2`
   with `positive`, `negative` or `neutral`. Don't look at each other's column.
2. Use `notes` for anything unclear, such as language you can't read, or sarcasm you're unsure about.
   If you can't read the language at all, write `skip` as the label.
3. When both are done, run `python scripts/evaluate_gold.py`. It reports agreement between the two
   annotators (the ceiling on accuracy) and the engine's accuracy on the rows where they agree.
4. Rows where the annotators disagree go to a third person, or get discussed and settled. Don't drop them
   silently, because they're the hard cases.

Expect roughly 15–20 seconds per post, so about 7–8 hours per annotator for 1,500 posts. Splitting it
over a few days helps keep judgement consistent.

## What the label means

Label **the sentiment the post expresses or reports**, as a typical reader would take it.

| Label | Use when |
|---|---|
| `positive` | Praise, happiness, gratitude, excitement, congratulations, recommendation, good news |
| `negative` | Complaint, anger, sadness, criticism, disappointment, mockery, bad news |
| `neutral` | Information, questions, announcements, ads with no emotional tone, hashtag-only posts, lists of products |

## Rules for the tricky cases

1. **News and factual headlines → the polarity of the event.**
   "Earthquake kills 120" is `negative`. "India wins the World Cup" is `positive`. "RBI keeps repo rate unchanged" is `neutral`.
   If the event is good for some people and bad for others ("Petrol prices hiked"), label it from the point of view
   of the general public.
2. **Sarcasm → the intended meaning.** "Great, another 2-hour delay 🙃" is `negative`.
3. **Mixed posts → the overall takeaway.** Use whichever side the post ends on or stresses more.
   Only use `neutral` if the two sides genuinely cancel out.
4. **Promotions and offers.** "50% off this weekend!" is `positive` when it's written with excitement.
   A plain listing ("Free shipping above ₹499", product specs, price lists) is `neutral`.
5. **Political posts.** Label the tone of the post, not whether you agree with it. A party celebrating its own
   win is `positive`. Attacking an opponent is `negative`.
6. **Condolences, RIP, tragedy** → `negative`, even when the post is respectful or grateful.
7. **Hashtag-only or emoji-only posts.** Go by the emoji if it's clear (😍 → `positive`, 😡 → `negative`).
   Otherwise `neutral`.
8. **Non-English posts.** Label them if you can read the language. Otherwise write `skip`.
9. **Truncated text.** Label what's there. Don't guess what the missing part said.
