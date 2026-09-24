import re
from typing import List, Set, Dict, Any

class SemanticTagger:
    """
    Extracts structured, high-value semantic tags from multimodal text and documents.
    Mines explicit hashtags, domain-specific intent tags, and salient keywords.
    """
    def __init__(self):
        self.domain_tag_rules = [
            # Bugs & Crash
            (r"\b(crash|crashed|crashing|freeze|restart|stuck|hang|हँग|क्रैश)\b", "#app_crash"),
            (r"\b(bug|bugs|glitch|error|exception|stack\s*trace)\b", "#bug_report"),
            (r"\b(memory\s*leak|outage|downtime|timeout|500\s*error|connection\s*pool)\b", "#system_failure"),
            
            # UI & Performance
            (r"\b(ui|ux|interface|design|dark\s*mode|animation|look\s*wise|दिसायला|दिसण्यात)\b", "#ui_ux"),
            (r"\b(slow|lag|latency|battery|draining|heating|गरम|स्लो)\b", "#performance_issue"),
            (r"\b(snappy|smooth|fast|fluid|zero\s*lag)\b", "#high_performance"),
            
            # Customer Service & Grievance
            (r"\b(customer\s*support|support\s*team|helpdesk|call\s*center|agent|executive)\b", "#customer_support"),
            (r"\b(worst\s*service|ghatiya|rude|no\s*response|ignored|ticket)\b", "#support_grievance"),
            
            # Finance & Billing
            (r"\b(refund|debited|deducted|payment\s*failed|money\s*lost|paise\s*cut|पैसे\s*कट)\b", "#payment_issue"),
            (r"\b(paisa\s*vasool|worth\s*every\s*penny|value\s*for\s*money)\b", "#value_for_money"),
            (r"\b(paise\s*fukat|paise\s*barbad|waste\s*of\s*money|scam|fraud)\b", "#money_wasted"),
            
            # Memes & Viral Slang
            (r"\b(cooked|who\s*let\s*him\s*cook|bro\s*cooked|never\s*cook)\b", "#meme_cooked"),
            (r"\b(aura|infinite\s*aura|aura\s*-1000)\b", "#aura_score"),
            (r"\b(ratio|skill\s*issue|demure|brainrot|gigachad)\b", "#viral_meme"),
            
            # Praise & Recommendations
            (r"\b(lai\s*bhari|ek\s*number|khup\s*chhan|bhaari|nadach\s*khula|tood\s*kam)\b", "#marathi_praise"),
            (r"\b(mast|badhiya|gazab|kamaal|ekdum\s*mast|कतई\s*जहर)\b", "#hindi_praise"),
            (r"\b(masterpiece|chef'?s\s*kiss|recommend|flawless|top\s*tier)\b", "#recommendation")
        ]

        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for", "with", "on", "at",
            "by", "from", "in", "out", "about", "into", "through", "after", "it", "this", "that", "these", "those",
            "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their",
            "hai", "ki", "ko", "ka", "ke", "me", "ahe", "aahe", "nahi", "hota", "hote", "tar", "pan", "ani", "app"
        }

    def extract_tags(self, text: str, category: str = "", sentiment_label: str = "") -> List[str]:
        if not text:
            default_tags = []
            if category:
                default_tags.append(f"#{category.lower().split()[0].replace('/', '')}")
            if sentiment_label:
                default_tags.append(f"#{sentiment_label}")
            return default_tags

        tags_set: Set[str] = set()
        text_lower = text.lower()

        # 1. Extract explicit user hashtags already in text (e.g. #PaisaVasool, #Crash)
        raw_hashtags = re.findall(r"#([a-zA-Z0-9_]+)", text)
        for h in raw_hashtags:
            clean_tag = f"#{h.lower()}"
            if len(clean_tag) > 2:
                tags_set.add(clean_tag)

        # 2. Extract Domain Rule Tags
        for pattern, tag in self.domain_tag_rules:
            if re.search(pattern, text_lower):
                tags_set.add(tag)

        # 3. Add Category and Sentiment Context Tags
        if category:
            cat_slug = re.sub(r"[^a-zA-Z0-9]", "_", category.lower()).strip("_")
            cat_tag = f"#{cat_slug.split('_')[0]}"
            tags_set.add(cat_tag)

        if sentiment_label:
            tags_set.add(f"#{sentiment_label}")

        # 4. Extract Top Significant Keywords as Tags (length >= 4)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text_lower)
        for w in words:
            if w not in self.stopwords and len(tags_set) < 8:
                tag = f"#{w}"
                tags_set.add(tag)

        # Format and sort list
        sorted_tags = sorted(list(tags_set), key=lambda x: (not x.startswith("#app_"), not x.startswith("#meme_"), len(x)))
        return sorted_tags[:10]  # Cap at top 10 most relevant tags
