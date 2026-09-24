import re
from typing import Dict, Any, List, Tuple

class DomainCategorizer:
    """
    Intelligent domain and topic categorizer for multimodal text and files.
    Supports English, Hindi, Marathi, Hinglish, and Maranglish.
    """
    def __init__(self):
        self.categories = {
            "Meme / Social Humor": {
                "keywords": [
                    "meme", "memes", "cooked", "bro cooked", "who let him cook", "aura", "infinite aura",
                    "ratio", "skill issue", "demure", "brainrot", "gigachad", "wojak", "no cap", "ong",
                    "fr fr", "💀", "😭", "😂", "🤣", "lmao", "lol", "peak", "tood kam", "nadach khula",
                    "wa re wa", "banger"
                ],
                "weight": 1.2
            },
            "Finance & Payment Issue": {
                "keywords": [
                    "payment", "payment failed", "transaction failed", "money debited", "refund", "transaction",
                    "deducted", "paise cut", "paise fukat", "paise doob", "billing", "invoice", "receipt",
                    "overcharged", "subscription", "bank", "wallet", "upi", "card", "paisa vasool",
                    "पैसे कट", "लूट लिया", "बुडवले", "scam", "confirm payment", "upi pin", "bank server"
                ],
                "weight": 1.4
            },
            "Tech & Software Bugs": {
                "keywords": [
                    "technical issue", "technical", "issue", "issues", "problem", "problems", "defect",
                    "crash", "crashed", "crashing", "bug", "bugs", "glitch", "freeze", "hang", "restart",
                    "restarting", "lag", "latency", "memory leak", "timeout", "timed out", "error", "exception",
                    "nullpointer", "stack trace", "watt laga", "band padto", "chalat nahi", "काम नहीं करता",
                    "अटक", "हँग", "क्रैश", "connection pool", "500 internal", "database", "not working", "failed"
                ],
                "weight": 1.2
            },
            "Customer Support & Service": {
                "keywords": [
                    "customer support", "customer care", "helpdesk", "support team", "ticket", "complaint",
                    "service", "agent", "executive", "no response", "ignored", "hold time", "call center",
                    "ghatiya service", "worst service", "काही मदत नाही", "सर्व्हिस"
                ],
                "weight": 1.2
            },
            "Product & E-Commerce Review": {
                "keywords": [
                    "review", "rating", "build quality", "battery", "camera", "display", "unboxing", "delivery",
                    "product", "packaging", "worth buying", "recommend", "order", "flipkart", "amazon",
                    "ek number", "lai bhari", "khup chhan", "chhan", "mast", "badhiya", "घटिया", "खूप छान"
                ],
                "weight": 1.1
            },
            "Entertainment & Media": {
                "keywords": [
                    "movie", "song", "trailer", "teaser", "music", "album", "gaming", "gameplay", "actor",
                    "actress", "cinema", "theatre", "netflix", "youtube", "cricket", "match", "गाणं", "चित्रपट"
                ],
                "weight": 1.1
            },
            "News & Corporate Announcement": {
                "keywords": [
                    "press release", "quarterly", "financial statements", "scheduled maintenance", "policy",
                    "official notice", "announcement", "board of directors", "investor relations", "मौसम",
                    "शासनाने", "नियमावली", "प्रसिद्ध करण्यात"
                ],
                "weight": 1.3
            }
        }

    def classify(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        text_lower = (text or "").lower()
        meta = metadata or {}
        
        # 1. Specialized Visual Signature Override: Payment App Failure UI
        is_payment_ui = meta.get("is_payment_ui", False)
        if is_payment_ui and any(x in text_lower for x in ["technical issue", "failed", "support", "try again", "payment", "bank", "request"]):
            gateway_name = meta.get("detected_gateway") or "UPI Gateway"
            return {
                "category": "Finance & Payment Issue",
                "subcategory": f"Payment Failure ({gateway_name})",
                "confidence": 0.98,
                "matched_keywords": ["payment_app_theme", "technical_issue_modal"]
            }

        if not text_lower:
            media_type = meta.get("media_type", "image")
            if is_payment_ui:
                return {"category": "Finance & Payment Issue", "subcategory": "Payment Application Screen", "confidence": 0.90, "matched_keywords": ["payment_theme"]}
            if media_type == "video":
                return {"category": "Entertainment & Media", "subcategory": "Video Clip", "confidence": 0.50, "matched_keywords": []}
            return {"category": "General Social", "subcategory": "Visual Media", "confidence": 0.50, "matched_keywords": []}

        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}

        for cat, data in self.categories.items():
            cat_score = 0.0
            cat_matches = []
            for kw in data["keywords"]:
                if kw in text_lower:
                    cat_score += data["weight"]
                    cat_matches.append(kw)
            if cat_score > 0:
                scores[cat] = cat_score
                matched[cat] = cat_matches

        if not scores:
            return {
                "category": "General Social",
                "subcategory": "Conversational Post",
                "confidence": 0.65,
                "matched_keywords": []
            }

        best_cat = max(scores, key=scores.get)
        raw_score = scores[best_cat]
        confidence = min(0.99, round(0.70 + (raw_score * 0.06), 2))

        subcategories = {
            "Meme / Social Humor": "Viral Internet Meme" if any(x in text_lower for x in ["meme", "cooked", "aura", "ratio"]) else "Humorous Reaction",
            "Finance & Payment Issue": "UPI & Payment Gateway Failure" if any(x in text_lower for x in ["debit", "failed", "deduct", "cut", "technical issue"]) else "Pricing & Invoice",
            "Tech & Software Bugs": "Crash & Stability" if any(x in text_lower for x in ["crash", "leak", "restart", "500"]) else "Functional Defect",
            "Customer Support & Service": "Escalated Grievance" if any(x in text_lower for x in ["worst", "ghatiya", "rude", "no response"]) else "General Inquiry",
            "Product & E-Commerce Review": "Positive Endorsement" if any(x in text_lower for x in ["best", "paisa vasool", "recommend", "chhan", "mast"]) else "Critique",
            "Entertainment & Media": "Music & Audio" if "song" in text_lower or "गाणं" in text_lower else "Pop Culture & Gaming",
            "News & Corporate Announcement": "Official Statement" if "press release" in text_lower or "शासनाने" in text_lower else "General Notice"
        }
        subcategory = subcategories.get(best_cat, "General Discussion")

        return {
            "category": best_cat,
            "subcategory": subcategory,
            "confidence": confidence,
            "matched_keywords": matched[best_cat]
        }
