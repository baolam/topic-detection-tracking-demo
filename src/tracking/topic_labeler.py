import re
from collections import Counter
from typing import List, Tuple

STOPWORDS_VI = {
    "là", "và", "của", "với", "đã", "đang", "sẽ", "cho", "trong", "khi",
    "được", "có", "này", "vẫn", "được", "người", "những", "các", "tại", "theo",
    "post", "user", "hôm", "nay", "thì", "mà", "từ", "trên", "ra", "đến"
}

class TopicLabeler:
    """Extracts top keywords and builds readable topic labels from sample texts."""

    def extract_keywords_and_label(self, texts: List[str], top_n: int = 5) -> Tuple[List[str], str]:
        if not texts:
            return ["No data"], "Chủ đề chưa xác định"

        # Tokenize and count words
        words = []
        hashtags = []
        for text in texts:
            # Extract hashtags
            ht = re.findall(r"#(\w+)", text)
            hashtags.extend([h.lower() for h in ht])

            # Extract words
            clean_text = re.sub(r"[^\w\s]", " ", text.lower())
            tokens = clean_text.split()
            filtered = [w for w in tokens if len(w) > 2 and w not in STOPWORDS_VI and not w.isdigit()]
            words.extend(filtered)

        # Prioritize hashtags + frequent words
        counter = Counter(words)
        hashtag_counter = Counter(hashtags)

        # Top keywords
        top_words = [w for w, _ in counter.most_common(top_n)]
        
        # Build readable label
        if hashtag_counter:
            best_tag = hashtag_counter.most_common(1)[0][0]
            label = f"Sự kiện #{best_tag.capitalize()}"
        elif top_words:
            label = f"Chủ đề: {', '.join(top_words[:3]).capitalize()}"
        else:
            label = "Chủ đề Mạng Xã hội"

        return top_words, label
