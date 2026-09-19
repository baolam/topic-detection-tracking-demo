import re
import math
from collections import Counter
from typing import List, Tuple, Optional

# Comprehensive English & Social Media Noise Stopwords
ENGLISH_STOPWORDS = {
    # Standard English Stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "isn't", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    # Social Media & Common Web Noise
    "rt", "http", "https", "com", "www", "amp", "via", "re", "co", "status", "user", "post", "tweet",
    "today", "yesterday", "tomorrow", "news", "update", "one", "two", "new", "get", "like", "also",
    "day", "please", "know", "see", "make", "say", "says", "said", "go", "going", "still", "back", "time",
    "first", "last", "much", "many", "well", "even", "us", "people", "need", "needs", "take", "help"
}

class TopicLabeler:
    """Extracts distinctive keywords using c-TF-IDF and builds clear English topic labels."""

    def __init__(self, stopwords: Optional[set] = None):
        self.stopwords = stopwords if stopwords is not None else ENGLISH_STOPWORDS

    def _tokenize(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extracts unigrams, bigrams, and hashtags from cleaned text."""
        # Extract hashtags
        hashtags = [h.lower() for h in re.findall(r"#(\w+)", text)]

        # Remove URLs, mentions, and non-alphanumeric chars
        clean_text = re.sub(r"http\S+|www\S+|@\w+|[^\w\s]", " ", text.lower())
        tokens = [
            w for w in clean_text.split()
            if len(w) > 2 and not w.isdigit() and w not in self.stopwords
        ]

        # Extract bigrams
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigrams.append(bigram)

        return tokens, bigrams, hashtags

    def extract_keywords_and_label(
        self,
        cluster_texts: List[str],
        all_background_texts: Optional[List[str]] = None,
        top_n: int = 5
    ) -> Tuple[List[str], str]:
        """
        Extracts top c-TF-IDF terms (unigrams + bigrams) and generates a readable topic label.
        """
        if not cluster_texts:
            return ["No data"], "Unidentified Topic"

        # 1. Collect terms from cluster texts
        cluster_unigrams = []
        cluster_bigrams = []
        cluster_hashtags = []

        for text in cluster_texts:
            unigrams, bigrams, hashtags = self._tokenize(text)
            cluster_unigrams.extend(unigrams)
            cluster_bigrams.extend(bigrams)
            cluster_hashtags.extend(hashtags)

        if not cluster_unigrams and not cluster_bigrams:
            return ["general_event"], "General Topic"

        # Count frequencies in cluster
        unigram_counts = Counter(cluster_unigrams)
        bigram_counts = Counter(cluster_bigrams)
        hashtag_counts = Counter(cluster_hashtags)

        # 2. Build background document frequencies for c-TF-IDF
        bg_df = Counter()
        bg_total_docs = 1

        if all_background_texts:
            bg_total_docs = max(1, len(all_background_texts))
            for text in all_background_texts:
                u_tokens, b_tokens, _ = self._tokenize(text)
                unique_terms = set(u_tokens + b_tokens)
                for term in unique_terms:
                    bg_df[term] += 1

        # 3. Calculate c-TF-IDF scores
        term_scores = {}

        # Score unigrams
        for term, tf in unigram_counts.items():
            df = bg_df.get(term, 1)
            idf = math.log((bg_total_docs + 1) / (df + 1)) + 1.0
            term_scores[term] = tf * idf

        # Score bigrams (boost bigram weight slightly for phrase readability)
        for term, tf in bigram_counts.items():
            df = bg_df.get(term, 1)
            idf = math.log((bg_total_docs + 1) / (df + 1)) + 1.0
            term_scores[term] = tf * idf * 1.5

        # Sort terms by c-TF-IDF score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [t for t, _ in sorted_terms[:top_n]]

        # 4. Generate human-readable Topic Label
        if hashtag_counts:
            best_hashtag = hashtag_counts.most_common(1)[0][0]
            label = f"Event #{best_hashtag.capitalize()}"
        elif sorted_terms:
            top_terms = [t.title() for t, _ in sorted_terms[:2]]
            label = f"Topic: {' & '.join(top_terms)}"
        else:
            label = "Emerging Event"

        return top_keywords, label
