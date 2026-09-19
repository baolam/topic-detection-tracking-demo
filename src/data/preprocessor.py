import re

class TextPreprocessor:
    """Preprocesses raw social media text for real-time embedding extraction."""

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r"@\w+", "", text)
        # Standardize spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_hashtags(text: str) -> list[str]:
        return re.findall(r"#(\w+)", text)
