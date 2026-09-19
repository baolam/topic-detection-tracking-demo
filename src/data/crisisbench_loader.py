import time
from typing import Generator, Optional, List, Dict, Any
from src.data.schemas import SocialPost

class CrisisBenchLoader:
    """Loads and streams real disaster tweets from QCRI/CrisisBench-english (humanitarian) via HuggingFace datasets."""

    def __init__(self, split: str = "test", max_samples: Optional[int] = None):
        self.split = split
        self.max_samples = max_samples
        self.dataset = None
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        try:
            from datasets import load_dataset
            print(f"[CrisisBench] Loading QCRI/CrisisBench-english ('humanitarian' subset, split='{self.split}')...")
            full_ds = load_dataset("QCRI/CrisisBench-english", "humanitarian")
            
            # Fallback to train/test split
            if self.split in full_ds:
                self.dataset = full_ds[self.split]
            else:
                available_split = list(full_ds.keys())[0]
                self.dataset = full_ds[available_split]

            if self.max_samples and len(self.dataset) > self.max_samples:
                self.dataset = self.dataset.select(range(self.max_samples))

            print(f"[CrisisBench] Successfully loaded {len(self.dataset)} crisis posts from HuggingFace.")
        except Exception as e:
            print(f"[CrisisBench ERROR] Failed to load dataset from HuggingFace: {e}")
            self.dataset = None

    def stream_posts(self, delay_sec: float = 0.05) -> Generator[SocialPost, None, None]:
        """Yields SocialPost objects from CrisisBench stream."""
        if self.dataset is None:
            print("[CrisisBench] Dataset is empty or not loaded.")
            return

        base_time = time.time()
        for idx, row in enumerate(self.dataset):
            # Resolve text column
            text = row.get("tweet_text") or row.get("text") or row.get("tweet") or ""
            if not text:
                continue

            # Resolve ground truth label
            label = row.get("label_text") or row.get("label") or row.get("class_label") or "unknown_humanitarian"
            if isinstance(label, int):
                label = f"category_{label}"

            post_id = str(row.get("tweet_id") or f"cb_{idx}")
            
            post = SocialPost(
                post_id=post_id,
                timestamp=base_time + idx * 0.5,  # Simulated streaming timestamp delta
                text=text,
                author_id=f"crisis_reporter_{idx % 20}",
                category_hint=str(label)
            )

            yield post
            if delay_sec > 0:
                time.sleep(delay_sec)
