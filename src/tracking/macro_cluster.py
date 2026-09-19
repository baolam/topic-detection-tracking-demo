import time
import numpy as np
from typing import List
from sklearn.cluster import DBSCAN
from src.engine.micro_cluster import MicroCluster
from src.data.schemas import TrendingTopic
from src.tracking.topic_labeler import TopicLabeler

class MacroClusterEngine:
    """Performs Macro-clustering over potential Micro-clusters to extract active Trending Topics."""

    def __init__(self, eps_macro: float = 0.40, min_samples: int = 1):
        self.eps_macro = eps_macro
        self.min_samples = min_samples
        self.labeler = TopicLabeler()

    def generate_topics(self, p_micro_clusters: List[MicroCluster]) -> List[TrendingTopic]:
        """Groups potential micro-clusters into macro-cluster trending topics."""
        if not p_micro_clusters:
            return []

        # Extract center vectors
        centers = np.array([mc.get_center() for mc in p_micro_clusters])

        # Run DBSCAN with Cosine metric
        db = DBSCAN(eps=self.eps_macro, min_samples=self.min_samples, metric="cosine")
        cluster_labels = db.fit_predict(centers)

        # Group micro-clusters by macro-label
        grouped: dict[int, list[MicroCluster]] = {}
        for label, mc in zip(cluster_labels, p_micro_clusters):
            grouped.setdefault(label, []).append(mc)

        topics: List[TrendingTopic] = []
        for idx, (label, mcs) in enumerate(grouped.items()):
            if label == -1 and len(p_micro_clusters) > 3:
                continue  # Skip unclustered noise if enough clusters exist

            # Aggregate texts & weights
            sample_texts = []
            total_weight = 0.0
            for mc in mcs:
                total_weight += mc.weight
                sample_texts.extend(mc.sample_texts)

            # Generate keyword summary
            keywords, topic_name = self.labeler.extract_keywords_and_label(sample_texts)

            topic_id = f"TOPIC-{idx+1:02d}"
            topics.append(
                TrendingTopic(
                    topic_id=topic_id,
                    label=topic_name,
                    keywords=keywords,
                    sample_posts=sample_texts[:3],
                    total_weight=round(total_weight, 2),
                    micro_cluster_count=len(mcs),
                    last_updated=time.time(),
                    is_bursting=(total_weight > 5.0)
                )
            )

        # Sort topics by weight descending
        topics.sort(key=lambda t: t.total_weight, reverse=True)
        return topics
