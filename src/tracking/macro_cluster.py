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
        all_background_texts = []
        for label, mc in zip(cluster_labels, p_micro_clusters):
            grouped.setdefault(label, []).append(mc)
            all_background_texts.extend(mc.sample_texts)

        topics: List[TrendingTopic] = []
        for idx, (label, mcs) in enumerate(grouped.items()):
            if label == -1 and len(p_micro_clusters) > 3:
                continue  # Skip unclustered noise if enough clusters exist

            # Aggregate texts & weights
            sample_texts = []
            total_weight = 0.0
            n_posts = 0
            burstiness_list = []
            for mc in mcs:
                total_weight += mc.weight
                n_posts += getattr(mc, 'n_posts', int(mc.weight))
                sample_texts.extend(mc.sample_texts)
                if hasattr(mc, 'get_burstiness'):
                    burstiness_list.append(mc.get_burstiness(time.time()))

            avg_burstiness = max(0.0, float(np.mean(burstiness_list))) if burstiness_list else 0.0

            # Calculate SADStream Burst Score: Score_k = w1*B_k + w2*N_k + w3*D_k + w4*U_k
            # w1=0.4 (burstiness), w2=0.3 (novelty/n_posts), w3=0.2 (density/weight), w4=0.1 (sample diversity)
            diversity = len(set(sample_texts)) / max(1, len(sample_texts))
            burst_score = 0.4 * avg_burstiness + 0.3 * (n_posts / 10.0) + 0.2 * total_weight + 0.1 * diversity

            # Generate keyword summary using c-TF-IDF
            keywords, topic_name = self.labeler.extract_keywords_and_label(
                sample_texts,
                all_background_texts=all_background_texts
            )

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
                    is_bursting=(burst_score > 2.0 or total_weight > 4.0),
                    burst_score=round(burst_score, 3)
                )
            )

        # Sort topics by burst score descending
        topics.sort(key=lambda t: t.burst_score, reverse=True)
        return topics

