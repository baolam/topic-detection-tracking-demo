import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class ClusteringEvaluator:
    """Evaluates Stream Clustering quality against ground-truth benchmark labels."""

    @staticmethod
    def calculate_purity(y_true: List[str], y_pred: List[int]) -> float:
        """Calculates Cluster Purity metric."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        clusters = np.unique(y_pred_arr)
        total_correct = 0

        for c in clusters:
            mask = (y_pred_arr == c)
            true_in_cluster = y_true_arr[mask]
            if len(true_in_cluster) == 0:
                continue
            # Find majority class in cluster
            unique_labels, counts = np.unique(true_in_cluster, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count

        return float(total_correct / len(y_true))

    @staticmethod
    def evaluate_benchmark(y_true: List[str], y_pred: List[int]) -> Dict[str, float]:
        """Calculates full benchmark metrics suite: Purity, NMI, ARI."""
        purity = ClusteringEvaluator.calculate_purity(y_true, y_pred)
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))

        return {
            "purity": round(purity, 4),
            "nmi": round(nmi, 4),
            "ari": round(ari, 4)
        }
