import time
import numpy as np
from typing import List, Tuple, Optional
from src.engine.micro_cluster import MicroCluster
from src.data.schemas import SocialPost

class DenStreamEngine:
    """DenStream Online Stream Clustering Engine with Time-Decay and Memory Pruning."""

    def __init__(
        self,
        lambda_decay: float = 0.03,
        epsilon: float = 0.38,
        mu: float = 2.5,
        beta: float = 0.3,
        pruning_period: float = 5.0
    ):
        self.lambda_decay = lambda_decay
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.pruning_period = pruning_period

        self.p_micro_clusters: List[MicroCluster] = []
        self.o_micro_clusters: List[MicroCluster] = []

        self.last_prun_time: float = time.time()
        self.total_processed: int = 0

    def process_post(self, post: SocialPost, vector: np.ndarray) -> str:
        """Processes an incoming social media post vector into micro-clusters."""
        self.total_processed += 1
        curr_time = post.timestamp

        # Step 1: Try merging into nearest Potential Micro-Cluster (p-MC)
        merged = self._try_merge(self.p_micro_clusters, vector, curr_time, post.text)
        if merged:
            self._check_prune(curr_time)
            return "merged_p_mc"

        # Step 2: Try merging into nearest Outlier Micro-Cluster (o-MC)
        merged_o = self._try_merge(self.o_micro_clusters, vector, curr_time, post.text)
        if merged_o:
            # Check if o-MC evolved into p-MC!
            o_mc = merged_o
            if o_mc.weight > self.beta * self.mu:
                self.o_micro_clusters.remove(o_mc)
                self.p_micro_clusters.append(o_mc)
                print(f"[DenStream EVENT] Outlier Cluster {o_mc.cluster_id} evolved into Potential Micro-Cluster! Weight={o_mc.weight:.2f}")
                self._check_prune(curr_time)
                return "promoted_to_p_mc"
            self._check_prune(curr_time)
            return "merged_o_mc"

        # Step 3: Create a new Outlier Micro-Cluster (o-MC)
        new_o_mc = MicroCluster(vector, curr_time)
        if post.text not in new_o_mc.sample_texts:
            new_o_mc.sample_texts.append(post.text)
        self.o_micro_clusters.append(new_o_mc)

        self._check_prune(curr_time)
        return "created_new_o_mc"

    def _try_merge(
        self, clusters: List[MicroCluster], vector: np.ndarray, curr_time: float, text: str
    ) -> Optional[MicroCluster]:
        """Finds nearest cluster and attempts merge if distance <= epsilon."""
        if not clusters:
            return None

        # Find closest cluster by cosine distance
        closest_mc = min(clusters, key=lambda mc: mc.distance_to_point(vector))
        dist = closest_mc.distance_to_point(vector)

        if dist <= self.epsilon:
            closest_mc.add_point(vector, curr_time, self.lambda_decay, text)
            return closest_mc
        return None

    def _check_prune(self, current_time: float) -> None:
        """Periodically prunes old/expired outlier micro-clusters."""
        if current_time - self.last_prun_time >= self.pruning_period:
            self.prune(current_time)
            self.last_prun_time = current_time

    def prune(self, current_time: float) -> None:
        """Prunes decayed outlier micro-clusters to free RAM."""
        # Decay all clusters
        for mc in self.p_micro_clusters:
            mc.decay(current_time, self.lambda_decay)
        for mc in self.o_micro_clusters:
            mc.decay(current_time, self.lambda_decay)

        # Evict decayed o-MCs with weight < threshold
        min_weight_threshold = 0.2
        initial_o_count = len(self.o_micro_clusters)
        self.o_micro_clusters = [mc for mc in self.o_micro_clusters if mc.weight >= min_weight_threshold]
        evicted = initial_o_count - len(self.o_micro_clusters)

        if evicted > 0:
            print(f"[Eviction Policy] Pruned {evicted} expired Outlier Micro-Clusters.")

    def get_potential_clusters(self) -> List[MicroCluster]:
        """Returns active potential micro-clusters."""
        return [mc for mc in self.p_micro_clusters if mc.weight >= self.beta * self.mu]
