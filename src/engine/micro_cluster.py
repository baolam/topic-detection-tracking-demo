import numpy as np
import uuid
import time
from typing import List, Optional

class MicroCluster:
    """Micro-Cluster feature (CF) with Exponential Time-Decay mechanism.

    Tuple: MC(t) = (Weight W, Linear Sum CF1, Square Sum CF2, Last Update Time t_c)
    Decay math: w(t) = 2^(-lambda * (t - t_0))
    """

    def __init__(self, first_point: np.ndarray, timestamp: float, cluster_id: Optional[str] = None):
        self.cluster_id = cluster_id or str(uuid.uuid4())[:8]
        self.creation_time = timestamp
        self.last_update_time = timestamp
        self.dim = len(first_point)

        # Statistical features
        self.weight: float = 1.0
        self.n_posts: int = 1
        self.CF1: np.ndarray = first_point.astype(np.float64).copy()
        self.CF2: np.ndarray = (first_point.astype(np.float64) ** 2)

        # Activity & Burst tracking for SADStream adaptive decay
        self.arrival_timestamps: List[float] = [timestamp]
        self.activity_history: List[float] = []

        # Auxiliary features (Keywords, Entities, Hashtags)
        self.keywords: set = set()
        self.entities: set = set()
        self.hashtags: set = set()

        # Track sample texts associated with this cluster
        self.sample_texts: List[str] = []

    def get_recent_activity(self, current_time: float, lambda_base: float = 0.03) -> float:
        """Calculates recent activity A_k(t) = sum_{t_i} e^(-lambda * (t - t_i))."""
        if not self.arrival_timestamps:
            return 0.0
        decay_sum = 0.0
        for t_i in self.arrival_timestamps:
            dt = max(0.0, current_time - t_i)
            decay_sum += np.exp(-lambda_base * dt)
        return float(decay_sum)

    def get_burstiness(self, current_time: float, lambda_base: float = 0.03) -> float:
        """Calculates normalized burstiness B_k(t) = (A_k(t) - mu_A) / (sigma_A + eps)."""
        act = self.get_recent_activity(current_time, lambda_base)
        self.activity_history.append(act)
        if len(self.activity_history) > 30:
            self.activity_history.pop(0)
        
        if len(self.activity_history) < 2:
            return 0.0
        
        mu_a = np.mean(self.activity_history)
        sigma_a = np.std(self.activity_history)
        return float((act - mu_a) / (sigma_a + 1e-5))

    def get_adaptive_lambda(self, current_time: float, lambda_0: float = 0.03, eta: float = 0.5, rho: float = 0.5) -> float:
        """
        Calculates cluster-specific adaptive decay rate:
        lambda_k(t) = lambda_0 / (1 + eta * B_k(t) + rho * A_k(t))
        Active or bursty topics decay more slowly (smaller lambda_k).
        """
        a_k = self.get_recent_activity(current_time, lambda_0)
        b_k = max(0.0, self.get_burstiness(current_time, lambda_0))
        denom = 1.0 + eta * b_k + rho * a_k
        adaptive_lambda = lambda_0 / max(0.01, denom)
        return float(adaptive_lambda)

    def decay(self, current_time: float, lambda_decay: float) -> None:
        """Applies exponential time-decay to cluster weight and feature vectors."""
        dt = current_time - self.last_update_time
        if dt > 0 and lambda_decay > 0:
            decay_factor = np.exp(-lambda_decay * dt)
            self.weight *= decay_factor
            self.CF1 *= decay_factor
            self.CF2 *= decay_factor
            self.last_update_time = current_time

    def decay_adaptive(self, current_time: float, lambda_0: float = 0.03, eta: float = 0.5, rho: float = 0.5) -> float:
        """Applies cluster-specific adaptive decay rate lambda_k(t). Returns used lambda."""
        lamb_k = self.get_adaptive_lambda(current_time, lambda_0, eta, rho)
        self.decay(current_time, lamb_k)
        return lamb_k

    def add_point(
        self,
        point: np.ndarray,
        timestamp: float,
        lambda_decay: float = 0.03,
        text: Optional[str] = None,
        is_adaptive: bool = False,
        lambda_0: float = 0.03,
        eta: float = 0.5,
        rho: float = 0.5
    ) -> None:
        """Decays cluster feature first, then incrementally adds the new point."""
        if is_adaptive:
            self.decay_adaptive(timestamp, lambda_0, eta, rho)
        else:
            self.decay(timestamp, lambda_decay)

        p = point.astype(np.float64)
        
        self.weight += 1.0
        self.n_posts += 1
        self.CF1 += p
        self.CF2 += (p ** 2)
        self.last_update_time = timestamp
        self.arrival_timestamps.append(timestamp)
        if len(self.arrival_timestamps) > 50:
            self.arrival_timestamps.pop(0)

        if text:
            if text not in self.sample_texts:
                self.sample_texts.append(text)
                if len(self.sample_texts) > 10:
                    self.sample_texts.pop(0)

    def get_center(self) -> np.ndarray:
        """Calculates the weighted centroid vector."""
        if self.weight <= 1e-9:
            return np.zeros(self.dim, dtype=np.float32)
        center = self.CF1 / self.weight
        norm = np.linalg.norm(center)
        if norm > 1e-6:
            center = center / norm
        return center.astype(np.float32)

    def get_radius(self) -> float:
        """Calculates the root mean square radius of the micro-cluster."""
        if self.weight <= 1.0:
            return 0.0
        center = self.CF1 / self.weight
        variance = (self.CF2 / self.weight) - (center ** 2)
        variance = np.maximum(variance, 0.0)
        radius = np.sqrt(np.sum(variance))
        return float(radius)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculates Cosine distance between cluster center and point vector."""
        center = self.get_center()
        norm_center = np.linalg.norm(center)
        norm_point = np.linalg.norm(point)
        if norm_center < 1e-6 or norm_point < 1e-6:
            return 1.0
        cosine_sim = np.dot(center, point) / (norm_center * norm_point)
        return float(1.0 - max(min(cosine_sim, 1.0), -1.0))

