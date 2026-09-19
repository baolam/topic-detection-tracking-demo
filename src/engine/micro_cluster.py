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
        self.CF1: np.ndarray = first_point.astype(np.float64).copy()
        self.CF2: np.ndarray = (first_point.astype(np.float64) ** 2)

        # Track sample texts associated with this cluster
        self.sample_texts: List[str] = []

    def decay(self, current_time: float, lambda_decay: float) -> None:
        """Applies exponential time-decay to cluster weight and feature vectors."""
        dt = current_time - self.last_update_time
        if dt > 0 and lambda_decay > 0:
            decay_factor = 2.0 ** (-lambda_decay * dt)
            self.weight *= decay_factor
            self.CF1 *= decay_factor
            self.CF2 *= decay_factor
            self.last_update_time = current_time

    def add_point(self, point: np.ndarray, timestamp: float, lambda_decay: float, text: Optional[str] = None) -> None:
        """Decays cluster feature first, then incrementally adds the new point."""
        self.decay(timestamp, lambda_decay)
        p = point.astype(np.float64)
        
        self.weight += 1.0
        self.CF1 += p
        self.CF2 += (p ** 2)
        self.last_update_time = timestamp

        if text:
            self.sample_texts.append(text)
            if len(self.sample_texts) > 10:
                self.sample_texts.pop(0)  # Keep recent 10 sample texts

    def get_center(self) -> np.ndarray:
        """Calculates the weighted centroid vector."""
        if self.weight <= 1e-9:
            return np.zeros(self.dim, dtype=np.float32)
        center = self.CF1 / self.weight
        # Normalize center for cosine space consistency
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
        variance = np.maximum(variance, 0.0)  # Numerical safety clip
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
