import time
import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from src.engine.micro_cluster import MicroCluster
from src.data.schemas import SocialPost

@dataclass
class CompressedPrototype:
    """Compressed Long-Term Memory Prototype L_k for expired/inactive topics."""
    cluster_id: str
    centroid: np.ndarray
    keywords: set
    entities: set
    hashtags: set
    t_start: float
    t_end: float

class SADStreamEngine:
    """
    SADStream: Adaptive Semantic Stream Clustering Engine.
    Combines sentence-level embeddings, cluster-specific adaptive decay rate lambda_k(t),
    compressed long-term prototype memory, and hybrid text-feature similarity.
    """

    def __init__(
        self,
        lambda_0: float = 0.03,
        eta: float = 0.5,
        rho: float = 0.5,
        epsilon: float = 0.38,
        mu: float = 2.5,
        beta: float = 0.3,
        theta_R: float = 0.60,
        theta_inactive: float = 0.2,
        alpha: float = 0.7,
        beta_lex: float = 0.1,
        gamma_ent: float = 0.1,
        delta_hash: float = 0.1
    ):
        self.lambda_0 = lambda_0
        self.eta = eta
        self.rho = rho
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.theta_R = theta_R
        self.theta_inactive = theta_inactive

        # Hybrid similarity weights
        self.alpha = alpha
        self.beta_lex = beta_lex
        self.gamma_ent = gamma_ent
        self.delta_hash = delta_hash

        self.p_micro_clusters: List[MicroCluster] = []
        self.o_micro_clusters: List[MicroCluster] = []
        self.compressed_archive: List[CompressedPrototype] = []

        self.last_pruning_time: float = time.time()
        self.total_processed: int = 0
        self.recurring_topic_count: int = 0

    def _extract_lexical_features(self, text: str) -> Tuple[set, set, set]:
        """Extracts words, entities (capitalized words), and hashtags."""
        hashtags = set(re.findall(r"#(\w+)", text.lower()))
        entities = set(re.findall(r"\b[A-Z][a-z]+\b", text))
        words = set(w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text))
        return words, entities, hashtags

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """Calculates Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def calculate_hybrid_similarity(self, vector: np.ndarray, text: str, mc: MicroCluster) -> float:
        """
        Calculates hybrid similarity S(x, C_k) = alpha*S_sem + beta*S_lex + gamma*S_ent + delta*S_hash.
        """
        # 1. Semantic Cosine Similarity
        dist = mc.distance_to_point(vector)
        s_sem = max(0.0, 1.0 - dist)

        # 2. Extract text features
        words, entities, hashtags = self._extract_lexical_features(text)

        # 3. Lexical, Entity, Hashtag Jaccard Similarities
        s_lex = self._jaccard_similarity(words, mc.keywords)
        s_ent = self._jaccard_similarity(entities, mc.entities)
        s_hash = self._jaccard_similarity(hashtags, mc.hashtags)

        hybrid_sim = (
            self.alpha * s_sem +
            self.beta_lex * s_lex +
            self.gamma_ent * s_ent +
            self.delta_hash * s_hash
        )
        return float(hybrid_sim)

    def process_post(self, post: SocialPost, vector: np.ndarray) -> str:
        """Processes an incoming post into micro-clusters using SADStream adaptive logic."""
        self.total_processed += 1
        curr_time = post.timestamp
        words, entities, hashtags = self._extract_lexical_features(post.text)

        # Check for recurring topic in compressed long-term archive
        recovered_cluster_id = self._check_recurring_topic(vector, words, entities, hashtags)

        # Step 1: Try merging into nearest Potential Micro-Cluster (p-MC)
        merged_p = self._try_merge_adaptive(self.p_micro_clusters, vector, post.text, curr_time, words, entities, hashtags)
        if merged_p:
            self._check_prune(curr_time)
            return "merged_p_mc"

        # Step 2: Try merging into nearest Outlier Micro-Cluster (o-MC)
        merged_o = self._try_merge_adaptive(self.o_micro_clusters, vector, post.text, curr_time, words, entities, hashtags)
        if merged_o:
            if merged_o.weight > self.beta * self.mu:
                self.o_micro_clusters.remove(merged_o)
                self.p_micro_clusters.append(merged_o)
                self._check_prune(curr_time)
                return "promoted_to_p_mc"
            self._check_prune(curr_time)
            return "merged_o_mc"

        # Step 3: Create a new Outlier Micro-Cluster
        cid = recovered_cluster_id if recovered_cluster_id else None
        new_o_mc = MicroCluster(vector, curr_time, cluster_id=cid)
        new_o_mc.keywords.update(words)
        new_o_mc.entities.update(entities)
        new_o_mc.hashtags.update(hashtags)
        if post.text not in new_o_mc.sample_texts:
            new_o_mc.sample_texts.append(post.text)
        self.o_micro_clusters.append(new_o_mc)

        self._check_prune(curr_time)
        return "created_recurring_o_mc" if recovered_cluster_id else "created_new_o_mc"

    def _try_merge_adaptive(
        self,
        clusters: List[MicroCluster],
        vector: np.ndarray,
        text: str,
        curr_time: float,
        words: set,
        entities: set,
        hashtags: set
    ) -> Optional[MicroCluster]:
        """Finds closest cluster using hybrid similarity and merges if distance <= epsilon."""
        if not clusters:
            return None

        # Find cluster with highest hybrid similarity
        best_mc = max(clusters, key=lambda mc: self.calculate_hybrid_similarity(vector, text, mc))
        best_sim = self.calculate_hybrid_similarity(vector, text, best_mc)
        best_dist = 1.0 - best_sim

        if best_dist <= self.epsilon:
            best_mc.add_point(
                point=vector,
                timestamp=curr_time,
                is_adaptive=True,
                lambda_0=self.lambda_0,
                eta=self.eta,
                rho=self.rho,
                text=text
            )
            best_mc.keywords.update(words)
            best_mc.entities.update(entities)
            best_mc.hashtags.update(hashtags)
            return best_mc
        return None

    def _check_recurring_topic(self, vector: np.ndarray, words: set, entities: set, hashtags: set) -> Optional[str]:
        """Checks if a new point matches an archived compressed prototype (Recurring Topic)."""
        if not self.compressed_archive:
            return None

        best_proto = None
        best_sim = 0.0

        for proto in self.compressed_archive:
            # Semantic cosine similarity
            norm_c = np.linalg.norm(proto.centroid)
            norm_v = np.linalg.norm(vector)
            sem_sim = np.dot(proto.centroid, vector) / (norm_c * norm_v + 1e-6) if norm_c > 0 and norm_v > 0 else 0.0
            
            lex_sim = self._jaccard_similarity(words, proto.keywords)
            ent_sim = self._jaccard_similarity(entities, proto.entities)
            hash_sim = self._jaccard_similarity(hashtags, proto.hashtags)

            total_sim = self.alpha * sem_sim + self.beta_lex * lex_sim + self.gamma_ent * ent_sim + self.delta_hash * hash_sim

            if total_sim > best_sim:
                best_sim = total_sim
                best_proto = proto

        if best_sim >= self.theta_R and best_proto:
            self.recurring_topic_count += 1
            return best_proto.cluster_id
        return None

    def _check_prune(self, current_time: float) -> None:
        """Periodically evicts inactive clusters into compressed prototype archive."""
        if current_time - self.last_pruning_time >= 5.0:
            self.prune(current_time)
            self.last_pruning_time = current_time

    def prune(self, current_time: float) -> None:
        """Decays active clusters with adaptive lambda_k(t) and archives expired ones."""
        # 1. Decay all micro-clusters adaptively
        for mc in self.p_micro_clusters:
            mc.decay_adaptive(current_time, self.lambda_0, self.eta, self.rho)
        for mc in self.o_micro_clusters:
            mc.decay_adaptive(current_time, self.lambda_0, self.eta, self.rho)

        # 2. Archive expired Outlier Micro-Clusters (W_k < theta_inactive)
        active_o = []
        for mc in self.o_micro_clusters:
            if mc.weight < self.theta_inactive:
                proto = CompressedPrototype(
                    cluster_id=mc.cluster_id,
                    centroid=mc.get_center(),
                    keywords=set(mc.keywords),
                    entities=set(mc.entities),
                    hashtags=set(mc.hashtags),
                    t_start=mc.creation_time,
                    t_end=mc.last_update_time
                )
                self.compressed_archive.append(proto)
            else:
                active_o.append(mc)
        self.o_micro_clusters = active_o

        # Limit archive size (bounded-memory requirement)
        if len(self.compressed_archive) > 200:
            self.compressed_archive = self.compressed_archive[-200:]

    def get_potential_clusters(self) -> List[MicroCluster]:
        """Returns potential micro-clusters with sufficient weight."""
        return [mc for mc in self.p_micro_clusters if mc.weight >= self.beta * self.mu]
