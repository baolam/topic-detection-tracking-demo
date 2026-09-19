import numpy as np
import time
from src.engine.micro_cluster import MicroCluster
from src.engine.denstream import DenStreamEngine
from src.data.schemas import SocialPost

def test_time_decay_math():
    """Verify exponential time-decay reduces cluster weight correctly."""
    v1 = np.ones(384, dtype=np.float32)
    t0 = time.time()
    mc = MicroCluster(v1, t0)
    assert mc.weight == 1.0

    # Simulate 10 seconds decay with lambda = 0.1
    # Decay factor = 2^(-0.1 * 10) = 2^(-1) = 0.5
    t1 = t0 + 10.0
    mc.decay(t1, lambda_decay=0.1)
    assert abs(mc.weight - 0.5) < 1e-4

def test_denstream_outlier_promotion():
    """Verify outlier micro-clusters evolve into potential micro-clusters when burst occurs."""
    engine = DenStreamEngine(lambda_decay=0.01, epsilon=0.5, mu=2.0, beta=0.3)
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # Process 5 identical posts on same topic
    t = time.time()
    for i in range(5):
        post = SocialPost(post_id=str(i), timestamp=t + i * 0.1, text="Storm event warning!")
        engine.process_post(post, vec)

    # Should have promoted to potential micro-cluster
    assert len(engine.p_micro_clusters) >= 1
    print("\n[TEST PASSED] DenStream Outlier Promotion Test Success!")

if __name__ == "__main__":
    test_time_decay_math()
    test_denstream_outlier_promotion()
