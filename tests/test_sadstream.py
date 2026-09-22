import numpy as np
import time
from src.data.schemas import SocialPost
from src.engine.micro_cluster import MicroCluster
from src.engine.sadstream_engine import SADStreamEngine, CompressedPrototype

def test_sadstream_adaptive_decay():
    """Verify adaptive decay rate lambda_k(t) adjusts based on cluster activity and burstiness."""
    v1 = np.ones(384, dtype=np.float32)
    t0 = time.time()
    mc = MicroCluster(v1, t0)

    # Initial adaptive decay rate should equal base lambda_0 when no history
    lamb_init = mc.get_adaptive_lambda(t0, lambda_0=0.03, eta=0.5, rho=0.5)
    assert lamb_init > 0

    # Add rapid posts to trigger high activity/burstiness
    for i in range(10):
        mc.add_point(v1, t0 + i * 0.1, is_adaptive=True, lambda_0=0.03, eta=0.5, rho=0.5)

    # Adaptive decay rate for bursty topic should be lower than base lambda_0 (slower decay)
    lamb_burst = mc.get_adaptive_lambda(t0 + 1.0, lambda_0=0.03, eta=0.5, rho=0.5)
    assert lamb_burst < 0.03

def test_sadstream_hybrid_similarity():
    """Verify hybrid similarity calculation combines semantic cosine and lexical features."""
    engine = SADStreamEngine(alpha=0.7, beta_lex=0.1, gamma_ent=0.1, delta_hash=0.1)
    
    vec1 = np.array([1.0] + [0.0]*383, dtype=np.float32)
    mc = MicroCluster(vec1, time.time())
    mc.keywords.update(["flood", "disaster"])
    mc.hashtags.update(["rescue"])

    text_match = "Severe flood disaster occurring #rescue"
    sim = engine.calculate_hybrid_similarity(vec1, text_match, mc)
    
    # Hybrid similarity should be higher than pure semantic similarity alone
    assert sim > 0.7

def test_sadstream_recurring_topic_archiving():
    """Verify expired outlier micro-clusters move to compressed archive and recover on recurring topic."""
    engine = SADStreamEngine(lambda_0=0.05, theta_inactive=0.5, theta_R=0.50)
    
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)

    post1 = SocialPost(post_id="1", timestamp=100.0, text="Breaking hurricane warning #hurricane")
    engine.process_post(post1, vec)

    assert len(engine.o_micro_clusters) == 1

    # Force pruning at a much later timestamp to decay and archive the inactive cluster
    engine.prune(current_time=200.0)
    
    assert len(engine.o_micro_clusters) == 0
    assert len(engine.compressed_archive) == 1
    assert engine.compressed_archive[0].hashtags == {"hurricane"}

    # Process new post matching archived prototype topic
    post2 = SocialPost(post_id="2", timestamp=205.0, text="Another hurricane warning #hurricane")
    status = engine.process_post(post2, vec)

    assert status == "created_recurring_o_mc"
    assert engine.recurring_topic_count == 1
