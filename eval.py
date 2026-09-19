import sys
import time
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.denstream import DenStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine
from src.evaluation.evaluator import ClusteringEvaluator
from src.utils.logger import setup_logger

def load_config():
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def main():
    logger = setup_logger("BenchmarkEvaluator")
    logger.info("Starting Evaluation on HuggingFace Dataset: QCRI/CrisisBench-english (humanitarian)...")

    config = load_config()
    den_cfg = config.get("denstream", {})
    macro_cfg = config.get("macro_clustering", {})

    # 1. Initialize Components
    embedder = TextEmbedder()
    engine = DenStreamEngine(
        lambda_decay=den_cfg.get("lambda_decay", 0.03),
        epsilon=den_cfg.get("epsilon", 0.38),
        mu=den_cfg.get("mu", 2.5),
        beta=den_cfg.get("beta", 0.3)
    )
    macro_engine = MacroClusterEngine(
        eps_macro=macro_cfg.get("eps_macro", 0.40),
        min_samples=macro_cfg.get("min_samples", 1)
    )

    # 2. Load HuggingFace CrisisBench Stream
    loader = CrisisBenchLoader(split="test", max_samples=150)
    if loader.dataset is None:
        logger.error("Could not load QCRI/CrisisBench-english. Aborting evaluation.")
        return

    print("\n" + "="*80)
    print(" >>> EVALUATING STREAM CLUSTERING ON QCRI/CrisisBench-english")
    print("="*80 + "\n")

    ground_truth_labels = []
    predicted_cluster_ids = []
    
    post_count = 0
    start_time = time.time()

    for post in loader.stream_posts(delay_sec=0.01):
        post_count += 1
        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        status = engine.process_post(post, vec)
        
        # Track ground truth vs assigned micro-cluster index
        ground_truth_labels.append(post.category_hint or "unknown")
        
        # Find closest micro-cluster ID as predicted cluster
        closest_id = -1
        p_mcs = engine.p_micro_clusters
        if p_mcs:
            dists = [mc.distance_to_point(vec) for mc in p_mcs]
            closest_idx = int(np_argmin(dists)) if hasattr(dists, '__len__') and len(dists) > 0 else 0
            closest_id = closest_idx
        predicted_cluster_ids.append(closest_id)

        if post_count % 30 == 0:
            print(f"[{post_count:03d} Posts] Processed... | Active p-MC: {len(engine.p_micro_clusters)} | Active o-MC: {len(engine.o_micro_clusters)}")

    elapsed = time.time() - start_time
    throughput = post_count / elapsed if elapsed > 0 else 0.0

    # 3. Perform Macro-Clustering
    p_mcs = engine.get_potential_clusters()
    topics = macro_engine.generate_topics(p_mcs)

    # 4. Calculate Metrics
    metrics = ClusteringEvaluator.evaluate_benchmark(ground_truth_labels, predicted_cluster_ids)

    print("\n" + "="*80)
    print(" 📊 CRISISBENCH EVALUATION RESULTS REPORT")
    print("="*80)
    print(f" ► Dataset Name:            QCRI/CrisisBench-english ('humanitarian')")
    print(f" ► Total Posts Evaluated:   {post_count}")
    print(f" ► Processing Time:         {elapsed:.2f}s ({throughput:.1f} posts/sec)")
    print(f" ► Discovered Topics:       {len(topics)} Active Macro-Clusters")
    print("-" * 80)
    print(f" ⭐ Cluster Purity:          {metrics['purity']:.4f}")
    print(f" ⭐ Normalized Mutual Info: {metrics['nmi']:.4f}")
    print(f" ⭐ Adjusted Rand Index:    {metrics['ari']:.4f}")
    print("="*80 + "\n")

    print(" 🔴 TOP DISCOVERED DISASTER TOPICS:")
    for t in topics[:5]:
        print(f"   > [{t.topic_id}] {t.label}")
        print(f"     Weight: {t.total_weight:.2f} | Keywords: {', '.join(t.keywords)}")
        if t.sample_posts:
            print(f"     Sample Tweet: \"{t.sample_posts[0][:80]}...\"")
    print("\n")

def np_argmin(lst):
    import numpy as np
    return np.argmin(lst)

if __name__ == "__main__":
    main()
