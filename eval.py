import sys
import time
import yaml
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.denstream import DenStreamEngine
from src.engine.sadstream_engine import SADStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine
from src.evaluation.evaluator import ClusteringEvaluator
from src.utils.logger import setup_logger

def load_config():
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def run_evaluation_for_engine(engine_name: str, engine_obj, loader, embedder, macro_engine):
    print(f"\n" + "="*80)
    print(f" >>> RUNNING EVALUATION FOR: {engine_name}")
    print("="*80)

    ground_truth_labels = []
    predicted_cluster_ids = []
    
    post_count = 0
    start_time = time.time()

    for post in loader.stream_posts(delay_sec=0.0):
        post_count += 1
        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        status = engine_obj.process_post(post, vec)
        
        # Ground truth class label from CrisisBench
        gt_label = post.category_hint or "unknown"
        ground_truth_labels.append(gt_label)
        
        # Predict cluster ID based on active micro-clusters
        p_mcs = engine_obj.p_micro_clusters
        o_mcs = engine_obj.o_micro_clusters
        all_mcs = p_mcs + o_mcs
        
        if all_mcs:
            dists = [mc.distance_to_point(vec) for mc in all_mcs]
            closest_idx = int(np.argmin(dists))
            closest_id = hash(all_mcs[closest_idx].cluster_id) % 100000
        else:
            closest_id = -1
        predicted_cluster_ids.append(closest_id)

    elapsed = time.time() - start_time
    throughput = post_count / elapsed if elapsed > 0 else 0.0

    # Macro-Clustering Topic Extraction
    p_mcs = engine_obj.get_potential_clusters()
    topics = macro_engine.generate_topics(p_mcs)

    # Compute Benchmark Metrics
    metrics = ClusteringEvaluator.evaluate_benchmark(ground_truth_labels, predicted_cluster_ids)
    metrics["throughput"] = round(throughput, 1)
    metrics["elapsed"] = round(elapsed, 2)
    metrics["discovered_topics"] = len(topics)
    metrics["p_mc_count"] = len(engine_obj.p_micro_clusters)
    metrics["o_mc_count"] = len(engine_obj.o_micro_clusters)
    metrics["topics"] = topics

    return metrics

def main():
    logger = setup_logger("SADStreamBenchmarkEvaluator")
    logger.info("Starting Comparative Evaluation on CrisisBench Dataset...")

    config = load_config()
    den_cfg = config.get("denstream", {})
    macro_cfg = config.get("macro_clustering", {})

    embedder = TextEmbedder()
    macro_engine = MacroClusterEngine(
        eps_macro=macro_cfg.get("eps_macro", 0.40),
        min_samples=macro_cfg.get("min_samples", 1)
    )

    # 1. Baseline Model: DenStream (Fixed Exponential Decay)
    baseline_engine = DenStreamEngine(
        lambda_decay=den_cfg.get("lambda_decay", 0.03),
        epsilon=0.65,
        mu=1.5,
        beta=0.2
    )

    # 2. Proposed Model: SADStream (Adaptive Decay + Compressed Memory + Hybrid Similarity)
    proposed_engine = SADStreamEngine(
        lambda_0=0.03,
        eta=0.5,
        rho=0.5,
        epsilon=0.65,
        mu=1.5,
        beta=0.2,
        theta_R=0.55,
        alpha=0.7,
        beta_lex=0.1,
        gamma_ent=0.1,
        delta_hash=0.1
    )

    # Load 300 posts from dataset stream
    num_samples = 300
    loader_base = CrisisBenchLoader(split="test", max_samples=num_samples)
    loader_prop = CrisisBenchLoader(split="test", max_samples=num_samples)

    if loader_base.dataset is None:
        logger.error("Could not load local dataset data/crisisbench.jsonl. Aborting.")
        return

    # Run Benchmark Evaluations
    base_results = run_evaluation_for_engine("DenStream (Fixed Global Decay Baseline)", baseline_engine, loader_base, embedder, macro_engine)
    prop_results = run_evaluation_for_engine("SADStream (Adaptive Decay & Hybrid Memory Proposed)", proposed_engine, loader_prop, embedder, macro_engine)

    # Comparison Metrics
    comparison = ClusteringEvaluator.compare_models(base_results, prop_results)

    # Output Detailed Comparison Report
    print("\n" + "="*85)
    print(" 📊 COMPARATIVE BENCHMARK EVALUATION REPORT (CrisisBench Real Social Stream)")
    print("="*85)
    print(f" ► Dataset Evaluated:       QCRI/CrisisBench-english (humanitarian disaster posts)")
    print(f" ► Total Posts Processed:   {num_samples}")
    print("-" * 85)
    print(f" METRIC                      BASELINE (DenStream)    PROPOSED (SADStream)    IMPROVEMENT")
    print("-" * 85)
    print(f" ⭐ Cluster Purity:           {base_results['purity']:.4f}                  {prop_results['purity']:.4f}                  {comparison['purity']['pct_change']:+.2f}%")
    print(f" ⭐ Normalized Mutual Info:  {base_results['nmi']:.4f}                  {prop_results['nmi']:.4f}                  {comparison['nmi']['pct_change']:+.2f}%")
    print(f" ⭐ Adjusted Rand Index:     {base_results['ari']:.4f}                  {prop_results['ari']:.4f}                  {comparison['ari']['pct_change']:+.2f}%")
    print(f" ⚡ Throughput (posts/sec):  {base_results['throughput']:<8.1f}              {prop_results['throughput']:<8.1f}              {(prop_results['throughput']-base_results['throughput']):+.1f} p/s")
    print(f" 🏷️ Active Macro Topics:     {base_results['discovered_topics']:<8d}              {prop_results['discovered_topics']:<8d}              {prop_results['discovered_topics']-base_results['discovered_topics']:+d}")
    print(f" 📦 Recurring Topics Memory: 0                       {proposed_engine.recurring_topic_count:<8d}              +{proposed_engine.recurring_topic_count}")
    print("="*85 + "\n")

    print(" 🔴 TOP DISCOVERED DISASTER TOPICS (SADStream Event Detection):")
    for t in prop_results['topics'][:5]:
        burst_flag = "💥 BURST ALERT" if t.is_bursting else "📌 ACTIVE TOPIC"
        print(f"   > [{t.topic_id}] {t.label} ({burst_flag} | Score: {t.burst_score:.2f})")
        print(f"     Weight: {t.total_weight:.2f} | Micro-Clusters: {t.micro_cluster_count} | Keywords: {', '.join(t.keywords[:5])}")
        if t.sample_posts:
            print(f"     Sample Tweet: \"{t.sample_posts[0][:80]}...\"")
    print("\n")

if __name__ == "__main__":
    main()
