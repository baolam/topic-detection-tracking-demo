import sys
import time
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from src.data.stream_generator import SocialStreamGenerator
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.denstream import DenStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine
from src.utils.logger import setup_logger

def load_config():
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def main():
    logger = setup_logger("MainRunner")
    logger.info("Starting Dynamic Topic Detection & Tracking Demo System...")

    config = load_config()
    den_cfg = config.get("denstream", {})
    macro_cfg = config.get("macro_clustering", {})

    # 1. Initialize Components
    embedder = TextEmbedder()
    engine = DenStreamEngine(
        lambda_decay=den_cfg.get("lambda_decay", 0.03),
        epsilon=den_cfg.get("epsilon", 0.38),
        mu=den_cfg.get("mu", 2.5),
        beta=den_cfg.get("beta", 0.3),
        pruning_period=den_cfg.get("pruning_period", 5.0)
    )
    macro_engine = MacroClusterEngine(
        eps_macro=macro_cfg.get("eps_macro", 0.40),
        min_samples=macro_cfg.get("min_samples", 1)
    )
    generator = SocialStreamGenerator(arrival_rate=5.0, burst_prob=0.25)

    print("\n" + "="*80)
    print(" >>> STREAMING DATA TOPIC TRACKING RUNNING (Simulating Social Media Feed)")
    print("="*80 + "\n")

    post_count = 0
    start_time = time.time()

    # Stream 35 real-time social media posts
    for post in generator.stream(max_posts=35, delay_sec=0.15):
        post_count += 1
        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        # Process post through DenStream Engine
        status = engine.process_post(post, vec)
        print(f"[{post_count:02d}] User: {post.author_id} | Status: {status:18s} | Text: {clean_text[:60]}...")

        # Periodically trigger Macro-clustering (Every 10 posts)
        if post_count % 10 == 0:
            p_mcs = engine.get_potential_clusters()
            topics = macro_engine.generate_topics(p_mcs)
            
            print("\n" + "-"*70)
            print(f" [MACRO-CLUSTERING QUERY AT POST #{post_count}]")
            print(f" Active Potential Micro-Clusters (p-MC): {len(p_mcs)}")
            print(f" Active Outlier Micro-Clusters (o-MC): {len(engine.o_micro_clusters)}")
            print(f" DISCOVERED TRENDING TOPICS ({len(topics)} Topics):")
            for t in topics:
                print(f"   > [{t.topic_id}] {t.label}")
                print(f"     Weight: {t.total_weight:.2f} | Keywords: {', '.join(t.keywords)}")
                print(f"     Sample: \"{t.sample_posts[0] if t.sample_posts else ''}\"")
            print("-" * 70 + "\n")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Successfully processed {post_count} posts in {elapsed:.2f}s ({post_count/elapsed:.1f} posts/sec).")

if __name__ == "__main__":
    main()
