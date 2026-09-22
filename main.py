import sys
import time
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.sadstream_engine import SADStreamEngine
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
    logger.info("Starting Dynamic Topic Detection & Tracking System...")

    config = load_config()
    sad_cfg = config.get("sadstream", {})
    macro_cfg = config.get("macro_clustering", {})

    # 1. Initialize Components
    embedder = TextEmbedder()
    engine = SADStreamEngine(
        lambda_0=sad_cfg.get("lambda_0", 0.03),
        eta=sad_cfg.get("eta", 0.5),
        rho=sad_cfg.get("rho", 0.5),
        epsilon=sad_cfg.get("epsilon", 0.65),
        mu=sad_cfg.get("mu", 1.5),
        beta=sad_cfg.get("beta", 0.2),
        theta_R=sad_cfg.get("theta_R", 0.55),
        alpha=sad_cfg.get("alpha", 0.7),
        beta_lex=sad_cfg.get("beta_lex", 0.1),
        gamma_ent=sad_cfg.get("gamma_ent", 0.1),
        delta_hash=sad_cfg.get("delta_hash", 0.1)
    )
    macro_engine = MacroClusterEngine(
        eps_macro=macro_cfg.get("eps_macro", 0.40),
        min_samples=macro_cfg.get("min_samples", 1)
    )

    # 2. Load Real Downloaded Dataset Stream
    loader = CrisisBenchLoader(split="test", max_samples=80)
    if loader.dataset is None:
        logger.error("Could not load local dataset data/crisisbench.jsonl. Aborting.")
        return

    print("\n" + "="*80)
    print(" >>> REAL-TIME DISASTER TOPIC TRACKING (QCRI/CrisisBench Dataset Stream)")
    print("="*80 + "\n")

    post_count = 0
    start_time = time.time()

    # Stream real social media posts from local downloaded dataset
    for post in loader.stream_posts(delay_sec=0.02):
        post_count += 1
        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        # Process post through DenStream Engine
        status = engine.process_post(post, vec)
        print(f"[{post_count:02d}] Category: {post.category_hint:25s} | Status: {status:18s} | Text: {clean_text[:50]}...")

        # Periodically trigger Macro-clustering (Every 20 posts)
        if post_count % 20 == 0:
            p_mcs = engine.get_potential_clusters()
            topics = macro_engine.generate_topics(p_mcs)
            
            print("\n" + "-"*75)
            print(f" [MACRO-CLUSTERING QUERY AT POST #{post_count}]")
            print(f" Active Potential Micro-Clusters (p-MC): {len(p_mcs)}")
            print(f" Active Outlier Micro-Clusters (o-MC): {len(engine.o_micro_clusters)}")
            print(f" DISCOVERED TRENDING TOPICS ({len(topics)} Topics):")
            for t in topics:
                print(f"   > [{t.topic_id}] {t.label}")
                print(f"     Weight: {t.total_weight:.2f} | Keywords: {', '.join(t.keywords)}")
                if t.sample_posts:
                    print(f"     Sample: \"{t.sample_posts[0][:75]}...\"")
            print("-" * 75 + "\n")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Successfully processed {post_count} dataset posts in {elapsed:.2f}s ({post_count/elapsed:.1f} posts/sec).")

if __name__ == "__main__":
    main()
