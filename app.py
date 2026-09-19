import streamlit as st
import time
import pandas as pd
from src.data.stream_generator import SocialStreamGenerator
from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.denstream import DenStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine
from src.evaluation.evaluator import ClusteringEvaluator

st.set_page_config(
    page_title="Real-Time Topic Detection & Tracking System",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Real-Time Stream Topic Detection & Tracking System")
st.caption("Powered by DenStream with Exponential Time-Decay & c-TF-IDF Topic Labeling | Benchmark: QCRI/CrisisBench Dataset")

# Sidebar Controls
st.sidebar.header("⚙️ Data Source & Parameters")
data_source = st.sidebar.radio(
    "Select Data Stream Source:",
    ["QCRI/CrisisBench (Local Downloaded Dataset)", "Synthetic Social Media Stream (Simulation)"]
)

lambda_decay = st.sidebar.slider("Decay Constant (Lambda λ)", 0.001, 0.1, 0.03, step=0.005, help="Time-decay weight penalty for older posts")
epsilon = st.sidebar.slider("Micro Cluster Radius (Epsilon ε)", 0.1, 0.6, 0.38, step=0.02, help="Cosine similarity radius")
mu = st.sidebar.slider("Potential Cluster Threshold (Mu μ)", 1.0, 5.0, 2.5, step=0.5)
stream_speed = st.sidebar.slider("Stream Speed (Posts/sec)", 1, 20, 5)

# Session State Initialization
if "engine" not in st.session_state:
    st.session_state.embedder = TextEmbedder()
    st.session_state.engine = DenStreamEngine(
        lambda_decay=lambda_decay,
        epsilon=epsilon,
        mu=mu,
        beta=0.3
    )
    st.session_state.macro_engine = MacroClusterEngine(eps_macro=0.40)
    st.session_state.posts_history = []
    st.session_state.y_true = []
    st.session_state.y_pred = []

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📡 Social Media Live Stream Feed")
    feed_placeholder = st.empty()

with col2:
    st.subheader("🔴 Discovered Topics & Breaking Events")
    topic_placeholder = st.empty()

# Dashboard Metrics Placeholder
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    m_processed = st.metric("Total Processed Posts", 0)
with metric_col2:
    m_pmc = st.metric("Potential Clusters (p-MC)", 0)
with metric_col3:
    m_omc = st.metric("Outlier Clusters (o-MC)", 0)
with metric_col4:
    m_purity = st.metric("Benchmark Purity", "N/A")

run_button = st.button("▶️ Start Streaming Real-Time Simulation")

if run_button:
    engine = st.session_state.engine
    embedder = st.session_state.embedder
    macro_engine = st.session_state.macro_engine

    # Dynamic update sliders
    engine.lambda_decay = lambda_decay
    engine.epsilon = epsilon
    engine.mu = mu

    # Select Stream Generator
    if "CrisisBench" in data_source:
        loader = CrisisBenchLoader(split="test", max_samples=80)
        stream_iter = loader.stream_posts(delay_sec=1.0 / stream_speed)
    else:
        generator = SocialStreamGenerator(arrival_rate=stream_speed)
        stream_iter = generator.stream(max_posts=50, delay_sec=1.0 / stream_speed)

    for post in stream_iter:
        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        status = engine.process_post(post, vec)
        
        # Track ground truth vs cluster
        st.session_state.y_true.append(post.category_hint or "unknown")
        closest_id = 0
        p_mcs = engine.p_micro_clusters
        if p_mcs:
            dists = [mc.distance_to_point(vec) for mc in p_mcs]
            closest_id = int(min(range(len(dists)), key=lambda i: dists[i]))
        st.session_state.y_pred.append(closest_id)

        st.session_state.posts_history.insert(0, {
            "Time": time.strftime("%H:%M:%S", time.localtime(post.timestamp)),
            "Author": post.author_id,
            "Category": post.category_hint or "N/A",
            "Text": clean_text,
            "Status": status
        })

        # Update Live Feed
        with feed_placeholder.container():
            df = pd.DataFrame(st.session_state.posts_history[:10])
            st.dataframe(df, use_container_width=True)

        # Periodically update Macro-Clusters
        p_mcs = engine.get_potential_clusters()
        topics = macro_engine.generate_topics(p_mcs)

        with topic_placeholder.container():
            if not topics:
                st.info("Accumulating post stream to form micro-clusters...")
            for t in topics:
                st.markdown(f"### 📌 [{t.topic_id}] {t.label}")
                st.progress(min(t.total_weight / 10.0, 1.0))
                st.write(f"**Weight:** `{t.total_weight:.2f}` | **c-TF-IDF Keywords:** `{', '.join(t.keywords)}`")
                if t.sample_posts:
                    st.caption(f"Sample Post: \"{t.sample_posts[0]}\"")
                st.divider()

        # Update Metrics
        purity_val = "N/A"
        if len(st.session_state.y_true) > 5:
            metrics = ClusteringEvaluator.evaluate_benchmark(st.session_state.y_true, st.session_state.y_pred)
            purity_val = f"{metrics['purity']:.4f}"

        m_processed.metric("Total Processed Posts", engine.total_processed)
        m_pmc.metric("Potential Clusters (p-MC)", len(p_mcs))
        m_omc.metric("Outlier Clusters (o-MC)", len(engine.o_micro_clusters))
        m_purity.metric("Benchmark Purity", purity_val)
