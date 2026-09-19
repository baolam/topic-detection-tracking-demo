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
    page_title="Dynamic Topic Detection & Tracking System",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Hệ Thống Phát Hiện & Theo Dõi Chủ Đề Động Trực Tuyến")
st.caption("Ứng dụng thuật toán DenStream với Cơ chế Lãng quên trên Luồng Mạng Xã Hội | Hỗ trợ Benchmark QCRI/CrisisBench")

# Sidebar Controls
st.sidebar.header("⚙️ Nguồn Dữ Liệu & Tham Số Stream")
data_source = st.sidebar.radio(
    "Chọn Nguồn Dữ Liệu Stream:",
    ["Synthetic Social Media Stream (Mô phỏng)", "QCRI/CrisisBench-english (HuggingFace Disaster Data)"]
)

lambda_decay = st.sidebar.slider("Hằng số suy giảm (Lambda λ)", 0.001, 0.1, 0.03, step=0.005, help="Tốc độ suy giảm trọng số bài đăng cũ")
epsilon = st.sidebar.slider("Bán kính cụm Micro (Epsilon ε)", 0.1, 0.6, 0.38, step=0.02, help="Bán kính tương đồng Cosine")
mu = st.sidebar.slider("Ngưỡng cụm tiềm năng (Mu μ)", 1.0, 5.0, 2.5, step=0.5)
stream_speed = st.sidebar.slider("Tốc độ phát luồng (Bài/giây)", 1, 20, 5)

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
    st.subheader("📡 Social Media Live Feed")
    feed_placeholder = st.empty()

with col2:
    st.subheader("🔴 Trending Topics & Breaking Events")
    topic_placeholder = st.empty()

# Dashboard Metrics Placeholder
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    m_processed = st.metric("Tổng bài đăng", 0)
with metric_col2:
    m_pmc = st.metric("Cụm tiềm năng (p-MC)", 0)
with metric_col3:
    m_omc = st.metric("Cụm nhiễu (o-MC)", 0)
with metric_col4:
    m_purity = st.metric("Benchmark Purity", "N/A")

run_button = st.button("▶️ Chạy Mô Phỏng Luồng Thời Gian Thực")

if run_button:
    engine = st.session_state.engine
    embedder = st.session_state.embedder
    macro_engine = st.session_state.macro_engine

    # Dynamic update sliders
    engine.lambda_decay = lambda_decay
    engine.epsilon = epsilon
    engine.mu = mu

    # Select Stream Generator
    if "HuggingFace" in data_source:
        loader = CrisisBenchLoader(split="test", max_samples=60)
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
                st.info("Đang tích lũy luồng bài đăng để hình thành cụm chủ đề...")
            for t in topics:
                st.markdown(f"### 📌 [{t.topic_id}] {t.label}")
                st.progress(min(t.total_weight / 10.0, 1.0))
                st.write(f"**Trọng số (Weight):** `{t.total_weight:.2f}` | **Từ khóa:** `{', '.join(t.keywords)}`")
                if t.sample_posts:
                    st.caption(f"Trích dẫn bài đăng: \"{t.sample_posts[0]}\"")
                st.divider()

        # Update Metrics
        purity_val = "N/A"
        if len(st.session_state.y_true) > 5:
            metrics = ClusteringEvaluator.evaluate_benchmark(st.session_state.y_true, st.session_state.y_pred)
            purity_val = f"{metrics['purity']:.4f}"

        m_processed.metric("Tổng bài đăng", engine.total_processed)
        m_pmc.metric("Cụm tiềm năng (p-MC)", len(p_mcs))
        m_omc.metric("Cụm nhiễu (o-MC)", len(engine.o_micro_clusters))
        m_purity.metric("Benchmark Purity", purity_val)
