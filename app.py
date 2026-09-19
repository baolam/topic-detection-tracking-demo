import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.data.stream_generator import SocialStreamGenerator
from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.denstream import DenStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine
from src.evaluation.evaluator import ClusteringEvaluator

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Real-Time Topic Intelligence & Tracking System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Dark Glassmorphism CSS
st.markdown("""
<style>
    /* Dark Obsidian Theme */
    .stApp {
        background: linear-gradient(180deg, #0b0e14 0%, #121824 100%);
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient Title */
    .header-title {
        background: linear-gradient(90deg, #ff4b4b 0%, #7928ca 50%, #00dfd8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.2rem;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    
    .header-sub {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* Glassmorphism Cards for Topics */
    .topic-card {
        background: rgba(22, 27, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease-in-out;
    }
    
    .topic-card:hover {
        border-color: #00dfd8;
        box-shadow: 0 0 20px rgba(0, 223, 216, 0.2);
        transform: translateY(-2px);
    }
    
    .topic-card-burst {
        border: 1px solid rgba(255, 75, 75, 0.6);
        background: rgba(35, 18, 25, 0.85);
        box-shadow: 0 0 25px rgba(255, 75, 75, 0.25);
    }

    /* Keyword Pills */
    .kw-pill {
        display: inline-block;
        background: rgba(0, 223, 216, 0.12);
        color: #00dfd8;
        border: 1px solid rgba(0, 223, 216, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.83rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Badges */
    .badge-burst {
        background: linear-gradient(90deg, #ff4b4b, #ff416c);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        box-shadow: 0 0 12px rgba(255, 75, 75, 0.5);
        display: inline-block;
    }

    .badge-emerging {
        background: linear-gradient(90deg, #00b4db, #0083b0);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }

    /* Stream Status Pills */
    .status-pmc {
        color: #00dfd8;
        font-weight: bold;
    }
    .status-omc {
        color: #ff7b72;
        font-weight: bold;
    }
    .status-promoted {
        color: #d2a8ff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# OPTIMAL STANDARD HYPERPARAMETERS
# ==========================================
DEFAULT_LAMBDA = 0.030   # Exponential decay constant
DEFAULT_EPSILON = 0.48   # Micro-cluster radius for text embeddings
DEFAULT_MU = 1.80        # Potential cluster weight threshold
DEFAULT_EPS_MACRO = 0.45 # DBSCAN macro-clustering radius threshold

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "embedder" not in st.session_state:
    with st.spinner("⚡ Initializing Sentence Embedding Model (all-MiniLM-L6-v2)..."):
        st.session_state.embedder = TextEmbedder()

if "engine" not in st.session_state:
    st.session_state.engine = DenStreamEngine(
        lambda_decay=DEFAULT_LAMBDA,
        epsilon=DEFAULT_EPSILON,
        mu=DEFAULT_MU,
        beta=0.3
    )
    st.session_state.macro_engine = MacroClusterEngine(eps_macro=DEFAULT_EPS_MACRO)
    st.session_state.posts_history = []
    st.session_state.vector_log = []   # List of dicts for 2D/3D scatter plot
    st.session_state.timeline_log = [] # List of dicts for dynamics line plot
    st.session_state.y_true = []
    st.session_state.y_pred = []
    st.session_state.is_running = False

# ==========================================
# SIDEBAR CONTROLS & HYPERPARAMETERS
# ==========================================
st.sidebar.markdown("## ⚙️ Control Center")

data_source = st.sidebar.radio(
    "🌐 Data Stream Source",
    ["QCRI/CrisisBench (Disaster Stream)", "Synthetic Social Media Stream"],
    help="Select dataset source to stream social posts."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌌 Map Visualization Settings")
dim_choice = st.sidebar.radio(
    "PCA Vector Projection Mode:",
    ["3D PCA Vector Plot", "2D PCA Vector Plot"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ DenStream Hyperparameters (Optimal Standard)")

lambda_decay = st.sidebar.slider(
    "Decay Constant (λ)", 0.001, 0.100, DEFAULT_LAMBDA, step=0.005,
    help="Time-decay penalty weight for older posts"
)
epsilon = st.sidebar.slider(
    "Micro Radius (ε)", 0.10, 0.70, DEFAULT_EPSILON, step=0.02,
    help="Max cosine distance for cluster merging (Optimal: 0.48 for text embeddings)"
)
mu = st.sidebar.slider(
    "Potential Threshold (μ)", 1.0, 5.0, DEFAULT_MU, step=0.1,
    help="Minimum weight threshold to form a Potential Micro-Cluster (Optimal: 1.8)"
)
eps_macro = st.sidebar.slider(
    "Macro DBSCAN (ε_macro)", 0.20, 0.60, DEFAULT_EPS_MACRO, step=0.05,
    help="Distance radius for grouping micro-clusters into macro topics (Optimal: 0.45)"
)
stream_speed = st.sidebar.slider("⚡ Stream Speed (Posts/sec)", 1, 20, 5)

st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("▶️ Start Stream", use_container_width=True, type="primary"):
        st.session_state.is_running = True

with col_btn2:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.is_running = False

if st.sidebar.button("🔄 Reset Engine State", use_container_width=True):
    st.session_state.engine = DenStreamEngine(lambda_decay=lambda_decay, epsilon=epsilon, mu=mu, beta=0.3)
    st.session_state.macro_engine = MacroClusterEngine(eps_macro=eps_macro)
    st.session_state.posts_history.clear()
    st.session_state.vector_log.clear()
    st.session_state.timeline_log.clear()
    st.session_state.y_true.clear()
    st.session_state.y_pred.clear()
    st.session_state.is_running = False
    st.rerun()

# ==========================================
# DASHBOARD HEADER & LIVE KPIS
# ==========================================
st.markdown('<div class="header-title">🔥 Real-Time Stream Topic Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Powered by DenStream Exponential Decay, Macro-DBSCAN Clustering & c-TF-IDF Topic Labeling</div>', unsafe_allow_html=True)

# Placeholder for Header Metric Cards
kpi_placeholder = st.empty()

def update_kpis():
    with kpi_placeholder.container():
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("📡 Total Posts", len(st.session_state.posts_history))
        with kpi2:
            p_mcs_active = st.session_state.engine.get_potential_clusters()
            st.metric("🟢 p-MC (Potential)", len(p_mcs_active))
        with kpi3:
            st.metric("🔴 o-MC (Outliers)", len(st.session_state.engine.o_micro_clusters))
        with kpi4:
            current_topics = st.session_state.macro_engine.generate_topics(p_mcs_active)
            st.metric("🔥 Active Topics", len(current_topics))
        with kpi5:
            purity_val = "N/A"
            if len(st.session_state.y_true) > 5:
                metrics = ClusteringEvaluator.evaluate_benchmark(st.session_state.y_true, st.session_state.y_pred)
                purity_val = f"{metrics['purity']*100:.1f}%"
            st.metric("🎯 Benchmark Purity", purity_val)

update_kpis()

st.markdown("---")

# ==========================================
# ACTIVE VIEW SELECTOR (LAZY RENDERING FOR HIGH FPS)
# ==========================================
active_view = st.radio(
    "📺 Active Live View (Chọn mục để xem thời gian thực, chống lag):",
    [
        "🔥 Active & Detected Topics",
        "🌌 2D/3D Cluster & Vector Map",
        "📈 Stream Dynamics & Velocity",
        "📡 Live Feed HUD",
        "🎯 Benchmark Diagnostics"
    ],
    horizontal=True,
    key="active_view_selector_radio"
)

main_content_placeholder = st.empty()

# ==========================================
# HELPER RENDERING FUNCTIONS
# ==========================================
def render_topics_tab(topics):
    """Renders dedicated detected topics grid and summary charts."""
    with main_content_placeholder.container():
        if not topics:
            st.info("💡 Accumulating social post stream to form micro-clusters and discover macro topics...")
            return

        # Top summary visual row
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            # Donut chart of Topic Weights
            df_topics = pd.DataFrame([
                {"Topic": f"[{t.topic_id}] {t.label}", "Weight": t.total_weight, "Burst": t.burst_score}
                for t in topics
            ])
            fig_donut = px.pie(
                df_topics, values="Weight", names="Topic", hole=0.4,
                title="📊 Topic Weight Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_chart2:
            # Top Keywords Bar Chart
            all_kw = []
            for t in topics:
                for kw in t.keywords[:4]:
                    all_kw.append({"Keyword": kw, "Topic": t.label, "Score": t.total_weight})
            if all_kw:
                df_kw = pd.DataFrame(all_kw).head(12)
                fig_kw = px.bar(
                    df_kw, x="Score", y="Keyword", color="Topic", orientation="h",
                    title="🔑 Top Trending Keywords",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_kw.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_kw, use_container_width=True)

        st.markdown("### 📌 Discovered Macro-Topics & Breaking Events")
        
        # Grid of Detected Topic Cards
        for t in topics:
            card_class = "topic-card topic-card-burst" if t.is_bursting else "topic-card"
            badge_html = f'<span class="badge-burst">🔥 BURSTING / BREAKING</span>' if t.is_bursting else f'<span class="badge-emerging">⚡ EMERGING TOPIC</span>'
            
            kw_pills = "".join([f'<span class="kw-pill">#{kw}</span>' for kw in t.keywords])
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin:0; color: #ffffff; font-size: 1.25rem;">📌 [{t.topic_id}] {t.label}</h3>
                    {badge_html}
                </div>
                <div style="margin-bottom: 12px;">{kw_pills}</div>
                <div style="display: flex; gap: 24px; font-size: 0.9rem; color: #8b949e; margin-bottom: 10px;">
                    <div>⚖️ Total Weight: <strong style="color: #00dfd8;">{t.total_weight:.2f}</strong></div>
                    <div>🟢 Micro-Clusters: <strong style="color: #58a6ff;">{t.micro_cluster_count}</strong></div>
                    <div>⚡ Burst Score: <strong style="color: #ff7b72;">{t.burst_score:.3f}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"💬 Sample Posts for [{t.topic_id}] ({len(t.sample_posts)} posts)"):
                for idx, sample in enumerate(t.sample_posts, 1):
                    st.markdown(f"**{idx}.** *\"{sample}\"*")


def render_map_tab(engine, vector_log, projection_choice):
    """Renders high-tech 2D or 3D PCA scatter plot of micro-clusters and post vectors."""
    with main_content_placeholder.container():
        st.markdown("### 🌌 2D / 3D Vector Space & Micro-Cluster Visualization")
        
        if not vector_log and not engine.p_micro_clusters:
            st.info("💡 Start streaming posts to visualize clusters in 2D/3D PCA vector space!")
            return

        n_components = 3 if "3D" in projection_choice else 2

        # Gather data points: Post vectors + Micro-cluster centers
        plot_rows = []
        raw_vectors = []

        # 1. Post Vectors History (sample up to 100 recent)
        recent_vecs = vector_log[-100:]
        for item in recent_vecs:
            raw_vectors.append(item["vector"])
            plot_rows.append({
                "Type": "Post Vector",
                "Label": f"Post: {item['text'][:30]}...",
                "Size": 6,
                "Text": item["text"],
                "ClusterID": item["status"]
            })

        # 2. Potential Micro Clusters (p-MCs)
        for mc in engine.get_potential_clusters():
            raw_vectors.append(mc.get_center())
            plot_rows.append({
                "Type": "Potential Micro-Cluster (p-MC)",
                "Label": f"p-MC [{mc.cluster_id}] (W={mc.weight:.1f})",
                "Size": max(12, int(mc.weight * 3)),
                "Text": f"Sample: {mc.sample_texts[0] if mc.sample_texts else 'N/A'}",
                "ClusterID": mc.cluster_id
            })

        # 3. Outlier Micro Clusters (o-MCs)
        for mc in engine.o_micro_clusters:
            raw_vectors.append(mc.get_center())
            plot_rows.append({
                "Type": "Outlier Micro-Cluster (o-MC)",
                "Label": f"o-MC [{mc.cluster_id}] (W={mc.weight:.1f})",
                "Size": 8,
                "Text": f"Sample: {mc.sample_texts[0] if mc.sample_texts else 'N/A'}",
                "ClusterID": mc.cluster_id
            })

        if len(raw_vectors) >= n_components:
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(np.array(raw_vectors))

            df_map = pd.DataFrame(plot_rows)
            df_map["PC1"] = coords[:, 0]
            df_map["PC2"] = coords[:, 1]
            if n_components == 3:
                df_map["PC3"] = coords[:, 2]

            if n_components == 3:
                fig_map = px.scatter_3d(
                    df_map, x="PC1", y="PC2", z="PC3",
                    color="Type", size="Size", hover_name="Label", hover_data=["Text"],
                    template="plotly_dark",
                    color_discrete_map={
                        "Potential Micro-Cluster (p-MC)": "#00dfd8",
                        "Outlier Micro-Cluster (o-MC)": "#ff4b4b",
                        "Post Vector": "#8b949e"
                    }
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=550)
            else:
                fig_map = px.scatter(
                    df_map, x="PC1", y="PC2",
                    color="Type", size="Size", hover_name="Label", hover_data=["Text"],
                    template="plotly_dark",
                    color_discrete_map={
                        "Potential Micro-Cluster (p-MC)": "#00dfd8",
                        "Outlier Micro-Cluster (o-MC)": "#ff4b4b",
                        "Post Vector": "#8b949e"
                    }
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=500)

            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Accumulating more vectors to perform PCA projection...")


def render_dynamics_tab(timeline_log):
    """Renders real-time stream dynamics and micro-cluster trajectory charts."""
    with main_content_placeholder.container():
        st.markdown("### 📈 Real-Time Stream Trajectory & Cluster Dynamics")
        
        if not timeline_log:
            st.info("💡 Dynamics timeline will appear as posts stream in.")
            return

        df_time = pd.DataFrame(timeline_log)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_clusters = px.line(
                df_time, x="PostIndex", y=["p_MC_Count", "o_MC_Count"],
                labels={"value": "Cluster Count", "variable": "Cluster Type"},
                title="🟢 Micro-Cluster Evolution over Stream (p-MC vs o-MC)",
                template="plotly_dark",
                color_discrete_map={"p_MC_Count": "#00dfd8", "o_MC_Count": "#ff4b4b"}
            )
            fig_clusters.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_clusters, use_container_width=True)

        with col_d2:
            fig_burst = px.area(
                df_time, x="PostIndex", y="MaxBurstScore",
                title="🔥 Peak Topic Burstiness Trajectory",
                template="plotly_dark",
                color_discrete_sequence=["#ff7b72"]
            )
            fig_burst.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_burst, use_container_width=True)


def render_feed_tab(posts_history):
    """Renders real-time social media post feed table."""
    with main_content_placeholder.container():
        st.markdown("### 📡 Real-Time Social Stream Feed HUD")
        if not posts_history:
            st.info("💡 No posts processed yet. Click 'Start Stream' in sidebar!")
            return

        df_posts = pd.DataFrame(posts_history)
        st.dataframe(
            df_posts,
            column_config={
                "Time": st.column_config.TextColumn("Timestamp", width="small"),
                "Author": st.column_config.TextColumn("Author ID", width="small"),
                "Category": st.column_config.TextColumn("Category / Ground Truth", width="medium"),
                "Status": st.column_config.TextColumn("DenStream Status", width="medium"),
                "Text": st.column_config.TextColumn("Cleaned Post Content", width="large")
            },
            use_container_width=True,
            height=480
        )


def render_eval_tab(y_true, y_pred):
    """Renders benchmark purity & evaluation breakdown."""
    with main_content_placeholder.container():
        st.markdown("### 🎯 Benchmark Evaluation & Purity Diagnostics")
        if len(y_true) <= 5:
            st.info("💡 Process at least 5 posts to compute benchmark purity metrics.")
            return

        metrics = ClusteringEvaluator.evaluate_benchmark(y_true, y_pred)
        
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Purity Score", f"{metrics['purity']*100:.2f}%")
        with col_e2:
            st.metric("Normalized Mutual Information (NMI)", f"{metrics['nmi']:.4f}")
        with col_e3:
            st.metric("Adjusted Rand Index (ARI)", f"{metrics['ari']:.4f}")

        st.markdown("---")
        st.markdown("#### Ground-Truth Category Breakdown")
        df_eval = pd.DataFrame({"Ground Truth Category": y_true, "Assigned Micro Cluster": y_pred})
        st.dataframe(df_eval.tail(25), use_container_width=True)


def render_selected_view(view_name):
    """Lazy-renders ONLY the active view selected by user for maximum streaming performance."""
    p_mcs = st.session_state.engine.get_potential_clusters()
    topics = st.session_state.macro_engine.generate_topics(p_mcs)

    if "Topics" in view_name:
        render_topics_tab(topics)
    elif "Map" in view_name:
        render_map_tab(st.session_state.engine, st.session_state.vector_log, dim_choice)
    elif "Dynamics" in view_name:
        render_dynamics_tab(st.session_state.timeline_log)
    elif "Feed" in view_name:
        render_feed_tab(st.session_state.posts_history)
    elif "Evaluation" in view_name or "Diagnostics" in view_name:
        render_eval_tab(st.session_state.y_true, st.session_state.y_pred)

# Initial static render when not running stream loop
if not st.session_state.is_running:
    render_selected_view(active_view)

# ==========================================
# STREAMING SIMULATION LOOP
# ==========================================
if st.session_state.is_running:
    engine = st.session_state.engine
    embedder = st.session_state.embedder
    macro_engine = st.session_state.macro_engine

    # Dynamic Hyperparameter Updates
    engine.lambda_decay = lambda_decay
    engine.epsilon = epsilon
    engine.mu = mu
    macro_engine.eps_macro = eps_macro

    # Select Stream Source Generator
    if "CrisisBench" in data_source:
        loader = CrisisBenchLoader(split="test", max_samples=100)
        stream_iter = loader.stream_posts(delay_sec=1.0 / stream_speed)
    else:
        generator = SocialStreamGenerator(arrival_rate=stream_speed)
        stream_iter = generator.stream(max_posts=80, delay_sec=1.0 / stream_speed)

    for post in stream_iter:
        if not st.session_state.is_running:
            break

        clean_text = TextPreprocessor.clean_text(post.text)
        vec = embedder.encode(clean_text)

        # Process post vector into DenStream Engine
        status = engine.process_post(post, vec)

        # Track Ground Truth vs Cluster Prediction
        ground_truth = post.category_hint or "unknown"
        st.session_state.y_true.append(ground_truth)
        
        closest_id = 0
        p_mcs = engine.get_potential_clusters()
        if p_mcs:
            dists = [mc.distance_to_point(vec) for mc in p_mcs]
            closest_id = int(min(range(len(dists)), key=lambda i: dists[i]))
        st.session_state.y_pred.append(closest_id)

        # Log post history
        st.session_state.posts_history.insert(0, {
            "Time": time.strftime("%H:%M:%S", time.localtime(post.timestamp)),
            "Author": post.author_id,
            "Category": ground_truth,
            "Status": status,
            "Text": clean_text
        })

        # Log vector data for 2D/3D map
        st.session_state.vector_log.append({
            "vector": vec,
            "status": status,
            "text": clean_text
        })

        # Generate Topics periodically
        topics = macro_engine.generate_topics(p_mcs)
        max_burst = max([t.burst_score for t in topics], default=0.0)

        # Log dynamics timeline
        st.session_state.timeline_log.append({
            "PostIndex": engine.total_processed,
            "p_MC_Count": len(p_mcs),
            "o_MC_Count": len(engine.o_micro_clusters),
            "TopicCount": len(topics),
            "MaxBurstScore": max_burst
        })

        # Update Top KPI Stats
        update_kpis()

        # LAZY-RENDER ONLY THE CURRENTLY SELECTED ACTIVE VIEW (SUPER FAST & NO LAG)
        render_selected_view(active_view)

        time.sleep(1.0 / stream_speed)

    st.session_state.is_running = False
    st.rerun()
