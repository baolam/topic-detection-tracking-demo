import time
import numpy as np
import streamlit as st

from src.data.stream_generator import SocialStreamGenerator
from src.data.crisisbench_loader import CrisisBenchLoader
from src.data.preprocessor import TextPreprocessor
from src.embeddings.embedder import TextEmbedder
from src.engine.sadstream_engine import SADStreamEngine
from src.engine.denstream import DenStreamEngine
from src.tracking.macro_cluster import MacroClusterEngine

from src.ui.styles import (
    apply_custom_styles,
    DEFAULT_LAMBDA,
    DEFAULT_EPSILON,
    DEFAULT_MU,
    DEFAULT_EPS_MACRO
)
from src.ui.components import (
    update_kpi_cards,
    render_selected_view,
    reset_render_counter,
)

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="SADStream — Real-Time Topic Detection & Tracking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()
reset_render_counter()

# Helper to load stream queue into session state
def init_stream_queue(data_source_name):
    if "CrisisBench" in data_source_name:
        loader = CrisisBenchLoader(split="test", max_samples=150)
        st.session_state.stream_posts_queue = list(loader.stream_posts(delay_sec=0))
    else:
        generator = SocialStreamGenerator(arrival_rate=5.0)
        st.session_state.stream_posts_queue = list(generator.stream(max_posts=150, delay_sec=0))
    st.session_state.stream_idx = 0
    st.session_state.active_data_source = data_source_name


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "embedder" not in st.session_state:
    with st.spinner("⚡ Initializing Sentence Embedding Model (all-MiniLM-L6-v2)..."):
        st.session_state.embedder = TextEmbedder()

if "engine" not in st.session_state:
    st.session_state.engine = SADStreamEngine(
        lambda_0=DEFAULT_LAMBDA,
        eta=0.8,
        rho=0.5,
        epsilon=DEFAULT_EPSILON,
        mu=DEFAULT_MU,
        beta=0.2,
        theta_R=0.50,
        alpha=0.7,
        beta_lex=0.1,
        gamma_ent=0.1,
        delta_hash=0.1
    )

    st.session_state.engine_name = "SADStream"
    st.session_state.macro_engine = MacroClusterEngine(eps_macro=DEFAULT_EPS_MACRO)
    st.session_state.posts_history = []
    st.session_state.vector_log   = []
    st.session_state.timeline_log = []
    st.session_state.y_true       = []
    st.session_state.y_pred       = []
    st.session_state.is_running   = False
    st.session_state.stream_idx   = 0
    st.session_state.stream_posts_queue = []
    st.session_state.active_data_source = None

# ==========================================
# 3. SIDEBAR CONTROLS & HYPERPARAMETERS
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Control Center")

    algo_choice = st.radio(
        "🧠 Algorithm Engine",
        [
            "🔥 SADStream — Adaptive Decay & Hybrid Memory",
            "⚡ DenStream — Fixed Exponential Decay (Baseline)"
        ],
        help="SADStream uses per-cluster adaptive decay λ_k(t), hybrid multi-modal similarity, and compressed long-term memory."
    )
    selected_algo_name = "SADStream" if "SADStream" in algo_choice else "DenStream"

    # Auto-switch engine if algorithm changed
    if st.session_state.engine_name != selected_algo_name:
        if selected_algo_name == "SADStream":
            st.session_state.engine = SADStreamEngine(
                lambda_0=DEFAULT_LAMBDA, eta=0.8, rho=0.5,
                epsilon=DEFAULT_EPSILON, mu=DEFAULT_MU, beta=0.2,
                theta_R=0.50, alpha=0.7, beta_lex=0.1, gamma_ent=0.1, delta_hash=0.1
            )

        else:
            st.session_state.engine = DenStreamEngine(
                lambda_decay=DEFAULT_LAMBDA, epsilon=DEFAULT_EPSILON, mu=DEFAULT_MU, beta=0.2
            )
        st.session_state.engine_name = selected_algo_name
        for lst in ["posts_history", "vector_log", "timeline_log", "y_true", "y_pred"]:
            st.session_state[lst].clear()
        st.session_state.stream_idx = 0

    data_source = st.radio(
        "🌐 Data Stream Source",
        ["QCRI/CrisisBench (Disaster Stream)", "Synthetic Social Media Stream"],
    )

    # Re-init queue if dataset source changed
    if st.session_state.active_data_source != data_source:
        init_stream_queue(data_source)

    st.markdown("---")
    st.markdown("### 🗺️ Map Projection")
    dim_choice = st.radio("PCA Mode:", ["3D PCA Vector Plot", "2D PCA Vector Plot"])

    st.markdown("---")
    st.markdown(f"### 🎛️ {selected_algo_name} Parameters")

    lambda_decay = st.slider("Base Decay (λ₀)", 0.001, 0.100, DEFAULT_LAMBDA, step=0.005)
    epsilon      = st.slider("Micro-Cluster Radius (ε)", 0.10, 0.80, DEFAULT_EPSILON, step=0.02)
    mu           = st.slider("Potential Threshold (μ)", 1.0, 5.0, DEFAULT_MU, step=0.1)
    eps_macro    = st.slider("Macro-DBSCAN Radius (ε_macro)", 0.20, 0.60, DEFAULT_EPS_MACRO, step=0.05)

    eta_val, rho_val, theta_R_val = 0.8, 0.5, 0.50
    alpha_val, beta_lex_val, gamma_ent_val, delta_hash_val = 0.7, 0.1, 0.1, 0.1

    if selected_algo_name == "SADStream":
        with st.expander("🔬 Advanced SADStream Parameters"):
            eta_val      = st.slider("Burstiness Weight (η)", 0.0, 2.0, 0.8, step=0.1)
            rho_val      = st.slider("Activity Weight (ρ)", 0.0, 2.0, 0.5, step=0.1)
            theta_R_val  = st.slider("Recurring Threshold (θ_R)", 0.30, 0.80, 0.50, step=0.05)

            st.markdown("**Hybrid Similarity Weights:**")
            alpha_val     = st.slider("Semantic (α)", 0.1, 1.0, 0.7, step=0.05)
            beta_lex_val  = st.slider("Lexical (β)", 0.0, 0.5, 0.1, step=0.05)
            gamma_ent_val = st.slider("Entity (γ)", 0.0, 0.5, 0.1, step=0.05)
            delta_hash_val= st.slider("Hashtag (δ)", 0.0, 0.5, 0.1, step=0.05)

    stream_speed = st.slider("⚡ Stream Speed (posts/sec)", 1, 20, 5)

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.is_running = True
    with col_btn2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.is_running = False

    if st.button("🔄 Reset Engine", use_container_width=True):
        if selected_algo_name == "SADStream":
            st.session_state.engine = SADStreamEngine(
                lambda_0=lambda_decay, eta=eta_val, rho=rho_val,
                epsilon=epsilon, mu=mu, beta=0.2, theta_R=theta_R_val,
                alpha=alpha_val, beta_lex=beta_lex_val,
                gamma_ent=gamma_ent_val, delta_hash=delta_hash_val
            )
        else:
            st.session_state.engine = DenStreamEngine(
                lambda_decay=lambda_decay, epsilon=epsilon, mu=mu, beta=0.2
            )
        st.session_state.macro_engine = MacroClusterEngine(eps_macro=eps_macro)
        for lst in ["posts_history", "vector_log", "timeline_log", "y_true", "y_pred"]:
            st.session_state[lst].clear()
        st.session_state.stream_idx = 0
        st.session_state.is_running = False
        st.rerun()

# ==========================================
# 4. DASHBOARD HEADER & LIVE KPIS
# ==========================================
status_indicator = "🟢 Running" if st.session_state.is_running else "⏸️ Paused"
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
    <div class="app-header-title">🔍 SADStream Topic Intelligence Dashboard</div>
    <span style="font-size:0.85rem;color:#64748b;background:white;border:1px solid #e2e8f0;padding:4px 12px;border-radius:20px">{status_indicator}</span>
</div>
<div class="app-header-sub">
    <span class="app-engine-badge">{selected_algo_name}</span>
    &nbsp;Adaptive Semantic Stream Clustering · Compressed Prototype Memory · Macro-DBSCAN · c-TF-IDF Topic Labeling
</div>
""", unsafe_allow_html=True)

update_kpi_cards()

st.markdown("---")

# ==========================================
# 5. ACTIVE VIEW SELECTOR & MAIN CONTENT
# ==========================================
active_view = st.radio(
    "📺 View",
    [
        "🔥 Detected Topics",
        "🗺️ Cluster Vector Map",
        "📈 Stream Dynamics",
        "📡 Live Feed",
        "🔬 Algorithm Simulation",
        "🎯 Benchmark Diagnostics",
    ],
    horizontal=True,
    key="active_view_selector_radio"
)

st.markdown("---")

# Static render when not streaming
if not st.session_state.is_running:
    render_selected_view(active_view, dim_choice)

# ==========================================
# 6. REAL-TIME STREAMING LOOP
# ==========================================
if st.session_state.is_running:
    engine        = st.session_state.engine
    embedder      = st.session_state.embedder
    macro_engine  = st.session_state.macro_engine

    # Push slider values into engine
    if isinstance(engine, SADStreamEngine):
        engine.lambda_0     = lambda_decay
        engine.eta          = eta_val
        engine.rho          = rho_val
        engine.theta_R      = theta_R_val
        engine.alpha        = alpha_val
        engine.beta_lex     = beta_lex_val
        engine.gamma_ent    = gamma_ent_val
        engine.delta_hash   = delta_hash_val
    elif isinstance(engine, DenStreamEngine):
        engine.lambda_decay = lambda_decay

    engine.epsilon         = epsilon
    engine.mu              = mu
    macro_engine.eps_macro = eps_macro

    # Populate dataset queue if empty or missing
    if not st.session_state.stream_posts_queue or st.session_state.active_data_source != data_source:
        init_stream_queue(data_source)

    if st.session_state.stream_posts_queue:
        # Loop around when reaching end of dataset
        if st.session_state.stream_idx >= len(st.session_state.stream_posts_queue):
            st.session_state.stream_idx = 0

        post = st.session_state.stream_posts_queue[st.session_state.stream_idx]
        st.session_state.stream_idx += 1

        clean_text = TextPreprocessor.clean_text(post.text)
        vec        = embedder.encode(clean_text)
        status     = engine.process_post(post, vec)

        ground_truth = post.category_hint or "unknown"
        st.session_state.y_true.append(ground_truth)

        p_mcs   = engine.p_micro_clusters
        o_mcs   = engine.o_micro_clusters
        all_mcs = p_mcs + o_mcs
        closest_id = hash(min(all_mcs, key=lambda mc: mc.distance_to_point(vec)).cluster_id) % 100000 \
                     if all_mcs else -1
        st.session_state.y_pred.append(closest_id)

        st.session_state.posts_history.insert(0, {
            "Time":     time.strftime("%H:%M:%S", time.localtime(post.timestamp)),
            "Author":   post.author_id,
            "Category": ground_truth,
            "Status":   status,
            "Text":     clean_text
        })
        st.session_state.vector_log.append({"vector": vec, "status": status, "text": clean_text})

        topics    = macro_engine.generate_topics(p_mcs)
        max_burst = max((t.burst_score for t in topics), default=0.0)
        st.session_state.timeline_log.append({
            "PostIndex":    engine.total_processed,
            "p_MC_Count":   len(p_mcs),
            "o_MC_Count":   len(engine.o_micro_clusters),
            "TopicCount":   len(topics),
            "MaxBurstScore": max_burst
        })

        # Render once for this post, then trigger rerun for next post
        update_kpi_cards()
        render_selected_view(active_view, dim_choice)

        time.sleep(1.0 / stream_speed)
        st.rerun()
