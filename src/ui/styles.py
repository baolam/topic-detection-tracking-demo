import streamlit as st

# ─── Sensitive & Responsive SADStream Hyperparameters ───────────────────────
DEFAULT_LAMBDA    = 0.020   # λ₀ base decay: nhạy (giữ tín hiệu mới lâu hơn)
DEFAULT_EPSILON   = 0.68    # ε micro radius: rộng vừa đủ để dễ gom bài viết tương đồng
DEFAULT_MU        = 1.20    # μ potential threshold: nhạy (1-2 bài viết hình thành topic ngay)
DEFAULT_EPS_MACRO = 0.45    # ε_macro DBSCAN radius: nhạy gom các micro-cluster lân cận


def apply_custom_styles():
    """Light theme — dark text, white sidebar, no dark top bar."""
    st.markdown("""
    <style>
        /* ══ GLOBAL BASE ══ */
        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="block-container"] {
            background-color: #f5f7fa !important;
            color: #1e293b !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Remove dark header/toolbar */
        [data-testid="stHeader"],
        header[data-testid="stHeader"],
        .stDeployButton,
        #MainMenu { display: none !important; }

        /* ══ SIDEBAR ══ */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #1e293b !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #0f172a !important;
            font-weight: 700;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #1e293b !important;
        }

        /* ══ MAIN CONTENT TEXT ══ */
        p, span, div, label, li,
        .stMarkdown, .stMarkdown p,
        .stMarkdown h1, .stMarkdown h2,
        .stMarkdown h3, .stMarkdown h4 {
            color: #1e293b;
        }

        /* Radio button labels (main + sidebar) */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        [data-testid="stRadio"] span {
            color: #1e293b !important;
            font-size: 0.88rem;
        }

        /* ══ HEADER BRAND ══ */
        .app-header-title {
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 50%, #0891b2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .app-header-sub {
            color: #64748b !important;
            font-size: 0.9rem;
            margin-bottom: 16px;
        }
        .app-engine-badge {
            display: inline-block;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        /* ══ KPI METRICS ══ */
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        [data-testid="stMetricLabel"] > div {
            color: #64748b !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] > div {
            color: #1e293b !important;
            font-weight: 800 !important;
            font-size: 1.4rem !important;
        }
        [data-testid="stMetricDelta"] > div {
            font-size: 0.77rem !important;
        }

        /* ══ PLOTLY CHARTS ══ */
        [data-testid="stPlotlyChart"] {
            background: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            padding: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* ══ EXPANDER ══ */
        [data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            background: #ffffff !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {
            color: #1e293b !important;
            font-weight: 600;
        }

        /* ══ DATAFRAME ══ */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        /* ══ INFO / ALERT ══ */
        [data-testid="stAlert"],
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            border-radius: 8px;
            color: #1e40af !important;
        }

        /* ══ BUTTONS ══ */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            background: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
        }
        .stButton > button:not([kind="primary"]) {
            background: #f1f5f9 !important;
            color: #1e293b !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* ══ SELECTBOX / DROPDOWN FIX ══ */
        [data-baseweb="select"],
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        ul[role="listbox"],
        li[role="option"] {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #f1f5f9 !important;
            color: #2563eb !important;
        }

        /* ══ DIVIDER ══ */
        hr { border-color: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

