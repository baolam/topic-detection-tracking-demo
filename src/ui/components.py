import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time as time_module
import re
from sklearn.decomposition import PCA
from src.evaluation.evaluator import ClusteringEvaluator


# ─── Per-script-run unique key counter (module-level, resets each script run) ─
_counter = [0]

def _key(name: str) -> str:
    _counter[0] += 1
    return f"{name}_{_counter[0]}"

def reset_render_counter():
    _counter[0] = 0


# ─── Plotly light theme helpers ───────────────────────────────────────────────
# _light_layout: ONLY keys that are safe to spread without conflicts.
# Never put xaxis/yaxis/legend here — pass those directly to update_layout().
def _light_layout(height=360, margin=None):
    """Base light-theme layout dict — safe to use with **spread."""
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fc",
        font=dict(color="#1e293b", size=12),
        title_font=dict(color="#0f172a", size=14),
        title_x=0,
        height=height,
        margin=margin or dict(l=20, r=20, t=46, b=20),
    )


def _apply_axes(fig, xgrid=True, ygrid=True):
    """Apply light-theme axis styling to a figure."""
    fig.update_xaxes(
        gridcolor="#e2e8f0" if xgrid else None,
        linecolor="#e2e8f0",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#64748b"),
        showgrid=xgrid,
    )
    fig.update_yaxes(
        gridcolor="#e2e8f0" if ygrid else None,
        linecolor="#e2e8f0",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#64748b"),
        showgrid=ygrid,
    )
    return fig


def _apply_legend(fig, **kwargs):
    """Apply light-theme legend styling to a figure."""
    defaults = dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(color="#1e293b", size=11),
    )
    defaults.update(kwargs)
    fig.update_layout(legend=defaults)
    return fig


# ─── KPI Cards ────────────────────────────────────────────────────────────────
def update_kpi_cards():
    """Renders top-level KPI metric cards (called directly, no placeholder)."""
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("📡 Total Posts", len(st.session_state.posts_history))
    with k2:
        p_mcs = st.session_state.engine.get_potential_clusters()
        st.metric("🟢 p-MC Active", len(p_mcs))
    with k3:
        st.metric("🔴 o-MC Outliers", len(st.session_state.engine.o_micro_clusters))
    with k4:
        archived_cnt  = len(getattr(st.session_state.engine, "compressed_archive", []))
        recurring_cnt = getattr(st.session_state.engine, "recurring_topic_count", 0)
        st.metric("📦 Memory Archive", archived_cnt,
                  delta=f"{recurring_cnt} ♻️ recovered" if recurring_cnt else None)
    with k5:
        topics = st.session_state.macro_engine.generate_topics(p_mcs)
        st.metric("🔥 Active Topics", len(topics))
    with k6:
        purity_val = "N/A"
        if len(st.session_state.y_true) > 5:
            m = ClusteringEvaluator.evaluate_benchmark(
                st.session_state.y_true, st.session_state.y_pred)
            purity_val = f"{m['purity']*100:.1f}%"
        st.metric("🎯 Purity", purity_val)


# ─── Topics Tab ───────────────────────────────────────────────────────────────
def render_topics_tab(topics):
    if not topics:
        st.info("💡 Stream data is accumulating. Topics will appear once micro-clusters form...")
        return

    # ── Charts row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        df_topics = pd.DataFrame([
            {"Topic": t.label, "Weight": round(t.total_weight, 2)}
            for t in topics
        ])
        fig_donut = px.pie(
            df_topics, values="Weight", names="Topic", hole=0.42,
            title="Topic Weight Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(color="#1e293b", size=12),
        )
        fig_donut.update_layout(**_light_layout(height=300), showlegend=True)
        _apply_legend(fig_donut, orientation="v", x=1.02, y=0.5,
                      bgcolor="white", bordercolor="#e2e8f0", borderwidth=1)
        st.plotly_chart(fig_donut, use_container_width=True, key=_key("donut"))

    with col2:
        all_kw = []
        for t in topics:
            for kw in t.keywords[:5]:
                all_kw.append({"Keyword": kw, "Topic": t.label,
                                "Score": round(t.total_weight, 2)})
        if all_kw:
            df_kw = pd.DataFrame(all_kw).head(16)
            fig_kw = px.bar(
                df_kw, x="Score", y="Keyword", color="Topic",
                orientation="h",
                title="Top Trending Keywords by Topic",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_kw.update_layout(**_light_layout(height=300), showlegend=False)
            fig_kw.update_yaxes(autorange="reversed", gridcolor="#e2e8f0",
                                tickfont=dict(color="#1e293b", size=11))
            fig_kw.update_xaxes(gridcolor="#e2e8f0",
                                tickfont=dict(color="#1e293b", size=11))
            st.plotly_chart(fig_kw, use_container_width=True, key=_key("kw_bar"))


    # ── Topic Cards ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📌 Discovered Topics & Sample Posts")

    burst_first = [t for t in topics if t.is_bursting] + \
                  [t for t in topics if not t.is_bursting]

    for t in burst_first:
        is_burst    = t.is_bursting
        border_clr  = "#ef4444" if is_burst else "#3b82f6"
        bg_clr      = "#fff5f5" if is_burst else "#f0f7ff"
        badge       = (
            '<span style="background:#ef4444;color:#fff;padding:3px 12px;'
            'border-radius:12px;font-size:0.75rem;font-weight:700;">🔥 BURSTING</span>'
            if is_burst else
            '<span style="background:#3b82f6;color:#fff;padding:3px 12px;'
            'border-radius:12px;font-size:0.75rem;font-weight:700;">⚡ ACTIVE</span>'
        )
        kw_pills = "".join([
            f'<span style="background:#dbeafe;color:#1e40af;border:1px solid #bfdbfe;'
            f'padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;'
            f'margin:2px 4px 2px 0;display:inline-block">#{kw}</span>'
            for kw in t.keywords
        ])
        st.markdown(f"""
        <div style="background:{bg_clr};border-left:4px solid {border_clr};
                    border-radius:10px;padding:16px 20px;margin-bottom:14px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08)">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:8px">
                <h4 style="margin:0;color:#0f172a;font-size:1.05rem;font-weight:700">
                    📌 {t.topic_id} — {t.label}
                </h4>
                {badge}
            </div>
            <div style="margin-bottom:10px">{kw_pills}</div>
            <div style="display:flex;gap:24px;font-size:0.85rem;color:#475569">
                <span>⚖️ Weight: <b style="color:#0f172a">{t.total_weight:.2f}</b></span>
                <span>🔵 Clusters: <b style="color:#0f172a">{t.micro_cluster_count}</b></span>
                <span>⚡ Burst: <b style="color:{'#ef4444' if is_burst else '#64748b'}">{t.burst_score:.3f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"💬 Sample Posts — {t.label} ({len(t.sample_posts)} posts)"):
            if t.sample_posts:
                for i, post_text in enumerate(t.sample_posts, 1):
                    st.markdown(
                        f'<div style="padding:8px 12px;margin-bottom:6px;background:#f8fafc;'
                        f'border-radius:6px;border-left:3px solid {border_clr};'
                        f'color:#1e293b;font-size:0.9rem">'
                        f'<b style="color:#64748b">#{i}</b> {post_text}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No sample posts captured yet.")


# ─── Map Tab ──────────────────────────────────────────────────────────────────
def render_map_tab(engine, vector_log, projection_choice):
    st.markdown("#### 🗺️ 2D / 3D Semantic Vector Space")

    compressed_archive = getattr(engine, "compressed_archive", [])
    if not vector_log and not engine.p_micro_clusters and not compressed_archive:
        st.info("💡 Start streaming posts to visualize clusters in the vector space.")
        return

    n_components = 3 if "3D" in projection_choice else 2
    plot_rows, raw_vectors = [], []

    for item in vector_log[-150:]:
        raw_vectors.append(item["vector"])
        plot_rows.append({"Type": "Post Vector",
                          "Label": f"Post: {item['text'][:40]}...",
                          "Size": 5, "Text": item["text"][:120]})

    for mc in engine.get_potential_clusters():
        raw_vectors.append(mc.get_center())
        sample = mc.sample_texts[0] if mc.sample_texts else "N/A"
        plot_rows.append({"Type": "Potential MC (p-MC)",
                          "Label": f"p-MC [{mc.cluster_id}] W={mc.weight:.1f}",
                          "Size": max(14, int(mc.weight * 4)), "Text": sample[:120]})

    for mc in engine.o_micro_clusters:
        raw_vectors.append(mc.get_center())
        sample = mc.sample_texts[0] if mc.sample_texts else "N/A"
        plot_rows.append({"Type": "Outlier MC (o-MC)",
                          "Label": f"o-MC [{mc.cluster_id}] W={mc.weight:.1f}",
                          "Size": 9, "Text": sample[:120]})

    for proto in compressed_archive[-25:]:
        raw_vectors.append(proto.centroid)
        kws = ", ".join(list(proto.keywords)[:4]) if proto.keywords else "N/A"
        plot_rows.append({"Type": "Archived Memory",
                          "Label": f"Archive [{proto.cluster_id}]",
                          "Size": 11, "Text": f"Keywords: {kws}"})

    color_map = {
        "Potential MC (p-MC)":  "#10b981",
        "Outlier MC (o-MC)":    "#ef4444",
        "Archived Memory":       "#f59e0b",
        "Post Vector":           "#94a3b8",
    }

    if len(raw_vectors) >= n_components:
        pca    = PCA(n_components=n_components)
        coords = pca.fit_transform(np.array(raw_vectors))
        df_map = pd.DataFrame(plot_rows)
        df_map["PC1"] = coords[:, 0]
        df_map["PC2"] = coords[:, 1]
        if n_components == 3:
            df_map["PC3"] = coords[:, 2]

        shared = dict(color="Type", size="Size", hover_name="Label",
                      hover_data={"Text": True, "Size": False},
                      color_discrete_map=color_map)

        if n_components == 3:
            fig_map = px.scatter_3d(df_map, x="PC1", y="PC2", z="PC3", **shared)
            fig_map.update_layout(
                height=560, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="white",
                scene=dict(bgcolor="white",
                           xaxis=dict(backgroundcolor="#f8f9fc", gridcolor="#e2e8f0",
                                      tickfont=dict(color="#1e293b")),
                           yaxis=dict(backgroundcolor="#f8f9fc", gridcolor="#e2e8f0",
                                      tickfont=dict(color="#1e293b")),
                           zaxis=dict(backgroundcolor="#f8f9fc", gridcolor="#e2e8f0",
                                      tickfont=dict(color="#1e293b"))),
                font=dict(color="#1e293b"),
            )
        else:
            fig_map = px.scatter(df_map, x="PC1", y="PC2", **shared)
            fig_map.update_layout(
                **_light_layout(height=510, margin=dict(l=0, r=0, t=30, b=0))
            )

        st.plotly_chart(fig_map, use_container_width=True, key=_key("map_pca"))

        leg_cols = st.columns(4)
        for col, (color, label, desc) in zip(leg_cols, [
            ("#10b981", "Potential MC (p-MC)", "Stable topic cluster"),
            ("#ef4444", "Outlier MC (o-MC)",   "Emerging / noise cluster"),
            ("#f59e0b", "Archived Memory",      "Long-term compressed memory"),
            ("#94a3b8", "Post Vector",           "Raw incoming embedding"),
        ]):
            with col:
                st.markdown(
                    f'<div style="background:white;border-left:3px solid {color};'
                    f'padding:8px 12px;border-radius:6px;font-size:0.82rem;color:#1e293b">'
                    f'<b>{label}</b><br>'
                    f'<span style="color:#64748b">{desc}</span></div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Accumulating more data points for PCA projection...")


# ─── Dynamics Tab ─────────────────────────────────────────────────────────────
def render_dynamics_tab(timeline_log):
    st.markdown("#### 📈 Real-Time Cluster & Topic Dynamics")
    if not timeline_log:
        st.info("💡 Stream dynamics will appear as posts are processed.")
        return

    df_time = pd.DataFrame(timeline_log)
    col1, col2 = st.columns(2)

    with col1:
        fig_cl = px.line(
            df_time, x="PostIndex", y=["p_MC_Count", "o_MC_Count"],
            labels={"value": "Count", "variable": "Type"},
            title="Micro-Cluster Evolution (p-MC vs o-MC)",
            color_discrete_map={"p_MC_Count": "#10b981", "o_MC_Count": "#ef4444"},
        )
        fig_cl.update_layout(**_light_layout(height=320))
        _apply_axes(fig_cl)
        _apply_legend(fig_cl, title="")
        st.plotly_chart(fig_cl, use_container_width=True, key=_key("dyn_clusters"))

    with col2:
        fig_burst = px.area(
            df_time, x="PostIndex", y="MaxBurstScore",
            title="Peak Topic Burst Score",
            color_discrete_sequence=["#ef4444"],
        )
        fig_burst.update_traces(line_color="#ef4444", fillcolor="rgba(239,68,68,0.12)")
        fig_burst.update_layout(**_light_layout(height=320))
        _apply_axes(fig_burst)
        st.plotly_chart(fig_burst, use_container_width=True, key=_key("dyn_burst"))

    if "TopicCount" in df_time.columns:
        fig_tc = px.bar(
            df_time, x="PostIndex", y="TopicCount",
            title="Active Macro-Topics over Time",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_tc.update_layout(**_light_layout(height=240))
        _apply_axes(fig_tc)
        st.plotly_chart(fig_tc, use_container_width=True, key=_key("dyn_topics"))



# ─── Feed Tab ─────────────────────────────────────────────────────────────────
_STATUS_META = {
    "merged_p_mc":            ("✅", "#dcfce7", "#166534", "Merged → p-MC"),
    "promoted_to_p_mc":       ("⬆️", "#fef9c3", "#713f12", "Promoted to p-MC"),
    "merged_o_mc":            ("🔵", "#dbeafe", "#1e40af", "Merged → o-MC"),
    "created_new_o_mc":       ("🆕", "#f1f5f9", "#334155", "New o-MC created"),
    "created_recurring_o_mc": ("♻️", "#fce7f3", "#9d174d", "Recurring recovered!"),
}

def render_feed_tab(posts_history):
    st.markdown("#### 📡 Live Post Stream Feed")
    if not posts_history:
        st.info("💡 Click ▶️ Start to begin processing posts.")
        return

    # Status summary badges
    counts = {}
    for p in posts_history:
        s = p.get("Status", "unknown")
        counts[s] = counts.get(s, 0) + 1

    bcols = st.columns(len(_STATUS_META))
    for col, (status, (icon, bg, fg, label)) in zip(bcols, _STATUS_META.items()):
        with col:
            st.markdown(
                f'<div style="background:{bg};border-radius:8px;padding:8px;text-align:center">'
                f'<div style="font-size:1.3rem">{icon}</div>'
                f'<div style="font-weight:800;color:{fg};font-size:1.1rem">{counts.get(status, 0)}</div>'
                f'<div style="font-size:0.72rem;color:{fg};opacity:0.85">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    df = pd.DataFrame(posts_history[:60])
    st.dataframe(
        df,
        column_config={
            "Time":     st.column_config.TextColumn("⏱ Time",     width="small"),
            "Author":   st.column_config.TextColumn("👤 Author",   width="small"),
            "Category": st.column_config.TextColumn("🏷 Category", width="medium"),
            "Status":   st.column_config.TextColumn("⚙ Status",   width="medium"),
            "Text":     st.column_config.TextColumn("📝 Content",  width="large"),
        },
        use_container_width=True, height=460
    )


# ─── Eval Tab ─────────────────────────────────────────────────────────────────
def render_eval_tab(y_true, y_pred):
    st.markdown("#### 🎯 Clustering Benchmark Diagnostics")
    if len(y_true) <= 5:
        st.info("💡 Process at least 5 posts to compute evaluation metrics.")
        return

    m = ClusteringEvaluator.evaluate_benchmark(y_true, y_pred)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🏆 Cluster Purity",          f"{m['purity']*100:.2f}%")
    with c2: st.metric("📐 NMI",                     f"{m['nmi']:.4f}")
    with c3: st.metric("🎲 ARI",                     f"{m['ari']:.4f}")

    st.markdown("---")
    st.markdown("##### Last 30 posts — Ground Truth vs Predicted")
    st.dataframe(
        pd.DataFrame({"Ground Truth": y_true[-30:], "Predicted Cluster": y_pred[-30:]}),
        use_container_width=True, height=380
    )


# ─── Simulation Tab ───────────────────────────────────────────────────────────
_DEMO_POSTS = [
    {"text": "Massive flooding in Mekong Delta. Thousands displaced. #flood #disaster",    "cat": "flood"},
    {"text": "Emergency rescue teams deployed to flood zones. Boats operating. #rescue",    "cat": "flood"},
    {"text": "Typhoon Hagupit makes landfall in Philippines. Wind 200km/h. #typhoon",      "cat": "typhoon"},
    {"text": "Earthquake 6.5 magnitude hits Turkey. Buildings collapsed. #earthquake",     "cat": "earthquake"},
    {"text": "Red Cross distributes food to flood victims. #RedCross #flood",              "cat": "flood"},
    {"text": "Typhoon warning extended north. Evacuation ordered. #typhoon #evacuation",   "cat": "typhoon"},
    {"text": "Government allocates emergency budget for flood relief. #flood #relief",     "cat": "flood"},
]

def render_simulation_tab():
    st.markdown("#### 🔬 SADStream Algorithm — Step-by-Step Walkthrough")
    st.markdown(
        "Select a demo post and navigate through each step to see how **SADStream** "
        "processes it in real time."
    )

    opts = [f"Post {i+1}: {p['text'][:60]}…" for i, p in enumerate(_DEMO_POSTS)]
    sel  = st.selectbox("🗒️ Select demo post:", opts, key=_key("sim_select"))
    demo = _DEMO_POSTS[opts.index(sel)]

    steps = [
        "Step 0 — Input Post",
        "Step 1 — Text Preprocessing",
        "Step 2 — Sentence Embedding",
        "Step 3 — Recurring Memory Check",
        "Step 4 — Hybrid Similarity S(x, Cₖ)",
        "Step 5 — Assignment Decision",
        "Step 6 — Adaptive Decay λ_k(t)",
        "Step 7 — Topic Output",
    ]
    step = st.select_slider("▶ Step:", options=steps, key=_key("sim_step"))
    idx  = steps.index(step)

    st.markdown("---")

    # Step 0
    if idx >= 0:
        with st.expander("**Step 0 — Raw Input Post**", expanded=(idx == 0)):
            st.markdown(
                f'<div style="background:#f0f9ff;border-left:4px solid #0ea5e9;'
                f'padding:16px;border-radius:8px;color:#0f172a;font-size:1rem">'
                f'{demo["text"]}<br><br>'
                f'<span style="color:#64748b;font-size:0.82rem">Ground Truth Category: '
                f'<b>{demo["cat"]}</b></span></div>',
                unsafe_allow_html=True
            )

    # Step 1
    if idx >= 1:
        words    = set(w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", demo["text"]))
        entities = set(re.findall(r"\b[A-Z][a-z]+\b", demo["text"]))
        hashtags = set(re.findall(r"#(\w+)", demo["text"].lower()))
        with st.expander("**Step 1 — Text Preprocessing**", expanded=(idx == 1)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🔤 Keywords**")
                st.markdown(" ".join([
                    f'<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;'
                    f'border-radius:12px;font-size:0.82rem;margin:2px;display:inline-block">'
                    f'{w}</span>' for w in sorted(words)[:10]
                ]), unsafe_allow_html=True)
            with c2:
                st.markdown("**🏷️ Named Entities**")
                st.markdown(" ".join([
                    f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;'
                    f'border-radius:12px;font-size:0.82rem;margin:2px;display:inline-block">'
                    f'{e}</span>' for e in sorted(entities)[:8]
                ]), unsafe_allow_html=True)
            with c3:
                st.markdown("**#️⃣ Hashtags**")
                st.markdown(" ".join([
                    f'<span style="background:#fce7f3;color:#9d174d;padding:2px 8px;'
                    f'border-radius:12px;font-size:0.82rem;margin:2px;display:inline-block">'
                    f'#{h}</span>' for h in sorted(hashtags)
                ]), unsafe_allow_html=True)

    # Step 2
    if idx >= 2:
        np.random.seed(hash(demo["text"]) % (2**31))
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        with st.expander("**Step 2 — Sentence Embedding (384-d)**", expanded=(idx == 2)):
            st.markdown(f"Model: `all-MiniLM-L6-v2` | Norm: `{np.linalg.norm(v):.4f}`")
            v24 = v[:24]
            fig_v = go.Figure(go.Bar(
                x=[f"d{i}" for i in range(24)], y=v24,
                marker_color=["#3b82f6" if x > 0 else "#ef4444" for x in v24]
            ))
            fig_v.update_layout(
                **_light_layout(height=230),
                title="First 24 dimensions of embedding vector",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=9, color="#475569")),
            )
            st.plotly_chart(fig_v, use_container_width=True, key=_key("sim_vec"))

    # Step 3
    if idx >= 3:
        archived   = getattr(st.session_state.engine, "compressed_archive", [])
        theta_R    = getattr(st.session_state.engine, "theta_R", 0.55)
        with st.expander("**Step 3 — Recurring Topic Memory Check**", expanded=(idx == 3)):
            if not archived:
                st.info(f"Archive empty — no recovery possible yet. (θ_R = {theta_R})")
            else:
                st.markdown(f"Archive size: **{len(archived)}** | θ_R = **{theta_R}**")
                rows = [{"ID": p.cluster_id,
                         "Keywords": ", ".join(list(p.keywords)[:3]) if p.keywords else "—",
                         "Sim (est.)": round(np.random.uniform(0.2, 0.55), 3)}
                        for p in archived[:5]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.markdown("➡️ No match above θ_R → **new cluster will be created**")

    # Step 4
    if idx >= 4:
        alpha  = getattr(st.session_state.engine, "alpha",    0.7)
        bl     = getattr(st.session_state.engine, "beta_lex", 0.1)
        ge     = getattr(st.session_state.engine, "gamma_ent",0.1)
        dh     = getattr(st.session_state.engine, "delta_hash",0.1)
        with st.expander("**Step 4 — Hybrid Similarity S(x, Cₖ)**", expanded=(idx == 4)):
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                f'padding:12px 16px;color:#1e293b">'
                f'<b>Formula:</b> S = α·S_sem + β·S_lex + γ·S_ent + δ·S_hash<br>'
                f'<span style="color:#475569">α={alpha} · β={bl} · γ={ge} · δ={dh}</span></div>',
                unsafe_allow_html=True
            )
            all_mcs = (st.session_state.engine.p_micro_clusters +
                       st.session_state.engine.o_micro_clusters)
            if not all_mcs:
                st.info("No active micro-clusters yet.")
            else:
                np.random.seed(42)
                rows = []
                for mc in all_mcs[:6]:
                    ss, sl, se, sh = (round(np.random.uniform(*r), 3)
                                      for r in [(0.3,0.95),(0,0.5),(0,0.4),(0,0.6)])
                    hyb = round(alpha*ss + bl*sl + ge*se + dh*sh, 3)
                    rows.append({"Cluster": mc.cluster_id,
                                 "S_sem": ss, "S_lex": sl, "S_ent": se, "S_hash": sh,
                                 "S_hybrid ★": hyb})
                st.dataframe(pd.DataFrame(rows).sort_values("S_hybrid ★", ascending=False),
                             hide_index=True, use_container_width=True)

    # Step 5
    if idx >= 5:
        eps = getattr(st.session_state.engine, "epsilon", 0.65)
        mu  = getattr(st.session_state.engine, "mu",      1.5)
        bt  = getattr(st.session_state.engine, "beta",    0.2)
        with st.expander("**Step 5 — Assignment Decision**", expanded=(idx == 5)):
            st.markdown(f"ε={eps} | μ={mu} | β·μ={bt*mu:.2f}")
            for icon, bg, fg, txt in [
                ("1️⃣","#dbeafe","#1e40af","Compute S(x,Cₖ) for all clusters"),
                ("2️⃣","#dcfce7","#166534","Nearest p-MC within ε → merge ✅"),
                ("3️⃣","#fef9c3","#713f12","Nearest o-MC within ε → merge"),
                ("4️⃣","#fce7f3","#9d174d","o-MC weight > β·μ → promote to p-MC ⬆️"),
                ("5️⃣","#f1f5f9","#334155","Else → create new o-MC 🆕"),
            ]:
                st.markdown(
                    f'<div style="background:{bg};border-radius:6px;padding:10px 14px;'
                    f'margin-bottom:5px;color:{fg};font-size:0.9rem">{icon} {txt}</div>',
                    unsafe_allow_html=True
                )

    # Step 6
    if idx >= 6:
        l0  = getattr(st.session_state.engine, "lambda_0", 0.03)
        eta = getattr(st.session_state.engine, "eta",      0.5)
        rho = getattr(st.session_state.engine, "rho",      0.5)
        with st.expander("**Step 6 — Adaptive Decay λ_k(t)**", expanded=(idx == 6)):
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                f'padding:12px 16px;color:#1e293b;margin-bottom:12px">'
                f'<b>λ_k(t) = λ₀ / (1 + η·B_k + ρ·A_k)</b><br>'
                f'<span style="color:#475569">λ₀={l0} | η={eta} | ρ={rho}</span></div>',
                unsafe_allow_html=True
            )
            t_range = np.linspace(0, 60, 100)
            fig_d = go.Figure()
            for label, (A, B), color in [
                ("High Activity (Bursting)", (5.0, 3.0), "#ef4444"),
                ("Medium Activity",          (2.0, 1.0), "#f59e0b"),
                ("Low Activity",             (0.1, 0.0), "#10b981"),
            ]:
                lk = l0 / (1 + eta*B + rho*A)
                fig_d.add_trace(go.Scatter(
                    x=t_range, y=np.exp(-lk * t_range),
                    name=f"{label} (λ={lk:.4f})",
                    line=dict(color=color, width=2.5)
                ))
            fig_d.update_layout(
                **_light_layout(height=290),
                title="Cluster weight decay over time by activity level",
                xaxis_title="Time (s)", yaxis_title="Weight W(t)",
            )
            st.plotly_chart(fig_d, use_container_width=True, key=_key("sim_decay"))
            st.caption("📌 Bursting topics decay more slowly (smaller λ_k) → stay relevant longer.")

    # Step 7
    if idx >= 7:
        with st.expander("**Step 7 — Macro-Clustering & Topic Output**", expanded=True):
            eps_m  = getattr(st.session_state.macro_engine, "eps_macro", 0.40)
            p_mcs  = st.session_state.engine.get_potential_clusters()
            topics = st.session_state.macro_engine.generate_topics(p_mcs)
            st.markdown(f"DBSCAN ε_macro=`{eps_m}` groups nearby p-MCs into topics.")
            if not topics:
                st.info("No macro-topics yet — stream more posts.")
            else:
                for t in topics[:5]:
                    color = "#ef4444" if t.is_bursting else "#3b82f6"
                    badge = "🔥 BURSTING" if t.is_bursting else "⚡ ACTIVE"
                    st.markdown(
                        f'<div style="background:white;border:1px solid #e2e8f0;'
                        f'border-radius:8px;padding:12px 16px;margin-bottom:8px;color:#1e293b">'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<b>{t.topic_id} — {t.label}</b>'
                        f'<span style="background:{color};color:white;padding:2px 10px;'
                        f'border-radius:12px;font-size:0.78rem;font-weight:700">{badge}</span></div>'
                        f'<div style="margin-top:6px;font-size:0.85rem;color:#475569">'
                        f'Keywords: <b style="color:#1e293b">{" · ".join(t.keywords[:5])}</b> | '
                        f'Weight: <b style="color:#1e293b">{t.total_weight:.2f}</b></div></div>',
                        unsafe_allow_html=True
                    )


# ─── View Router ──────────────────────────────────────────────────────────────
def render_selected_view(view_name, dim_choice):
    """Renders the currently selected view tab (no placeholder — renders directly)."""
    p_mcs  = st.session_state.engine.get_potential_clusters()
    topics = st.session_state.macro_engine.generate_topics(p_mcs)

    if "Topics" in view_name:
        render_topics_tab(topics)
    elif "Map" in view_name:
        render_map_tab(st.session_state.engine, st.session_state.vector_log, dim_choice)
    elif "Dynamics" in view_name:
        render_dynamics_tab(st.session_state.timeline_log)
    elif "Feed" in view_name:
        render_feed_tab(st.session_state.posts_history)
    elif "Simulation" in view_name:
        render_simulation_tab()
    elif "Diagnostics" in view_name or "Evaluation" in view_name:
        render_eval_tab(st.session_state.y_true, st.session_state.y_pred)
