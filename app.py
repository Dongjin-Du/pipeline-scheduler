"""
Pipeline Inspection Scheduler — Streamlit App
==============================================
Inputs:
  - Number of pipe segments
  - RUL distribution (type + parameters)
  - Inspection threshold (fraction of RUL remaining that triggers replacement)
  - Planning horizon, costs

Outputs:
  - Best inspection interval (optimized or adaptive)
  - Full schedule table
  - Cost estimation plots
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os

# Ensure the project root (the folder containing 'core/') is on the path,
# regardless of where streamlit is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import (
    RULDistribution, DistributionParams, CostParams,
    make_bs, make_weibull, make_normal, make_lognormal,
    optimize_inspection_schedule, run_adaptive_simulation, results_to_dataframe,
    plot_rul_distribution, plot_replacements_per_inspection,
    plot_cumulative_cost, plot_cost_breakdown, plot_inspection_intervals,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pipeline Inspection Scheduler",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title { font-size: 2rem; font-weight: 800; color: #065A82; margin-bottom: 0; }
  .sub-title  { font-size: 1rem; color: #4A7A8A; margin-top: 0; }
  .metric-box { background: #F0F7FB; border-left: 4px solid #065A82;
                padding: 0.8rem 1.2rem; border-radius: 6px; margin: 0.4rem 0; }
  .section-header { font-size: 1.1rem; font-weight: 700; color: #065A82;
                    border-bottom: 2px solid #e0eef4; padding-bottom: 4px; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Inputs ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # --- System ---
    st.markdown("### 🔩 Pipeline System")
    n_segments = st.number_input("Number of pipe segments", min_value=10, max_value=10000,
                                  value=400, step=10)
    horizon = st.number_input("Planning horizon (years)", min_value=5, max_value=200,
                               value=50, step=5)

    # --- RUL Distribution ---
    st.markdown("### 📊 RUL Distribution")
    dist_type = st.selectbox("Distribution type", [
        "Birnbaum-Saunders (Fatigue Life)",
        "Weibull",
        "Normal",
        "Lognormal",
    ])

    rul_dist = None

    if dist_type == "Birnbaum-Saunders (Fatigue Life)":
        st.caption("Shape (α) controls variability; Scale (β) ≈ median RUL.")
        bs_shape = st.slider("Shape α", 0.1, 2.0, 0.5, 0.05)
        bs_scale = st.number_input("Scale β (years)", min_value=1.0, max_value=500.0, value=60.0)
        rul_dist = make_bs(shape=bs_shape, scale=bs_scale)

    elif dist_type == "Weibull":
        st.caption("Shape k>1 = wear-out failure; Scale λ ≈ characteristic life.")
        w_shape = st.slider("Shape k", 0.5, 5.0, 2.0, 0.1)
        w_scale = st.number_input("Scale λ (years)", min_value=1.0, max_value=500.0, value=70.0)
        rul_dist = make_weibull(shape=w_shape, scale=w_scale)

    elif dist_type == "Normal":
        st.caption("Symmetric bell curve around mean RUL.")
        n_mean = st.number_input("Mean RUL (years)", min_value=1.0, max_value=500.0, value=60.0)
        n_std  = st.number_input("Std dev (years)",  min_value=0.1, max_value=200.0, value=15.0)
        rul_dist = make_normal(mean=n_mean, std=n_std)

    elif dist_type == "Lognormal":
        st.caption("μ and σ are the mean and std of ln(RUL).")
        ln_mu    = st.number_input("μ (log-mean)", min_value=0.0, max_value=10.0, value=4.1, step=0.1)
        ln_sigma = st.slider("σ (log-std)", 0.05, 2.0, 0.4, 0.05)
        rul_dist = make_lognormal(mu=ln_mu, sigma=ln_sigma)

    # --- Inspection threshold ---
    st.markdown("### 🔍 Inspection Policy")
    threshold = st.slider(
        "Replacement threshold (fraction of RUL remaining)",
        min_value=0.05, max_value=0.60, value=0.20, step=0.05,
        help="Replace a segment at inspection if its remaining RUL / original RUL is below this value."
    )
    mode = st.radio("Scheduling mode", ["Adaptive (paper method)", "Optimized (Monte Carlo)"],
                    help="Adaptive mirrors the paper's convergence behavior. Optimized finds the best fixed interval.")

    # --- Costs ---
    st.markdown("### 💰 Costs")
    insp_cost = st.number_input("Inspection cost per segment ($)", min_value=10, max_value=50000,
                                 value=500, step=50)
    repl_cost = st.number_input("Replacement cost per segment ($)", min_value=100, max_value=500000,
                                 value=5000, step=500)

    # --- Advanced ---
    with st.expander("Advanced options"):
        mc_trials = st.slider("Monte Carlo trials (optimizer)", 10, 100, 30, 5)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)


# ── Main panel ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🔧 Pipeline Inspection Scheduler</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Reliability-constrained, cost-optimal inspection planning · Bricks Model</p>',
            unsafe_allow_html=True)

# Always show distribution preview
col_dist, col_stats = st.columns([3, 1])
with col_dist:
    st.markdown('<p class="section-header">RUL Distribution Preview</p>', unsafe_allow_html=True)
    if rul_dist:
        st.pyplot(plot_rul_distribution(rul_dist), use_container_width=True)

with col_stats:
    st.markdown('<p class="section-header">Distribution Stats</p>', unsafe_allow_html=True)
    if rul_dist:
        st.markdown(f"""
        <div class="metric-box">
          <b>Mean RUL</b><br>{rul_dist.mean():.1f} years
        </div>
        <div class="metric-box">
          <b>Std Dev</b><br>{rul_dist.std():.1f} years
        </div>
        <div class="metric-box">
          <b>5th pct</b><br>{rul_dist.ppf(0.05):.1f} years
        </div>
        <div class="metric-box">
          <b>50th pct</b><br>{rul_dist.ppf(0.50):.1f} years
        </div>
        <div class="metric-box">
          <b>95th pct</b><br>{rul_dist.ppf(0.95):.1f} years
        </div>
        """, unsafe_allow_html=True)


# ── Run simulation ───────────────────────────────────────────────────────────

if run_btn and rul_dist:
    cost_params = CostParams(
        inspection_cost_per_segment=insp_cost,
        replacement_cost_per_segment=repl_cost,
    )

    with st.spinner("Running simulation…"):
        if mode == "Adaptive (paper method)":
            result = run_adaptive_simulation(
                n_segments=int(n_segments),
                rul_dist=rul_dist,
                inspection_threshold=threshold,
                cost_params=cost_params,
                horizon_years=float(horizon),
                seed=int(seed),
            )
        else:
            progress = st.progress(0, text="Optimizing inspection interval…")
            def cb(frac):
                progress.progress(frac, text=f"Optimizing… {int(frac*100)}%")
            result = optimize_inspection_schedule(
                n_segments=int(n_segments),
                rul_dist=rul_dist,
                inspection_threshold=threshold,
                cost_params=cost_params,
                horizon_years=float(horizon),
                n_mc_trials=mc_trials,
                seed=int(seed),
                progress_callback=cb,
            )
            progress.empty()

    # ── Summary KPIs ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Inspection Plan Summary</p>', unsafe_allow_html=True)

    total_inspections = len(result.events)
    total_replacements = sum(result.replacements_per_inspection)
    avg_replaced = np.mean(result.replacements_per_inspection)
    avg_rate = 100 * avg_replaced / n_segments

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("First Interval",     f"{result.first_interval:.2f} yrs")
    k2.metric("Converged Interval", f"{result.converged_interval:.2f} yrs")
    k3.metric("Total Inspections",  total_inspections)
    k4.metric("Total Replacements", f"{total_replacements:,}")
    k5.metric("Avg Replaced/Inspection", f"{avg_replaced:.0f} ({avg_rate:.1f}%)")
    k6.metric("Total Cost",         f"${result.total_cost:,.0f}")

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Visualizations</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_replacements_per_inspection(result), use_container_width=True)
    with c2:
        st.pyplot(plot_inspection_intervals(result), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_cumulative_cost(result), use_container_width=True)
    with c4:
        st.pyplot(plot_cost_breakdown(result), use_container_width=True)

    # ── Schedule table ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📅 Full Inspection Schedule</p>', unsafe_allow_html=True)

    df = results_to_dataframe(result)

    col_filter, col_dl = st.columns([3, 1])
    with col_filter:
        show_all = st.checkbox("Show all rows", value=False)
    with col_dl:
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "inspection_schedule.csv", "text/csv",
                           use_container_width=True)

    if not show_all:
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)
        if len(df) > 30:
            st.caption(f"Showing 30 of {len(df)} inspections. Check 'Show all rows' for full table.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Cost summary ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">💰 Cost Summary</p>', unsafe_allow_html=True)

    total_insp_cost = sum(ev.inspection_cost  for ev in result.events)
    total_repl_cost = sum(ev.replacement_cost for ev in result.events)

    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Total Inspection Cost", f"${total_insp_cost:,.0f}",
               delta=f"{100*total_insp_cost/result.total_cost:.0f}% of total")
    cs2.metric("Total Replacement Cost", f"${total_repl_cost:,.0f}",
               delta=f"{100*total_repl_cost/result.total_cost:.0f}% of total")
    cs3.metric("Annualized Cost", f"${result.total_cost/horizon:,.0f} / year")

elif not run_btn:
    st.info("👈 Configure your pipeline system in the sidebar and click **Run Simulation** to generate the inspection plan.")
