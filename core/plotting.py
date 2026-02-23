"""
Plotting utilities — all return matplotlib Figure objects for Streamlit.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .simulation import SimulationResult
from .distributions import RULDistribution


PALETTE = {
    "primary": "#065A82",
    "accent":  "#02C39A",
    "warn":    "#F96167",
    "muted":   "#8FA8B8",
    "bg":      "#F7FAFC",
    "dark":    "#1C2B36",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["dark"], labelsize=9)
    ax.set_title(title, color=PALETTE["dark"], fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=PALETTE["dark"], fontsize=9)
    ax.set_ylabel(ylabel, color=PALETTE["dark"], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def plot_rul_distribution(dist: RULDistribution) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    x, y = dist.pdf_range()
    ax.fill_between(x, y, alpha=0.25, color=PALETTE["primary"])
    ax.plot(x, y, color=PALETTE["primary"], linewidth=2)

    mean = dist.mean()
    ax.axvline(mean, color=PALETTE["accent"], linestyle="--", linewidth=1.5,
               label=f"Mean = {mean:.1f} yrs")
    p5  = dist.ppf(0.05)
    p95 = dist.ppf(0.95)
    ax.axvspan(p5, p95, alpha=0.08, color=PALETTE["accent"], label="5th–95th pct")

    ax.legend(fontsize=8, framealpha=0.5)
    _style_ax(ax, title=f"RUL Distribution — {dist.params.label()}",
              xlabel="Remaining Useful Life (years)", ylabel="Probability Density")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    return fig


def plot_replacements_per_inspection(result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    years = [ev.time for ev in result.events]
    replaced = result.replacements_per_inspection
    rates = [100 * r / result.n_segments for r in replaced]

    ax.bar(years, rates, width=result.converged_interval * 0.7,
           color=PALETTE["primary"], alpha=0.8, label="Replacement rate (%)")

    mean_rate = np.mean(rates)
    ax.axhline(mean_rate, color=PALETTE["accent"], linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_rate:.1f}%")

    _style_ax(ax, title="Replacement Rate per Inspection",
              xlabel="Year", ylabel="% Segments Replaced")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cumulative_cost(result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    years = [ev.time for ev in result.events]
    cum = result.cumulative_cost

    ax.plot(years, cum, color=PALETTE["primary"], linewidth=2.5, zorder=3)
    ax.fill_between(years, cum, alpha=0.12, color=PALETTE["primary"])

    ax.scatter(years[-1], cum[-1], color=PALETTE["warn"], zorder=5, s=60,
               label=f"Total: ${cum[-1]:,.0f}")
    ax.legend(fontsize=9)

    _style_ax(ax, title="Cumulative Cost Over Planning Horizon",
              xlabel="Year", ylabel="Cumulative Cost ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_cost_breakdown(result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    years = [ev.time for ev in result.events]
    insp_costs  = [ev.inspection_cost  for ev in result.events]
    repl_costs  = [ev.replacement_cost for ev in result.events]
    width = result.converged_interval * 0.7

    ax.bar(years, insp_costs, width=width, color=PALETTE["primary"],
           alpha=0.85, label="Inspection cost")
    ax.bar(years, repl_costs, width=width, bottom=insp_costs,
           color=PALETTE["warn"], alpha=0.85, label="Replacement cost")

    _style_ax(ax, title="Cost Breakdown per Inspection",
              xlabel="Year", ylabel="Cost ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_inspection_intervals(result: SimulationResult) -> plt.Figure:
    """Show the interval between each inspection (useful for adaptive mode)."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(PALETTE["bg"])

    times = [ev.time for ev in result.events]
    if len(times) < 2:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["muted"])
    else:
        intervals = np.diff([0] + times)
        ax.plot(times, intervals, color=PALETTE["primary"], linewidth=2,
                marker="o", markersize=4)
        ax.axhline(result.converged_interval, color=PALETTE["accent"],
                   linestyle="--", linewidth=1.5,
                   label=f"Converged = {result.converged_interval:.2f} yrs")
        ax.legend(fontsize=8)

    _style_ax(ax, title="Inspection Interval Over Time",
              xlabel="Year", ylabel="Interval (years)")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    return fig
