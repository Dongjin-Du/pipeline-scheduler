"""
Pipeline Inspection Simulation Engine
--------------------------------------
Implements the Bricks Model from Davis et al. 2021, extended with:
  - Pluggable RUL distributions
  - Inspection-threshold-driven replacement decisions
  - Monte Carlo optimization of inspection intervals
  - Cost estimation over a planning horizon
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .distributions import RULDistribution


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CostParams:
    inspection_cost_per_segment: float = 500.0   # $ per segment inspected
    replacement_cost_per_segment: float = 5000.0  # $ per segment replaced

@dataclass
class InspectionEvent:
    time: float               # Year of inspection
    n_inspected: int
    n_replaced: int
    inspection_cost: float
    replacement_cost: float
    total_cost: float
    avg_remaining_rul: float  # mean RUL of segments that passed (not replaced)

@dataclass
class SimulationResult:
    inspection_schedule: List[float]          # Inspection times (years)
    events: List[InspectionEvent]
    total_cost: float
    converged_interval: float                 # Steady-state inspection interval
    first_interval: float
    n_segments: int
    horizon_years: float
    cost_params: CostParams
    # Time series for plotting
    cumulative_cost: List[float] = field(default_factory=list)
    replacements_per_inspection: List[int] = field(default_factory=list)


# ── Segment-level health tracker ────────────────────────────────────────────

class PipelineSystem:
    """
    Tracks the RUL of each segment over time.
    Segments are replaced (new RUL drawn) when:
      (a) They fail (RUL hits 0 before inspection)
      (b) At inspection, their remaining RUL < threshold * original_RUL
    """

    def __init__(
        self,
        n_segments: int,
        rul_dist: RULDistribution,
        inspection_threshold: float,   # fraction 0-1: replace if RUL_remaining/RUL_original < threshold
        cost_params: CostParams,
        seed: int = 42,
    ):
        self.n = n_segments
        self.dist = rul_dist
        self.threshold = inspection_threshold
        self.costs = cost_params
        self.rng = np.random.default_rng(seed)

        # Draw initial RULs
        self.original_rul = self.dist.sample(n_segments, self.rng)
        self.installed_at = np.zeros(n_segments)   # time each segment was last installed/replaced
        self.current_rul = self.original_rul.copy()
        self.current_time = 0.0

    def advance_time(self, dt: float):
        """Move the clock forward by dt years. Segments that fail mid-interval are replaced."""
        self.current_time += dt
        elapsed = self.current_time - self.installed_at  # time each segment has been running

        # Segments that have exceeded their RUL are failures — replace them immediately
        failed = elapsed >= self.current_rul
        n_failed = int(np.sum(failed))
        if n_failed > 0:
            new_rul = self.dist.sample(n_failed, self.rng)
            self.current_rul[failed] = new_rul
            self.installed_at[failed] = self.current_time
            self.original_rul[failed] = new_rul

        return n_failed

    def inspect_and_replace(self) -> InspectionEvent:
        """
        Run an inspection at current_time.
        Replace segments whose remaining RUL fraction is below threshold.
        Returns a detailed InspectionEvent.
        """
        elapsed = self.current_time - self.installed_at
        remaining = np.maximum(self.current_rul - elapsed, 0.0)
        fraction_remaining = np.where(self.original_rul > 0, remaining / self.original_rul, 0.0)

        # Decision: replace if fraction remaining < threshold
        to_replace = fraction_remaining < self.threshold
        n_replaced = int(np.sum(to_replace))
        n_inspected = self.n

        if n_replaced > 0:
            new_rul = self.dist.sample(n_replaced, self.rng)
            self.current_rul[to_replace] = new_rul
            self.installed_at[to_replace] = self.current_time
            self.original_rul[to_replace] = new_rul

        kept = ~to_replace
        avg_remaining = float(np.mean(remaining[kept])) if np.sum(kept) > 0 else 0.0

        insp_cost = n_inspected * self.costs.inspection_cost_per_segment
        repl_cost = n_replaced * self.costs.replacement_cost_per_segment
        total = insp_cost + repl_cost

        return InspectionEvent(
            time=self.current_time,
            n_inspected=n_inspected,
            n_replaced=n_replaced,
            inspection_cost=insp_cost,
            replacement_cost=repl_cost,
            total_cost=total,
            avg_remaining_rul=avg_remaining,
        )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.original_rul = self.dist.sample(self.n, self.rng)
        self.installed_at = np.zeros(self.n)
        self.current_rul = self.original_rul.copy()
        self.current_time = 0.0


# ── Monte Carlo optimizer ────────────────────────────────────────────────────

def optimize_inspection_schedule(
    n_segments: int,
    rul_dist: RULDistribution,
    inspection_threshold: float,
    cost_params: CostParams,
    horizon_years: float = 50.0,
    n_mc_trials: int = 30,
    candidate_intervals: Optional[np.ndarray] = None,
    seed: int = 42,
    progress_callback=None,
) -> SimulationResult:
    """
    Find the inspection interval (fixed) that minimizes total cost over the
    planning horizon, using Monte Carlo simulation.

    Strategy:
      1. Try a range of fixed inspection intervals.
      2. For each, run n_mc_trials simulations and average total cost.
      3. Pick the interval with minimum average cost.
      4. Re-simulate with the best interval to produce the detailed schedule.

    Also detects the convergence behavior (first vs. steady-state interval)
    by using an adaptive schedule: start with short intervals, widen once the
    replacement rate stabilizes.
    """
    mean_rul = rul_dist.mean()

    if candidate_intervals is None:
        lo = max(0.5, mean_rul * 0.02)
        hi = mean_rul * 0.20
        candidate_intervals = np.linspace(lo, hi, 40)

    best_interval = candidate_intervals[len(candidate_intervals) // 2]
    best_cost = np.inf

    total_steps = len(candidate_intervals)

    for i, interval in enumerate(candidate_intervals):
        if progress_callback:
            progress_callback(i / total_steps)

        trial_costs = []
        for trial in range(n_mc_trials):
            system = PipelineSystem(
                n_segments, rul_dist, inspection_threshold, cost_params,
                seed=seed + trial * 1000 + i
            )
            t = interval
            cost = 0.0
            while t <= horizon_years:
                system.advance_time(interval if t == interval else interval)
                ev = system.inspect_and_replace()
                cost += ev.total_cost
                t += interval
            trial_costs.append(cost)

        avg_cost = float(np.mean(trial_costs))
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_interval = interval

    # ── Detailed simulation with best interval ────────────────────────────
    result = _run_detailed_simulation(
        n_segments, rul_dist, inspection_threshold, cost_params,
        best_interval, horizon_years, seed=seed
    )
    return result


def _run_detailed_simulation(
    n_segments: int,
    rul_dist: RULDistribution,
    inspection_threshold: float,
    cost_params: CostParams,
    interval: float,
    horizon_years: float,
    seed: int = 42,
) -> SimulationResult:
    """
    Run a single detailed simulation with a fixed interval and collect all events.
    """
    system = PipelineSystem(n_segments, rul_dist, inspection_threshold, cost_params, seed=seed)

    events: List[InspectionEvent] = []
    schedule: List[float] = []
    cum_cost = 0.0
    cum_costs: List[float] = []
    replacements: List[int] = []

    t = interval
    while t <= horizon_years + 1e-9:
        system.advance_time(interval)
        ev = system.inspect_and_replace()
        events.append(ev)
        schedule.append(round(t, 4))
        cum_cost += ev.total_cost
        cum_costs.append(cum_cost)
        replacements.append(ev.n_replaced)
        t = round(t + interval, 6)

    # Detect convergence: first interval vs. steady state
    # In adaptive mode the interval is fixed here, but we report it anyway
    first_interval = interval
    converged_interval = interval  # fixed interval — same throughout

    return SimulationResult(
        inspection_schedule=schedule,
        events=events,
        total_cost=cum_cost,
        converged_interval=converged_interval,
        first_interval=first_interval,
        n_segments=n_segments,
        horizon_years=horizon_years,
        cost_params=cost_params,
        cumulative_cost=cum_costs,
        replacements_per_inspection=replacements,
    )


def run_adaptive_simulation(
    n_segments: int,
    rul_dist: RULDistribution,
    inspection_threshold: float,
    cost_params: CostParams,
    horizon_years: float = 50.0,
    target_replacement_rate: float = 0.10,  # ~10% per inspection is "normal"
    seed: int = 42,
) -> SimulationResult:
    """
    Adaptive schedule replicating the paper's behavior:
    - Start with short intervals (more frequent early on)
    - Widen intervals once replacement rate stabilizes near target
    - Converges to a steady-state interval
    """
    mean_rul = rul_dist.mean()
    initial_interval = max(0.5, mean_rul * 0.03)

    system = PipelineSystem(n_segments, rul_dist, inspection_threshold, cost_params, seed=seed)

    events: List[InspectionEvent] = []
    schedule: List[float] = []
    cum_cost = 0.0
    cum_costs: List[float] = []
    replacements: List[int] = []

    current_interval = initial_interval
    current_time = 0.0
    window = []  # recent replacement rates for convergence detection
    converged = False
    converged_interval = initial_interval
    first_interval = initial_interval

    while current_time + current_interval <= horizon_years + 1e-9:
        system.advance_time(current_interval)
        current_time = round(current_time + current_interval, 6)
        ev = system.inspect_and_replace()

        events.append(ev)
        schedule.append(round(current_time, 4))
        cum_cost += ev.total_cost
        cum_costs.append(cum_cost)
        replacements.append(ev.n_replaced)

        # Adaptive interval adjustment
        rate = ev.n_replaced / n_segments
        window.append(rate)
        if len(window) > 5:
            window.pop(0)

        if len(window) == 5 and not converged:
            avg_rate = np.mean(window)
            std_rate = np.std(window)
            if std_rate < 0.02:  # stable
                converged = True
                converged_interval = current_interval
            elif avg_rate < target_replacement_rate * 0.7:
                # Too few replacements — extend interval
                current_interval = min(current_interval * 1.05, mean_rul * 0.15)
            elif avg_rate > target_replacement_rate * 1.5:
                # Too many replacements — shorten interval
                current_interval = max(current_interval * 0.95, 0.3)

    return SimulationResult(
        inspection_schedule=schedule,
        events=events,
        total_cost=cum_cost,
        converged_interval=converged_interval,
        first_interval=first_interval,
        n_segments=n_segments,
        horizon_years=horizon_years,
        cost_params=cost_params,
        cumulative_cost=cum_costs,
        replacements_per_inspection=replacements,
    )


def results_to_dataframe(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for i, ev in enumerate(result.events):
        rows.append({
            "Inspection #": i + 1,
            "Year": round(ev.time, 2),
            "Segments Inspected": ev.n_inspected,
            "Segments Replaced": ev.n_replaced,
            "Replacement Rate (%)": round(100 * ev.n_replaced / ev.n_inspected, 1),
            "Inspection Cost ($)": f"${ev.inspection_cost:,.0f}",
            "Replacement Cost ($)": f"${ev.replacement_cost:,.0f}",
            "Total Cost ($)": f"${ev.total_cost:,.0f}",
            "Avg Remaining RUL (yrs)": round(ev.avg_remaining_rul, 2),
            "Cumulative Cost ($)": f"${result.cumulative_cost[i]:,.0f}",
        })
    return pd.DataFrame(rows)
