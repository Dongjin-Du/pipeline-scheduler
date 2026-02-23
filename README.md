# 🔧 Pipeline Inspection Scheduler

> Reliability-constrained, cost-optimal inspection and replacement planning for pipeline systems — based on the **Bricks Model** (Davis et al. 2021).

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Given a pipeline system, this tool computes the **optimal inspection schedule** and provides a **full cost estimate** over a planning horizon by:

1. Modeling each pipe segment's remaining useful life (RUL) as a random draw from a chosen distribution
2. Simulating the system over time using the **Bricks Model** — segments age, fail, or get replaced at inspection
3. Finding the best inspection interval via **Monte Carlo optimization** or an **adaptive convergence** algorithm
4. Reporting how many inspections and replacements are needed each year and the total cost

---

## Inputs

| Parameter | Description |
|---|---|
| **Number of segments** | Total pipe segments in the system |
| **RUL distribution** | Birnbaum-Saunders, Weibull, Normal, or Lognormal |
| **Distribution parameters** | Shape, scale, mean, std, etc. depending on type |
| **Replacement threshold** | Replace at inspection if remaining RUL / original RUL < threshold |
| **Planning horizon** | Number of years to simulate (e.g. 50 years) |
| **Costs** | Inspection cost per segment, replacement cost per segment |

## Outputs

- **Optimal inspection interval** (first interval + converged steady-state interval)
- **Full inspection schedule** — year-by-year table with # inspected, # replaced, costs
- **Cost estimation** — cumulative cost, breakdown by inspection vs. replacement
- **Charts** — replacement rate over time, cost over time, interval convergence
- **CSV export** of the full schedule

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Dongjin-Du/pipeline-scheduler.git
cd pipeline-inspection-scheduler

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
pipeline-inspection-scheduler/
│
├── app.py                   # Streamlit web UI
├── requirements.txt
│
└── core/
    ├── __init__.py
    ├── distributions.py     # RUL distribution models (BS, Weibull, Normal, Lognormal)
    ├── simulation.py        # Bricks model, Monte Carlo optimizer, adaptive scheduler
    └── plotting.py          # Matplotlib chart generators
```

---

## Supported RUL Distributions

| Distribution | Parameters | Best for |
|---|---|---|
| **Birnbaum-Saunders** (Fatigue Life) | Shape α, Scale β | Corrosion fatigue — the model from Jangala et al. 2024 |
| **Weibull** | Shape k, Scale λ | General wear-out failure (k > 1) |
| **Normal** | Mean μ, Std σ | Symmetric aging around a known mean life |
| **Lognormal** | Log-mean μ, Log-std σ | Right-skewed lifetimes |

---

## Scheduling Modes

### Adaptive (Paper Method)
Mirrors the behavior described in the paper: starts with short inspection intervals, then adaptively widens them as the replacement rate stabilizes. Reports the *first interval* and the *converged steady-state interval*.

### Optimized (Monte Carlo)
Tries a grid of fixed inspection intervals, runs `n_mc_trials` simulations for each, picks the interval with the lowest average total cost over the planning horizon.

---

## Bricks Model (brief)

Each segment is assigned an independent RUL drawn from the chosen distribution at installation. At each inspection:
- Segments whose `remaining_RUL / original_RUL < threshold` are **replaced** (new RUL drawn)
- Segments that fail between inspections are replaced immediately (emergency)

The system converges to a steady-state health distribution as old cohorts of pipes are replaced over time.

---

## References

- Davis, P. et al. (2021). *Pipeline service life estimation based on the bricks model*
- Jangala, R. et al. (2024). *Probabilistic external corrosion fatigue life (RUL) estimation*
- Birnbaum, Z.W. & Saunders, S.C. (1969). *A new family of life distributions*

---

## License

MIT
