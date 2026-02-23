"""
RUL Distribution Models
Supports: Birnbaum-Saunders (Fatigue Life), Weibull, Normal, Lognormal
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Literal, Dict, Any


DistributionType = Literal["birnbaum_saunders", "weibull", "normal", "lognormal"]


@dataclass
class DistributionParams:
    dist_type: DistributionType
    params: Dict[str, float]

    def label(self) -> str:
        labels = {
            "birnbaum_saunders": "Birnbaum-Saunders (Fatigue Life)",
            "weibull": "Weibull",
            "normal": "Normal",
            "lognormal": "Lognormal",
        }
        return labels[self.dist_type]


class RULDistribution:
    """
    Wraps scipy distributions and provides a unified interface
    for sampling and CDF evaluation.
    """

    def __init__(self, params: DistributionParams):
        self.params = params
        self._dist = self._build(params)

    def _build(self, p: DistributionParams):
        t = p.dist_type
        d = p.params

        if t == "birnbaum_saunders":
            # scipy: fatiguelife(c, loc, scale)  — c=shape, scale=scale
            return stats.fatiguelife(c=d["shape"], loc=d.get("loc", 0), scale=d["scale"])

        elif t == "weibull":
            # scipy: weibull_min(c, loc, scale) — c=shape (k), scale=scale (lambda)
            return stats.weibull_min(c=d["shape"], loc=d.get("loc", 0), scale=d["scale"])

        elif t == "normal":
            return stats.norm(loc=d["mean"], scale=d["std"])

        elif t == "lognormal":
            # scipy lognorm: shape=sigma (log-space std), scale=exp(mu)
            return stats.lognorm(s=d["sigma"], scale=np.exp(d["mu"]))

        else:
            raise ValueError(f"Unknown distribution type: {t}")

    def sample(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Draw n RUL samples (clipped to positive values)."""
        if rng is not None:
            samples = self._dist.rvs(size=n, random_state=rng)
        else:
            samples = self._dist.rvs(size=n)
        return np.clip(samples, 0.01, None)

    def cdf(self, t: float) -> float:
        """P(RUL <= t), i.e. probability of failure by time t."""
        return float(self._dist.cdf(t))

    def sf(self, t: float) -> float:
        """Survival function: P(RUL > t)."""
        return float(self._dist.sf(t))

    def mean(self) -> float:
        return float(self._dist.mean())

    def std(self) -> float:
        return float(self._dist.std())

    def var(self) -> float:
        return float(self._dist.var())

    def ppf(self, q: float) -> float:
        """Percent point function (inverse CDF)."""
        return float(self._dist.ppf(q))

    def pdf_range(self, n_points: int = 300):
        """Return (x, pdf) for plotting."""
        lo = max(0, self.ppf(0.001))
        hi = self.ppf(0.999)
        x = np.linspace(lo, hi, n_points)
        y = self._dist.pdf(x)
        return x, y


# ── convenience constructors ────────────────────────────────────────────────

def make_bs(shape: float, scale: float, loc: float = 0.0) -> RULDistribution:
    return RULDistribution(DistributionParams(
        "birnbaum_saunders", {"shape": shape, "scale": scale, "loc": loc}
    ))

def make_weibull(shape: float, scale: float, loc: float = 0.0) -> RULDistribution:
    return RULDistribution(DistributionParams(
        "weibull", {"shape": shape, "scale": scale, "loc": loc}
    ))

def make_normal(mean: float, std: float) -> RULDistribution:
    return RULDistribution(DistributionParams(
        "normal", {"mean": mean, "std": std}
    ))

def make_lognormal(mu: float, sigma: float) -> RULDistribution:
    return RULDistribution(DistributionParams(
        "lognormal", {"mu": mu, "sigma": sigma}
    ))
