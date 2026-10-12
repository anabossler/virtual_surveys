"""Profile queries against the transfer model, for the interface.

Loads the coefficient table and the CIS design once, precomputes Monte Carlo
draws of the coefficients, the base rate and the calibrated intercept, and
returns the projected probability with a 95% interval for a profile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from transfer_model import A0, P0, P0_N, PREDICTORS, SEED, calibrate_intercept, load_design, worry_score

WORRY_CODING = "centered"
N_DRAWS = 2000


class TransferQuery:
    def __init__(self, cis_path: Path, coefficients_path: Path, a0: float = A0, p0: float = P0):
        table = pd.read_csv(coefficients_path).set_index("predictor").loc[list(PREDICTORS)]
        self.beta = table["coefficient"].to_numpy()
        self.se = table["std_error"].to_numpy()
        design = load_design(cis_path, WORRY_CODING)
        x = design[list(PREDICTORS)].to_numpy()
        weights = design["weight"].to_numpy()
        self.a0 = a0
        self.intercept = calibrate_intercept(a0 * (x @ self.beta), p0, weights)

        rng = np.random.default_rng(SEED)
        self.draw_beta = rng.normal(self.beta, self.se, size=(N_DRAWS, len(self.beta)))
        draw_p0 = rng.beta(p0 * P0_N + 0.5, (1 - p0) * P0_N + 0.5, size=N_DRAWS)
        self.draw_intercept = np.array(
            [calibrate_intercept(a0 * (x @ b), p, weights) for b, p in zip(self.draw_beta, draw_p0)]
        )

    @staticmethod
    def vector(female: bool, under_30: bool, higher_education: bool, recycles: bool, worry: int) -> np.ndarray:
        """Predictor vector; worry on the CIS V15 scale (1 to 5)."""
        return np.array(
            [float(female), float(under_30), float(higher_education), float(recycles),
             float(worry_score(np.array([worry]), WORRY_CODING)[0])]
        )

    def predict(self, **profile) -> tuple[float, float, float]:
        x = self.vector(**profile)
        point = float(norm.cdf(self.intercept + self.a0 * (x @ self.beta)))
        draws = norm.cdf(self.draw_intercept + self.a0 * (self.draw_beta @ x))
        low, high = np.percentile(draws, [2.5, 97.5])
        return point, float(low), float(high)
