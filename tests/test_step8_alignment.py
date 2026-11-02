"""The CIS reference in step8 must not depend on the categories a model generates.

Run: python -m pytest tests/test_step8_alignment.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step8  # noqa: E402


def _cis_and_model_without_rare_category():
    rng = np.random.default_rng(0)
    n = 2000
    sex = pd.Series(rng.choice([1, 2, 7], size=n, p=[0.48, 0.517, 0.003]))
    concern = pd.Series(np.where(sex == 2, rng.choice([4, 5], n, p=[0.3, 0.7]),
                                 rng.choice([4, 5], n, p=[0.5, 0.5])))
    cis = pd.DataFrame({"SEX": sex, "V15": concern})
    synth = cis[cis["SEX"] != 7].sample(5000, replace=True, random_state=1).reset_index(drop=True)
    return cis, synth


def test_raw_scale_keeps_cis_reference():
    cis, synth = _cis_and_model_without_rare_category()
    x_cis, x_synth = step8.align_to_0based(cis["SEX"], synth["SEX"], raw_scale=True)
    pd.testing.assert_series_equal(x_cis, cis["SEX"].dropna())
    pd.testing.assert_series_equal(x_synth, synth["SEX"].dropna())


def test_reference_is_identical_across_models():
    cis, synth = _cis_and_model_without_rare_category()
    pairs = [("SEX", "V15", "gender -> env_concern")]
    cols = ["SEX", "V15"]
    _, _, full = step8.compute_chi2_preservation(cis, cis.copy(), pairs, cols, raw_scale=True)
    _, _, partial = step8.compute_chi2_preservation(cis, synth, pairs, cols, raw_scale=True)
    assert full.loc[0, "V_CIS"] == partial.loc[0, "V_CIS"]
    assert full.loc[0, "p_CIS"] == partial.loc[0, "p_CIS"]
