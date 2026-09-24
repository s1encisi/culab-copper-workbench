import numpy as np
from scipy.stats import norm

from copper_mvp.specialized_evaluation import gaussian_scores


def test_gaussian_probability_scores_match_centered_normal_and_unit_changes():
    score = gaussian_scores([0.0], [0.0], [1.0])
    assert np.isclose(score["gaussian_crps"], (np.sqrt(2) - 1) / np.sqrt(np.pi))
    expected = sum(a * norm.ppf(1 - a / 2) for a in (0.5, 0.2, 0.1)) / 3.5
    assert np.isclose(score["weighted_interval_score"], expected)
    original = gaussian_scores([1.0, -1.0], [0.2, 0.3], [0.4, 0.7])
    changed = gaussian_scores([1000.0, -1000.0], [200.0, 300.0], [400.0, 700.0])
    for name in ("gaussian_crps", "weighted_interval_score"):
        assert np.isclose(changed[name], 1000 * original[name])
