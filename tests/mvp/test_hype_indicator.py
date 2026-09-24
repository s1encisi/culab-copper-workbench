import numpy as np

from copper_mvp.hype import ExactHypEGeometry


def test_hype_exact_weights_match_manual_rectangle_areas():
    geometry = ExactHypEGeometry([[1, 3], [2, 2], [3, 1]], [4, 4])
    # Three exclusive unit squares; shared strips and the triple-covered square.
    np.testing.assert_allclose(geometry.fitness(1), [1, 1, 1])
    np.testing.assert_allclose(geometry.fitness(2), [1.25, 1.5, 1.25])
    np.testing.assert_allclose(geometry.fitness(3), [11 / 6, 7 / 3, 11 / 6])
    assert np.isclose(geometry.fitness(3).sum(), 6)


def test_hype_duplicate_allocation_updates_after_deletion_and_excludes_outside_box():
    geometry = ExactHypEGeometry([[1, 1], [1, 1], [3, 0]], [2, 2])
    np.testing.assert_allclose(geometry.fitness(1), [0, 0, 0])
    np.testing.assert_allclose(geometry.fitness(3), [0.5, 0.5, 0])
    geometry.remove(0)
    np.testing.assert_allclose(geometry.fitness(1)[geometry.active], [1, 0])
