import numpy as np
from copper_mvp.neural_replay import neural_replay_error

def test_float32_replay_tolerance_is_not_prediction_quality_tolerance():
    origin=np.array([4.,6000.]);scale=np.array([.4,500.]);expected=np.array([4.7,7000.])
    okay,detail=neural_replay_error(expected+np.array([3e-8,6.1e-5]),expected,origin,scale)
    assert okay and detail["maximum_tolerance_fraction"]<1
    bad,_=neural_replay_error(expected+np.array([.01,1.]),expected,origin,scale)
    assert not bad
