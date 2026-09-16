"""Numerical replay criteria for float32 neural predictions in physical units."""
import numpy as np
NEURAL_METHODS={"TabNet","FTTransformer","NODE","GRU","LSTM","CausalTCN","TabPFN"}


def neural_replay_error(actual,expected,current,target_scale,method=None):
    """Use 1e-5 relative/absolute tolerance on standardized target increments.

    TabPFN uses 1e-4 for its float32 attention/reduction path; the other neural
    adapters retain 1e-5. Full-batch replay stays strict under the declared
    single-thread numerical runtime.

    This is a floating-point replay check, not a predictive-accuracy threshold.
    Full-batch OOF reproduction is checked separately with float64-level tolerance.
    """
    scale=np.asarray(target_scale,float)
    actual_delta=(np.asarray(actual)-np.asarray(current))/scale
    expected_delta=(np.asarray(expected)-np.asarray(current))/scale
    error=np.abs(actual_delta-expected_delta)
    tolerance=1e-4 if method=="TabPFN" else 1e-5
    allowance=tolerance+tolerance*np.abs(expected_delta)
    return bool(np.isfinite(actual_delta).all() and np.all(error<=allowance)),{
        "dtype":"float32","absolute_tolerance":tolerance,"relative_tolerance":tolerance,"method":method,
        "standardized_max_error":float(np.max(error)),
        "maximum_tolerance_fraction":float(np.max(error/allowance))}
