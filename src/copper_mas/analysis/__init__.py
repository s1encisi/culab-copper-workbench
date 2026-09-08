"""只读研究诊断模块。"""

from .covariate_shift import (
    CovariateShiftResult,
    SecurityBoundaryError,
    materialize_safe_process_view,
    run_covariate_shift_audit,
)

__all__ = [
    "CovariateShiftResult",
    "SecurityBoundaryError",
    "materialize_safe_process_view",
    "run_covariate_shift_audit",
]
