"""数据准入与泄漏防护。"""

from .leakage import (
    FutureInformationError,
    assert_no_future_information,
    find_forbidden_paths,
)

__all__ = [
    "FutureInformationError",
    "assert_no_future_information",
    "find_forbidden_paths",
]
