"""LLM 请求适配层；默认仅构造载荷，不进行网络调用。"""

from .adapters import DeepSeekV4ProAdapter, KimiK3Adapter, StructuredTask

__all__ = ["DeepSeekV4ProAdapter", "KimiK3Adapter", "StructuredTask"]
