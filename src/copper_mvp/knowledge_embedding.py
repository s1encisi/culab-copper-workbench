"""Pinned local ONNX embeddings; no model download or HTTP client at inference."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

from copper_mvp.common import PROJECT_ROOT, WorkbenchError, digest


def load_dependencies():
    directory = Path(
        os.environ.get("COPPER_KNOWLEDGE_RUNTIME_DIR", str(PROJECT_ROOT / "runs/dependencies/knowledge-v1"))
    )
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.append(str(directory))


class LocalEmbedding:
    def __init__(self, model_dir=None):
        self.model_dir = Path(
            model_dir
            or os.environ.get("COPPER_KNOWLEDGE_MODEL_DIR", str(PROJECT_ROOT / "runs/dependencies/bge-small-zh-v1.5"))
        )
        self.lock = threading.RLock()
        self.session = None
        self.manifest = None

    def _load(self):
        if self.session is not None:
            return
        load_dependencies()
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config = PROJECT_ROOT / "configs/runtime/knowledge_embedding.json"
        if not config.is_file():
            raise WorkbenchError("未配置本地嵌入模型的固定版本", "KNOWLEDGE_MODEL_NOT_FOUND")
        manifest = json.loads(config.read_text(encoding="utf-8"))
        for name, entry in manifest["files"].items():
            path = self.model_dir / name
            if not path.is_file():
                raise WorkbenchError("请先配置本地知识检索模型", "KNOWLEDGE_MODEL_NOT_FOUND")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise WorkbenchError("本地嵌入模型文件校验失败", "KNOWLEDGE_MODEL_HASH")
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512, stride=80)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(self.model_dir / "onnx/model_quantized.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.manifest = {
            **manifest,
            "runtime": "onnxruntime",
            "runtime_version": ort.__version__,
            "adapter": "g6m.cls-window.v1",
            "long_text": "mean_of_overlapping_cls_vectors",
        }
        self.dimension = manifest["dimension"]

    @property
    def signature(self):
        with self.lock:
            self._load()
            return digest(self.manifest)

    def encode(self, texts, *, query=False):
        with self.lock:
            self._load()
            result = []
            for text in texts:
                if query:
                    text = "为这个句子生成表示以用于检索相关文章：" + text
                first = self.tokenizer.encode(text)
                windows = [first] + list(first.overflowing)
                vectors = []
                for offset in range(0, len(windows), 4):
                    batch = windows[offset : offset + 4]
                    length = max(len(item.ids) for item in batch)
                    fields = {
                        "input_ids": [v.ids + [0] * (length - len(v.ids)) for v in batch],
                        "attention_mask": [v.attention_mask + [0] * (length - len(v.ids)) for v in batch],
                        "token_type_ids": [v.type_ids + [0] * (length - len(v.ids)) for v in batch],
                    }
                    feed = {
                        node.name: np.asarray(fields[node.name], dtype=np.int64) for node in self.session.get_inputs()
                    }
                    output = self.session.run(None, feed)[0]
                    cls = output[:, 0, :] if output.ndim == 3 else output
                    vectors.extend(cls)
                vector = np.asarray(vectors, dtype=np.float32).mean(axis=0)
                norm = float(np.linalg.norm(vector))
                if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or norm == 0:
                    raise WorkbenchError("本地嵌入输出无效", "KNOWLEDGE_EMBEDDING")
                result.append(vector / norm)
            return np.asarray(result, dtype=np.float32).reshape(len(texts), self.dimension)
