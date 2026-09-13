"""Pinned local multilingual cross-encoder for retrieval candidate ranking."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from time import perf_counter

import numpy as np

from copper_mvp.common import PROJECT_ROOT, WorkbenchError, digest, file_hash
from copper_mvp.knowledge_embedding import load_dependencies


class LocalReranker:
    def __init__(self, model_dir=None):
        self.directory=Path(model_dir or os.environ.get("COPPER_KNOWLEDGE_RERANKER_DIR",
                            str(PROJECT_ROOT/"runs/dependencies/mmarco-minilm-reranker")))
        self.lock=threading.RLock()
        self.session=None

    def _load(self):
        if self.session is not None:
            return
        load_dependencies()
        import onnxruntime as ort
        from tokenizers import Tokenizer
        path=PROJECT_ROOT/"configs/runtime/knowledge_reranker.json"
        if not path.is_file():
            raise WorkbenchError("缺少本地重排模型配置", "KNOWLEDGE_RERANKER_NOT_FOUND")
        self.manifest=json.loads(path.read_text(encoding="utf-8"))
        for name,record in self.manifest["files"].items():
            target=self.directory/name
            if not target.is_file():
                raise WorkbenchError("请先安装本地重排模型", "KNOWLEDGE_RERANKER_NOT_FOUND")
            if hashlib.sha256(target.read_bytes()).hexdigest()!=record["sha256"]:
                raise WorkbenchError("本地重排模型哈希不一致", "KNOWLEDGE_RERANKER_HASH")
        config=json.loads((self.directory/"config.json").read_text(encoding="utf-8"))
        self.pad_id=config["pad_token_id"]
        self.tokenizer=Tokenizer.from_file(str(self.directory/"tokenizer.json"))
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        options.enable_mem_pattern=False
        self.session=ort.InferenceSession(str(self.directory/self.manifest["model_file"]),
                                          sess_options=options,providers=["CPUExecutionProvider"])
        self.identity={**self.manifest,"runtime":ort.__version__,"adapter":"g6m.cross-encoder.v1",
                       "truncation":"longest_first","score_kind":"uncalibrated_relevance_logit"}

    @property
    def cache_key(self):
        return digest({"manifest": file_hash(PROJECT_ROOT/"configs/runtime/knowledge_reranker.json"),
                       "adapter": "g6m.cross-encoder.v1"})

    @property
    def signature(self):
        with self.lock:
            self._load()
            return digest(self.identity)

    def rank(self, query, passages):
        with self.lock:
            self._load()
            start=perf_counter()
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            lengths=[len(self.tokenizer.encode(query,text).ids) for text in passages]
            self.tokenizer.enable_truncation(max_length=self.manifest["max_length"],strategy="longest_first")
            self.tokenizer.enable_padding(pad_id=self.pad_id,pad_token="<pad>")
            scores=[]
            for offset in range(0,len(passages),4):
                batch=self.tokenizer.encode_batch([(query,text) for text in passages[offset:offset+4]])
                fields={"input_ids":[v.ids for v in batch],"attention_mask":[v.attention_mask for v in batch],
                        "token_type_ids":[v.type_ids for v in batch]}
                feed={item.name:np.asarray(fields[item.name],dtype=np.int64) for item in self.session.get_inputs()}
                output=np.asarray(self.session.run(None,feed)[0]).reshape(-1)
                if output.shape!=(len(batch),) or not np.isfinite(output).all():
                    raise WorkbenchError("重排输出无效", "KNOWLEDGE_RERANKER_OUTPUT")
                scores.extend(float(value) for value in output)
            return {"scores":scores,"pairs":len(passages),"truncated_pairs":sum(n>self.manifest["max_length"] for n in lengths),
                    "input_token_counts":lengths,"elapsed_ms":(perf_counter()-start)*1000,
                    "model_signature":self.signature,"score_kind":"uncalibrated_relevance_logit"}
