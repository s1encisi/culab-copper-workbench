"""Offline PDF rendering and OCR with page-space provenance."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from copper_mvp.common import PROJECT_ROOT, WorkbenchError, digest
from copper_mvp.knowledge_embedding import load_dependencies
from copper_mvp.knowledge_parsing import block

PDFIUM_LOCK = threading.RLock()


def render_page(payload, page_number, dpi=180):
    load_dependencies()
    import pypdfium2 as pdfium

    with PDFIUM_LOCK, pdfium.PdfDocument(payload) as document:
        if not 1 <= page_number <= len(document):
            raise WorkbenchError("PDF 页码不存在", "DOCUMENT_PAGE_NOT_FOUND")
        page = document[page_number - 1]
        try:
            width, height = page.get_size()
            scale = dpi / 72
            if width * height * scale * scale > 16_000_000:
                raise WorkbenchError("页面过大，请选择较低的渲染分辨率", "DOCUMENT_RENDER_SIZE")
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("RGB")
                try:
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    size = list(image.size)
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    return stream.getvalue(), {
        "page": page_number,
        "rendered_size": size,
        "pdf_size_points": [width, height],
        "dpi": dpi,
        "coordinate_space": "rendered_pixels_top_left",
    }


class LocalOCR:
    def __init__(self):
        self.engine = None
        self.manifest = None
        self.lock = threading.RLock()

    def _load(self):
        if self.engine is not None:
            return
        load_dependencies()
        import onnxruntime
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR

        config = json.loads((PROJECT_ROOT / "configs/runtime/knowledge_ocr.json").read_text(encoding="utf-8"))
        if version("rapidocr-onnxruntime") != config["version"] or version("pypdfium2") != config["renderer_version"]:
            raise WorkbenchError("本地 OCR 依赖版本与协议不一致", "KNOWLEDGE_OCR_VERSION")
        package = Path(rapidocr_onnxruntime.__file__).parent
        for filename, expected in config["model_files"].items():
            if hashlib.sha256((package / filename).read_bytes()).hexdigest() != expected:
                raise WorkbenchError("本地 OCR 模型校验失败", "KNOWLEDGE_OCR_HASH")
        self.engine = RapidOCR(
            text_score=0.0,
            intra_op_num_threads=2,
            inter_op_num_threads=1,
            det_model_path=str(package / config["det_model"]),
            cls_model_path=str(package / config["cls_model"]),
            rec_model_path=str(package / config["rec_model"]),
            det_use_cuda=False,
            cls_use_cuda=False,
            rec_use_cuda=False,
            det_use_dml=False,
            cls_use_dml=False,
            rec_use_dml=False,
        )
        import cv2

        self.manifest = {
            **config,
            "onnxruntime": onnxruntime.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }

    @property
    def signature(self):
        with self.lock:
            self._load()
            return digest(self.manifest)

    def recognize(self, payload, page_number, dpi=180):
        with self.lock:
            self._load()
            rendered, location = render_page(payload, page_number, dpi)
            start = perf_counter()
            rows, timings = self.engine(rendered)
            lines = []
            for box, text, confidence in rows or []:
                lines.append(
                    {"text": text, "score": float(confidence), "polygon": np.asarray(box, dtype=float).tolist()}
                )
            content = "\n".join(row["text"] for row in lines)
            scores = [row["score"] for row in lines]
            return block(
                "ocr_page",
                content,
                location,
                ocr={
                    "engine_signature": self.signature,
                    "engine": self.manifest,
                    "score_kind": "recognizer_score_not_calibrated_probability",
                    "minimum_score": min(scores) if scores else None,
                    "low_score_lines": [i for i, value in enumerate(scores) if value < 0.9],
                    "lines": lines,
                    "elapsed_ms": (perf_counter() - start) * 1000,
                    "rendered_sha256": hashlib.sha256(rendered).hexdigest(),
                    "review_required": True,
                    "executable": False,
                },
            )
