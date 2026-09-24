"""Local comparison jobs, immutable result lookup and explicit model replay."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from copper_mvp.classical_registry import CLASSICAL_METHODS
from copper_mvp.common import WorkbenchError, digest, file_hash, safe, utc_now, write_json
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.model_evaluation import evaluate_comparison, read_training
from copper_mvp.model_registry import ComparisonPredictionRequest, ComparisonRequest, method_spec
from copper_mvp.model_training import load_registered_model, source_signature, train_comparison
from copper_mvp.specialized_registry import SPECIALIZED_METHODS
from copper_mvp.statistical_registry import STATISTICAL_METHODS


def process_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.GetLastError() == 5
        try:
            status = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def observed_job_state(state):
    """Report a stopped worker without rewriting the saved experiment record."""
    if (
        state.get("status") in ("queued", "running")
        and state.get("owner_pid") is not None
        and not process_alive(state["owner_pid"])
    ):
        return {
            **state,
            "status": "interrupted",
            "error": {"code": "WORKER_NOT_RUNNING", "message": "原运行进程已停止；已保留原始记录和工件。"},
        }
    return state


class ComparisonService:
    def __init__(self, root: Path, data):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.lock = threading.RLock()

    def directory(self, run_id):
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise WorkbenchError("比较编号无效", "COMPARISON_NOT_FOUND")
        return self.root / run_id

    def submit(self, request: ComparisonRequest, executor):
        run_id = digest(request.request_key)[:32]
        root = self.directory(run_id)
        source = source_signature(self.data)
        fingerprint_data = {
            "request": request.model_dump(mode="json"),
            "source": source,
            "methods": [method_spec(m, request.seed) for m in request.methods],
            "adapters": file_hash(Path(__file__).with_name("model_adapters.py")),
        }
        if any(m in CLASSICAL_METHODS for m in request.methods):
            fingerprint_data["classical_adapters"] = {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("classical_registry.py", "classical_models.py", "classical_evaluation.py")
            }
        if any(m in STATISTICAL_METHODS for m in request.methods):
            fingerprint_data["statistical_adapters"] = {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("statistical_registry.py", "statistical_models.py", "statistical_runtime.py")
            }
        if any(m in SPECIALIZED_METHODS for m in request.methods):
            fingerprint_data["specialized_adapters"] = {
                name: file_hash(Path(__file__).with_name(name))
                for name in (
                    "specialized_registry.py",
                    "specialized_models.py",
                    "specialized_evaluation.py",
                    "cubist_model.py",
                )
            }
        if "BART" in request.methods:
            from copper_mvp.bart_registry import bart_source_hashes

            fingerprint_data["bart_adapters"] = bart_source_hashes()
        if "SymbolicRegression" in request.methods:
            from copper_mvp.symbolic_registry import symbolic_source_hashes

            fingerprint_data["symbolic_adapters"] = symbolic_source_hashes()
        if any(m in ("TabNet", "FTTransformer", "NODE") for m in request.methods):
            from copper_mvp.tabular_registry import tabular_source_hashes

            fingerprint_data["tabular_adapters"] = tabular_source_hashes()
        if any(method_spec(m, request.seed).get("input_kind") == "anchored_process_sequence" for m in request.methods):
            from copper_mvp.temporal_registry import temporal_source_hashes

            fingerprint_data["temporal_adapters"] = temporal_source_hashes()
        if "TabPFN" in request.methods:
            from copper_mvp.tabpfn_registry import tabpfn_source_hashes

            fingerprint_data["tabpfn_adapters"] = tabpfn_source_hashes()
        fingerprint = digest(fingerprint_data)
        with self.lock:
            if (root / "state.json").exists():
                state = self.get(run_id)
                if state["fingerprint"] != fingerprint:
                    raise WorkbenchError("同一请求键的配置或数据不同", "REQUEST_CONFLICT")
                return {**state, "reused": True}
            root.mkdir(parents=True, exist_ok=True)
            state = {
                "run_id": run_id,
                "status": "queued",
                "created_at": utc_now(),
                "owner_pid": os.getpid(),
                "request": request.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "automatic_promotion": False,
            }
            write_json(root / "state.json", state)
            executor.submit(self.execute, run_id, request)
            return {**state, "reused": False}

    def execute(self, run_id, request):
        root = self.directory(run_id)

        def update(**values):
            with self.lock:
                state = json.loads((root / "state.json").read_text(encoding="utf-8"))
                state.update(values)
                write_json(root / "state.json", state)

        update(status="running", started_at=utc_now())
        try:
            train_comparison(self.data, request, root, progress=lambda detail: update(progress=detail))
            update(progress={"phase": "independent_evaluation"})
            result = evaluate_comparison(self.data, root)
            update(
                status="completed",
                finished_at=utc_now(),
                evaluation_status=result["status"],
                evaluation_sha256=file_hash(root / "evaluation.json"),
                progress={"phase": "completed"},
            )
        except Exception as exc:
            update(
                status="failed",
                finished_at=utc_now(),
                error={
                    "code": getattr(exc, "code", type(exc).__name__),
                    "message": "比较未完成，已保留现有工件和失败状态",
                },
            )

    def get(self, run_id, *, result=True):
        root = self.directory(run_id)
        if not (root / "state.json").is_file():
            raise WorkbenchError("没有找到该比较", "COMPARISON_NOT_FOUND")
        with self.lock:
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if state["status"] in ("queued", "running") and not process_alive(state.get("owner_pid")):
            state = {
                **state,
                "status": "interrupted",
                "error": {
                    "code": "COMPARISON_INTERRUPTED",
                    "message": "原进程已结束；请使用新的请求键重新执行，旧工件保留。",
                },
            }
        if result and state["status"] == "completed":
            path = root / "evaluation.json"
            if file_hash(path) != state.get("evaluation_sha256"):
                raise WorkbenchError("比较结果哈希不匹配", "COMPARISON_HASH_MISMATCH")
            evaluated = json.loads(path.read_text(encoding="utf-8"))
            if file_hash(root / "training_manifest.json") != evaluated["training_manifest_sha256"]:
                raise WorkbenchError("训练清单与独立评价的绑定失效", "COMPARISON_HASH_MISMATCH")
            manifest, _ = read_training(root)
            if (
                evaluated["protocol_sha256"] != manifest["protocol_sha256"]
                or evaluated["oof_sha256"] != manifest["oof_sha256"]
            ):
                raise WorkbenchError("评价与预测协议的绑定失效", "COMPARISON_HASH_MISMATCH")
            state["result"] = evaluated
        return state

    def list(self):
        states = [self.get(p.parent.name, result=False) for p in self.root.glob("*/state.json")]
        return sorted(states, key=lambda s: s["created_at"], reverse=True)

    def predict(self, run_id, request: ComparisonPredictionRequest):
        state = self.get(run_id)
        if state["status"] != "completed":
            raise WorkbenchError("比较尚未完成", "COMPARISON_NOT_READY")
        root = self.directory(run_id)
        manifest, protocol = read_training(root)
        if source_signature(self.data) != manifest["source"]:
            raise WorkbenchError("当前数据与比较工件不匹配", "COMPARISON_DATA_VERSION")
        snapshot = DataService(self.data).snapshot(request.event_id)
        decision = source_time(self.data.row(request.event_id).decision_at)
        fold = self.data.fold_for.get(request.event_id) if request.scope == "oof_replay" else "DEVELOPMENT"
        if fold is None:
            raise WorkbenchError("该事件没有 OOF 模型", "NO_OOF_MODEL")
        model, artifact = load_registered_model(root, manifest, protocol, request.method_id, fold)
        if request.scope == "oof_replay" and source_time(artifact["fit_cutoff_at"]) > decision:
            raise WorkbenchError("模型训练截止晚于事件", "FUTURE_MODEL")
        values = model.predict_context(self.data, [request.event_id])[0]
        result = {
            "schema_version": "comparison-prediction.g2a.v1",
            "run_id": run_id,
            "event_id": request.event_id,
            "method_id": request.method_id,
            "scope": request.scope,
            "evaluation_mode": "historical_replay" if request.scope == "oof_replay" else "development_analysis",
            "virtual_prediction_at": decision.isoformat() if request.scope == "oof_replay" else None,
            "context_at": decision.isoformat(),
            "computed_at": utc_now(),
            "online_prediction_claim": False,
            "snapshot_id": snapshot["id"],
            "artifact_sha256": artifact["sha256"],
            "fit_cutoff_at": artifact["fit_cutoff_at"],
            "predictions": {
                t: {"value": float(values[i]), "unit": unit}
                for i, (t, unit) in enumerate((("cu", "g/L"), ("as", "mg/L")))
            },
            "warnings": ["NEGATIVE_MODEL_PREDICTION"] if any(values < 0) else [],
            "automatic_promotion": False,
            "optimization_proxy_approval": False,
        }

        if request.include_uncertainty:
            distribution = model.predict_uncertainty_context(self.data, [request.event_id])
            result["uncertainty"] = safe(
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in distribution.items()}
            )
        return result
