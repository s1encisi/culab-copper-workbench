"""Small physical-variable symbolic models with train-only affine transforms."""

import json
import os
import subprocess
import sys
import uuid

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from copper_mvp.classical_models import StableNumericScale
from copper_mvp.common import PROJECT_ROOT, WorkbenchError, file_hash, write_json
from copper_mvp.symbolic_expression import evaluate_expression
from copper_mvp.symbolic_registry import (
    SYMBOLIC_FEATURE_INDICES,
    SYMBOLIC_FEATURE_ROLES,
    SYMBOLIC_PARAMETERS,
    symbolic_source_hashes,
)


class SymbolicModel:
    def __init__(self, seed, settings=None):
        self.seed = seed
        self.settings = dict(SYMBOLIC_PARAMETERS if settings is None else settings)
        self.fit_warnings = []

    def fit(self, X, y, sample_weight=None, max_wall_seconds=None):
        if sample_weight is not None:
            raise WorkbenchError("符号回归首版不开放样本权重", "MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.preprocessor = Pipeline(
            [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StableNumericScale())]
        )
        inputs = self.preprocessor.fit_transform(X[:, SYMBOLIC_FEATURE_INDICES])
        self.target_scaler = StandardScaler().fit(y - X[:, :2])
        target = self.target_scaler.transform(y - X[:, :2])
        directory = PROJECT_ROOT / "runs/mvp/symbolic_fits" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        np.savez(directory / "input.npz", X=inputs, y=target)
        write_json(
            directory / "request.json",
            {
                "seed": self.seed,
                "settings": self.settings,
                "input_sha256": file_hash(directory / "input.npz"),
                "code_hashes": symbolic_source_hashes(),
            },
        )
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[name] = "1"
        environment["LANGSMITH_TRACING"] = environment["LANGCHAIN_TRACING_V2"] = "false"
        timeout = 1800 if max_wall_seconds is None else max(1.0, min(1800, max_wall_seconds))
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    str(PROJECT_ROOT / "scripts/run_symbolic_fit.py"),
                    "--directory",
                    str(directory),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=out,
                stderr=err,
                **kwargs,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                    else:
                        import signal

                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                write_json(directory / "timeout.json", {"status": "timeout", "seconds": timeout})
                raise WorkbenchError("符号回归达到拟合预算，已有工件保留", "SYMBOLIC_TIME_BUDGET") from None
        if process.returncode:
            raise WorkbenchError("符号回归拟合失败，记录编号 " + directory.name, "SYMBOLIC_FIT_FAILED")
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state["status"] != "completed" or file_hash(directory / "result.json") != state["result_sha256"]:
            raise WorkbenchError("符号表达式工件校验失败", "SYMBOLIC_ARTIFACT_HASH")
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        self.models = result["models"]
        self.fit_warnings = result["fit_warnings"]
        self.fit_metadata = {
            "training_rows": len(X),
            "selected_feature_indices": list(SYMBOLIC_FEATURE_INDICES),
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "fit_run_id": directory.name,
            "runtime": result["runtime"],
            "source_hashes": result["code_hashes"],
            "dimensionally_valid_by_normalization": True,
            "native_loss_parity": True,
            "elapsed_seconds": result["elapsed_seconds"],
            "complexities": [m["complexity"] for m in self.models],
        }
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        inputs = self.preprocessor.transform(X[:, SYMBOLIC_FEATURE_INDICES])
        values = np.column_stack([evaluate_expression(m["ast"], inputs) for m in self.models])
        return X[:, :2] + self.target_scaler.inverse_transform(values)

    def equations(self):
        scale = self.preprocessor.named_steps["scale"]
        return {
            "input_roles": list(SYMBOLIC_FEATURE_ROLES),
            "input_indices": list(SYMBOLIC_FEATURE_INDICES),
            "input_mean": scale.mean_.tolist(),
            "input_scale": scale.scale_.tolist(),
            "imputation_values": self.preprocessor.named_steps["impute"].statistics_.tolist(),
            "active_input_mask": scale.active_.tolist(),
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "form": (
                "next_target = current_target + delta_mean + delta_scale * f(z); in"
                "active z=0, active z=(imputed_input-mean)/scale"
            ),
            "models": self.models,
            "scope": "model associations; no causal-control interpretation",
        }
