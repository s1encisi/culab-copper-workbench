"""Train-only BART preprocessing and physical-unit posterior prediction."""

import json
import os
import subprocess
import sys
import uuid

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from copper_mvp.bart_registry import BART_PARAMETERS, bart_source_hashes
from copper_mvp.bart_trees import draw_predictions, mixture_quantiles
from copper_mvp.classical_models import classical_preprocess
from copper_mvp.common import PROJECT_ROOT, WorkbenchError, file_hash, write_json


class BartModel:
    def __init__(self, seed, settings=None):
        self.seed = seed
        self.settings = dict(BART_PARAMETERS if settings is None else settings)
        self.fit_warnings = []

    def fit(self, X, y, sample_weight=None, max_wall_seconds=None):
        if sample_weight is not None:
            raise WorkbenchError("BART 首版不开放样本权重", "MODEL_SAMPLE_WEIGHT_UNSUPPORTED")
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.preprocessor = Pipeline(
            [
                ("columns", classical_preprocess("BART")),
                ("reduce", PCA(n_components=self.settings["components"], svd_solver="full")),
            ]
        )
        inputs = np.ascontiguousarray(self.preprocessor.fit_transform(X))
        self.target_scaler = StandardScaler().fit(y - X[:, :2])
        target = self.target_scaler.transform(y - X[:, :2])
        directory = PROJECT_ROOT / "runs/mvp/bart_fits" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        np.savez(directory / "input.npz", X=inputs, y=target)
        write_json(
            directory / "request.json",
            {
                "settings": self.settings,
                "seed": self.seed,
                "input_sha256": file_hash(directory / "input.npz"),
                "code_hashes": bart_source_hashes(),
            },
        )
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[key] = "1"
        environment["LANGSMITH_TRACING"] = environment["LANGCHAIN_TRACING_V2"] = "false"
        timeout = self.settings["max_fit_seconds"]
        if max_wall_seconds is not None:
            timeout = min(timeout, max(1.0, max_wall_seconds))
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    str(PROJECT_ROOT / "scripts/run_bart_fit.py"),
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
                raise WorkbenchError("BART 到达拟合时间预算，已有工件保留", "BART_TIME_BUDGET") from None
        if process.returncode:
            raise WorkbenchError("BART 拟合失败，详细记录保留在本地拟合目录", "BART_FIT_FAILED")
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state["status"] != "completed" or file_hash(directory / "posterior.joblib") != state["posterior_sha256"]:
            raise WorkbenchError("BART 后验工件校验失败", "BART_ARTIFACT_HASH")
        result = joblib.load(directory / "posterior.joblib")
        self.models = result["targets"]
        self.fit_warnings = result["fit_warnings"]
        self.fit_metadata = {
            "training_rows": len(X),
            "projected_features": inputs.shape[1],
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "fit_run_id": directory.name,
            "runtime": result["runtime"],
            "source_hashes": result["code_hashes"],
            "diagnostics": [t["diagnostics"] for t in self.models],
            "diagnostics_passed": result["diagnostics_passed"],
            "elapsed_seconds": result["elapsed_seconds"],
            "chains_per_target": self.settings["chains"],
            "draws_per_chain": self.settings["draws"],
            "fresh_sampler_per_chain": True,
        }
        return self

    def _posterior_means(self, X, target):
        return np.concatenate([draw_predictions(chain, X) for chain in self.models[target]["chains"]], axis=0)

    def predict(self, X):
        X = np.asarray(X, float)
        inputs = np.ascontiguousarray(self.preprocessor.transform(X))
        result = np.empty((len(X), 2))
        for target in (0, 1):
            for start in range(0, len(X), 128):
                result[start : start + 128, target] = self._posterior_means(inputs[start : start + 128], target).mean(
                    axis=0
                )
        return X[:, :2] + self.target_scaler.inverse_transform(result)

    def uncertainty(self, X):
        X = np.asarray(X, float)
        inputs = np.ascontiguousarray(self.preprocessor.transform(X))
        result = np.empty((len(X), 3, 2))
        for target in (0, 1):
            sigma = np.concatenate([c["sigma"] for c in self.models[target]["chains"]])
            for start in range(0, len(X), 32):
                means = self._posterior_means(inputs[start : start + 32], target)
                quantiles = mixture_quantiles(means, sigma)
                result[start : start + 32, :, target] = (
                    X[start : start + 32, target, None]
                    + self.target_scaler.mean_[target]
                    + self.target_scaler.scale_[target] * quantiles
                )
        return {
            "kind": "raw_marginal_quantiles",
            "levels": [0.1, 0.5, 0.9],
            "values": result,
            "distribution": "posterior_normal_mixture",
            "calibrated": False,
            "joint_region": False,
        }
