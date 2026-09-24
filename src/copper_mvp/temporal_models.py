"""Context-aware fixed-window neural forecasts without cross-event recurrent state."""

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from copper_mvp.common import WorkbenchError, digest
from copper_mvp.data_contracts import source_time
from copper_mvp.data_service import DataService
from copper_mvp.neural_runtime import load_tensor_runtime, tensor_random_scope
from copper_mvp.temporal_inputs import TemporalInputs
from copper_mvp.temporal_registry import TEMPORAL_ARCHITECTURES, TEMPORAL_TRAINING


class TemporalModel:
    def __init__(self, method, seed):
        self.method_id, self.seed = method, seed
        self.training = dict(TEMPORAL_TRAINING)
        self.architecture = dict(TEMPORAL_ARCHITECTURES[method])
        self.fit_warnings = []

    def fit_context(self, data, train_ids, cutoff, max_wall_seconds=None):
        torch = load_tensor_runtime()
        from copper_mvp.temporal_networks import TemporalNetwork

        cutoff = source_time(cutoff)
        if any(source_time(data.row(e).decision_at) >= cutoff for e in train_ids):
            raise WorkbenchError("时序模型训练事件越过截止时间", "TEMPORAL_TRAINING_CUTOFF")
        labels = DataService(data).labels.latest(cutoff)
        if not all(
            (e, t) in labels and labels[(e, t)].quality_eligible and labels[(e, t)].value is not None
            for e in train_ids
            for t in ("cu", "as")
        ):
            raise WorkbenchError("时序模型训练标签尚未成熟", "IMMATURE_TRAINING_LABEL")
        y = np.array([[labels[(e, t)].value for t in ("cu", "as")] for e in train_ids])
        current = data.X.loc[train_ids].to_numpy(float)[:, :2]
        self.preprocessor = TemporalInputs().fit(data, train_ids)
        sequence, context = self.preprocessor.transform(data, train_ids)
        self.target_scaler = StandardScaler().fit(y - current)
        target = self.target_scaler.transform(y - current).astype(np.float32)
        inputs = (torch.from_numpy(sequence), torch.from_numpy(context), torch.from_numpy(target))
        started = time.perf_counter()
        history = []
        updates = 0
        with tensor_random_scope(torch, self.seed):
            network = TemporalNetwork(self.method_id, self.architecture, sequence.shape[-1], context.shape[-1])
            initial = {name: p.detach().clone() for name, p in network.named_parameters()}
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=self.training["learning_rate"], weight_decay=self.training["weight_decay"]
            )
            generator = torch.Generator().manual_seed(self.seed)
            for epoch in range(self.training["epochs"]):
                network.train()
                total = 0.0
                for indices in torch.randperm(len(train_ids), generator=generator).split(self.training["batch_size"]):
                    if max_wall_seconds is not None and time.perf_counter() - started > max_wall_seconds:
                        raise WorkbenchError("时序模型达到训练预算", "NEURAL_TIME_BUDGET")
                    optimizer.zero_grad(set_to_none=True)
                    prediction = network(inputs[0][indices], inputs[1][indices])
                    loss = (prediction - inputs[2][indices]).square().mean()
                    if not torch.isfinite(loss):
                        raise WorkbenchError("时序模型损失非有限", "NEURAL_LOSS")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), self.training["gradient_clip"])
                    if not torch.isfinite(norm):
                        raise WorkbenchError("时序模型梯度非有限", "NEURAL_GRADIENT")
                    optimizer.step()
                    updates += 1
                    total += float(loss.detach()) * len(indices)
                history.append({"epoch": epoch + 1, "standardized_mse": total / len(train_ids)})
            network.eval()
            self.state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            changes = {name: float((p.detach() - initial[name]).abs().max()) for name, p in network.named_parameters()}
            self._network = network
        self.models = [{"method": self.method_id, "state_dict": self.state}]
        self.fit_metadata = {
            "training_rows": len(train_ids),
            "training_event_hash": digest(train_ids),
            "fit_cutoff": cutoff.isoformat(),
            "sequence_steps": 7,
            "sequence_features": sequence.shape[-1],
            "context_features": context.shape[-1],
            "target_delta_mean": self.target_scaler.mean_.tolist(),
            "target_delta_scale": self.target_scaler.scale_.tolist(),
            "epochs": self.training["epochs"],
            "optimizer_updates": updates,
            "training_history": history,
            "parameter_max_changes": changes,
            "parameters": sum(p.numel() for p in network.parameters()),
            "elapsed_seconds": time.perf_counter() - started,
            "hidden_state_carried_between_events": False,
            "training_labels_in_sequence": False,
            "device": "cpu",
            "native_threads": 1,
            "early_stopping": False,
        }
        return self

    def __getstate__(self):
        values = self.__dict__.copy()
        values.pop("_network", None)
        return values

    def network(self):
        if not hasattr(self, "_network"):
            torch = load_tensor_runtime()
            from copper_mvp.temporal_networks import TemporalNetwork

            with tensor_random_scope(torch, self.seed):
                self._network = TemporalNetwork(
                    self.method_id,
                    self.architecture,
                    self.fit_metadata["sequence_features"],
                    self.fit_metadata["context_features"],
                )
            self._network.load_state_dict(self.state)
            self._network.eval()
        return self._network

    def predict_context(self, data, event_ids):
        torch = load_tensor_runtime()
        sequence, context = self.preprocessor.transform(data, event_ids)
        network = self.network()
        network.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(event_ids), 128):
                result.append(
                    network(
                        torch.from_numpy(sequence[start : start + 128]), torch.from_numpy(context[start : start + 128])
                    ).numpy()
                )
        delta = np.concatenate(result, axis=0)
        current = data.X.loc[event_ids].to_numpy(float)[:, :2]
        values = current + self.target_scaler.inverse_transform(delta)
        if not np.isfinite(values).all():
            raise WorkbenchError("时序预测非有限", "MODEL_OUTPUT_VALUES")
        return values
