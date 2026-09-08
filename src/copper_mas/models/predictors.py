"""可接入 A4 的本地预测器；不包含任何外部 LLM 调用。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from copper_mas.agents.runtime import NumericPrediction
from copper_mas.contracts.cards import ProcessModeCardV2
from copper_mas.data.leakage import assert_no_future_information


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PersistencePredictor:
    """论文中必须保留的当前值持续性基线。"""

    model_id = "PERSISTENCE_CURRENT_RESULT_V1"

    def __call__(
        self, feature_row: Mapping[str, Any], process_mode: ProcessModeCardV2
    ) -> NumericPrediction:
        del process_mode
        assert_no_future_information(feature_row)
        try:
            cu = float(feature_row["origin_cu_g_l"])
            arsenic = float(feature_row["origin_as_mg_l"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("持续性基线需要当前 origin_cu_g_l 和 origin_as_mg_l") from exc
        if not math.isfinite(cu) or not math.isfinite(arsenic):
            raise ValueError("当前 Cu/As 不是有限数，A4 必须拒绝预测")
        return NumericPrediction(predicted_cu_g_l=cu, predicted_as_mg_l=arsenic)


def load_selected_predictor(manifest_path: str | Path) -> PersistencePredictor:
    """从冻结选择清单加载当前 A4；未知选择一律拒绝而非猜测。"""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("external_2026_read") is not False:
        raise ValueError("最终选择清单必须明确声明 external_2026_read=false")
    selected = payload.get("selected_per_target") or {}
    models = {
        (selected.get(target) or {}).get("model_name")
        for target in ("target_cu_g_l", "target_as_mg_l")
    }
    if models == {"Persistence"}:
        predictor = PersistencePredictor()
        expected_id = (payload.get("a4_predictor") or {}).get("model_id")
        if expected_id != predictor.model_id:
            raise ValueError("选择清单的 model_id 与 PersistencePredictor 不一致")
        return predictor
    raise ValueError(f"当前运行时尚未实现该冻结模型组合: {sorted(str(x) for x in models)}")


class CandidateBundlePredictor:
    """加载显式指定的本地候选模型；加载前逐文件核对 SHA-256。"""

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        cu_model_name: str,
        as_model_name: str,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        manifest_path = self.bundle_dir / "bundle_manifest_v1.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("external_2026_used") is not False:
            raise ValueError("候选包必须明确声明 external_2026_used=false")
        if manifest.get("bundle_status") != "FROZEN_DEVELOPMENT_CANDIDATES_NO_WINNER":
            raise ValueError("当前加载器只接受冻结但未选胜者的开发期候选包")
        features = manifest.get("features") or {}
        if features.get("known_at_decision_only") is not True:
            raise ValueError("候选包未声明全部特征在决策时已知")
        self.feature_names = tuple(features.get("ordered_names") or ())
        if not self.feature_names:
            raise ValueError("候选包缺少冻结特征顺序")
        expected_feature_hash = features.get("ordered_names_sha256")
        feature_payload = json.dumps(
            list(self.feature_names), ensure_ascii=False, separators=(",", ":")
        )
        actual_feature_hash = hashlib.sha256(feature_payload.encode("utf-8")).hexdigest()
        if actual_feature_hash != expected_feature_hash:
            raise ValueError("冻结特征顺序哈希不一致")

        candidates = manifest.get("serialized_candidates") or []
        self._cu_model = self._load_one(candidates, cu_model_name, "target_cu_g_l")
        self._as_model = self._load_one(candidates, as_model_name, "target_as_mg_l")
        self.model_id = (
            f"{manifest['bundle_id']}::Cu={cu_model_name}::As={as_model_name}"
        )

    def _load_one(self, candidates: list[dict[str, Any]], model_name: str, target: str) -> Any:
        matches = [
            item
            for item in candidates
            if item.get("model_name") == model_name and item.get("target_name") == target
        ]
        if len(matches) != 1:
            raise ValueError(f"候选包中找不到唯一模型: {model_name}/{target}")
        item = matches[0]
        artifact = str(item.get("artifact") or "")
        if not artifact or Path(artifact).name != artifact:
            raise ValueError("候选模型 artifact 必须是包内单一文件名")
        path = (self.bundle_dir / artifact).resolve()
        if path.parent != self.bundle_dir or not path.is_file():
            raise ValueError("候选模型文件不在冻结包内")
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"候选模型 SHA-256 校验失败: {artifact}")
        return joblib.load(path)

    def __call__(
        self, feature_row: Mapping[str, Any], process_mode: ProcessModeCardV2
    ) -> NumericPrediction:
        del process_mode
        assert_no_future_information(feature_row)
        missing = [name for name in self.feature_names if name not in feature_row]
        if missing:
            raise ValueError(f"A4 输入缺少冻结特征: {missing[:5]}")
        matrix = pd.DataFrame(
            [[feature_row[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        cu = float(self._cu_model.predict(matrix)[0])
        arsenic = float(self._as_model.predict(matrix)[0])
        if not math.isfinite(cu) or not math.isfinite(arsenic):
            raise ValueError("候选模型输出非有限数")
        return NumericPrediction(predicted_cu_g_l=cu, predicted_as_mg_l=arsenic)
