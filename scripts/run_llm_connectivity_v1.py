"""Kimi K3 中国站 + DeepSeek V4 Pro 的脱敏首次连通脚本。

不带 --execute-live 时仅做本地预检，不发起网络请求。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_mas.config.settings import (  # noqa: E402
    load_local_env,
    load_provider_registry,
    redact_text,
    resolve_budget_runtime,
)
from copper_mas.llm.connectivity import (  # noqa: E402
    CONNECTIVITY_PROVIDERS,
    run_connectivity_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="默认只加载本正式工程根目录的 .env（该文件不随工程复制）",
    )
    parser.add_argument(
        "--provider",
        choices=("all", *CONNECTIVITY_PROVIDERS),
        default="all",
    )
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--acknowledge-max-cost-cny", type=float)
    parser.add_argument("--declared-month-spend-cny", type=float, default=0.0)
    args = parser.parse_args()

    local_env: dict[str, str] = {}
    loaded_names = load_local_env(
        args.env_file,
        environ=local_env,
        ignore_unrelated=True,
    )
    registry = load_provider_registry(PROJECT_ROOT / "configs/llm/providers.yaml")
    budget = resolve_budget_runtime(registry, environ=local_env)
    if args.execute_live:
        acknowledged = args.acknowledge_max_cost_cny
        if acknowledged is None or abs(acknowledged - budget.max_cny_per_run) > 1e-12:
            parser.error(
                "真实连通必须显式传入 "
                f"--acknowledge-max-cost-cny {budget.max_cny_per_run:g}"
            )

    provider_names = (
        CONNECTIVITY_PROVIDERS if args.provider == "all" else (args.provider,)
    )
    run_id = "llm-connectivity-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        report = run_connectivity_v1(
            registry=registry,
            budget_runtime=budget,
            environ=local_env,
            execute_live=args.execute_live,
            run_id=run_id,
            provider_names=provider_names,
            declared_month_spend_cny=args.declared_month_spend_cny,
        )
        report["env_file"] = str(args.env_file.resolve())
        report["loaded_variable_names"] = list(loaded_names)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # 顶层仅输出脱敏错误，不回显运行对象或环境值。
        error_report = {
            "schema_version": "1.0",
            "execution_status": "FAILED_CLOSED",
            "error_type": type(exc).__name__,
            "error": redact_text(str(exc)),
            "loaded_variable_names": list(loaded_names),
            "secret_values_printed": False,
        }
        print(json.dumps(error_report, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
