"""安全核验本机 .env；只输出变量名和脱敏后的运行元数据。"""

from __future__ import annotations

import argparse
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
    resolve_budget_runtime,
    resolve_provider_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()

    local_env: dict[str, str] = {}
    loaded_names = load_local_env(
        args.env_file,
        environ=local_env,
        ignore_unrelated=True,
    )
    registry = load_provider_registry(PROJECT_ROOT / "configs/llm/providers.yaml")
    runtimes = {
        name: resolve_provider_runtime(registry, name, environ=local_env)
        for name in ("kimi_orchestrator", "deepseek_executor")
    }
    budget = resolve_budget_runtime(registry, environ=local_env)

    required = {
        "KIMI_API_KEY",
        "DEEPSEEK_API_KEY",
    }
    payload = {
        "env_file_exists": args.env_file.is_file(),
        "loaded_variable_names": list(loaded_names),
        "missing_required_variable_names": sorted(required - set(local_env)),
        "providers": {
            name: {
                "live_calls": runtime.live_calls,
                "provider": runtime.provider,
                "model": runtime.model,
                "base_url": runtime.base_url,
                "api_key_configured": runtime.api_key is not None,
                "max_calls_per_run": runtime.max_calls_per_run,
            }
            for name, runtime in runtimes.items()
        },
        "budgets": budget.model_dump(),
        "secret_values_printed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not payload["missing_required_variable_names"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
