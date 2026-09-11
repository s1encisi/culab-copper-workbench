# 脚本导航

## 当前 MVP

| 脚本 | 用途 |
|---|---|
| [run_mvp.py](run_mvp.py) | 启动本机 FastAPI 和工作台，默认 127.0.0.1:8765 |
| [setup_mvp.ps1](setup_mvp.ps1) | 创建 Python 环境、安装锁定依赖、构建前端；仅新环境需要 |
| [verify_project.py](verify_project.py) | 检查冻结输入、当前 MVP 入口、诊断配置、应用版本、静态清单和当前文档本地链接 |
| [build_file_manifest.py](build_file_manifest.py) | 变更正式文件后重建当前文件清单；不更新历史验收或冻结输入哈希 |
| [benchmark_mvp.py](benchmark_mvp.py) | 本地固定 Python 与 LangGraph 路径的同模型对照；不测大模型诊断正确率 |

环境已准备好时直接启动。不要把环境安装脚本当作日常启动脚本。

## G1 验收

[verify_g1.py](verify_g1.py)检查全开发集的输入与标签时间边界，并输出真实事件的成熟前后证据。--before 可指定改造前的本地基线，--output 只能指向 runs/ 下的新验收位置；默认不会覆盖已有产物。

## G2a 比较

[run_model_comparison.py](run_model_comparison.py)运行六方法的固定协议比较，保存训练工件、OOF 预测、独立指标和报告。相同请求键只在相同配置与数据下复用；工件保留于 runs/mvp/model_comparisons。

## 保留的 V1/V2 兼容与研究入口

| 脚本 | 当前角色 |
|---|---|
| [run_v1_offline_smoke_v1.py](run_v1_offline_smoke_v1.py) | 原冻结预测基线的离线检查 |
| [run_v1_fault_injection_v1.py](run_v1_fault_injection_v1.py) | 原 V1 故障注入实验 |
| [run_langgraph_synthetic_v1.py](run_langgraph_synthetic_v1.py) | 原 V2 合成案例检查 |
| [run_langgraph_oof_parity_v1.py](run_langgraph_oof_parity_v1.py) | 原 V1/V2 的 OOF 数值一致性检查 |
| [validate_local_env_v1.py](validate_local_env_v1.py) | 旧版环境与提供商配置检查 |
| [run_llm_connectivity_v1.py](run_llm_connectivity_v1.py) | 旧角色配置的 LLM 连接实验；当前 Flash 诊断从工作台入口运行 |

这些路径由既有说明或验收引用，保留原位。[当前使用指南](../docs/MVP使用与验收.md)说明哪条入口用于日常工作。

## G2b 比较

[run_optimizer_comparison.py](run_optimizer_comparison.py)运行统一问题下的优化器对照。--mode benchmark 使用解析前沿数学案例；--mode plant --cases 8 使用八个历史工况，--seeds 与 --budget 明确重复和总求值上限。结果包含完整求值 CSV 和逐轮复核前沿。
