# CuLab 铜电解电积研究工作台

本私密仓库保存 CuLab 的程序代码、前端、测试与通用配置。

## 功能与结构

- `src/copper_mvp/`：FastAPI 服务、模型训练与预测、多目标优化、运行管理和工具型诊断。
- `src/copper_mas/`、`src/copper_langgraph_v2/`：保留的历史实现与兼容路径。
- `web/`：React、TypeScript 和 Vite 前端。
- `scripts/`、`tests/`：启动、验证和测试入口。
- `configs/llm/`、`configs/agents/`：通用模型提供商和编排配置；密钥通过本地环境变量提供。

## 本地运行

运行需要 Python 3.11、项目依赖和已获授权的本地数据及合同文件。当前工作区可沿用已准备的 Python 环境：

```powershell
& '../.analysis_work/venvs/copper-mvp/Scripts/python.exe' -X utf8 -B ./scripts/run_mvp.py
```

默认地址为 http://127.0.0.1:8765。新环境可参考 `scripts/setup_mvp.ps1`；该脚本会安装依赖并构建前端。

本仓库不包含工厂数据、内部工艺合同、训练模型或原始验收证据。克隆代码后，需通过经授权的本地途径恢复这些资源，方可执行真实数据训练、优化与完整集成测试。冻结资源保持原始字节，不通过修改源文件哈希来消除验证失败。内部使用说明及实验记录在原工作区保留。

## 数据与 Git 边界

Excel、CSV/TSV、数据库、模型工件、`.env`、本机配置、内部文档、工艺映射和验收记录不进入远端。`.env.example` 仅提供空密钥示例。推送前检查会检查待推送提交的完整可达历史，阻止受保护路径、疑似凭据及个人绝对路径。

本机已安装推送前检查；在其他克隆中启用：

```powershell
git config core.hooksPath .githooks
python -B scripts/check_git_upload.py --ref HEAD
```

测试中的密钥形状样例为合成值，检查器只对已审阅的精确路径和精确样例放行。新增文件仍需内容审阅，自动检查不能识别所有商业秘密。请勿使用 `--no-verify` 绕过推送检查。

## G1 数据与证据查询

后端版本 0.3.0 新增 /api/v2/data 下的只读依赖清单、TaskSpec、输入快照、成熟标签和事件证据接口。证据查询默认使用原事件的虚拟决策时间和 Persistence；实际计算时间另行记录。

本地资源可通过 COPPER_MVP_DATA_DIR、COPPER_MVP_CONTRACT_DIR、COPPER_MVP_LABEL_DIR 和 COPPER_MVP_EVIDENCE_DIR 配置。COPPER_MVP_G1_ENABLED=false 可关闭新增路由。所有路径由进程环境或本地代码设置，HTTP 不接收路径。

运行 scripts/verify_g1.py 可生成本地验收报告和一条真实事件的时间线；含真实事件和路径的产物只写入被 Git 忽略的 runs/。当前 G1 没有自动换模或新增模型，现有运行接口继续兼容。

## G2a 模型库与比较

后端版本 0.4.0 新增六种方法的统一入口：Persistence、DeltaRidge、DeltaHGB、ElasticNet、Huber、PLS。GET /api/v2/models 返回方法目录；POST /api/v2/model-comparisons 创建固定协议比较，可查询历史、显式回放模型并导出结果。

在项目目录运行 scripts/run_model_comparison.py，可完成本地五折比较和全开发重拟合。协议先保存，指标由独立模块从 OOF 预测及成熟标签重算；结果保留覆盖、警告、失败及计时。产物只写入 runs/。

G2a 不写入旧模型目录或自动换模，当前通过脚本与新 API 使用。COPPER_MVP_G2A_ENABLED=false 可关闭新路由。推理工件会核验版本、哈希和历史截止时间。

## 升级开发分支

本轮系统升级统一在 codex/system-upgrade 分支进行，G1、G2a 及后续阶段按完成的业务闭环提交。main 保留改进前的 34933e0 基线。完整设计与内部验收材料继续保存在本地受保护目录；开发按真实功能、实验与接口验收推进。

## G2b 优化器比较

版本 0.5.0 将旧优化与新比较统一到同一目标/约束计算。NSGA-II、SPEA2、SMS-EMOA 按相同初始点、总预算和参考点比较；运行 scripts/run_optimizer_comparison.py，或通过 /api/v2/optimizers 和 /api/v2/optimizer-comparisons 读取结果。工件仅保存本地，候选不会自动成为设备命令。
