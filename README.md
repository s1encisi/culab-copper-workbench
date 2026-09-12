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

## G3 研究对话

版本 0.6.0 新增本机访问码登录、连续研究对话、工具证据、任务暂停/恢复/取消。首次启动后，访问码位于运行目录的 owner_access.key；原 API 也通过身份与权限检查。

研究对话使用只读工具访问已选资源。默认关闭真实模型调用；明确设置进程变量 COPPER_ASSISTANT_LIVE_CALLS=1 后才调用现有 DeepSeek 密钥。会话、任务、证据和费用留在本地数据库，首次迁移保留备份。

## G4 Mock 控制

版本 0.7.0 新增独立的合成 PLC 服务，以及工作台中的 Mock 控制页面。点表使用合成 0–200 A 量程；化验历史与模拟电流互相独立。

先运行 `python -B scripts/run_mock.py --root runs/mock`，再运行 `python -B scripts/run_mvp.py --mock-root runs/mock`。两者分别监听本机 8766、8765 端口；沿用当前工作区 Python 环境即可。`--manual-clock` 可让 Mock 仅通过测试管理员接口推进时间。

工作台 owner 可创建、审批、取消提案并查询回读。审批绑定完整参数、控制版本、启动代次和有效期；修改参数需重新提案。发送记录和设备账本分别持久化，进程中断后继续协调原命令。取消已提交命令不恢复设定值；补偿需独立提案与审批。

访问码、独立 Mock 驱动与测试管理员凭据、命令记录和验收证据全部保存在被 Git 忽略的运行目录。研究对话未注册设备写入、时钟推进或故障注入工具。

## G5a 成熟标签与影子路由

版本 0.8.0 新增 /api/v2/prediction-studies：从已完成的模型比较创建历史选模回放，查询共同成熟样本、分层误差、逐事件决策和确认窗口证据。全局与工况路由同固定模型比较，当前默认模型保持不变。

scripts/run_routing_replay.py 提供本地执行入口；必需参数为 --comparison-id 和 --request-key。回放、标签引用和预测账本只保存在本地运行目录。G5 的嵌套 OOF 集成、模型发布回退与优化器组合继续单独实施。

## G5b 嵌套 OOF 集成

版本 0.9.0 新增 /api/v2/ensemble-studies 与 scripts/run_ensemble_study.py，支持嵌套 Stacking、尾段 Blending、OOF 固定融合、成熟标签动态融合和时间块 Bagging。参数在各外层训练段内选择，保留种子、原时间折、训练证据与独立评价。

集成研究可从已核验的完成折恢复；所有模型、预测和证据保存在本地运行目录。创建、恢复需要计算权限，默认模型与优化代理不会因研究完成自动替换。

## G5c 模型生命周期与发布

版本 0.10.0 新增模型工件登记、历史运行影子检查、按目标的模型指针、精确发布审批和回退。发布提案绑定工件/证据哈希与指针版本；版本变化后需要重新提案。

/api/v2/release-predictions 保留模型、发布与输入快照引用，并区分当前研究回看和原事件时的选择。完整性失败会记录原因并使用参考模型；旧预测记录保持原样。现有候选尚未因此改变正式默认模型。

## G5d 优化器组合

版本 0.11.0 新增 /api/v2/optimizer-portfolios 和 scripts/run_portfolio_study.py。固定 NSGA-II、固定 SPEA2、轮流与自适应组合在同一初始种群和总预算下比较；各臂 pilot 独立等额，后续只交换同签名的已评估前沿。

比较保存求值、分配轨迹、独立复核和逐次尝试，结果按工况与种子配对。失败恢复保留原成本并与单次预算比较分开；当前默认优化器与设备执行资格保持不变。

## G6a 经典方法扩展

版本 0.12.0 将注册目录扩展为 20 种方法，包含设计中的 17 个新增模型。新增方法支持训练期数值稳定缩放、缺失指示、明确的样本权重能力，以及部分方法的未校准边际概率输出。

模型比较和预测接口可显式选择新方法；原比较默认仍为旧六方法。scripts/run_classical_study.py 与 /api/v2/classical-studies 提供固定时间折、多种子目录研究和报告读取。模型数量与实际运行状态分开登记，不自动更换默认模型。
