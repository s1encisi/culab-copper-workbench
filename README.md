# CuLab 铜电解电积研究工作台

本私密仓库保存 CuLab 的程序代码、前端、测试与通用配置。

[查看升级阶段、分支依赖与逐阶段差异 →](UPGRADE_BRANCHES.md)

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

本轮升级按 G 阶段分别保存为递进分支，每个分支继承前一阶段。查看 [升级分支导航](UPGRADE_BRANCHES.md)，可直接打开各阶段源码、提交和相邻阶段差异。

codex/system-upgrade 是已集成升级的汇总入口，当前集成到 G6g。后续源码保存在各阶段分支；本地 codex/g8-ui 保存十工作区界面检查点，codex/g8 继续界面整合。main 保留升级前的 34933e0 基线。依赖、验收与远端同步状态见升级分支导航。

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

## G6b 统计与事件序列方法

版本 0.13.0 增加 LocalLinearKernel、PSplineGAM、SARIMAX、VAR、ETS。目录共 25 种方法，覆盖设计中的 22 个新增模型。序列方法保留完整事件索引，按时间更新状态，训练参数与预测时可得信息分开处理。

可选统计依赖通过 scripts/setup_statistical_models.ps1 按固定版本与哈希安装到项目运行目录。统一比较、历史回放和概率查询接口支持这些方法，默认比较范围与既有模型行为保持兼容。

## G6c 优化方法扩展

版本 0.14.0 增加 NSGA-III、MOEA/D、RVEA、AGE-MOEA、C-TAEA、GDE3 和 Omni-Optimizer。目录共 10 种方法，覆盖设计中的 9 个新增优化方法；原三算法仍为默认比较范围。

scripts/run_optimizer_comparison.py 可通过 --optimizers 显式选择这些方法，MOEA/D 的接口标识为 MOEA-D。新方法使用共享初始点、固定目标尺度和完整求值预算；MOEA/D 显式执行约束优先的邻域替换，GDE3 支持最后不足一代的预算。AGE-MOEA 可选依赖通过 scripts/setup_optimizer_methods.ps1 安装到本地运行目录。

## G6d 指标、档案与群体搜索方法

版本 0.15.0 接入 IBEA、Epsilon-MOEA、SMPSO、PAES、PESA-II 和 MO-CMA-ES。目录共 16 种算法，原三算法仍为默认比较范围。可选 Platypus 依赖通过 scripts/setup_platypus_methods.ps1 按固定版本与哈希安装。

各方法复用共同初始样本及原始 F/G 求值账本；克隆候选按比较协议重新求值，尾批次使用剩余预算。IBEA 使用约束优先的指标比较，多目标 CMA-ES 按非支配层级和拥挤度更新分布。方法配置、结果与独立验收沿用现有优化比较接口。

## G6e HypE 与标量化前沿

版本 0.16.0 接入 HypE、Epsilon-Constraint、Augmented-Chebyshev、Weighted-Sum、NBI 和 NNC，目录共 22 个算法。HypE 采用双目标精确体积分配；五种标量化方法执行重复锚点与偏好子问题求解，联合计算 F/G 并计费复核返回点。

scripts/run_optimizer_comparison.py 新增 --model-profile。NBI 的历史比较明确选择 DeltaRidge；原默认三算法和 DeltaHGB 配置保留。报告同时列出候选可行性、停止原因和内层收敛数，失败运行保留已计费求值账本。

## G6f 贝叶斯优化方法

版本 0.17.0 接入 ParEGO、NEHVI、MES 和 JES，采用项目内固定 CPU 张量运行时。每次拟合高斯过程，使用原生 BoTorch 采集函数选择候选，并保存模型工件、采集开销和实际求值账本。

ParEGO/NEHVI 支持已接入的约束概率路径；MES/JES 首版通过 --benchmark-problem unconstrained_quadratic 显式运行无约束数学问题。后者不会接受工厂 As 约束请求。原默认三算法及默认有约束数学问题保持不变。

## G6g 专用预测方法

版本 0.18.0 接入 CatBoost、NGBoost、EBM 和 Cubist，目录共 29 种预测方法。CatBoost 使用输入时间顺序；NGBoost 提供正态边际分布；EBM 保留加性形状和交互项；Cubist 提供规则内的线性模型。

运行 scripts/run_model_inventory_study.py，可通过既有模型比较服务执行固定时间折、多种子研究。报告包含误差、配对时间块区间、工况分层、NLL/CRPS/WIS 和资源记录。scripts/verify_model_inventory.py 重载保存工件，核对预测回放、EBM 加性重构及跨种子稳定性、Cubist 规则，以及模型登记和影子预测。

可选依赖使用 scripts/setup_specialized_models.ps1 按固定版本和哈希安装。原始预测、模型、解释与验收证据保存在本地 runs/ 目录，模型工件记录实际依赖及许可证。

## G6m 本地文档知识库

版本 0.24.1 增加文档登记、版本化解析、关键词和本地语义检索。文档、向量与引用全部保存在运行目录的 knowledge/ 中。HTTP 不接受服务器文件路径，文档内容通过登记后的上传接口提供。

先运行 `python -B scripts/setup_knowledge.py`，安装独立依赖并下载固定哈希的公开中文嵌入模型。已有独立环境可以设置 COPPER_KNOWLEDGE_RUNTIME_DIR 与 COPPER_KNOWLEDGE_MODEL_DIR。检索时只读取本地文件，不下载权重或调用外部 API。嵌入来源：[BAAI BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[Xenova ONNX 转换](https://huggingface.co/Xenova/bge-small-zh-v1.5)，固定版本与哈希见 configs/runtime/knowledge_embedding.json。

接口位于 /api/v2/knowledge：先 POST /documents 登记元数据与访问者，再 PUT /documents/{doc_id}/versions/{version}/content 上传原文，必要时审核，最后 POST 同版本的 /index 建立索引。POST /search 返回带版本、页码或段落、内容哈希的引用；引用和原文下载每次重新检查权限。更改访问者或撤销文档立即影响后续检索；新版本索引成功前保留旧索引。

MD 表格保留完整行、表头与单位；DOCX 保留合并单元格结构、原始公式 XML 和图片位置。PDF 的版面、表格和公式需要审核。扫描页通过同一版本的 /ocr 接口调用本机 RapidOCR；请求需携带当前 parse_hash。识别结果保存分数、文字框、页图哈希及模型版本，随后通过 /corrections 校正表格或公式，再以 /review 审核准确的解析哈希。长表格按完整行组切块并重复表头与脚注。修改解析时保留旧索引，完成审核和重建后原子切换；旧解析保留为可回查修订，删除文档时一并清除。会话记忆和本地文档引用工具的接入见下一节；领域评价与模型侧片段推理继续推进。文档文字是证据，不能授予审批或设备操作权限。

运行 `python -B scripts/verify_knowledge.py --output runs/g6m/verification-新编号` 可执行断网条件下的合成检索对照、引用回读与权限撤销验证。该结果用于接口验收；经审核的真实领域问答集评价仍待后续完成。

扫描验证使用本地合成页。可用含 reportlab 的 Python 运行 `scripts/create_ocr_fixture.py --font <本地中文字体> --output runs/g6m/fixture-新编号`，再运行 `scripts/verify_knowledge_ocr.py --fixture runs/g6m/fixture-新编号/scan.pdf --output runs/g6m/ocr-新编号`。模型来源：[RapidOCR](https://github.com/RapidAI/RapidOCR)，检测/方向/识别模型的固定哈希位于 configs/runtime/knowledge_ocr.json。OCR 分数不是经过校准的正确率；合成页验证只证明该接口链路，不能替代真实领域文档评价。

## G6m 会话记忆与文档引用

版本 0.24.2 增加基于任务和证据引用的会话记忆。POST /api/v2/sessions/{id}/memory 创建有有效期的记忆，GET 重新读取权威记录，DELETE 删除该记忆。自动恢复会检查来源变化；删除后的记忆不会自动重建，直到用户再次创建。初始研究目标、最近任务和未完成任务分别保留；数值与单位从原证据重读，摘要不能代替审批。

Context 可选择 knowledge_refs 和 control_command_ids。文档引用需要 chunk_id、hash、parse_hash，可指定历史 as_of；命令审批状态通过实际命令记录读取。文档被撤销、修订或失效时，恢复结果会标记不可用；文档工具缓存随内容、权限和有效版本变化而失效。

研究助手增加 search_documents 和 read_document。当前模式由模型选择查询、本机插入文档原文，提供商只得到引用和本地事实名称。会话中保存引用模板，查看回答时再检查文档权限；删除文档后不会继续展示旧正文。模型读取正文的受控 RAG 接口见下一节；真实模型答案质量、领域评价和后续适配仍待验收。

## G6m 受控文档推理

版本 0.24.3 为模型读取正文增加独立授权。owner 在 /api/v2/knowledge/documents/{doc_id}/versions/{version}/disclosures 创建记录，绑定 document_hash、parse_hash、deepseek-flash 提供商、片段长度估算上限、有效期和用途。授权只适用于已审核索引，当前访问权限仍需通过；真实模型调用继续受现有开关和费用预算控制。

每次请求临时装配已授权片段，工具证据库只保存引用。模型通过 document_refs 指定 evidence_id 和片段 field，服务器校验该片段确实提供过，并返回文档版本、内容哈希和原文位置。未授权的文档继续通过本地引用展示。

撤销正文授权使用 /api/v2/knowledge/documents/{doc_id}/disclosures/{consent_id}/revoke。撤销、文档删除或读者权限移除会清理相关账号的派生回答和待提交结果，并保留不含正文的审计记录；服务启动时也会检查离线期间失效的来源。历史文档回答通过本轮工具重新取证，正文不会直接从旧对话拼接进下一次模型请求。

受控 RAG 的管线验证使用合成文档与 MockTransport，覆盖准确版本授权、过期、调用中撤销、账号隔离、重启清理、文档指令隔离及片段级引用。该验证不代表真实模型的领域回答质量已经通过验收。

## G6m 可选本地重排与条款解析

版本 0.24.4 支持本地多语言交叉编码器重排。默认使用 BM25、向量检索及倒数排名融合；POST /api/v2/knowledge/search 可显式设置 rerank=true，研究工具 search_documents 也提供该选项。权限和生效时间先筛选候选，重排只处理已获访问权限的内容。

[官方 mMARCO MiniLM 交叉编码器](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) 的固定版本、权重和 tokenizer 哈希保存在 configs/runtime/knowledge_reranker.json。scripts/setup_knowledge.py 可安装本地资产，并支持中断续传。重排分数是相关性排序值；响应另记录候选数量、输入截断数量、模型签名及耗时。

现有项目文档上的预设查询对照显示：该通用模型提高了首位条款命中率，但前五条覆盖率和延迟未改善，因此保留为可选模式。该小规模工程查询集不能代替独立领域评价。

Markdown 列表现在按单条规则保留位置。已有文档可通过 /api/v2/knowledge/documents/{doc_id}/versions/{version}/reparse 重新解析，请求携带 expected_parse_hash。重新解析会保留原文件、历史解析和原活动索引，审核并重建后再切换；旧正文授权不会自动转到新的解析哈希。

## G6n 独立时序校准

版本 0.25.0 为注册预测方法提供独立时序校准研究。每个既有外层训练窗口再分为基础拟合段和后续校准段；插补、缩放和模型参数只在基础拟合段拟合。训练标签需在基础拟合截止前已可获得，校准标签需在外层截止前成熟，外层验证标签由独立评价器读取。

提供两种名义 90% 区间：每目标绝对残差校准，以及使用训练段目标变化量标准差归一化的最大残差联合区域。尺度为零时明确关闭联合区域。区间保持未裁剪实数边界，并记录负下界；两个边际 90% 区间不被当作联合 90%。时间相关和漂移使交换性条件不成立，界面与记录仅报告实测覆盖，不声明无条件保证。

默认对照包含 Persistence、DeltaRidge、BayesianRidge、MultiTaskElasticNet、Quantile 和 NGBoost。固定确定性配置只保留一个真实种子运行；随机配置使用请求预先指定的种子。原生概率区间与 C90 使用同一个基础拟合模型，报告覆盖、宽度、WIS、可用时的 NLL/CRPS、按时间折结果和配对时间块区间。

运行 scripts/run_calibration_study.py 可创建研究；scripts/verify_calibration_study.py 从保存的模型和校准器重新推理。结果均在 runs/ 下。API 位于 /api/v2/calibrations，可查询研究并以 /{id}/predict 回放已有外层验证事件；不会自动改变发布指针或获得控制资格。

专用模型可通过 COPPER_SPECIALIZED_RUNTIME_DIR 和 COPPER_CUBIST_RUNTIME_DIR 指向经验证的本地依赖。基础模型和校准器分别保存，并记录数据段、标签版本、参数、源码及工件哈希。

方法依据：[保形预测入门](https://arxiv.org/abs/2107.07511)；时间漂移下的保证边界及自适应方案背景见 [Adaptive Conformal Inference](https://arxiv.org/abs/2106.00170)。当前实现采用固定独立时序校准段，没有将自适应方法的理论保证套用到本实现。

## G6n 校准工件生命周期

校准研究可使用 source_kind=calibration_study 进入现有模型工件登记、影子验证和发布提案流程。工件同时绑定基础模型、校准器、数据段、来源预测和独立评价的哈希；校准器被修改时，该工件不能继续通过校验。

发布资格仍使用原有误差、覆盖、分组稳定性和时延门槛。选择一个保存种子不丢弃其他预设种子的负面结果，资格评价保留全部种子的证据。校准能力不会自动更新发布指针，也不授予响应代理或因果控制资格。

已发布预测记录可保留边际区间和联合区域的投影。只有 Cu、As 来自同一个模型、种子、时间折及校准器时，记录才提供联合二维区域；来自不同校准组的两个区间不会被标记为联合 90%。名义水平与实际覆盖分别记录。

## G8 工作区界面检查点

界面按总览、数据与质量、模型池与比较、优化实验室、自由诊断助手、任务与轨迹、Mock 控制与审批、文档知识、领域适配、评估与治理组织。已有模型、知识和任务接口通过对应工作区使用；统一顶部显示实际项目、环境和版本，证据通过侧栏查看。领域适配服务尚未实现时显示未接入状态，真实助手调用沿用进程配置和授权。

此检查点完成构建、账号隔离与三种屏幕尺寸的导航检查；完整领域适配、付费诊断和全部生命周期操作的界面验收仍在后续阶段。

## 当前收尾状态

完整升级已按用户要求暂停。最新模型流程源码保存在 codex/g8-models-checkpoint，依赖和验证范围见 [升级分支导航](UPGRADE_BRANCHES.md)。该分支为可继续工作的源码检查点；完整领域适配和端到端验收仍未完成。

2026-09-15 对已实现功能的复查与四项缺陷修复保存在 codex/g8-review-20260915。74 个独立自动测试用例、构建与对应浏览器复测通过；详见分支导航中的本轮复查记录。
