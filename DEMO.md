# CuLab 演示与运行说明

## 面试展示顺序

1. 用 README 的问题表和架构图说明价值：把研究查询、可复核证据和受控执行接成闭环。
2. 运行公开合成演示，展示正常执行与 ACK 丢失恢复的两份记录。
3. 展示关键代码：研究工具循环、权限检查、审批哈希、outbox 与设备账本。
4. 根据岗位展开模型评估、RAG/记忆或前端工作区，具体完成状态以 CLOSEOUT.md 为准。

## 无私密资料的公开演示

环境：Python 3.11。建议在独立虚拟环境中安装项目。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[mvp,dev]"
.\.venv\Scripts\python.exe -B scripts/demo_public.py
```

Linux/macOS 可使用 .venv/bin/python 运行相同命令。脚本为每次运行创建新目录，不覆盖已有记录。

输出包含：

- report.md：可直接阅读的场景结果。
- report.json：状态、权限阻断、回读与写入次数。
- workbench/device 数据库：两个独立持久化状态，方便追踪审批与设备结果。

正常场景和 ACK 丢失场景的最终状态均应为 VERIFIED，设备写入次数均为 1。ACK 丢失场景的 ack_received 为 false，结果通过账本和回读确认。

### 演示的含义

输入、设备参数和事实数值均为合成样例，决策步骤采用确定性回放。证据渲染、授权、幂等、审批及设备协调使用项目实际服务实现。该演示不访问外部模型接口，不生成真实模型质量分数。

## 公开源码可运行的协议测试

```powershell
python -B -m pytest tests/mvp/test_g4_control.py tests/mvp/test_g5_releases.py -k "not workbench_api and not release_api"
```

这组测试使用合成输入检验执行协议与发布规则。完整项目测试还包括依赖经授权历史资源的集成测试，不能将缺少资料导致的失败误解为公开演示需要上传工厂数据。

## 完整工作台

1. 安装项目依赖与相应方法的可选运行时。
2. 在本机设置 COPPER_MVP_DATA_DIR、COPPER_MVP_EVIDENCE_DIR、COPPER_MVP_CONTRACT_DIR、COPPER_MVP_LABEL_DIR，指向已获授权的资源。
3. 构建前端：在 web 目录执行 npm ci、npm run build。
4. 分别运行现有服务：

```powershell
python -B scripts/run_mock.py --root runs/mock --port 8766
python -B scripts/run_mvp.py --run-dir runs/mvp --mock-root runs/mock --port 8765 --mock-port 8766
```

访问 http://127.0.0.1:8765。首次启动的本机访问码位于对应运行目录的 owner_access.key，访问码不进入版本控制。

新研究助手的真实调用由 COPPER_ASSISTANT_LIVE_CALLS 配置控制。供应商配置、数据外发范围和预算应事先明确；公开演示不需要启用此开关。

## 页面流程

- 数据与质量：选择事件，核对单位、时间与来源，再进入预测或优化。
- 模型池与比较：比较、登记、影子回放、资格判定、提案及审批分别可查。
- 文档知识：接入、解析核对、索引与检索；失败时重试同一版本，已登记版本可补充原文。
- 任务与轨迹：查看真实执行状态；已不存在的运行进程显示为中断，原工件保留。
- Mock 控制与审批：核对具体参数和反馈；研究结论不会自行获得设备写入权限。

若首次更新后仍显示旧界面，执行一次 Ctrl+F5。后续入口页面通过缓存重新验证获得新构建。
