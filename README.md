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
