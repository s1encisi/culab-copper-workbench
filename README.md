# CuLab — 铜电积研究工作台

面向铜电积研究的 Python / FastAPI / React 项目，包含代理建模、多目标优化、证据查询、人工确认与 Mock 设备执行链路。

## 运行

使用 Python 3.11，在项目根目录执行：

```bash
python -m pip install -e ".[mvp,dev]"
python -B scripts/demo_public.py
```

演示使用合成输入和 Mock 设备，无需工厂数据或付费模型调用。前端源码位于 `web/`，后端位于 `src/`，测试位于 `tests/`。可选模型需要另行安装对应运行时；本仓库不分发私有数据或模型权重。

第三方模型许可保留在 `third_party/`。
