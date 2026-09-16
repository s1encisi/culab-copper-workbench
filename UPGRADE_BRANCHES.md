# 分支导航与依赖

项目沿用设计中的 G 编号。每个阶段保存源码节点，阶段接口实现、研究验收和发布资格分别记录。

## 入口

- main：固定在 34933e0 的升级前基线，未合入升级内容。
- codex/portfolio-release-20260916：完整交付源码、公开演示和项目展示入口。
- codex/system-upgrade：综合升级入口，快进到交付节点。
- 实验未完成项与前置条件见 [CLOSEOUT.md](CLOSEOUT.md)。

## 阶段分支

比较使用实际共同依赖的固定提交，避免后续分支补丁改变早期阶段的比较口径。

| 分支 | 实际继承基点 | 内容 | 本阶段差异 |
| --- | --- | --- | --- |
| [codex/g1](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g1) | 34933e0 | 数据、时间边界与事件证据 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/34933e0308e7fda71e6cc3807302cbb674c97bf0...codex/g1) |
| [codex/g2a](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g2a) | d2ea1eb | 模型注册与固定协议比较 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/d2ea1eb873e989b41feb5b70c46cf60965dd1b7d...codex/g2a) |
| [codex/g2b](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g2b) | 24949dd | 优化器注册与统一评价 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/24949ddfa2b497e4287753823d798833e175f58d...codex/g2b) |
| [codex/g3](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g3) | 0a86817 | 研究对话、权限和任务恢复 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/0a868173f88566cefd4116823a44ec9d4fc63ef6...codex/g3) |
| [codex/g4](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g4) | 80b85e3 | Mock 控制、审批与回读 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/80b85e3f482438e8017793fdd65ba4b317099ada...codex/g4) |
| [codex/g5a](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5a) | e4c5ef5 | 成熟标签评价与影子路由 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/e4c5ef54c4cd443835590e835479822fcec3d947...codex/g5a) |
| [codex/g5b](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5b) | 27923fd | 嵌套集成与时间块研究 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/27923fdd31a46a161038ca5fa100c4a36b26c748...codex/g5b) |
| [codex/g5c](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5c) | 37f47c2 | 模型工件、发布与回退 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/37f47c22dbcba68b40d94169a2ce13b1a5a7a6f0...codex/g5c) |
| [codex/g5d](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5d) | a0ba976 | 优化器组合与配对比较 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/a0ba976e6f2f6b84f160ec0958210935f3a46cef...codex/g5d) |
| [codex/g6a](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6a) | 1c46bf1 | 经典预测方法 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/1c46bf1083f401c9455d4b3d1ed06c1c2ad544b8...codex/g6a) |
| [codex/g6b](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6b) | 19bcef6 | 统计与事件序列方法 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/19bcef6d7ba497bfe396a1e408129d4f90bc886d...codex/g6b) |
| [codex/g6c](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6c) | 6149b91 | pymoo 方法扩展 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/6149b910993013aa3864a11942482c925b143582...codex/g6c) |
| [codex/g6d](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6d) | 00f231e | Platypus 方法扩展 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/00f231ed0975551c9c3de2918910129eb3d694a1...codex/g6d) |
| [codex/g6e](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6e) | 79f75c7 | HypE 与标量化 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/79f75c7946f26de8646f8187c3dd626627bd762f...codex/g6e) |
| [codex/g6f](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6f) | 0d4d3e5 | 贝叶斯多目标优化 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/0d4d3e5a9d09a582d978055fe887bfbb4ce70e84...codex/g6f) |
| [codex/g6g](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6g) | 941e658 | 专门化预测与概率评分 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/941e6585e6e2ad5ac7c3e3ed197f3be7906a648f...codex/g6g) |
| [codex/g6h](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6h) | 9765f3e | BART 与采样诊断 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/9765f3e230169d2f474e91a8976d8e6d91206a3e...codex/g6h) |
| [codex/g6i](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6i) | 7addfb8 | 符号回归与公式导出 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/7addfb856da3a78d101599054de76c993c978870...codex/g6i) |
| [codex/g6j](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6j) | 8a846b4 | 表格神经模型与选择性恢复 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/8a846b4e7f3cbc7faf27a0702ab6294e3d39cd75...codex/g6j) |
| [codex/g6k](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6k) | ce9bd72 | GRU、LSTM、因果 TCN | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/ce9bd720ec9aadef572dbb2678ed3f24a8ea35bc...codex/g6k) |
| [codex/g6l](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6l) | 96bd78f | TabPFN 与数值回放 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/96bd78fdbc3e49e5962a5107081959bb606acf5e...codex/g6l) |
| [codex/g6m](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m) | 966d5ed | 文档、OCR、记忆与受控 RAG | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/966d5ed972899ffa99831a3bb19d0c2302080a0a...codex/g6m) |
| [codex/g6n](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6n) | 1cde3e3 | 共享时序校准 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/1cde3e3f59777a22da17edfe84497a9d3f2e7d53...codex/g6n) |
| [codex/g6n-release](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6n-release) | 1711b51 | 校准工件生命周期 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/1711b51753c8b6a4c06b6d6f9ce36e19286fe56a...codex/g6n-release) |
| [codex/g8](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g8) | 6ef5379 | 十工作区与模型流程 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/6ef537990853078805078d2e4b5914ec86472558...codex/g8) |
| [codex/g8-review-20260915](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g8-review-20260915) | f0e9283 | 四项已复现缺陷修复 | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/f0e92836be7667cbd07e7485d139fa1045433d07...codex/g8-review-20260915) |

## 固定功能检查点

| 分支 | 提交 | 内容 |
| --- | --- | --- |
| [codex/g6m-core](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m-core) | [b3e0951](https://github.com/s1encisi/culab-copper-workbench/commit/b3e0951e1a104c858f16d01c6ba98c0a1ca3048f) | 文档登记、结构解析与混合检索 |
| [codex/g6m-ocr](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m-ocr) | [3edf0f9](https://github.com/s1encisi/culab-copper-workbench/commit/3edf0f9c788ad5bc6a3d7d40815281e80b29074b) | OCR 与解析修订 |
| [codex/g6m-memory](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m-memory) | [a658297](https://github.com/s1encisi/culab-copper-workbench/commit/a658297e8f0f59506cf3b9d58c95ed1640f68a2d) | 权威记忆与引用 |
| [codex/g6m-rag](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m-rag) | [ccc3dc6](https://github.com/s1encisi/culab-copper-workbench/commit/ccc3dc604e85538460d42638cced440fc0262525) | 正文授权、引用与派生回答清理 |
| [codex/g6m-rerank](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6m-rerank) | [1cde3e3](https://github.com/s1encisi/culab-copper-workbench/commit/1cde3e3f59777a22da17edfe84497a9d3f2e7d53) | 可选重排与条款解析 |
| [codex/g6l-replay](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6l-replay) | [65c2f5e](https://github.com/s1encisi/culab-copper-workbench/commit/65c2f5ed5d090c364795002d625c697531d940a6) | TabPFN 数值回放精度 |
| [codex/g8-ui](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g8-ui) | [a04d31c](https://github.com/s1encisi/culab-copper-workbench/commit/a04d31c432f32046e231a62880c140b442e0c2ac) | 十工作区初始界面 |
| [codex/g8-models-checkpoint](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g8-models-checkpoint) | [f0e9283](https://github.com/s1encisi/culab-copper-workbench/commit/f0e92836be7667cbd07e7485d139fa1045433d07) | 模型生命周期界面 |

## 交付分支的合并关系

交付分支从 G8 复查节点 aeba226 建立，通过明确合并纳入 G6j 的恢复工具 9cea2f9 与 G6l 的回放修正 65c2f5e，随后增加中断状态处理、公开演示和展示材料。

这些合并保留原阶段来源，不改变 main，也不把研究候选自动批准为默认模型。

## 本地保留的历史

refs/local-only 和 refs/codex 下的原始备份、工作快照保留在本机，不属于公开阶段分支，不推送。公开同步只使用审查后的 refs/heads 引用；不使用 mirror、强推或历史重写。
