# 升级分支导航

每个阶段都有独立分支，后一个阶段继承前一个阶段的源码。查看某阶段的全部文件，打开分支；只看该阶段新增或修改的内容，打开表中的“本阶段差异”。

## 主线与当前工作

- [main](https://github.com/s1encisi/culab-copper-workbench/tree/main)：升级前基线，固定为 `34933e0`。
- [codex/system-upgrade](https://github.com/s1encisi/culab-copper-workbench/tree/codex/system-upgrade)：升级汇总入口，本次整理后与 `codex/g6g` 指向同一最新提交。
- [codex/g6g](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6g)：当前已验证阶段，已完成五折、五种子比较及工件/接口回放；完整升级继续按后续阶段推进。
- G7、G8 尚未开始；开始实施时再从其所依赖的阶段建立分支。

## 阶段顺序与差异

“已归档”表示这个源码节点已保存，不代表整轮升级全部验收完成。G1 至 G5a 保留原有提交；G5b 至 G6f 根据当时保存的文件哈希恢复，130 个阶段文件版本全部一致。G6g 的源代码和验收已保存；后续 BART、符号与神经预测方法、共享校准和知识能力继续分阶段实现。

| 阶段 | 分支 | 继承阶段 | 改动主题 | 阶段源码提交 | 本阶段差异 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | [`codex/g1`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g1) | main | 数据、时间边界与事件证据 | [`d2ea1eb`](https://github.com/s1encisi/culab-copper-workbench/commit/d2ea1eb873e989b41feb5b70c46cf60965dd1b7d) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/main...codex/g1) | 已归档 |
| G2a | [`codex/g2a`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g2a) | G1 | 模型注册与固定协议比较 | [`24949dd`](https://github.com/s1encisi/culab-copper-workbench/commit/24949ddfa2b497e4287753823d798833e175f58d) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g1...codex/g2a) | 已归档 |
| G2b | [`codex/g2b`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g2b) | G2a | 优化器注册与统一评价 | [`0a86817`](https://github.com/s1encisi/culab-copper-workbench/commit/0a868173f88566cefd4116823a44ec9d4fc63ef6) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g2a...codex/g2b) | 已归档 |
| G3 | [`codex/g3`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g3) | G2b | 研究对话、权限与任务恢复 | [`80b85e3`](https://github.com/s1encisi/culab-copper-workbench/commit/80b85e3f482438e8017793fdd65ba4b317099ada) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g2b...codex/g3) | 已归档 |
| G4 | [`codex/g4`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g4) | G3 | Mock 控制、精确审批与回读 | [`e4c5ef5`](https://github.com/s1encisi/culab-copper-workbench/commit/e4c5ef54c4cd443835590e835479822fcec3d947) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g3...codex/g4) | 已归档 |
| G5a | [`codex/g5a`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5a) | G4 | 成熟标签评价与影子路由 | [`27923fd`](https://github.com/s1encisi/culab-copper-workbench/commit/27923fdd31a46a161038ca5fa100c4a36b26c748) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g4...codex/g5a) | 已归档 |
| G5b | [`codex/g5b`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5b) | G5a | 嵌套 OOF 集成与时间块 Bagging | [`37f47c2`](https://github.com/s1encisi/culab-copper-workbench/commit/37f47c22dbcba68b40d94169a2ce13b1a5a7a6f0) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g5a...codex/g5b) | 已归档 |
| G5c | [`codex/g5c`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5c) | G5b | 模型工件登记、发布与回退 | [`a0ba976`](https://github.com/s1encisi/culab-copper-workbench/commit/a0ba976e6f2f6b84f160ec0958210935f3a46cef) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g5b...codex/g5c) | 已归档 |
| G5d | [`codex/g5d`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g5d) | G5c | 优化器组合与配对比较 | [`1c46bf1`](https://github.com/s1encisi/culab-copper-workbench/commit/1c46bf1083f401c9455d4b3d1ed06c1c2ad544b8) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g5c...codex/g5d) | 已归档 |
| G6a | [`codex/g6a`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6a) | G5d | 经典预测方法扩展 | [`19bcef6`](https://github.com/s1encisi/culab-copper-workbench/commit/19bcef6d7ba497bfe396a1e408129d4f90bc886d) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g5d...codex/g6a) | 已归档 |
| G6b | [`codex/g6b`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6b) | G6a | 统计与事件序列方法 | [`6149b91`](https://github.com/s1encisi/culab-copper-workbench/commit/6149b910993013aa3864a11942482c925b143582) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6a...codex/g6b) | 已归档 |
| G6c | [`codex/g6c`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6c) | G6b | pymoo 优化方法扩展 | [`00f231e`](https://github.com/s1encisi/culab-copper-workbench/commit/00f231ed0975551c9c3de2918910129eb3d694a1) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6b...codex/g6c) | 已归档 |
| G6d | [`codex/g6d`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6d) | G6c | Platypus 优化方法扩展 | [`79f75c7`](https://github.com/s1encisi/culab-copper-workbench/commit/79f75c7946f26de8646f8187c3dd626627bd762f) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6c...codex/g6d) | 已归档 |
| G6e | [`codex/g6e`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6e) | G6d | HypE 与标量化前沿 | [`0d4d3e5`](https://github.com/s1encisi/culab-copper-workbench/commit/0d4d3e5a9d09a582d978055fe887bfbb4ce70e84) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6d...codex/g6e) | 已归档 |
| G6f | [`codex/g6f`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6f) | G6e | 贝叶斯多目标优化与独立复核 | [`941e658`](https://github.com/s1encisi/culab-copper-workbench/commit/941e6585e6e2ad5ac7c3e3ed197f3be7906a648f) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6e...codex/g6f) | 已归档 |
| G6g | [`codex/g6g`](https://github.com/s1encisi/culab-copper-workbench/tree/codex/g6g) | G6f | CatBoost、NGBoost、EBM、Cubist 与概率评分 | [`31d6cab`](https://github.com/s1encisi/culab-copper-workbench/commit/31d6cab4cf0db4aab7309cba94d350a5f3bc88af) | [查看](https://github.com/s1encisi/culab-copper-workbench/compare/codex/g6f...codex/g6g) | 已归档 |

## 后续使用

1. 每开始一个新阶段，先从依赖的上一阶段建立分支；该阶段的修改只提交到自己的分支。
2. 已归档阶段保留为稳定节点。完成下一阶段后，更新此表，并将升级汇总分支快进到新节点。
3. 当前链上的阶段共享模型接口、工件格式或评价模块，因此采用顺序继承。若将来出现互不依赖的工作，可从共同基点分出，再按实际依赖集成。
4. 查看累计升级用 `main...codex/system-upgrade`；查看单阶段用表中相邻分支的比较。后续修正通过新提交记录，保留已经共享的历史。

## 本次整理的验证范围

- 拟上传历史的 17 个源码树、345 个文本文件版本通过上传检查；279 个 Python/TOML 文件版本通过语法解析。新增导航文档会再纳入推送前检查。
- G6g 正式研究与工件验收已完成：120 个拟合工件、800 个回放点、8 组生命周期案例通过验证；完整回归 242 项通过。
- G3 的真实付费模型调用验收仍待单独授权；分支存档不代表该项已执行。
- Excel、行级数据、模型、密钥、内部设计文档与原始验收证据继续保留在本地被忽略的目录中。
