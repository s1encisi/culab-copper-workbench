# 公开发布与数据边界

目标仓库：s1encisi/culab-copper-workbench。main 保留 34933e0 基线，公开展示以交付分支为入口。

## 审查范围

- 所有拟推送阶段分支及其完整可达提交历史，而非只检查当前目录。
- 文件路径、二进制类型、凭据形状、个人绝对路径、提交消息和作者元数据。
- GitHub 分支/标签、Release 附件、Issue/PR 及评论、Actions 运行/产物、Pages、Wiki 与自定义社交预览图。
- 合成测试中的凭据形状样例使用精确路径与精确内容白名单；不放宽普通源码检查。

审查时，远端没有标签、Release/附件、Issue/PR 内容、Actions 运行/产物或 Pages，未发现已初始化的 Wiki，也未设置自定义社交预览图。拟公开分支的提交作者使用 GitHub noreply 地址。

## 公开与私密内容

| 可进入仓库 | 留在本地或私密存储 |
| --- | --- |
| 通用源码、配置模板、合成测试、公开演示、项目文档 | Excel、行级数据、训练模型、数据库、内部资料、运行截图、访问会话、密钥与详细私密证据 |

本地 refs/local-only 和 refs/codex 下的备份/工作快照不属于阶段分支，不推送。同步使用明确分支引用，保留历史，不使用 mirror、强推或历史重写。

当前资料已生成 AES-256 加密备份，随机恢复密钥受 Windows 当前账户 DPAPI 保护，并通过逐文件解密哈希校验。原件仍保留在本地；这不等于原文件或整个磁盘已加密。加密归档、恢复材料和内部清单均不上传。

## 持续检查

```powershell
git config core.hooksPath .githooks
python -B scripts/check_git_upload.py --ref HEAD
```

推送前钩子检查完整待推送历史。发布后以远端分支提交比对和匿名读取仓库入口验证结果。新内容仍需按相同边界审查，不能仅依赖 .gitignore。

GitHub 公开后，Actions 历史与日志也会公开，因此它们纳入审查范围。[GitHub 官方说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)

DPAPI 当前账户保护依赖 Windows 用户凭据和配置，恢复材料需与原账户环境妥善保存。[Microsoft 官方说明](https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.protecteddata)
