# Project Status

- **状态：PLANNING**
- **日期：2026-11-22**
- **当前阶段：R0 — 环境与上游审计（未开始）**
- **真实训练：0 次（仓库初始化规划时）**
- **真实评估：0 次（仓库初始化规划时）**
- **支持结论：仅规划，不包含任何 AMP 或 G1 实验性能声明。**

## 已完成

- 建立项目目标与任务边界。
- 写入 R0–R4 路线图、实验协议、文献索引与移交指令。

## 下一步

1. 新 Codex 会话执行 `docs/HANDOFF.md` 中的 R0（仅环境/上游审计）。
2. 在 `reports/R0_environment_audit.md` 记录真实检测与结论。
3. 决定 R1 采用的 AMP 实现、版本隔离方式及 smoke 命令。
4. 更新本状态文件，并提交证据。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | NOT_STARTED | — |
| R1 官方 AMP baseline | NOT_STARTED | — |
| R2 G1 单技能 | NOT_STARTED | — |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## 每轮必须补充的状态字段

```text
date:
stage:
status: PASS | PARTIAL | BLOCKED
code_sha:
environment:
commands_run:
evidence_paths:
key_metrics:
blockers:
next_action:
```
