# Learning-for-AMP

> **当前阶段为 R2 / G1 单技能 AMP** · 4.19M checkpoint 在 100 个新初态中均完成 10 秒仿真行走 · 更新于 2026-11-25

## 项目目标

围绕人形机器人运控岗位，形成可复现、可解释、可量化的 AMP 项目经历，而不是仅仅“跑通一个算法”。

**主线：官方 AMP 基线 → 宇树 G1 单技能 → 多技能覆盖与模式坍塌诊断 → 有证据后开展条件 AMP / 采样消融。**

1. 理解并验证 Actor / Critic / Discriminator、任务奖励与风格奖励的训练与部署边界。
2. 在一致实验预算下对比 AMP 与任务奖励 PPO 的效果，**不预设 AMP 一定更优**。
3. 将方法迁移到 **宇树 G1 仿真模型**；优先实现可重复的单技能运动。
4. 诊断多技能混合数据引发的覆盖不足：区分数据不足、奖励冲突、条件忽略、动力学不可行与真正的模式坍塌。
5. 仅在基线成立后，评估技能条件 Actor、条件/多判别器、均衡采样及自适应采样。
6. 产出 GitHub 可复现实验、消融数据、视频、失败分析与面试解释材料。

## 当前边界与优先级

- **本仓库只负责 AMP 仿真实验**，不修改既有 PegInsert、FR3 双臂及机械臂 Sim2Real 仓库。
- G1 基线优先复用成熟开源工程，先核查资产、步态数据和权重，再做独立环境的小规模验证。
- 当前已知本地设备：RTX 4070 12GB（主要）／RTX 3070 8GB（备用），Ubuntu 22.04，Isaac Sim 4.5 / Isaac Lab 2.1.1 历史环境。以上为用户既有环境信息，**接手时必须重新实测确认**。
- 如 G1 上游项目需要其他 Isaac Lab / RSL-RL 版本，请使用独立环境，不能直接升级/污染旧环境。
- 本项目仿真优先；**不以真实宇树机器人部署为验收条件**，除非后续新增硬件、授权和独立安全门禁。
- 这是论文学习与工程复现项目；不能把社区已有的条件 AMP / Multi-AMP 方法描述为独创算法。

## 导航

| 文件 | 内容 |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | R0–R4 分阶段任务、依赖、交付物、验收门 |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 公平对照、训练日志、统一评估、诊断与消融 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 新 Codex 窗口接手指令、首轮执行范围、禁止事项 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 原始论文、Isaac Lab 与候选仓库、版本审计要求 |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | 项目状态和下一动作（后续每步维护） |
| [R1 速度跟踪对照](reports/R1_speed_tracking_seed42.md) | seed42 的阶段评估、资源、运动视频与学习曲线 |
| [R1 奖励漏洞诊断与归档](reports/R1_reward_hacking_audit.md) | 接触实测、冻结模型及阶段限制 |
| [R2 G1 单技能 AMP](reports/R2_g1_single_skill.md) | 上游选择、资产与动作溯源、复现记录 |

## 规划中的目录结构（不代表已有实现）

```text
Learning-for-AMP/
├── README.md
├── PROJECT_STATUS.md
├── docs/
│   ├── ROADMAP.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── HANDOFF.md
│   └── REFERENCES.md
├── configs/             # R0 确认技术栈后创建
├── scripts/             # 小规模训练/评估封装
├── src/                 # 仅存必要的原创适配/改动
├── tests/
├── reports/
└── artifacts/           # 大型日志/模型/视频不直接提交 Git
```

## 成功标准

成功不是“训练曲线上升”。至少需要：
- 可复现的运行命令与锁定版本；
- 已固定的评估协议、任务指令和种子；
- AMP vs PPO 对照；
- 各技能表现与技能覆盖率，包含失败案例；
- 有针对性的消融实验（若进入 R3）；
- 仿真结果与真实机器人结果严格分开陈述。

当前实验记录与后续阶段安排见 [项目状态](PROJECT_STATUS.md)。

## 术语提示

- **专家数据清洗** ≠ **各技能采样均衡** ≠ **覆盖了所有技能**。
- **Actor 的技能条件必须在部署时可获得**（遥控器、任务指令或上层选择器），不能偷偷使用仿真真值。
- **条件判别器**帮助评价“是否像指定技能”；**自适应采样**调节“困难技能获得多少训练机会”；两者均不保证消灭模式坍塌。
