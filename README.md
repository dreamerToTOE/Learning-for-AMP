# Learning-for-AMP

> **第一轮 AMP 学习与实验已归档，模型已冻结。** 当前进入 R5 Motion Tracking / Motion Retargeting。AMP-G1 最终性能目标尚未达到，简历核心项目尚未验收。更新于 2026-11-26。 [当前状态](docs/AMP_G1_CURRENT_STATE.md)

## 项目目标

复现 AMP 算法并将其用于宇树 G1 的多技能运动学习。长期目标是形成有实测指标、失败证据和复现记录的人形机器人运控求职项目。

**主线为官方 AMP → G1 单技能 → 多技能与条件控制 → 第一轮归档 → 跟踪、重定向与鲁棒部署学习 → AMP-G1 第二轮 → 最终工程验收。** 首轮在仿真中复现单技能行走，并用失败回放检验奖励与技能判据。 [第一轮中期报告](reports/FINAL_REPORT.md)

1. 理解并验证 Actor / Critic / Discriminator、任务奖励与风格奖励的训练与部署边界。
2. 在一致实验预算下对比 AMP 与任务奖励 PPO 的效果，**不预设 AMP 一定更优**。
3. 将方法迁移到 **宇树 G1 仿真模型**；优先实现可重复的单技能运动。
4. 诊断多技能混合数据引发的覆盖不足：区分数据不足、奖励冲突、条件忽略、动力学不可行与真正的模式坍塌。
5. 优先验证技能条件 Actor、一次条件判别器对照；对称性轻量尝试，采样和多判别器按证据与预算选做。
6. 产出 GitHub 可复现实验、消融数据、视频、失败分析与面试解释材料。

## 当前实测结果

- R1 官方 Humanoid28 纯风格 AMP 在 40.96M 步后有 97/100 个新初态跑满约 10 s。速度任务对照记录了 PPO 爬行漏洞，两组直立行走均为 0/100。 [官方冻结评估](reports/R1_eval_freeze_40m.md) · [对照与视频](reports/R1_speed_tracking_seed42.md)
- R2 G1 4.194M 冻结基线在 100 个参考相位初态全部行走 10 s，平均前进 9.21 m、横向漂移 2.51 m。R1 与 G1 实验在 RTX 3070 8 GiB 上测得。 [G1 报告](reports/R2_g1_single_skill.md)
- R3 两个条件控制模型对 1.0/1.9 m/s 有速度响应。低速持续步行为 25/32 和 24/32。持续慢跑及双向完整切换均为 0/32。 [同预算对照与视频](reports/R3_speed_amp.md)
- 稳定慢跑、低漂移方向控制、双向完整切换、系统鲁棒性、Actor 独立推理和 Sim2Sim 留给后续阶段。现阶段只有仿真记录。 [技术债务](docs/AMP_G1_FUTURE_PLAN.md)

## 导航

| 文件 | 内容 |
|---|---|
| [当前状态](docs/AMP_G1_CURRENT_STATE.md) | 七份冻结模型、可复现行为、工程验收边界 |
| [第一轮中期报告](reports/FINAL_REPORT.md) | R0–R3 实验结果、失败经验和结论范围 |
| [问题驱动的迭代史](docs/AMP_G1_PROBLEM_HISTORY.md) | 训练与诊断背景、方法、结果、原始证据 |
| [技术债务与第二轮计划](docs/AMP_G1_FUTURE_PLAN.md) | 先决技能、最小恢复实验、验收指标 |
| [面试复习](docs/AMP_G1_INTERVIEW_GUIDE.md) | 以实际实验为例的算法和工程问答 |
| [工件与复现索引](docs/AMP_G1_ARTIFACT_INDEX.md) | 本地文件、字节数、SHA-256、远程副本及复跑命令 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | R0–R8 分阶段任务、优先级、停止条件 |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 公平对照、训练日志、统一评估、诊断与消融 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 新 Codex 窗口接手、R5 当前入口、R7 恢复顺序 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 原始论文、Isaac Lab 与候选仓库、版本审计要求 |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | 项目状态和下一动作（后续每步维护） |
| [R1 速度跟踪对照](reports/R1_speed_tracking_seed42.md) | seed42 的阶段评估、资源、运动视频与学习曲线 |
| [R1 奖励漏洞诊断与归档](reports/R1_reward_hacking_audit.md) | 接触实测、冻结模型及阶段限制 |
| [R2 G1 单技能 AMP](reports/R2_g1_single_skill.md) | 上游选择、资产与动作溯源、复现记录 |
| [R3 G1 步行与慢跑基线](reports/R3_walk_jog_baseline.md) | HOLD 数据、原始多技能 AMP 与独立评估 |
| [R3 G1 速度指令与条件判别器](reports/R3_speed_amp.md) | 两档指令、交叉初态、走跑切换与仿真视频 |

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

## 工程验收和阶段衔接

R4 归档保留 R1–R3 的阶段成果和限制。R5/R6 先完成跟踪、重定向、适应和部署学习，后续 R7 从冻结基准接手。R8 检查走跑、速度方向与漂移、切换、运动质量、鲁棒性、独立闭环和完整复现。量化门槛在 R7 开始前固定。真机另需硬件与授权。实验运行环境与 PegInsert、FR3 和机械臂 Sim2Real 各自隔离。 [项目状态](PROJECT_STATUS.md) · [R7/R8 路线](docs/ROADMAP.md#r7--amp-g1-第二轮综合完善)

## 术语提示

- **专家数据清洗** ≠ **各技能采样均衡** ≠ **覆盖了所有技能**。
- **Actor 的技能条件必须在部署时可获得**（遥控器、任务指令或上层选择器），不能偷偷使用仿真真值。
- **条件判别器**帮助评价“是否像指定技能”；**自适应采样**调节“困难技能获得多少训练机会”；两者均不保证消灭模式坍塌。
