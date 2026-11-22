# 实验与评价协议（预注册草案）

状态：PLANNED。具体阈值需在 R0/R1 的实际平台确认后冻结，避免看过结果再改标准。

## 1. 基本记录

每次独立训练生成唯一 `run_id`；保存：
- 源代码 Git SHA、diff 是否干净、上游仓库 URL+commit、Python/Isaac Sim/Isaac Lab/skrl/RSL-RL/PyTorch 版本；
- 机器人模型/资产版本、专家数据来源与 hash、训练/评估数据分割；
- seed、环境数量、dt、控制频率、episode 长度、训练总环境步数、奖励分量权重；
- Actor/Critic/Discriminator 网络与输入、命令分布、轨迹采样分布；
- GPU 型号/峰值显存、耗时、退出原因、checkpoint hash。

## 2. 评估的公平性

- 固定评估 seed、初始状态分布、目标指令分布、扰动分布和评估预算。
- 不得用训练 seed 的单次最好结果代替跨 seed 结果。
- 普通 PPO vs AMP：尽量保持**环境、机器人、任务奖励、训练环境交互步数、观测/动作、随机化、评估条件**一致；AMP 的 D 计算成本另行报告。
- 条件 AMP vs 原始 AMP：明确条件输入增加、标签可用性和训练数据配对改变所带来的额外信息。
- 强化学习不保证某技能完全可控；为任务指令和技能条件制定**不同指标**。
- 每个主要组至少输出独立评估 100 episodes/seed 的建议目标；核心结论争取 3 seeds。资源不足时明确写“探索性结果”，不作显著性断言。

## 3. 指标定义

1. **任务完成/稳定率**：事先写明成功判据、episode 终止与超时处理。
2. **速度跟踪误差**：`MAE(vx)`、`MAE(vy)`、`MAE(wz)`，分别报告指令速度范围。
3. **姿态/动作自然性**：采用不依赖训练判别器的相似度，如参考动作运动学误差、脚步时序差异、速度分布等；训练用判别器打分只作为辅助诊断。
4. **接触质量**：足端滑移距离/速度、无意接触/穿透、支撑切换、跌倒率。
5. **控制平滑性**：相邻动作变化 `||a_t-a_(t-1)||`、关节速度/加速度峰值，若模型可用则记录力矩峰值。
6. **技能覆盖率**：定义每技能是否达标 `I_i=1[success_rate_i>=tau_i AND stability_i>=sigma_i]`；覆盖率 `sum(I_i)/N`；`tau_i,sigma_i` 在观察测试结果前冻结。
7. **切换指标**：切换命令后达到新目标的时间、跌倒与滑移率、切换窗口内动作跃变。
8. **效率**：训练 wall time、GPU-hour、env steps/sec、峰值显存；跨不同 GPU 谨慎比较。

## 4. 模式坍塌诊断树

发现某技能成功率低时，按顺序排查：
- 任务定义/指令：该技能是否明确且正确传给 Actor？部署是否有相同信息？
- 数据覆盖：是否有足量、高质量且机器人可执行的对应专家数据？
- 采样：专家数据与 rollout 的技能比例、各技能判别器置信度/损失是否失衡？
- 奖励：任务与风格奖励是否竞争，是否奖励了“另一技能但看起来自然”的动作？
- 策略行为：是“始终输出同一技能”还是“试图执行但跌倒/受力不足”？
- 物理条件：模型、关节驱动、足底摩擦、接触求解、动作/观测延迟是否导致失败？

只有证据表明输出运动模式向少数模式收缩，才使用 **mode collapse** 描述；其他失败应如实分类。

## 5. 实验命名和报表

建议记录字段：

```text
run_id, stage, variant, seed, env_steps, eval_episodes,
skill, command_vx, command_vy, command_wz,
success_rate, fall_rate, vx_mae, vy_mae, wz_mae,
foot_slip, smoothness, style_metric, wall_seconds,
peak_gpu_mem_mb, code_sha, checkpoint_path, notes
```

建议产物：
- `reports/R1_amp_baseline.md`：单技能 AMP vs PPO。
- `reports/R2_g1_single_skill.md`：G1 单技能可行性。
- `reports/R3_multiskill_ablation.md`：多技能诊断与对照。
- `reports/failures/`：每个典型失败关联配置、日志、视频位置。

不要上传未经授权的人体动作数据、宇树资产、第三方模型权重；大型 checkpoint/视频放在外部工件存储并记录 SHA/URL。
