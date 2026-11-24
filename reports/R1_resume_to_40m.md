# R1 20.48M 到 40.96M 恢复训练归档

日期：2026-11-25

本阶段从 20.48M checkpoint 恢复，保持 2048 并行环境、seed42、官方 Humanoid AMP Walk 网络、奖励和动作映射。恢复段完成 625 次更新，对应 20.48M 新环境交互步。由于 checkpoint 间隔落在 620 次更新，最后 5 次更新从恢复段的 `agent_9920.pt` 使用官方原始 Isaac Lab skrl 入口完成，最终累计为 40.96M 环境交互步。
最终 checkpoint 为 `logs/skrl/r1_amp_resume_seed42_20251125_final5/2026-11-25_02-40-09_amp_torch_resume40m_final5/checkpoints/agent_80.pt`，SHA256 为 `22f389039512e97a542403702e391b3d7efde619a78008026d849144dadd860c`。

## 恢复内容检查

20.48M checkpoint 包含 Policy、Value、Discriminator、Optimizer、state preprocessor、value preprocessor 和 AMP state preprocessor。seed42、rollouts16、amp_batch_size512 与官方配置一致。checkpoint 没有保存 motion dataset 和 reply buffer，因此这两类缓冲区在恢复边界重新初始化。恢复后的前几次更新会重新填充专家数据和策略 replay 数据。

这次运行记录为带边界的恢复训练，不能称为严格无缝连续训练。训练动力学和策略参数从 checkpoint 继续，缓冲区状态和随机采样历史没有完全连续保留。

## 资源和监控

恢复段正常退出，没有 OOM、NaN 或 Inf。外部 GPU 采样峰值为 7832 MiB，最低空闲采样为 0 MiB，GPU 利用率最高 98%。额外 AMP 监控每 31 次更新执行一次，只抽取 128 个 AMP 状态，记录 21 个边界点。监控末端的风格奖励约 2.66，策略和专家判别器评分约 0.276 和 0.731，平均存活约 8.53 秒，跌倒率约 17.8%。

最低空闲为 0 MiB 表明恢复段显存余量非常紧。训练本身完成，但后续继续训练前需要保留外部显存监控并避免增加额外前向操作。

## 结果

40.96M checkpoint 的 100 个新增 seed 中 97 个完整运行约 10 秒，3 个跌倒，完整运行率 97%，95% Wilson 区间为 91.5%–99.0%。与 20.48M 阶段新增 seed 的 89% 完整运行率相比，稳定行走趋势继续改善。

- [40.96M 冻结评估报告](R1_eval_freeze_40m.md)
- [恢复摘要](r1_resume_to_40m_summary.json)
- [checkpoint 恢复审计](r1_resume_checkpoint_audit_20m.json)
