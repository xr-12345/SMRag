# Phase 8E — Prior Provenance（protocol_fixed_prior 的来源与数值）

## 结论（一句话）

`protocol_fixed_prior`（即 Phase 8D 的 `train_fixed`）**不是从训练数据学习的**，
而是根据实验协议冻结的噪声集合 `{0.2, 0.3}` 与机制份额
`(uncertain=0.40, unknown=0.30, misreported=0.30)` 人工计算的**算术平均**。
因此按 Prompt #21 §四 要求，把它从 `train_fixed` 改称为 `protocol_fixed_prior`，
不声称「由训练数据学习」。

## 来源链

1. `PatientProfile.from_noise_rate(noise)`（`src/powerful_medrag/simulator.py:40`）
   把总不可靠质量 `noise` 按三个固定份额拆成四种报告模式先验：

   | 报告模式 | from_noise_rate(n) |
   |---|---|
   | CERTAIN     | `1 - n` |
   | UNCERTAIN   | `n * 0.40` |
   | UNKNOWN     | `n * 0.30` |
   | MISREPORTED | `n * 0.30` |

   份额 `(0.40, 0.30, 0.30)` 是 `from_noise_rate` 的**默认参数**（协议冻结，非数据拟合）。

2. 实验协议噪声集合 = `{0.2, 0.3}`（Phase 8C/8D 一致）。

3. `protocol_fixed_prior()`（`src/powerful_medrag/channel.py`，新增）对两个噪声
   水平取平均，得到**单一、非 cue 条件**的先验：

   | 报告模式 | noise=0.2 | noise=0.3 | 平均（protocol_fixed） |
   |---|---|---|---|
   | CERTAIN     | 0.80 | 0.70 | **0.75** |
   | UNCERTAIN   | 0.08 | 0.12 | **0.10** |
   | UNKNOWN     | 0.06 | 0.09 | **0.075** |
   | MISREPORTED | 0.06 | 0.09 | **0.075** |

   数值与 Phase 8D `run_offline_paired_audit.py::fixed_cross_noise_prior()` 逐位一致
   （Phase 8D `CONFIG.json` 的 `cross_noise_prior` 记录同样为 `.75/.10/.075/.075`）。

## 与训练数据的关系

- **未使用** train/validation/test 的任何标签、真值、噪声类型或报告模式。
- 它只依赖「协议声明的噪声水平 + 协议冻结的机制份额」两个先验设计参数，
  是**参数化先验**，不是经验贝叶斯估计。
- Phase 8D 已证明它与 **oracle**（读环境真实噪声、特权信息）的校准几乎持平，
  但本身**不读**任何 case 级噪声——只读全局协议噪声集合。

## 三条红线的核对

| 要求（§四） | 是否满足 |
|---|---|
| 只来自 train split / 训练协议 / 预先冻结的训练噪声分布 | ✅ 只来自「预先冻结的协议噪声分布 {0.2,0.3}」 |
| 没有使用 validation/test 标签 | ✅ 未使用 |
| 没有根据 N=5 结果调参 | ✅ 数值在 Phase 8D 已冻结，Phase 8E 未改动 |

## 配置机制（不覆盖 legacy）

- `protocol_fixed_prior()` 返回先验字典。
- `protocol_fixed_channel_parameters()` 返回 `ChannelParameters(rates=默认, cue_priors=全 cue 同一先验)`。
- legacy 默认 `ChannelParameters()`（MISREPORTED=0.03）**保持不变**；train_fixed 通过
  显式构造 `AnswerChannel(protocol_fixed_channel_parameters())` 启用。
- `cue_conditioned` 本阶段只保留为 Phase 8D 的消融结论，不作为默认部署配置。

## 测试锁定

`tests/test_unknown_semantics.py::test_09_protocol_fixed_prior_source_fixed` 锁定：
`.75/.10/.075/.075` 的数值、跨 noise 平均来源、全 cue 一致、以及 legacy 默认未被覆盖。
