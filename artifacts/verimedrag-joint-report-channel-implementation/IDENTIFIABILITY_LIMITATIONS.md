# Phase 8B — 可辨识性局限 (Identifiability Limitations)

本文件诚实记录联合报告模型 `P(D, Z, E, Y, Y')` 在 DDXPlus / 本仓库数据条件下
**无法从数据辨识**的部分，以及因此作出的假设。这些是已知局限，不是隐藏缺陷。

## 1. 重复回答不可辨识（核心局限）

DDXPlus **没有**同一特征被同一患者回答两次的记录。因此：

* **`rho = repeat_mode_persistence` 无法从任何 split 拟合。** 联合通道的
  `T_v(e'|e) = rho·1[e'=e] + (1-rho)P(e')` 中的 `rho` 是**仿真假设**，
  默认 `0.5`，可配置，但**绝不**声称是真实患者行为或临床相关性参数。
* 所有关于「首答与再问的模式相关性」的结论（如 Case C 的「重复确认模式」）都是
  **模型内一致性的展示**，不是对真实重复问诊行为的实证。

## 2. 潜在报告模式 `E_i` 不可观测

`ReportMode`（CERTAIN/UNCERTAIN/UNKNOWN/MISREPORTED）是 latent，只能由可观测的
`CertaintyCue` 与回答值间接推断。`p_i^mode` / `p_i^wrong` 是**后验**，其绝对值
取决于冻结的 `AnswerChannel` 先验（如 MISREPORTED 先验 0.03）。因此这些量用于
**相对比较 / 门控**，不应解读为校准好的真实概率。

## 3. 模式转移与单答边际的等价性

由恒等式 `Σ_e P(e)T_v(e'|e)=P(e')`，**再问的边际模式分布与首答相同，与 rho 无关**。
这意味着：

* 仅凭再问的**边际**答案分布无法辨识 rho；
* rho 只影响首答与再问的**联合相关性**，而这一相关性在本数据中无法被观测验证。

## 4. 验证类型仅实现 REPEAT

`VerificationType` 枚举了 REPEAT / REPHRASE / DEFINITION / TEMPORAL_ANCHOR / CONTRASTIVE，
但只有 `REPEAT` 有形式化的 `T_v` 转移（其余在 `mode_transition` 中显式
`NotImplementedError`）。其余类型是**可扩展接口**，不是已实现语义。

## 5. 决策层仍是「一步前瞻」

`JointChannelBrierAuditPolicy` 沿用 Phase 8A 的一步 Brier 价值，不做多步 POMDP 求解。
「统一」指四个量来自同一模型，不指全局最优。

## 6. 与仿真器的一致性缺口

`StructuredPatientSimulator.answer()` 每次按 `hashlib` 每键每出现独立采样（等价
`rho=0`）。而联合模型默认 `rho=0.5`。因此：**联合模型的再问预测与仿真器的实际
再问采样在相关性上不一致**。这影响的是「再问价值」的**期望**估计，不影响本阶段
要验证的**机制一致性**（归一化、单因子、无 UNKNOWN 压缩、门控接线）。正式 N=5/N=20
（红线禁止本阶段运行）若要做，需先对齐仿真器与模型的 rho 或明确标注该缺口。

## 7. 未做校准

`p_mode_misreported` / `p_wrong` 未做 ECE / 可靠性图校准。它们作为门控阈值
（`suspicious_report_threshold=0.05`）的相对信号使用，绝对概率含义有限。
