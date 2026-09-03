# Phase 8C — 复问环境与推断模型对齐报告 (ALIGNMENT_REPORT)

本报告记录 **JointStructuredPatientSimulator（复问相关患者模拟器）** 与
**JointReportChannel（联合推断通道）** 之间的对齐工作：把 Phase 8B 遗留的
「环境 `rho_env=0`（每次回答独立采样） vs 推断假设 `rho_model=0.5`
（`repeat_mode_persistence=0.5`）」不匹配消除，并在 toy 模型上做了机制级验证。

## 1. 问题与目标

Phase 8B 的联合推断模型 `JointReportChannel` 用 `repeat_mode_persistence`
（记作 `rho_model`）描述复问时患者真实报告模式的相关性：

    T_v(e' | e) = rho_model * 1[e' = e] + (1 - rho_model) * P(e')

但患者模拟器（旧 `StructuredPatientSimulator`）每次 `answer()` 独立采样，
等价于 `rho_env = 0`。推断假设与数据生成环境不一致，会使复问的第二次回答在
模型中被当作「与第一次回答按 `rho_model` 相关」的证据，而模拟器实际按
`rho_env = 0` 独立生成。

本阶段目标（spec §二）：
1. 新增带相关复问的患者模拟器；
2. 保留旧模拟器及其默认行为；
3. `rho_env = 0` 时与旧模拟器严格兼容（优先 byte-identical）；
4. 验证模拟器频率与 `JointReportChannel` 理论概率一致；
5. 对齐 vs 错配的相关性比较（N=5 筛选，见 N5_SCREEN_REPORT）；
6-8. 无新 RAG / 无 N=20 / 不访问 test split / 不从数据拟合 rho。

## 2. 复问相关模拟器（`joint_patient_simulator.py`）

`JointStructuredPatientSimulator` 是 `StructuredPatientSimulator` 的 drop-in 替代：

    E_i^(1) ~ P(E)
    Y_i^(1) ~ P(Y | Z_i, E_i^(1))
    E_i^(k) ~ T_v(E_i^(k) | E_i^(k-1))          (k >= 2)
    Y_i^(k) ~ P(Y | Z_i, E_i^(k))

    T_v(e' | e) = rho_env * 1[e' = e] + (1 - rho_env) * P(e')

关键设计：
- `rho_env` 是**环境**相关性（模拟器的真实机制），`rho_model` 是**推断**假设
  （`JointReportChannel.repeat_mode_persistence`）。两者在 harness 中分开记录，
  永不混用。
- 首次回答（occurrence 0）恒为独立采样，与旧模拟器 byte-identical，与
  `rho_env` 无关。
- 复问（occurrence >= 1）在 `rho_env = 0` 时退化为 `T_v(e'|e) = P(e')`，与旧
  模拟器 byte-identical；在 `rho_env > 0` 时按相关性采样 `e'`。
- 确定性：回答是 `(case → latent_states, seed, key.token, occurrence,
  verification_type, rho_env)` 的纯函数。独立路径复用旧模拟器的
  `blake2b(seed|key|occurrence)` 派生，相关路径用
  `blake2b(seed|key|occurrence|verification_type|rho_env)`。无全局 RNG。
- 隐藏模式链 `hidden_mode_chain` / `mode_chain_for` / `answer_chain_for`
  仅用于评估，永不进入任何决策。

## 3. 与旧模拟器的字节级兼容（红线 #3）

`tests/test_joint_patient_simulator.py::test_19_old_simulator_unchanged_byte_identical`
逐字段断言：对相同的 `(seed, key, occurrence)`，`rho_env=0` 的
`JointStructuredPatientSimulator.answer()` 与旧 `StructuredPatientSimulator.answer()`
产出相同的 `(Observation.value, Observation.certainty, oracle_report_mode)`。

`baseline_regression_check.txt` 记录该回归检查通过（3/3 tests OK）。

## 4. 模拟器频率 vs 理论概率（spec 目标 4）

`run_smoke.py` 在 toy 模型上对每个真值状态 `z ∈ {present, absent}` 与
`rho ∈ {0.0, 0.5, 0.9}` 采样 N=20000 对 `(Y, Y')`，与
`JointReportChannel.joint_probability(..., repeat_mode_persistence=rho)` 比较。

结果（`probability_errors.csv`）：

    max_absolute_probability_error = 0.005191

出现在 `(true_state=present, rho=0.5, Y=present, Y'=present)` 处。0.0052 的绝对
误差在 N=20000 的蒙特卡洛误差（标准误差 ~ sqrt(p(1-p)/N) ≈ 0.0035，99.7% 置信
区间约 ±0.01）之内，确认模拟器实现了与联合通道一致的联合概率。

`simulator_mode_chain_demo` 显示模式链行为：
- `rho_env=0` 链：`[certain, certain, uncertain, certain, certain, uncertain]`（每次独立从先验采样）
- `rho_env=1` 链：`[certain, certain, certain, certain, certain, certain]`（全持久化）

## 5. 机制 smoke（spec §八 Cases A–E）

每个 case 在 toy 模型 `influenza`（真值 `fever=present`）上，用显式回答对驱动
tracker，同时用 noisy profile 采样模拟器记录真实模式链（仅评估）。

| Case | rho_env/rho_model | 回答 | 病后验 top（前→后） | 解释 |
|---|---|---|---|---|
| A 独立复问 | 0.0 / 0.0 | present→present | influenza 0.712 → 0.754 | 与旧独立通道一致 |
| B 高相关一致 | 0.9 / 0.9 | present→present | influenza 0.712 → 0.731 | 一致回答未被重复计成两份独立强证据 |
| C 高相关模型误判独立 | 0.9 / 0.0 | present→present | influenza 0.712 → 0.754 | 与 A 相同 → 过度自信 |
| D 独立环境模型误判相关 | 0.0 / 0.5 | present→present | influenza 0.712 → 0.740 | 低估第二次回答信息量 |
| E 冲突回答 | 0.9 / 0.9 | absent→present | allergic_rhinitis 0.474 → 0.339 | 两份回答都保留，联合似然更新 |

关键量化（`smoke_metrics.json`）：

- **Case B vs C vs D 的同一对回答 present→present**：
  - B（对齐 0.9/0.9）后验 0.731 —— 相关下第二次回答的边际信息被正确折减；
  - C（错配 0.9/0.0）后验 0.754 —— 模型把相关回答当独立，产生过度自信；
  - D（错配 0.0/0.5）后验 0.740 —— 模型把独立回答当相关，低估信息量。
  三者 `joint_over_independent_ratio`：B=1.233、C=1.105、D=1.176，方向符合
  `rho_model` 对第二次回答证据权重的调节。

- **Case E 冲突**：`joint_over_independent_ratio = 0.099`。在高相关（0.9）下，
  联合似然对「absent→present」冲突的联合概率远低于独立乘积，模型正确地把冲突
  识别为低概率事件，`p_mode` 0.018→0.172、`p_wrong` 0.023→0.350 同步上升，两份
  回答都被保留（无 UNKNOWN 压缩）。

- **p_mode / p_wrong 一致性**：所有 case 的 `p_mode`、`p_wrong` 与病后验同源于
  一个联合通道，`p_wrong_after = 1 - P(Z=Y | H)`，与 `state_posterior` 一致。

## 6. 对齐结论（机制层）

机制 smoke 确认：
1. 复问相关模拟器在 `rho_env=0` 时与旧模拟器字节一致；
2. 模拟器频率与 `JointReportChannel` 理论联合概率一致（误差 0.0052）；
3. 推断模型的 `rho_model` 正确调节第二次回答的证据权重（一致回答不重复计数、
   冲突回答联合更新、不产生 UNKNOWN 压缩）；
4. 错配的两种方向（环境相关而模型当独立 → 过度自信；环境独立而模型当相关 →
   低估信息量）都被演示。

对齐在机制层成立。诊断层面的对齐敏感性（`rho_model` 是否影响最终诊断、错配的
诊断代价）在 N=5 筛选（`N5_SCREEN_REPORT.md`）中验证。
