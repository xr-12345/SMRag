# Phase 8B — 失败日志 (Failure Log)

本文件记录实现过程中遇到的失败、根因与处置。最终状态：**无未解决失败**。

## 1. `ReliabilityMemory` 缺 `values()`（已修复）

**症状**：`JointChannelBrierAuditPolicy._verification_count()` 调用
`self.tracker.memory.values()` 时抛 `AttributeError: 'ReliabilityMemory' object has
no attribute 'values'`。

**根因**：`ReliabilityMemory` 实现了 `items()` / `keys()` / `__iter__` / `__len__`，
但漏了对称的 `values()`。

**修复**：给 `ReliabilityMemory` 增加 `values()` 方法（与 `keys()` 对称）。

**验证**：修复后对话端到端跑通（seed 0/1/2 各正常终止）。

## 2. 决策测试 test_19 / test_20 返回 STOP 而非 NEW（已修复，测试缺陷）

**症状**：`test_19_high_p_mode_blocks_confident_stop` 与
`test_20_gross_gain_blocks_stop_but_does_not_force_verify` 断言 `ActionKind.NEW`，
实际返回 `STOP`。

**根因**：测试 patch 的是 `policy.rank_actions`，但 `choose_action` 的
`best_new / best_verify` 来自 `self._last_best_new / _last_best_verify`，而这两个
属性是在 `rank_actions` 内部由 `_asknew_brier_values` / `_verifyold_brier_values`
设置的。patch 掉 `rank_actions` 后，`_last_best_new / _last_best_verify` 保持
`None`，导致 fallback 走到「无正效用」的 uncertainty STOP。

**修复**：测试改为 patch `_asknew_brier_values` / `_verifyold_brier_values`
（即 `_last_best_new / _last_best_verify` 的真实来源），与 Phase 8A 修正套件的
`_StubChooser` 口径一致。这是**测试写法缺陷**，非实现缺陷。

**验证**：两测试转绿；全套 207 项 OK。

## 3. 无运行时失败 / 无 OOM / 无收敛异常

* 本阶段无正式 N=5/N=20（红线禁止），故无长运行、无 OOM、无收敛问题。
* smoke（toy）与单测均在秒级完成，无 traceback。

## 4. 未发生但需警惕的缺口（记录，非失败）

* 仿真器 `answer()` 独立采样（`rho=0`）与联合模型默认 `rho=0.5` 的再问相关性
  不一致——已记入 `IDENTIFIABILITY_LIMITATIONS.md` §6，正式 N=5/N=20 前需处理。
