# Prompt #18 — Failure Log

本文件记录实现与 N=5 筛选过程中遇到的失败、根因与处置。运行时失败
（traceback / OOM / 收敛异常）在 full run 完成后补充。

## 1. 启发式基线回归失败（已修复）

**症状**：N=5 harness 中 `heuristic_baseline` 与冻结的 drop-in 启发式
（Phase 15）逐轨迹对比出现 **42 处不匹配**。

**根因**：`_make_policy(HEURISTIC_BASELINE)` 分支调用
`build_policy(WorthinessStrategy.HEURISTIC_VERIFY, config=config)` 时**未传
`retriever=retriever`**，导致 heuristic 策略 `retriever=None`，DYNAMIC_RAG
检索永不执行，进而改变 VerifyOld 排序 / joint gate。

**修复**：该分支改为 `build_policy(..., retriever=retriever)`。

**验证**：修复后与 drop-in 启发式 **0 处不匹配（byte-identical）**。

## 2. `action_seq` 列缺失（对比口径，非失败）

Phase 5 的 `case_outcomes.csv` 无 `action_seq` 列（None）。回归对比仅比较
`predicted_diagnosis / correct_top1 / brier_score / new_questions /
verification_questions / unnecessary_verifications / resolved_wrong_reports`
七列，排除 `action_seq`。

## 3. corrected vs corrected_dynamic_rag 轨迹恒等（预期现象，非失败）

RAG 检索信号被红线锁死、**不进入 Brier 价值**，故 `unified_brier_audit_corrected`
与 `unified_brier_audit_corrected_dynamic_rag` 产生 byte-identical 轨迹。这是
「RAG adds nothing」的诚实结论，不是实现错误（`test_14` 亦断言此不变量）。

## 4. 运行时失败（full run）

* 第一次 full run（16 worker）在上一会话进程退出时被 teardown **连同杀掉**（无
  traceback，日志被清空，`case_outcomes.csv` 仍是 smoke 的 245 行旧输出）。已
  **重新启动**，第二次 full run 完整跑完。
* 第二次 full run：**exit 0**，7350 轨迹 / 25318 turn 行 / 3672 stop-audit 行，
  无 traceback、无 OOM（峰值 1966MB < 分配）、无收敛异常。
* smoke 阶段无失败。
