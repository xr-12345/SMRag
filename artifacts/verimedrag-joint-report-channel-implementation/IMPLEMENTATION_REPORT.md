# Phase 8B — 联合报告通道实现报告

## 0. 目标

把 Phase 8A 遗留的理论缺口补上：疾病后验、逐回答可靠度、VerifyOld 结果预测
此前来自不同概率模型，且矛盾回答被压缩为 UNKNOWN。本阶段实现一个共享的
联合报告模型 `P(D, Z, E, Y, Y')`，并接入策略。

## 1. 新增文件

| 文件 | 职责 |
| --- | --- |
| `src/powerful_medrag/joint_report_channel.py` | `VerificationType` 枚举 + `JointReportChannel`（单答边际 / 联合概率 / 模式转移 `T_v`） |
| `src/powerful_medrag/reliability_memory.py` | `ReportBundle`（首答 + 再问，不覆盖、不压缩）+ `ReliabilityMemory` |
| `src/powerful_medrag/joint_reliability_belief.py` | `JointReliabilityBeliefTracker`（单因子似然、疾病/状态/模式后验、p_mode/p_wrong、再问预测） |
| `src/powerful_medrag/joint_channel_policy.py` | `JointChannelBrierAuditPolicy`（joint_channel_brier_audit 策略）+ `run_joint_channel_dialogue` |
| `tests/test_joint_report_channel.py` | 26 项测试（概率正确性 / 联合更新 / 可靠度一致 / 决策安全 / 泄漏 / 回归） |

## 2. 修改文件

* `src/powerful_medrag/worthiness_policy.py`：`WorthinessStrategy` 增加
  `JOINT_CHANNEL_BRIER_AUDIT` 成员；`build_policy` 增加对应分支（该分支
  `NotImplementedError`，指引 standalone 入口）。**未改动任何旧策略行为。**

## 3. 关键设计

* **单特征一个因子**：`L_i(d) = Σ_z P(z|d) P(y_i, y_i'|z, v)`，被问两次的特征
  只贡献一个因子，杜绝重复计数与 UNKNOWN 压缩。
* **模式转移**：`T_v(e'|e) = rho·1[e'=e] + (1-rho)P(e')`，`rho` 可配置，满足
  边际恒等式 `Σ_e P(e)T_v(e'|e)=P(e')`（再问边际与首答相同，与 rho 无关）。
* **四量同源**：`b_t(d)`、`p_i^mode`、`p_i^wrong`、`P(y'|H_t,v)` 全部经
  `JointReportChannel` + `JointReliabilityBeliefTracker` 从同一模型推导。
* **决策复用 Phase 8A 修正逻辑**：总收益仅阻止 Stop（严格 `>`），净 Brier 比较
  带全局边际 `τ_A`；Stop 可靠度改用 `max_i p_i^mode ≤ τ_suspicious`。

## 4. 验证结果

* **单测**：新增 26 项全绿；全套 **207 项 OK**（旧 181 无回归）。
* **基线回归**：`heuristic_baseline == build_policy(HEURISTIC_VERIFY)` 逐轨迹
  byte-identical（见 `baseline_regression_check.txt`）。
* **机制 smoke**：Cases A–E **5/5 通过**（见 `SMOKE_REPORT.md` / `smoke_metrics.json`）。

## 5. 红线合规

* 不访问 test split；不跑 N=5/N=20；不覆盖旧 artifacts（新目录）；不删除旧策略
  （`resolve` 的 UNKNOWN 压缩与旧策略全部保留，test_25 锁定）；不重训 learned gate；
  不加 LLM/dense/SFT/RL；不用 val/test 真值拟合 `rho`（仿真假设，明确标注）；
  不声称 `rho` 临床真实；不修改预注册成功标准；未 git commit；未伪造结果。

## 6. 已知局限

见 `IDENTIFIABILITY_LIMITATIONS.md`：DDXPlus 无重复回答记录，`rho` 不可辨识；
仿真器独立采样（`rho=0`）与模型默认（`rho=0.5`）在再问相关性上不一致（正式
N=5/N=20 前需对齐或标注）；`p_mode`/`p_wrong` 未做校准。
