# 汇报 / 交接 prompt —— SafeMedRAG 结构化研究底座恢复与验证

> 把本文原样交给一个新的 Claude 会话（或同事），即可继续本阶段未竟工作。
> 这是「恢复并验证结构化研究底座」阶段的交接 prompt，不是研究报告本身。

## 0. 一句话现状

结构化底座（Bayesian 主动问诊 + 两层回答通道 + new/verify/stop 联合策略）代码与测试已就绪、
`oracle_verify` 验证上界已实现并通过 48 项测试；**唯一阻塞是 DDXPlus 原始患者数据尚未落盘**，
数据一到即可 fit 模型并跑 validation 六方法。

## 1. 项目与硬约束（不可违反）

项目：SafeMedRAG / PowerfulMedRAG —— 结构化贝叶斯医学问诊研究（DDXPlus），另有 MedExAgent NL-LLM 基线（本阶段不涉及）。

本阶段四条红线：
1. **不加入自然语言 LLM**；
2. **不加入正式 RAG**（不做文档检索）；
3. **不修改已有实验结果**（`artifacts/` 下已提交结果保持原样）；
4. **不伪造缺失数据**（DDXPlus 数据拿不到就如实报告，绝不编造）。

同时：**保留所有旧策略作为 baseline**。

## 2. 当前快照

- **HEAD**：`e661311`（`Add reliability-aware gating, joint decision policy, and gate learning`）。
- **工作区**：main 分支干净，仅新增未跟踪目录 `docs/medexagent-baseline/`。
- **测试**：`PYTHONPATH=src python -m unittest discover -s tests -v` → **48 tests OK**
  （本阶段新增 3 个 oracle 测试前是 45）。
- **环境**：
  - 解释器：`/Users/xr-12345/miniforge3/bin/python3`（Python 3.13.13，numpy 2.4.6）。
  - 注意：`/usr/bin/python3` 是 3.9.6，太旧，pyproject 要求 ≥3.10，**不要用**。
  - 平台：Apple M5 Pro arm64，无 CUDA。
  - 网络：huggingface.co 与 figshare.com 均被墙；github.com / pypi.org 可达。
- **数据现状**（`data/ddxplus/`）：
  - ✅ 在位：`release_conditions.json`、`release_evidences.json`（元数据）。
  - ❌ 缺失：`release_train_patients.zip`（约 134 MB）、`release_validate_patients.zip`（约 18 MB）、
    `release_test_patients.zip`（约 18 MB）、`data/ddxplus/model-full.json`。
  - 结论：DDXPlus 患者数据仅托管在 figshare（API 返回 403，全盘 find 无结果），本机无法自行获取，
    需同事按 DDXPlus 授权/分发方式放入 `data/ddxplus/`。**不要伪造这些文件。**

## 3. 本阶段已完成

1. **实现了 `oracle_verify` 策略**（oracle VerifyOld 上界，纯代码、不依赖数据）：
   - `src/powerful_medrag/decision.py`：
     - `ReliabilityAwareActionPolicy.__init__` 增加 `oracle_verification: bool = False`；
     - `rank_actions` / `choose_action` 增加 `oracle_states` 入参；
     - 新增模块级 `_oracle_wrong_reports()`：仅对「真值与报告值不一致」的报告给出 VERIFY（utility=1.0）；
     - `run_reliability_aware_dialogue` 在 oracle 模式下用真实 latent state 构造澄清 Observation。
   - `src/powerful_medrag/reliability_experiment.py`：`allowed_strategies` 增加 `"oracle_verify"`，
     派发分支改为 `if strategy in ("joint_new_verify_stop", "oracle_verify"):` 并传 `oracle_verification=...`。
   - `src/powerful_medrag/cli.py`：`benchmark-reliability-ddxplus --strategies` 的 default 与 choices 增加 `"oracle_verify"`。
   - 关键点：oracle 仅读取 `patient.latent_states`（模拟器特权状态），**绝不能作为可部署方法列出**。
2. **新增 3 个单元测试**（`tests/test_decision.py`）：`_oracle_wrong_reports` 判定、oracle 对错误报告给出 VERIFY、
   oracle 对话可解析错误报告。
3. **端到端 smoke**（toy 数据，噪声 0.2，seed 2026）确认派发链路可跑：

   | strategy | top1 | new | verify | total | Brier |
   |---|---|---:|---:|---:|---:|
   | full_two_layer | 0.767 | 3.93 | 0.00 | 3.93 | 0.275 |
   | joint_new_verify_stop | 0.800 | 4.57 | 0.37 | 4.93 | 0.227 |
   | **oracle_verify** | **0.900** | 4.50 | 0.20 | 4.70 | **0.133** |

   toy 上界符合预期：oracle 只验证真错报告（verify 0.20 vs 0.37），精度 +0.10、Brier 减半、问题数反而更低。

## 4. 下一步待办（按顺序，数据到位后执行）

**步骤 0 —— 等待数据落盘**：确认 `data/ddxplus/` 出现三个 `release_*_patients.zip`。
（本阶段已用后台监视任务等待；数据一到自动继续。）

**步骤 1 —— fit 模型**（用训练集，仅此一次）：
```bash
PYTHONPATH=src /Users/xr-12345/miniforge3/bin/python3 -m powerful_medrag.cli fit-ddxplus \
  --patients data/ddxplus/release_train_patients.zip \
  --evidences data/ddxplus/release_evidences.json \
  --output data/ddxplus/model-full.json
```

**步骤 2 —— 运行 validation 六方法**（详见第 5 节，两条 runner）。

**步骤 3 —— 回答三问 + 产出报告**（详见第 6、7、8 节）。

## 5. 六方法的 runner 映射（关键：不在同一个 runner 里）

用户要求的六方法分属**两个实验入口**，这是本阶段必须处理好的整合点：

| # | 方法 | runner / CLI | 说明 |
|---|---|---|---|
| 1 | `ordinary_eig_reliable` | `benchmark-reliability-ddxplus` | 普通 EIG 基线 |
| 2 | `full_two_layer` | `benchmark-reliability-ddxplus` | 两层回答通道，无 gate |
| 3 | `adaptive_history` | `benchmark-reliability-ddxplus` | 历史可靠性 gate |
| 4 | `retro_utility_u050_b1` | `clarify-ddxplus --variants retro_utility_u050_b1` | **PaMis 风格回溯澄清，另一 runner** |
| 5 | `joint_new_verify_stop` | `benchmark-reliability-ddxplus` | 联合 new/verify/stop 策略 |
| 6 | `oracle_verify` | `benchmark-reliability-ddxplus` | oracle VerifyOld 上界（本阶段新增） |

**统一预算的整合点**（用户要求「所有方法用相同病例、回答、总原子问题预算」）：
- 5 个 reliability 策略走 `run_reliability_experiment`，已天然共享
  `sample_balanced_ddxplus_cases(seed=2026)` + `PatientProfile` 配对种子，且 `total_atomic_questions =
  new + verify` 已内建。
- `retro_utility_u050_b1` 走 `run_clarification_curve_experiment`，采样与计账口径不同。
  **对齐要求**：固定相同病例列表（`--sample-seed 2026`）、相同噪声种子（`--seed/--seeds 2026 2027 2028`）、
  相同总原子问题预算（`--max-questions 15` / `--max-total-turns 15`），并在最终报告里把两套结果合并到
  **同一张 risk–questions Pareto 表**，明确写出每个方法的原子问题计数口径。

冻结配置（六方法一致，来自 `docs/EXPERIMENT_PROTOCOL.md` 与 `configs/preregistered_success_criteria.json`）：
`sample-seed 2026`；`seeds 2026 2027 2028`；`cases-per-disease 20`；`noise-rates 0 0.1 0.2 0.3`；
`max-total-turns 15`；`posterior 0.85`；`margin 0.70`；`minimum-action-utility 0.08`；
`verification-cost 0.03`；`minimum-history-cues 1`。

可靠性六策略命令（validation）：
```bash
PYTHONPATH=src /Users/xr-12345/miniforge3/bin/python3 -m powerful_medrag.cli benchmark-reliability-ddxplus \
  --model data/ddxplus/model-full.json \
  --patients data/ddxplus/release_validate_patients.zip \
  --evidences data/ddxplus/release_evidences.json \
  --output-dir artifacts/reliability-policy-validation-n20 \
  --cases-per-disease 20 --seeds 2026 2027 2028 \
  --noise-rates 0 0.1 0.2 0.3 --max-total-turns 15 --minimum-action-utility 0.08 \
  --strategies ordinary_eig_reliable full_two_layer adaptive_history joint_new_verify_stop oracle_verify
```

回溯澄清方法命令（validation，`retro_utility_u050_b1`）：
```bash
PYTHONPATH=src /Users/xr-12345/miniforge3/bin/python3 -m powerful_medrag.cli clarify-ddxplus \
  --model data/ddxplus/model-full.json \
  --patients data/ddxplus/release_validate_patients.zip \
  --evidences data/ddxplus/release_evidences.json \
  --output-dir artifacts/retrospective-u050-validation-curve-n20 \
  --cases-per-disease 20 --max-questions 15 \
  --noise-rates 0 0.1 0.2 0.3 --thresholds 0.85 \
  --variants retro_utility_u050_b1 --seed 2026 --sample-seed 2026
```

## 6. 要回答的三个研究问题

1. **VerifyOld 是否优于 matched-budget AskNew**：在相同总原子问题预算下，把一次验证的花费与一次新问题的
   信息增益对齐比较——验证旧回答是否比「多问一个新问题」更划算。
2. **oracle VerifyOld 是否存在明确收益**：oracle_verify 是否能在中高噪声（0.2、0.3）下，相对
   `full_two_layer` / `joint_new_verify_stop` 明显改善「风险—问题数」Pareto 前沿。
3. **joint policy 是否减少临床错误而不是仅增加问题**：`joint_new_verify_stop` 相比 `full_two_layer`，
   top1/Brier 是否改善，且不是靠简单堆问题数（要看 `unnecessary_verification_rate`、`premature_stop_rate`、
   `uncertain_output_rate`）。

## 7. 验收标准（冻结）与停止条件

验收标准（`configs/preregistered_success_criteria.json`，**测试评估后不得更改**）：
- `maximum_top1_accuracy_drop_absolute = 0.005`
- `maximum_relative_brier_degradation = 0.05`
- `minimum_average_total_questions_saved = 1.0`

**停止条件**：如果 **oracle VerifyOld 也无法在中高噪声下改善「风险—问题数」Pareto 前沿**，
则停止扩展 VeriMedRAG，并报告「核心假设暂不成立」。即：连 oracle 上界都不成立，则真实可部署的
verify 策略没有改进空间，应停止投入。

## 8. 产物清单（必须保存）

在 `artifacts/reliability-policy-validation-n20/`（及回溯澄清对应目录）下保存：
1. **病例级结果**：每个 (strategy, noise_rate, seed, case) 一行的 outcomes CSV（Top-1/Top-3、new、
   verify、total atomic、interaction turns、Brier、ECE、unnecessary verification、premature stop、
   uncertain output、stop_reason）。
2. **配置**：本次运行用的全部超参与冻结种子，落成 `CONFIG.md` 或 JSON。
3. **依赖**：`pip freeze` 或 `python -m pip list` 快照 + 解释器版本。
4. **失败日志**：任何命令的非零退出、traceback、数据缺失/校验失败，单独落成 `FAILURE_LOG.md`。
5. **合并报告**：三问的明确结论 + 六方法的统一 risk–questions Pareto 表 + 是否触发停止条件。

## 9. 交接时最需警惕的三点

1. `retro_utility_u050_b1` 与其余五法**不在同一 runner**，统一预算口径是易错点，不要各跑各的就下结论。
2. `oracle_verify` 读的是 `patient.latent_states` 特权状态，**只能作为上界**，绝不能写进「可部署方法」结论。
3. 数据拿不到时**如实报告缺失**，不要用 toy 数据冒充 DDXPlus、不要手写任何患者记录来「补数」。
