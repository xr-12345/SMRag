# Phase 8D — FAILURE_LOG

记录本阶段遇到并已修复的问题，供复现与审计。

## 1. 复现环境：system python3 测试失败（已修复）

**现象**：`python3 -m unittest`（macOS 系统 Python 3.9）8 个 error：
- 2 × `TypeError: unsupported operand type(s) for |`（`ablation.py:96` 的 `X | None` 语法，
  需 Python ≥3.10）；
- 6 × `joblib.load` 失败（numpy `PCG64` 不是已知 BitGenerator，numpy 版本不匹配）。

**修复**：改用 `/Users/xr-12345/miniforge3/bin/python3`（Python 3.13.13），229 tests 全绿。
系统 python3 不可用于本仓库。

## 2. 后台命令退出码被 `| tail` 吞掉（已避免）

首次测试命令用 `| tail -20` 把非零退出码掩盖成 0，误判"通过"。后续改用不接 tail
的重跑看真实错误。教训：测试/运行命令不要用 `| tail` 掩盖退出码。

## 3. 复现口径：曾误排除 UNKNOWN（关键方法学修正）

**问题**：Phase 8C 的 p_wrong ECE 0.1555 是在**全 first-answer 行（含 UNKNOWN）**上、
标签 `true_wrong = wrong_report AND value != UNKNOWN` 算的。初版复现用了
`first & (value != UNKNOWN)` 掩码，得到 ECE 0.030，与 0.156 差 5×。

**修正**：复现掩码 = `first`（含 UNKNOWN），标签对 UNKNOWN 行取 0 而 p_wrong 返 1.0。
修正后 ECE = **0.155523**，与 Phase 8C 逐位一致。

**根因提示**：UNKNOWN 混入是 0.156 的主成分，**排除 UNKNOWN 后复现必错**——这本身就
是诊断结论的关键证据。

## 4. value=unknown 分层标签错误（关键方法学修正）

**问题**：初版 `value=unknown` 分层用 `wrong_report`（=1[Z≠Y]，对 UNKNOWN 值在 binary
真值下取 1）作标签，得到 ECE 0.0，误导为"未知层已完美校准"。

**修正**：改用 `true_wrong`（= wrong_report AND value≠UNKNOWN，UNKNOWN 行取 0）作标签。
修正后 UNKNOWN 层 ECE = 1.0（四配置一致），正确反映 p_wrong=1.0 vs 标签=0 的混入。

## 5. 子集命名错误（已修正）

初版 `all_first_nonunknown` 子集实际用了 `first`（含 UNKNOWN），命名与语义不符。
已拆分为 `all_first_incl_unknown`（=8C 口径）与 `all_first_nonunknown`（排除 UNKNOWN）
两个独立子集。

## 6. 绘图函数残留死代码（已清理）

初版 `_plot_reliability` 的 `curve` 内有一处 `colors[...] if False else ...` 死代码，
已改为显式传 `color` 参数。

---

## 无遗留问题

以上均已修复；`analyze_paired_audit.py` 输出与 Phase 8C 复现逐位一致，指标可信。
