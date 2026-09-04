# Phase 8E — FAILURE_LOG

记录本阶段遇到并已解决/避免的问题，供复现与审计。

## 1. unittest verbose 输出走 stderr（已修正）

**现象**：首条 `python -m unittest -v > TEST_LOG.txt 2>&1` 因把 `2>&1` 放在
`> file` 之前，导致 `unittest -v` 的逐条 `... ok`（它写 stderr）被导向终端而非文件，
`TEST_LOG.txt` 为空（0 行）。

**修正**：改为 `... -v > TEST_LOG.txt 2>&1`（先重定向 stdout 到文件，再 `2>&1`）。
修正后 `TEST_LOG.txt` 记录 240 tests OK。

**教训**：`unittest`/`pytest` 的逐条结果写 **stderr**，`> file 2>&1` 顺序必须正确。

## 2. 后台运行避免 `| tail` 吞退出码（沿用 Phase 8D 教训）

全量测试与 smoke/full 运行均不接 `| tail`，直接看真实退出码与末尾输出。

## 3. p_wrong=None 需在 CSV 中显式空值（已处理）

`run_n5_screen.py` 对 `p_wrong is None` 写空字符串 `""`（不写 `NaN`、不写 `0.0`），
`is_nonresponse` 独立记录。分析脚本对空字符串跳过（不进入 wrong-report ECE/Brier）。
smoke 已验证：UNKNOWN 行 `p_wrong=''` 且 `is_nonresponse=1`，明确回答行 `p_wrong` 非空。

## 4. 无遗留问题

以上均已处理；无其它失败。
