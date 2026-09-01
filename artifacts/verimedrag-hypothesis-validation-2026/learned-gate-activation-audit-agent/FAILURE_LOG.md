# 失败日志（learned-gate activation 主点审核）

## 1. activation 修复未改变 NO-GO 核心（阴性结果，非 bug）

- 现象：用 `activation_threshold=0.01` 主动打开 VerifyOld 后，`joint_learned_gate` 的核验效率指标反向恶化——无谓核验率 +13~20 pp、冲突解决率 −12.7~18.6 pp、总问题 +0.5；唯一改善是 Top-1 +0.4~1.05 pp、Brier 改善。
- 判定：精度/Brier 反超是「多核验 ~0.5 次」堆出来的，gate 的选择性（该核验哪份报告）未变强；不满足进入六阈值 sweep 的条件（不必要核验率未降、冲突解决率未升），维持 NO-GO，转四信号离线机制诊断。
- 依据：`paired_deltas.csv`、`summary_by_noise.csv`、`ACTIVATION_AUDIT_REPORT.md`。

## 2. 分析脚本写回 bug（已修，不影响 runner 产物）

- 现象：`config/analysis_script.py` 的 `paired_deltas` 对 `noise_rate`（str）误用 `f"{v:.6f}"` → `ValueError: Unknown format code 'f' for object of type 'str'`。
- 处置：写回改为 str 直通、float 格式化。纯分析层，未触碰 src/，不影响 runner 产物（baseline 回归已 PASS）。

## 3. 其他

- 无其他失败。3 进程 exit 0，baseline 回归 byte-identical 11,760/11,760，未访问 test split，未覆盖旧结果。
