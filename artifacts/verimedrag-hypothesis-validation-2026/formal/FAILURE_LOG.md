# 失败日志（prompt #5 正式 validation 阶段）

## Runner 失败

- 无。`benchmark-reliability-ddxplus`（58,800 条）与 `clarify-ddxplus`（3 seeds × 3,920 条）均退出码 0，无 traceback。
- 依赖：matplotlib 已在 prompt #3 阶段补齐，本阶段未新增依赖。

## 分析脚本失败（已修复，不影响 runner 产物）

- 现象：`config/analysis_script.py` 首次运行在「配对对比」段抛
  `TypeError: tuple indices must be integers or slices, not str`。
- 原因：归一化建索引时误把 `(case_id, dict)` 元组整体 append 进按 case_id 分组的列表，
  随后在 `mean_over_seeds` 中把元组当 dict 按下标取列。
- 影响：完整性检查、归一化 CSV、summary CSV 均已正确写出；仅配对对比与重复敏感性中断。
- 处置：将 append 值由 `(r["case_id"], r)` 改为 `r`（dict），重跑成功，四个产物全部落盘。
- 备注：属分析脚本 bug，非 runner 或 src/ 代码改动；不重跑实验，仅重跑后处理。

## 其他

- 无。数据/模型/manifest SHA256 与 prompt #3 冻结一致；未访问 test split。
