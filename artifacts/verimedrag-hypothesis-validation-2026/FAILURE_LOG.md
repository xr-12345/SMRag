# 失败日志（prompt #3 阶段）

## 1. clarify-ddxplus（retro runner）首次 smoke 失败：matplotlib 缺失

- 现象：`clarify-ddxplus` 在写完全部 CSV 后，调用 `plot_ablation_curves` 抛出
  `ModuleNotFoundError: No module named 'matplotlib'` → `RuntimeError: ablation plotting requires matplotlib`。
- 原因：matplotlib 是 `pyproject.toml` 中 `[project.optional-dependencies].experiment` 声明的依赖（`matplotlib>=3.7`），
  但 miniforge 环境此前未安装；reliability runner 不绘图故未触发。
- 影响：retro runner 的曲线 CSV / 逐病例 CSV / 诊断 CSV 已写入，仅 PNG 绘制失败，命令非零退出。
- 处置：安装声明依赖 `matplotlib`（pip，3.11.1，连同 contourpy/fonttools/cycler/kiwisolver/pyparsing），
  重新运行 `clarify-ddxplus` → 成功完成 49/49 任务并生成 PNG。
- 备注：属依赖补齐，非代码改动；未修改任何 src/ 文件。

## 2. 其他

- 无其他失败。reliability runner 一次通过；fit-ddxplus 一次通过。

---

# 失败日志（prompt #8 阶段：learned gate 接入 joint policy）

## 1. learned gate 集成结果阴性（No-Go）—— 非 bug，是信号不匹配

- 现象：`joint_learned_gate`（VerifyOld 用 `learned_risk_i × diagnostic_influence_i` 替换
  retrospective error probability）在全部 4 噪声 × 6 阈值上严格劣于 `joint_new_verify_stop`。
  参照点（noise 0.2 / 阈值 0.85）：精度 0.8378→0.8327（−0.0051）、不必要核验率 0.688→0.825
  （**+0.1365**）、冲突解决率 0.294→0.167（**−0.1261**）；matched-budget 13/13 被 baseline 离散支配。
- 原因（假说）：learned gate 的 AUROC 0.844 是「是否 harmful misreport」的二分类排序能力，而 VerifyOld
  需要的是「核验哪份报告最大化诊断改变量」的影响力加权已校准信号；gate 输出被夹在
  `[minimum_probability, cap]` 窄区间且未按诊断影响力加权，替换后稀释了与 `diagnostic_influence` 的乘积判别力。
- 影响：集成代码正确（回归 byte-identical、无泄漏、18 进程全过），阴性结果是科学结论而非实现缺陷。
- 处置：No-Go，不部署 `joint_learned_gate`；不 retrain/重校准 gate；落盘 `learned_risk` 分布以证伪假说
  （当前 outcomes CSV 未落该字段）留待后续 prompt。详见 `learned-gate-integration/LEARNED_GATE_INTEGRATION_REPORT.md`。

## 2. 分析脚本 CSV 写回 bug（已修，不影响 runner 产物）

- 现象：`analysis_script.py` 的 `write_summary` 对 `noise_rate`/`posterior_threshold`（已格式化为 str）
  与 `None` 字段误用 `f"{v:.6f}"` → `ValueError: Unknown format code 'f' for object of type 'str'`。
- 处置：加 `_csv_fmt`（str 直通 / None 留空 / 数字格式化）；`learned_deltas` 对可空的
  `unnecessary_verification_rate`/`conflict_resolution_rate` 改用 `_delta` 安全求差。纯分析层，未触碰 src/。
