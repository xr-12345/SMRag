# 恢复与验证尝试 — 失败日志

日期：2026-08-30（由 Claude 会话记录；未改动任何既有实验产物）

## 环境 / 依赖（实测可用）
- git HEAD：e661311（main，工作区仅新增本目录与 docs/medexagent-baseline/）
- Python：/Users/xr-12345/miniforge3/bin/python3 = 3.13.13
- numpy：2.4.6（仅 benchmark.py / questioning.py 使用；核心无第三方运行时依赖）
- 单元测试：`PYTHONPATH=src python -m unittest discover -s tests -v` → **45 tests OK**（README/RESEARCH_AUDIT 所记 35 已过时，e661311 新增了 gate_learning / decision 等测试）
- smoke test：`python -m powerful_medrag demo` → 通过（单病例 influenza，1 问，后验 0.916；EIG 0.767 acc / 4.41 问 vs 流行度 0.767 / 5.27）

## 冻结配置（来自仓库既有约定，未改动）
- 病例采样：sample-seed 2026；cases-per-disease 20（validation）
- 回答/噪声种子：seeds 2026 2027 2028
- 噪声率：0.0 0.1 0.2 0.3
- 预算：max-total-turns 15
- 联合策略默认：posterior 0.85，margin 0.70，minimum-action-utility 0.08，verification-cost 0.03
- 验收：见 configs/preregistered_success_criteria.json（Top-1 降幅 ≤0.005；Brier 相对退化 ≤5%；平均总原子问题节省 ≥1.0）

## 阻塞（无法完成 DDXPlus 恢复与 validation 实验）
1. 数据缺失：data/ddxplus/ 仅 release_conditions.json + release_evidences.json；
   release_{train,validate,test}_patients.zip 与 model-full.json 均不在本机（全盘 find 无结果，.gitignore 排除、不在 git 内）。
2. 官方源不可达：DDXPlus 患者 ZIP 仅托管于 figshare（法文版 article 20043374 / 英文版 22687585）。
   - api.figshare.com/v2/articles/* → HTTP 403（含浏览器 UA 仍 403）
   - figshare.com 主站 → 202（JS SPA/挑战页，curl 无法直取文件）
   - WebFetch figshare.com → "Unable to verify if domain figshare.com is safe to fetch"
   - ndownloader.figshare.com → 404（缺 file_id，无法从 API 取得）
   - github.com/mila-iqia/ddxplus → 200 可达，但只托管代码/元数据，不含患者数据
3. oracle 上界方法未实现为策略：benchmark-reliability-ddxplus 的 --strategies 仅
   {random_reliable, ordinary_eig_reliable, full_two_layer, adaptive_history, joint_new_verify_stop}，
   无 oracle 验证策略。仓库仅有 OracleMisreportGate（oracle 门控，读 simulator 特权 oracle_report_mode），
   且只接入 ablate-ddxplus 的报告层消融，未接入联合动作 benchmark。即“oracle VerifyOld 上界”当前无法运行。

## 结论
- 可运行部分（单元测试、demo、toy pilot 管线）全部就绪。
- DDXPlus validation 六方法实验因数据缺失 + 官方源不可达而阻塞；oracle 验证上界因未实现而阻塞。
- 已有 toy 证据（artifacts/toy-reliability-pilot/）显示 joint_new_verify_stop 未达效率标准
  （四种噪声下总原子问题数 +0.07~+0.42，未节省 ≥1；unnecessary_verification_rate 0.80~0.97）。
