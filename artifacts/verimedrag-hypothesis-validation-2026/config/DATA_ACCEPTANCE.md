# DDXPlus 数据验收 + 模型拟合记录

日期：2026-08-30（prompt #3 阶段，仅验收与 smoke，未运行正式 N=20 validation）

## 数据文件（来源：Desktop/22687585，DDXPlus 英语版 figshare article 22687585）

| 文件 | 大小 (bytes) | SHA256 |
|---|---|---|
| release_train_patients.zip | 140,923,730 | 174ae1d56f36a15b7144838ecd214e1e1748a5369fc90c559529b6dc6ecfb218 |
| release_validate_patients.zip | 18,706,053 | 625a96855becf0efe5da339fd3d440aa99597d48f784957a97dacc3fceaa7e40 |
| release_test_patients.zip | 18,986,243 | ad3bb981f3453fd7abffb35cc595a1c02197e409c581bff97c1265c33eb00d23 |
| release_evidences.json | 120,263 | 281c78b044ae60514e28ecc9052d08a8225dd3e2db3f00f8788b99c47b05e0c0 |
| release_conditions.json | 20,889 | 56edf4682d8e86a4209fc3a33932e50ce03d1cc1fecc326c26d5ef8fa5c24890 |

## ZIP 完整性

三个患者 ZIP 均通过 `unzip -t`：`No errors detected in compressed data`。
每个 ZIP 内含**单个无扩展名文件**（`release_train_patients` / `release_validate_patients` / `release_test_patients`），
loader 的 `_open_patient_csv` 以 `csv_names or file_names` 回退正确读取。

## CSV 字段与行数

字段：`AGE, DIFFERENTIAL_DIAGNOSIS, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE`（**无患者 ID 列**）。
证据名使用英语代码（`E_48`, `E_54_@_V_161` 等）。

| split | 行数 | 疾病数 |
|---|---:|---:|
| train | 1,025,602 | 49 |
| validate | 132,448 | 49 |
| test | 134,529 | 49 |

## 元数据一致性

- `release_evidences.json`：仓库 `data/ddxplus/` 与 Desktop 下载版 **SHA256 完全一致**（281c78b0…），
  已是英语版（E_ 代码 + `question_en`），与患者 CSV 命名一致。
- `release_conditions.json`：仓库（61e16706…）与英语版下载（56edf468…）**不一致**（差约 140 字节）。
  该文件仅用于 RDC 基线疾病—证据图（`load_ddxplus_condition_graph`），**不在本阶段 fit / reliability / clarify 关键路径**；
  留待后续 RDC 基线阶段核对，本阶段未覆盖仓库文件。

## 病例 ID 重叠检查（train vs validate）

- **无患者 ID 列**：loader 以 `case_id = str(row_index)`（分文件行号）作为病例标识，仅在同一切分内唯一，
  跨切分数值必然重复，无法用 ID 判断跨 split 患者身份。
- 内容级检查（`AGE|SEX|PATHOLOGY|EVIDENCES|INITIAL_EVIDENCE` 全字段 SHA256）：
  validate 内 132,373 个唯一全记录哈希；train 中有 **2,208 行**的全记录与 validate 完全一致。
  这是合成数据**疾病—症状模板有限导致的画像复现**，不是患者泄漏证据；但正式结论应注明“无法用 ID 验证跨切分患者独立性”。

## 模型拟合（fit-ddxplus）

- 命令：`PYTHONPATH=src python -m powerful_medrag.cli fit-ddxplus --patients data/ddxplus/release_train_patients.zip --evidences data/ddxplus/release_evidences.json --output data/ddxplus/model-full.json`
- 返回码：0（输出“已保存 DDXPlus 模型…”）
- 运行时间：42.96 s real / 42.69 s user；峰值内存 ~123 MB
- Python：3.13.13；Git HEAD：e661311f91510df39018d04d9c966a110b836018（工作区含 4 个未提交 oracle_verify 修改）
- 模型文件：`data/ddxplus/model-full.json`，12,112,977 bytes，SHA256 a8c95e7f5a41987a5cc60fb334455ecf819b5aea267e1503abf3064a4cf290c8
- 结果：49 疾病，889 原子临床变量（specs），其中 218 askable
- 训练病例数：1,025,602

## 病例 manifest

- `validation_case_manifest.csv`：980 行（49 疾病 × 20），case_id 唯一，split=validate，sample_seed=2026。
- `manifest_meta.json`：数据/模型 SHA256、commit、python 版本、生成时间。
