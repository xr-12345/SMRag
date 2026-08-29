# PowerfulMedRAG：少轮次、不可靠回答感知的问诊原型

这是一个面向研究验证的最小实现。它没有复刻 MedRAG 的全部 RAG/KG
流水线，而是先实现我们讨论的核心增量：

1. 从训练病例估计疾病条件下的真实临床状态
   `P(Z_j | D)`；
2. 用隐变量 `R` 区分确定、不确定、不知道和误报四种回答机制；
3. 对回答机制求和后更新疾病后验，而不是把“不知道”当阴性、把矛盾直接判错；
4. 用期望信息增益（EIG）选择下一个问题，以疾病后验阈值和信息量停止问诊；
5. 输出准确率、平均问题数、Brier 分数、未知回答率和误报识别 AUROC。

> 这是研究代码，不是医疗器械，不应用于真实诊疗决策。目前按我们的约定，
> 尚未实现安全红线/分诊约束。

## 快速运行

项目本身没有第三方运行时依赖。直接在项目根目录执行：

```powershell
$env:PYTHONPATH = "src"
python -m powerful_medrag demo
```

运行测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 概率模型

对疾病 `D=d` 和临床变量 `Z_j`，训练阶段使用平滑后的计数：

```text
P(Z_j=v | D=d) = (n[d,j,v] + alpha) /
                  (sum_v n[d,j,v] + alpha * |V_j|)
```

二元症状时，这就是 Beta--Bernoulli；多分类变量时是
Dirichlet--Categorical。若启用 `hierarchical_strength=kappa`，还会加入全局
特征频率作为经验贝叶斯伪计数，避免罕见病的小样本概率过于极端。

患者回答 `O` 由真实状态和隐含回答模式共同产生：

```text
P(O | Z, cue) = sum_R P(O | Z, R) P(R | cue)
P(D | O)      ∝ P(D) sum_Z P(O | Z, cue) P(Z | D)
```

`cue` 只是从语言中观察到的“肯定/好像”等确定性线索；`R=misreported`
不会直接暴露给系统。代码输出的是误报后验概率，而不是“患者说错了”的布尔标签。

`__unknown__` 是报告值，不是真实临床状态。默认通道下，它对各个真实状态的似然
相同，因此不会把疾病后验往“症状不存在”方向推。

## 训练数据格式

模型刻意要求把 EHR 预处理成“明确状态”，防止把未记录误当阴性。

`schema.json`：

```json
{
  "variables": [
    {
      "key": "fever",
      "values": ["absent", "present"],
      "question": "是否发热？",
      "cost": 1.0
    },
    {
      "key": "pain|activity=walking",
      "values": ["absent", "present"],
      "question": "走路时是否疼痛？",
      "cost": 1.2
    }
  ]
}
```

`cases.jsonl` 每行一个病例：

```json
{"case_id":"1","diagnosis":"influenza","states":{"fever":"present","myalgia":"present"}}
{"case_id":"2","diagnosis":"common_cold","states":{"fever":"absent","myalgia":"__unknown__"}}
```

缺失键和 `__unknown__` 都不进入该变量的分母。只有数据集明确保证完整时，数据适配器
才可以把未激活的二元证据转换成 `absent`。

训练并检查参数：

```powershell
python -m powerful_medrag fit `
  --cases cases.jsonl `
  --schema schema.json `
  --output model.json `
  --hierarchical-strength 5

python -m powerful_medrag inspect --model model.json --feature fever
```

### DDXPlus

官方 DDXPlus 的病例文件可直接是 `.csv` 或包含单个 CSV 的 `.zip`：

```powershell
python -m powerful_medrag fit-ddxplus `
  --patients release_train_patients.zip `
  --evidences release_evidences.json `
  --output ddxplus-model.json `
  --hierarchical-strength 5
```

适配器是流式的，不会把全部患者同时保存在内存中。二元证据映射为
`absent/present`，分类证据用 Dirichlet--Categorical，多选证据的每个非默认选项
被原子化为一个二元变量。默认仅对官方完整合成数据把未激活值解释为默认/阴性；
该快捷命令只适用于官方完整合成数据。若使用删减或二次加工数据，应先转换成前述
显式状态 JSONL，再使用通用 `fit` 命令，避免把缺失值当阴性。

独立测试集曲线：

```powershell
python -m powerful_medrag benchmark-ddxplus `
  --model ddxplus-model.json `
  --patients release_test_patients.zip `
  --evidences release_evidences.json `
  --conditions release_conditions.json `
  --output-dir artifacts/ddxplus-benchmark `
  --cases-per-disease 100 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.60 0.70 0.80 0.85 0.90 0.95 `
  --seeds 2026 2027 2028 `
  --sample-seed 2026 `
  --workers 12 `
  --executor process `
  --no-prevalence
```

该实验使用按疾病分层的独立测试样本。每个病例—变量的噪声由固定哈希种子决定，
所以 EIG 与基线无论以什么顺序提问，遇到同一变量时都会得到同一个模拟回答。
一次最长轨迹会复用于所有停止阈值。
传入官方 `release_conditions.json` 后还会运行 `medrag_rdc` 基线：它把论文公式
`(n-1)/degree` 原样用于 DDXPlus 疾病—证据图，并只在当前 Top-5 候选疾病关联的
证据中选问。由于公开 MedRAG 代码没有结构化多轮选问循环，这是一项明确标注的
“MedRAG-RDC（公式 15 适配）”，不是对未公开实现的冒充复现。多个 `--seeds`
始终复用由 `--sample-seed` 固定的同一批病例，输出病例聚类 bootstrap 置信区间和
逐种子精确 McNemar 检验。

报告层消融：

```powershell
python -m powerful_medrag ablate-ddxplus `
  --model data/ddxplus/model-full.json `
  --patients data/ddxplus/release_test_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output-dir artifacts/ddxplus-ablation-n100-s3 `
  --cases-per-disease 100 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.60 0.70 0.80 0.85 0.90 0.95 `
  --seeds 2026 2027 2028 `
  --sample-seed 2026 `
  --workers 12 `
  --executor process
```

四组均使用 EIG，只改变模型如何解释同一个患者回答：完整两层模型、直接相信已知
回答、把不知道/不确定强制转成阴性，以及删除误报隐状态。患者端始终使用同一个
完整回答通道和特征级随机种子，因此这是推断层消融，而不是更换噪声数据。

PaMis-style 异常检测—澄清基线（验证集）：

```powershell
python -m powerful_medrag clarify-ddxplus `
  --model data/ddxplus/model-full.json `
  --patients data/ddxplus/release_validate_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output-dir artifacts/pamis-style-validation `
  --cases-per-disease 20 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.60 0.70 0.80 0.85 0.90 0.95 `
  --variants pamis_style_s30 `
  --workers 12 `
  --executor process
```

该基线在纳入回答前以 `-log P(answer | history)` 检测异常；超过阈值时重复询问同一
证据，回答一致则只纳入一次，矛盾或不知道则弃权。澄清严格占用一个额外轮次，且
复问仍通过同一个患者噪声通道。它是用于机制对照的 PaMis-style 结构化代理，不是
对 PaMis 实体图、结构熵检测器或自然语言生成器的完整复现。

回溯澄清版本使用 `retro_utility_u050_b1`：在准备停止或即将耗尽问题预算时，逐一
删除历史回答，计算固定报告通道下的潜在错误概率，以及该回答对疾病后验造成的总
变差影响；二者乘积超过 `0.05` 时复问得分最高的一项，每例最多回溯一次。唯一特征
的 leave-one-out 后验通过“完整后验除以该回答似然”精确计算；重复特征自动回退到
完整历史重放。检测器从不读取模拟器的真实噪声率或真实临床状态。

补充消融包括高精度、低成本的 `retro_risk_r50_b1`，以及追求最高召回率的
`hybrid_s30_u025_b2`。前者只使用 leave-one-out 错误概率；后者先进行在线异常澄清，
再允许最多两次回溯审计。验证结果将三者分别定位为“最少额外轮数”“最佳中高噪声
效率”和“最高召回率”，主方法仍采用 `retro_utility_u050_b1`。

## 代码结构

- `schema.py`：带时间/活动/部位上下文的变量、病例和回答；
- `estimation.py`：疾病先验和 `P(Z|D)` 的估计、平滑及序列化；
- `ddxplus.py`：官方 DDXPlus CSV/ZIP 的流式适配器；
- `channel.py`：四种隐回答模式和混淆通道；
- `belief.py`：疾病后验、误报后验和同上下文冲突检测；
- `questioning.py`：单位问题成本的期望信息增益；
- `simulator.py`：结构化患者和停止规则；
- `evaluation.py`：队列级研究指标；
- `benchmark.py`：配对测试集实验、置信区间、CSV 与曲线图；
- `ablation.py`：报告层消融、病例聚类统计与四面板 Pareto 曲线；
- `clarification.py`：异常检测、同证据澄清及检测精度/召回率诊断；
- `curve_analysis.py`：按平均轮数插值的同预算曲线比较与 Pareto 检查；
- `cli.py`：训练、参数检查和演示命令。

## 当前实验边界

- 结构化回答用于隔离验证概率模型，尚未加入 LLM 文本解析误差；
- 各临床变量在给定疾病后条件独立，这是第一版的朴素贝叶斯假设；
- 默认通道参数是可解释的初始值，正式实验应从带复核标签的问诊数据估计；
- 当前“少轮次”来自 EIG 排序和置信度停止；已经加入 MedRAG 度中心性公式适配，
  后续还应加入固定顺序与随机提问等补充基线。
