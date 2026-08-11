# 轻量厨房安全巡检助手

本项目使用 `HuggingFaceTB/SmolVLM-500M-Instruct` 分析单张厨房 RGB 图片，判断画面中是否存在三类可见问题：地面或通道障碍、台面明显杂乱、物品摆放位置不合理。没有发现明显问题时输出 `normal`；画面严重模糊、遮挡或无法判断时输出 `uncertain`。

项目不训练、不微调模型，也不调用付费接口。主要工作包括 NYU Depth V2 厨房图像筛选、人工标注、提示词迭代、结构化输出解析、视觉证据二次确认、批量评价和 Gradio Demo。

## 1. 功能与完成情况

| 内容 | 实现位置 |
|---|---|
| 下载 NYU Depth V2 标注数据 | `download_data.py` |
| 筛选并导出 Kitchen RGB 图片 | `prepare_data.py` |
| 人工剔除无效图片并标注 | `curate_data.py`、`review_annotations.py` |
| 划分 Debug/Test 子集并统计数据 | `select_subset.py`、`analyze_data.py` |
| 直接提问、检查清单、结构化输出、二次确认 | `inspection/prompts.py`、`inspection/pipeline.py` |
| 批量推理与指标计算 | `run_model.py`、`evaluate.py` |
| 上传图片并显示巡检结果 | `app.py` |
| 正确与失败案例分析 | `results/case_analysis.md` |

模型每次只接收一张图片。深度图不参与推理，权重文件也不保存在仓库中。

## 2. 研究背景与相关工作

家庭厨房巡检与通用图像描述不同。程序不需要描述画面中的所有物体，而要围绕少量、边界明确的问题给出可以核查的结论。实际应用更关心模型是否看到了直接证据，以及正常场景会不会被频繁误报。

室内场景研究早期常结合 RGB 与深度信息完成分割和支撑关系推断，NYU Depth V2 是其中常用的数据集。近年来，Florence-2、Qwen2-VL 等模型把多种视觉任务统一到文本生成框架中，但完整模型的计算成本较高。本项目选用参数量不超过 0.5B 的 SmolVLM，在普通消费级设备上完成单图推理。根据官方模型卡，500M 版本处理单张图片约需 1.23 GB GPU 显存，适合做轻量原型。

本任务的主要难点不是模型能否识别“碗”“毛巾”或“炉灶”，而是能否判断物体与地面、通道、热源、水源和台面边缘之间的关系。实验中还观察到以下问题：

- 小模型对提示词长度、类别顺序和输出格式较敏感；
- 识别出物体不等于理解物体所在位置及其危险关系；
- `floor_obstruction` 和 `unsafe_object_placement` 样本较少，评价容易受单个样本影响；
- 单人标注和单标签设定会引入边界差异，一张图片也可能同时包含多种问题；
- 模型有时会复述规则、输出不完整 JSON，或给出图中不存在的证据。

因此，本项目将类别边界、可见证据和输出格式写入提示词，并在第一轮报告问题后增加独立的证据确认。

## 3. 任务定义

| 标签 | 判定含义 | 不应计入的情况 |
|---|---|---|
| `floor_obstruction` | 可移动物体明确位于地面或行走通道，并影响通行 | 台面、水槽、炉灶或家电上的物品；固定家具；地砖纹理和阴影 |
| `countertop_clutter` | 多个松散物品明显占据台面或水槽的可用空间 | 少量日常用品、单个容器、固定家电、正常使用的沥水架 |
| `unsafe_object_placement` | 物体与热源、水源或台面边缘之间存在直接可见的不安全关系 | 只看见物体但看不见危险位置关系；一般杂乱 |
| `normal` | 图片清晰，且未发现上述三类明显问题 | — |
| `uncertain` | 严重模糊、遮挡或非厨房画面导致无法判断 | 仅因场景复杂而不愿作出判断 |

## 4. 整体技术方案

<p align="center">
  <img src="docs/model_architecture.png" width="100%" alt="模型与推理流程架构图" />
</p>

一次巡检按以下顺序执行：

1. 读取一张厨房 RGB 图片，并统一转换为三通道格式；
2. `AutoProcessor` 将图片、任务提示和对话模板整理为模型输入；
3. `SmolVLM-500M-Instruct` 以确定性方式生成文本；
4. `inspection/parsing.py` 依次尝试严格 JSON 解析、标签解析和常见回答回退；
5. `inspection/pipeline.py` 将输出统一为 `result`、`issue_type`、`evidence`、`suggestion`；
6. 若启用二次确认且第一轮报告问题，再让模型检查该类别是否有清晰视觉证据；
7. 批量脚本把逐图结果保存为 JSONL，Demo 则显示中文结果和处理建议。

模型加载代码位于 `inspection/model.py`。CUDA 推理根据显卡能力使用 `float16` 或 `bfloat16`，CPU 使用 `float32`。生成阶段设置 `do_sample=False` 并固定随机种子，减少重复实验之间的随机差异。

## 5. 数据准备与人工标注

### 5.1 数据来源

实验使用 [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) 标注版 `.mat` 文件。程序读取其中的 `images` 和 `sceneTypes`，只导出场景类型为 `kitchen` 的 RGB 图片，不下载约 428 GB 的原始视频。

### 5.2 处理流程

```powershell
# 1. 下载标注版数据
.\.venv\Scripts\python.exe download_data.py

# 2. 导出 Kitchen RGB 候选图片
.\.venv\Scripts\python.exe prepare_data.py

# 3. 浏览候选图片，选择标签或 reject
.\.venv\Scripts\python.exe curate_data.py

# 4. 复查已有标注（可选）
.\.venv\Scripts\python.exe review_annotations.py

# 5. 保存 128 条有效标注，再生成固定子集
Copy-Item .\data\annotations.jsonl .\data\annotations_all_128.jsonl -Force
.\.venv\Scripts\python.exe select_subset.py

# 6. 如有需要，复查固定子集中的边界样本，然后统计数据
.\.venv\Scripts\python.exe review_annotations.py
.\.venv\Scripts\python.exe analyze_data.py --representatives 8
```

人工审核时，严重模糊、损坏或不适合评价的图片标为 `reject`；其余图片从五个任务标签中选择一个主要标签。项目共审核 225 张厨房候选图，保留 128 条有效标注记录，再固定 15 张 Debug 图片和 30 张 Test 图片。

最终 45 张实验图片的分布如下。标签复查后个别样本发生调整，因此最终统计与 `select_subset.py` 中最初设定的抽样目标略有不同；报告和指标均以复查后的标注为准。

| 标签 | Debug | Test | 合计 |
|---|---:|---:|---:|
| `floor_obstruction` | 2 | 4 | 6 |
| `countertop_clutter` | 3 | 8 | 11 |
| `unsafe_object_placement` | 1 | 2 | 3 |
| `normal` | 7 | 14 | 21 |
| `uncertain` | 2 | 2 | 4 |
| **合计** | **15** | **30** | **45** |

Debug 集只用于检查程序和调整提示词。提示词确定后冻结，再在 Test 集上运行最终实验。原始数据、筛选图片和标注文件不随仓库公开，仓库仅保留数据处理代码、目录占位文件、统计结果和模型输出。

## 6. 提示词设计

四种正式方法使用同一模型、同一批图片和相同生成参数，只改变提示方式。

| 方法 | 作用 |
|---|---|
| `direct` | 不给出类别清单，观察模型开放式判断能力 |
| `checklist` | 明确检查区域和类别边界，只输出一个标签 |
| `structured` | 在类别规则之外约束 JSON 字段和取值范围 |
| `verified` | 复用 Checklist 初判；只有第二轮确认有证据时才保留问题 |

实际运行的完整提示词位于 `inspection/prompts.py`，最终实验副本位于 `results/prompts_final.py`。下面列出正式实验使用的核心文本。

<details>
<summary>直接提问提示词</summary>

```text
Please inspect this kitchen image and determine whether there is any obvious problem.
Briefly state your judgment and visible evidence.
```

</details>

<details>
<summary>最终检查清单提示词</summary>

```text
Inspect one kitchen image using visible evidence only.

Silently check three places before answering:
1. Are many loose items covering a substantial part of the counter or sink?
2. Is an object in a directly visible dangerous relation to heat, water, or an edge?
3. Is a movable object actually located on the floor or walking path?

Objects on a counter, sink, table, stove, or appliance are never floor obstructions.
A few normally used items are not countertop clutter. An object without a visible
dangerous relation is not unsafe placement. If none applies, choose normal. Choose
uncertain only when severe blur or occlusion prevents judgment.

Reply with one label only and no explanation.
Decision: exactly one label from normal, countertop_clutter, unsafe_object_placement,
floor_obstruction, uncertain.
```

</details>

<details>
<summary>结构化 JSON 提示词</summary>

```text
Inspect one kitchen image using visible evidence only.

Decision rule:
- Use normal when the image is clear and no strong listed issue is visible.
- Use countertop_clutter only when multiple loose items substantially occupy the counter/sink.
- Use unsafe_object_placement only for a directly visible dangerous object-location relation.
- Use floor_obstruction only when a named object is visibly on the floor/walking path.
- Use uncertain only when severe blur or occlusion prevents judgment.

Return one compact JSON object and nothing else. Use result=attention for an issue and put
exactly one issue label in issue_type. For normal or uncertain, issue_type must be empty.
Required keys: result, issue_type, evidence, suggestion. Keep the last two values short.
Allowed result values: normal, attention, uncertain.
Allowed issue_type values: countertop_clutter, unsafe_object_placement,
floor_obstruction, or an empty string.
```

</details>

<details>
<summary>二次确认提示词模板</summary>

```text
Independently verify one candidate issue in this kitchen image.

Candidate: <issue_type>
First-pass note (it may be wrong): <evidence>
Condition to check in the image: <required_evidence>.

The candidate label is not evidence. Silently locate the exact object and its location.
Answer yes only when the image itself clearly satisfies the whole condition.
Answer no when the relevant region is visible but the whole condition is not satisfied.
Answer uncertain if the relevant region cannot be judged.
Reply with exactly one word: yes, no, or uncertain.
```

</details>

### 6.1 提示词迭代记录

提示词不是一次确定的。中文版本容易出现复述和乱码，英文长提示会放大首类偏置，过强的证据约束又会让大量图片退化为 `normal` 或 `uncertain`。最终版本是在 Debug 集上逐步取舍后的结果。

| 版本 | Debug 集现象 |
|---|---|
| 中文第一版 | 单图输出出现乱码、复述，难以解析 |
| 中文第二版 | 清单偶尔判断正确，但直接提问和结构化输出仍重复、跑题 |
| 初始英文 | 容易把不同问题都判断为地面障碍 |
| 证据优先 | 二次确认开始降低误报，但格式不稳定 |
| 英文详细版 | 定义和例子过长，首类偏置更明显 |
| 英文精简版 | 输出更短，但类别偏置仍存在 |
| 偏置修复第一版 | 重排类别并强调 `normal`，结构化输出明显改善 |
| 偏置修复第二版 | 证据门槛过高，Checklist 和 Verification 输出退化 |
| 偏置修复第三版 | 恢复较保守的确认规则，误报下降 |
| 区域检查版本 | 分别检查地面、台面和危险关系，误报减少但召回不足 |
| v1.1 / v1.2 | 分别尝试完整规则和四行观察格式，出现类别塌缩或大量 `uncertain` |
| **v1.3（最终）** | 放宽格式但保留证据边界，在准确率和误报之间取得较稳定结果 |

## 7. 输出格式与解析

结构化方法要求模型输出四个字段：

```json
{
  "result": "attention",
  "issue_type": "floor_obstruction",
  "evidence": "A plate is visible on the walking path.",
  "suggestion": "Remove the plate from the walking path."
}
```

- `result`：`normal`、`attention` 或 `uncertain`；
- `issue_type`：三类问题之一，正常或无法判断时为空；
- `evidence`：模型给出的可见依据；
- `suggestion`：对应的简单处理建议。

小模型有时不能严格遵守 JSON 格式。解析程序先尝试标准 JSON，再处理标签文本和常见回答形式。`raw_json_valid` 只记录原始模型输出是否能被 JSON 解析器直接读取；后处理得到的合法字典不会被计入原始 JSON 合法率。

## 8. 环境安装

实验在 Windows 11、Python 3.11、RTX 4050 Laptop GPU（6 GB）上完成。CPU 也可以运行，但推理速度较慢。

```powershell
git clone https://github.com/RuichenLiu222/home-inspection.git
cd home-inspection

python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需使用 NVIDIA GPU，请先按本机 CUDA 环境安装对应的 PyTorch 版本，再安装其余依赖。当前实验环境使用 `PyTorch 2.13.0+cu126`。可用下面的命令检查 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

模型首次运行时会从 Hugging Face 下载。若系统盘空间不足，可以把缓存放到其他磁盘：

```powershell
$env:HF_HOME="D:\hf-cache"
setx HF_HOME "D:\hf-cache"
```

## 9. 批量实验与评价

### 9.1 调试集冒烟测试

```powershell
.\.venv\Scripts\python.exe run_model.py `
    --split debug `
    --methods all `
    --device cuda `
    --limit 1 `
    --output results\debug_smoke.jsonl
```

### 9.2 运行测试集

```powershell
.\.venv\Scripts\python.exe run_model.py `
    --split test `
    --methods direct,checklist,verified,structured `
    --device cuda `
    --output results\test_final_predictions.jsonl
```

### 9.3 计算指标

```powershell
.\.venv\Scripts\python.exe evaluate.py `
    --predictions results\test_final_predictions.jsonl `
    --output results\test_final_metrics.json
```

评价指标与任务要求一致：

- **Accuracy**：预测标签与人工标签一致的图片比例；
- **正常图误报数**：人工标注为 `normal`，但模型判断存在问题的次数；
- **JSON 合法率**：结构化方法的原始输出能够被程序直接解析的比例。

所有方法在同一张测试图片上运行，使用同一模型和相同生成参数。最终指标只统计冻结提示词后的 30 张 Test 图片。

## 10. 最终结果

| 方法 | Accuracy | 正常图误报数 | JSON 合法率 |
|---|---:|---:|---:|
| Direct | 46.67%（14/30） | 0 | — |
| Checklist | 20.00%（6/30） | 8 | — |
| Checklist + Verification | 30.00%（9/30） | 5 | — |
| Structured | **56.67%（17/30）** | **3** | **86.67%（26/30）** |

结构化提示在本次测试中准确率最高。它没有彻底解决细粒度分类问题，但固定字段和较明确的类别边界使输出更容易被程序使用。

二次确认把 Checklist 的正常图误报从 8 次降到 5 次，并将正确数从 6 张提高到 9 张。它能够驳回一部分缺少视觉证据的初判，但不会重新比较所有类别：第一轮如果把 `unsafe_object_placement` 错分为 `countertop_clutter`，第二轮只能确认或否定 `countertop_clutter`，不能主动改成正确类别。

逐图输出、汇总指标和提示词快照保存在：

```text
results/
├── test_final_predictions.jsonl
├── test_final_metrics.json
├── test_final_metrics.csv
├── prompts_final.py
├── case_analysis.md
└── data_analysis/
    └── dataset_summary.json
```

## 11. 案例与误差分析

完整说明见 [`results/case_analysis.md`](results/case_analysis.md)。实践报告选取了两类正确现象和两类失败现象：

- 地面宠物碗被识别为行走障碍；
- 第一轮误报整洁厨房，二次确认后改为 `normal`；
- 正常图片仍被反复判断为地面障碍；
- 炉具附近的危险摆放被错分为台面杂乱。

这些案例反映出二次确认主要改善误报，不能解决所有类别错分。误差还受到以下因素影响：

1. **人工标注边界**：目前由一人完成标注，台面“明显杂乱”和“正常使用”、轻微障碍和真实通道阻塞之间存在主观边界；
2. **类别分布**：`unsafe_object_placement` 只有 3 张，Test 集只有 2 张，单个样本就会显著改变该类结果；
3. **单标签简化**：一张图片可能同时存在台面杂乱和危险摆放，强制选择一个标签会丢失信息；
4. **图像质量与视角**：NYU 图片分辨率有限，地面、台面边缘或危险关系可能只占很小区域；
5. **模型规模**：500M 模型能识别常见物体，但对空间关系和长提示的稳定性有限；
6. **确认锚定**：第二轮围绕第一轮类别提问，容易受到初判影响。

后续可以采用双人独立标注和分歧复核，补充证据明确的少数类图片；如果任务允许，也可以改用多标签和问题严重程度标注。模型侧可尝试区域裁剪、目标位置描述或受约束解码，但仍保持单模型、无微调的基本设定。

## 12. Gradio Demo

```powershell
.\.venv\Scripts\python.exe app.py --device cuda
```

打开 `http://127.0.0.1:7860`，上传一张厨房图片并点击“开始巡检”。

- 未勾选“启用视觉证据二次确认”：使用结构化提示直接生成结果；
- 勾选后：先进行 Checklist 判断，再执行证据确认。

页面显示是否存在问题、问题类别、判断依据、处理建议以及可展开的模型调试信息。Demo 只用于本地展示，不需要部署到服务器。

## 13. 项目结构

```text
home-inspection/
├── inspection/
│   ├── dataset.py           # 标注读取和图片路径处理
│   ├── model.py             # SmolVLM 加载与单图生成
│   ├── pipeline.py          # 四种方法和二次确认流程
│   ├── prompts.py           # 正式提示词及实验提示词
│   ├── parsing.py           # JSON、标签和确认结果解析
│   └── schemas.py           # 标签与结果数据结构
├── data/
│   ├── raw/                 # 原始 .mat 文件（不上传）
│   ├── candidates/          # Kitchen 候选图片（不上传）
│   └── selected/            # Debug/Test 图片（不上传）
├── results/                 # 预测记录、指标、提示词快照和案例分析
├── tests/                   # 单元测试
├── docs/
│   └── model_architecture.png
├── download_data.py
├── prepare_data.py
├── curate_data.py
├── review_annotations.py
├── select_subset.py
├── analyze_data.py
├── run_model.py
├── evaluate.py
├── app.py
├── requirements.txt
└── requirements-dev.txt
```

## 14. 测试与代码检查

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## 15. 参考资料

1. N. Silberman, D. Hoiem, P. Kohli, et al. [Indoor Segmentation and Support Inference from RGBD Images](https://doi.org/10.1007/978-3-642-33715-4_54). ECCV, 2012.
2. A. Marafioti, O. Zohar, M. Farré, et al. [SmolVLM: Redefining Small and Efficient Multimodal Models](https://arxiv.org/abs/2504.05299), 2025.
3. B. Xiao, H. Wu, W. Xu, et al. [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242), 2023.
4. Y. Li, Y. Du, K. Zhou, et al. [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/). EMNLP, 2023: 292–305.
5. P. Wang, S. Bai, S. Tan, et al. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191), 2024.
6. HuggingFaceTB. [SmolVLM-500M-Instruct Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).
7. New York University. [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
