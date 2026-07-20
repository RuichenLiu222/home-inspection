# 数据目录

本仓库不提交 NYU Depth V2 图片或模型权重。

- `raw/`：约 2.8 GB 的 `nyu_depth_v2_labeled.mat`。
- `candidates/`：从 `.mat` 中导出的全部 `kitchen` RGB 候选图片。
- `debug/`：用于提示词设计的人工筛选图片。
- `test/`：仅用于最终评价的人工筛选图片。
- `annotations.jsonl`：由 `curate_data.py` 生成的人工标签。

每条标注格式如下：

```json
{"image":"data/test/kitchen_0001.jpg","label":"normal","split":"test"}
```

合法标签为：

- `floor_obstruction`
- `countertop_clutter`
- `unsafe_object_placement`
- `normal`
- `uncertain`
