from __future__ import annotations

import argparse
import html
from dataclasses import replace

import gradio as gr
from PIL import Image

from inspection.model import DEFAULT_MODEL_ID
from inspection.pipeline import InspectionPipeline

CATEGORY_ZH = {
    "floor_obstruction": "地面或通道存在杂物",
    "countertop_clutter": "厨房台面明显杂乱",
    "unsafe_object_placement": "物品摆放位置不合理",
    "normal": "未发现明显问题",
    "uncertain": "无法可靠判断",
}

SUGGESTION_ZH = {
    "floor_obstruction": "移走地面或通道中的物品，保持行走区域畅通。",
    "countertop_clutter": "整理台面物品，将暂时不用的物品收纳归位。",
    "unsafe_object_placement": "将相关物品移至远离热源、水源或台面边缘的安全位置。",
}


def _demo_method(use_confirmation: bool) -> str:
    """Keep the checkbox comparison on one checklist-based pipeline."""
    return "verified" if use_confirmation else "checklist"


def _with_demo_suggestion(result):
    """Fill a missing UI suggestion without changing experiment predictions."""
    if result.result != "attention" or result.suggestion:
        return result
    return replace(result, suggestion=SUGGESTION_ZH[result.issue_type])

CUSTOM_CSS = """
:root {
  --page: #f4f7fb;
  --surface: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #f97316;
  --accent-hover: #ea580c;
  --success: #15803d;
  --warning: #b45309;
  --danger: #c2410c;
}

body, .gradio-container {
  background: var(--page) !important;
  color: var(--text) !important;
  font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
}

.gradio-container {
  max-width: none !important;
  padding: 30px 24px 40px !important;
}

#app-shell {
  max-width: 1460px;
  margin: 0 auto;
  gap: 22px;
}

#app-header {
  padding: 2px 2px 4px;
}

#app-header h1 {
  margin: 0 0 8px !important;
  color: var(--text) !important;
  font-size: clamp(28px, 3vw, 42px) !important;
  line-height: 1.18 !important;
  letter-spacing: -0.03em !important;
}

#app-header p {
  margin: 0 !important;
  color: var(--muted) !important;
  font-size: 16px !important;
  line-height: 1.7 !important;
}

#workspace {
  gap: 24px !important;
  align-items: stretch !important;
}

#image-panel, #inspection-panel {
  min-width: 0 !important;
}

#kitchen-image {
  overflow: hidden;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  background: var(--surface) !important;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
}

#kitchen-image .image-container {
  background: #f8fafc !important;
}

#kitchen-image .label-wrap {
  color: var(--text) !important;
  font-weight: 600 !important;
}

#inspection-panel {
  gap: 16px !important;
}

#confirm-control {
  margin: 0 !important;
  padding: 16px 18px !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  background: var(--surface) !important;
  box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04) !important;
}

#confirm-control label {
  color: var(--text) !important;
  font-size: 16px !important;
  font-weight: 600 !important;
}

#inspect-button {
  min-height: 52px !important;
  border: 0 !important;
  border-radius: 13px !important;
  background: var(--accent) !important;
  color: white !important;
  font-size: 17px !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 20px rgba(249, 115, 22, 0.22) !important;
  transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease !important;
}

#inspect-button:hover {
  background: var(--accent-hover) !important;
  box-shadow: 0 10px 24px rgba(234, 88, 12, 0.28) !important;
  transform: translateY(-1px);
}

#result-card {
  flex: 1;
  min-height: 390px;
  overflow: hidden;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  background: var(--surface) !important;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
}

.result-shell {
  height: 100%;
  padding: 24px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 18px;
}

.result-title {
  color: var(--text);
  font-size: 20px;
  font-weight: 750;
}

.status-label {
  flex: none;
  padding: 6px 11px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
}

.status-normal { background: #ecfdf3; color: var(--success); }
.status-attention { background: #fff7ed; color: var(--danger); }
.status-uncertain { background: #fffbeb; color: var(--warning); }
.status-waiting { background: #f1f5f9; color: var(--muted); }

.result-list {
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 13px;
}

.result-row {
  display: grid;
  grid-template-columns: minmax(120px, 30%) 1fr;
  gap: 22px;
  align-items: start;
  padding: 18px 20px;
  border-bottom: 1px solid var(--border);
}

.result-row:last-child { border-bottom: 0; }
.result-key { color: var(--muted); font-size: 14px; font-weight: 600; }
.result-value { color: var(--text); font-size: 15px; font-weight: 600; line-height: 1.65; }
.result-value.normal { color: var(--success); }
.result-value.attention { color: var(--danger); }
.result-value.uncertain { color: var(--warning); }

.result-empty {
  display: flex;
  min-height: 310px;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  text-align: center;
  line-height: 1.7;
}

#technical-details {
  overflow: hidden;
  border: 1px solid var(--border) !important;
  border-radius: 15px !important;
  background: var(--surface) !important;
  box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04) !important;
}

#technical-details > .label-wrap {
  padding: 15px 18px !important;
  color: var(--text) !important;
  font-size: 15px !important;
  font-weight: 700 !important;
}

#technical-details .json-holder {
  border-color: var(--border) !important;
  border-radius: 12px !important;
}

@media (max-width: 900px) {
  .gradio-container { padding: 20px 14px 28px !important; }
  #workspace { flex-direction: column !important; }
  #result-card { min-height: 340px; }
  .result-row { grid-template-columns: 1fr; gap: 6px; }
}
"""


def _empty_result(message: str = "上传厨房图片后，点击“开始巡检”查看结果。") -> str:
    return f"""<div class="result-shell">
      <div class="result-header">
        <span class="result-title">巡检结果</span>
        <span class="status-label status-waiting">等待巡检</span>
      </div>
      <div class="result-empty">{html.escape(message)}</div>
    </div>"""


def _result_card(result) -> str:
    has_issue = {
        "attention": "是",
        "normal": "否",
        "uncertain": "无法判断",
    }[result.result]
    status_text = {
        "attention": "需要关注",
        "normal": "未见异常",
        "uncertain": "证据不足",
    }[result.result]
    category = CATEGORY_ZH[result.label]
    evidence = html.escape(result.evidence) or "无"
    suggestion = html.escape(result.suggestion) or "无"
    return f"""<div class="result-shell">
      <div class="result-header">
        <span class="result-title">巡检结果</span>
        <span class="status-label status-{result.result}">{status_text}</span>
      </div>
      <div class="result-list">
        <div class="result-row">
          <span class="result-key">是否存在明显问题</span>
          <span class="result-value {result.result}">{has_issue}</span>
        </div>
        <div class="result-row">
          <span class="result-key">问题类别</span>
          <span class="result-value">{html.escape(category)}</span>
        </div>
        <div class="result-row">
          <span class="result-key">判断依据</span>
          <span class="result-value">{evidence}</span>
        </div>
        <div class="result-row">
          <span class="result-key">处理建议</span>
          <span class="result-value">{suggestion}</span>
        </div>
      </div>
    </div>"""


def build_app(model_id: str, device: str) -> gr.Blocks:
    pipeline = InspectionPipeline(model_id=model_id, device=device)

    def inspect(
        image: Image.Image | None,
        use_confirmation: bool,
    ) -> tuple[dict, str, dict]:
        if image is None:
            normalized = {"result": "", "issue_type": "", "evidence": "", "suggestion": ""}
            return normalized, _empty_result("请先上传一张厨房图片。"), {"status": "waiting"}

        method = _demo_method(use_confirmation)
        trace = pipeline.inspect(image, method)
        result = _with_demo_suggestion(trace.parsed)
        normalized = result.to_dict()
        debug_payload = {
            "method": method,
            "confirmation": trace.confirmation_decision or "not_required",
            "raw_json_valid": trace.raw_json_valid,
            "parse_strategy": trace.parse_strategy,
            "raw_model_output": trace.raw_output,
            "confirmation_output": trace.confirmation_output,
            "focused_check_outputs": getattr(trace, "component_outputs", None),
        }
        return normalized, _result_card(result), debug_payload

    with gr.Blocks(
        title="轻量厨房安全巡检助手",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        fill_width=True,
    ) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.Markdown(
                "# 轻量厨房安全巡检助手\n上传一张厨房图片，使用 SmolVLM 检查三类可见问题。",
                elem_id="app-header",
            )
            with gr.Row(elem_id="workspace"):
                with gr.Column(scale=6, elem_id="image-panel"):
                    image = gr.Image(
                        type="pil",
                        label="厨房图片",
                        height=560,
                        elem_id="kitchen-image",
                    )
                with gr.Column(scale=5, elem_id="inspection-panel"):
                    confirmation = gr.Checkbox(
                        value=True,
                        label="启用视觉证据二次确认",
                        elem_id="confirm-control",
                    )
                    button = gr.Button(
                        "开始巡检",
                        variant="primary",
                        elem_id="inspect-button",
                    )
                    summary = gr.HTML(_empty_result(), elem_id="result-card")

            with gr.Accordion("技术详情", open=False, elem_id="technical-details"):
                with gr.Row():
                    normalized_result = gr.JSON(label="标准化 JSON")
                    debug_result = gr.JSON(label="模型调试信息")

            button.click(
                inspect,
                [image, confirmation],
                [normalized_result, summary, debug_result],
            )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the kitchen-inspection Gradio demo")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.model, args.device).launch(
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
