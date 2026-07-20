from __future__ import annotations

import argparse
import html

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


def build_app(model_id: str, device: str) -> gr.Blocks:
    pipeline = InspectionPipeline(model_id=model_id, device=device)

    def inspect(image: Image.Image | None, use_confirmation: bool) -> tuple[dict, str]:
        if image is None:
            return {}, "请先上传一张厨房图片。"
        method = "verified" if use_confirmation else "structured"
        trace = pipeline.inspect(image, method)
        result = trace.parsed
        payload = {
            "result": result.result,
            "issue_type": result.issue_type,
            "evidence": result.evidence,
            "suggestion": result.suggestion,
            "confirmation": trace.confirmation_decision or "not_required",
            "raw_json_valid": trace.raw_json_valid,
            "parse_strategy": trace.parse_strategy,
            "raw_model_output": trace.raw_output,
        }
        has_issue = {
            "attention": "是",
            "normal": "否",
            "uncertain": "无法判断",
        }[result.result]
        category = CATEGORY_ZH[result.label]
        summary = (
            f"### 巡检结果\n\n"
            f"- 是否存在明显问题：{has_issue}\n"
            f"- 问题类别：{category}\n"
            f"- 判断依据：{html.escape(result.evidence) or '无'}\n"
            f"- 处理建议：{html.escape(result.suggestion) or '无'}"
        )
        return payload, summary

    with gr.Blocks(title="轻量厨房安全巡检助手") as demo:
        gr.Markdown("# 轻量厨房安全巡检助手")
        gr.Markdown("上传一张厨房图片，使用 SmolVLM 检查三类可见问题。")
        with gr.Row():
            image = gr.Image(type="pil", label="厨房图片")
            with gr.Column():
                confirmation = gr.Checkbox(value=True, label="启用视觉证据二次确认")
                button = gr.Button("开始巡检", variant="primary")
                summary = gr.Markdown("等待上传图片。")
        raw_result = gr.JSON(label="结构化结果")
        button.click(inspect, [image, confirmation], [raw_result, summary])
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
