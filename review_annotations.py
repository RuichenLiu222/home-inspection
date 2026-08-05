from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspection.dataset import write_jsonl
from inspection.schemas import LABELS


class AnnotationReviewer:
    def __init__(self, annotations_path: Path, project_root: Path) -> None:
        self.annotations_path = annotations_path
        self.project_root = project_root
        self.rows = [
            json.loads(line)
            for line in annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"No annotations found in {annotations_path}")
        self.index = 0

    def _image_path(self) -> str:
        path = self.project_root / str(self.rows[self.index]["image"])
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return str(path)

    def _status(self, message: str = "") -> str:
        row = self.rows[self.index]
        prefix = f"**{self.index + 1}/{len(self.rows)}**"
        details = (
            f"`{Path(str(row['image'])).name}` · "
            f"split: `{row['split']}` · 当前标签: `{row['label']}`"
        )
        return f"{prefix}　{details}" + (f"  \n{message}" if message else "")

    def current(self) -> tuple[str, str, str]:
        row = self.rows[self.index]
        return self._image_path(), str(row["label"]), self._status()

    def save_and_next(self, label: str) -> tuple[str, str, str]:
        if label not in LABELS:
            return (*self.current()[:2], self._status("请先选择有效标签。"))
        self.rows[self.index]["label"] = label
        write_jsonl(self.annotations_path, self.rows)
        if self.index < len(self.rows) - 1:
            self.index += 1
            message = "上一张标签已保存。"
        else:
            message = "最后一张标签已保存，复核完成。"
        image, current_label, _ = self.current()
        return image, current_label, self._status(message)

    def previous(self) -> tuple[str, str, str]:
        self.index = max(0, self.index - 1)
        return self.current()

    def next_without_change(self) -> tuple[str, str, str]:
        self.index = min(len(self.rows) - 1, self.index + 1)
        return self.current()


def build_app(reviewer: AnnotationReviewer):
    import gradio as gr

    image_value, label_value, status_value = reviewer.current()
    with gr.Blocks(title="最终标注复核") as demo:
        gr.Markdown("# 最终45张图片标注复核")
        gr.Markdown(
            "逐张检查当前标签。标签正确也要点击“保存并下一张”；"
            "该工具只修改 `data/annotations.jsonl`，不会删除图片。"
        )
        image = gr.Image(value=image_value, interactive=False, label="待复核图片")
        label = gr.Radio(LABELS, value=label_value, label="人工标准标签")
        status = gr.Markdown(status_value)
        with gr.Row():
            previous_button = gr.Button("上一张")
            save_button = gr.Button("保存并下一张", variant="primary")
            next_button = gr.Button("跳到下一张（不修改）")
        outputs = [image, label, status]
        previous_button.click(reviewer.previous, None, outputs)
        save_button.click(reviewer.save_and_next, label, outputs)
        next_button.click(reviewer.next_without_change, None, outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and correct final selected labels")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations.jsonl"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    annotations_path = (
        args.annotations
        if args.annotations.is_absolute()
        else project_root / args.annotations
    )
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")
    reviewer = AnnotationReviewer(annotations_path, project_root)
    build_app(reviewer).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
