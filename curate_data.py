from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import gradio as gr

from inspection.dataset import write_jsonl
from inspection.schemas import LABELS

CHOICES = [*LABELS, "reject"]


class Curator:
    def __init__(self, candidates: Path, data_root: Path) -> None:
        self.candidates = sorted(
            path
            for path in candidates.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.data_root = data_root
        self.annotations_path = data_root / "annotations.jsonl"
        self.review_path = data_root / "curation.jsonl"
        self.annotations: dict[str, dict] = {}
        self.reviews: dict[str, dict] = {}
        self.index = 0
        self._load_existing()
        self._advance_to_unreviewed()

    def _load_existing(self) -> None:
        for path, destination in (
            (self.annotations_path, self.annotations),
            (self.review_path, self.reviews),
        ):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    destination[Path(record["image"]).name] = record

    def _advance_to_unreviewed(self) -> None:
        while self.index < len(self.candidates):
            if self.candidates[self.index].name not in self.reviews:
                break
            self.index += 1

    def current(self) -> str | None:
        return str(self.candidates[self.index]) if self.index < len(self.candidates) else None

    def status(self) -> str:
        split_counts = Counter(record["split"] for record in self.annotations.values())
        label_counts = Counter(record["label"] for record in self.annotations.values())
        labels = "，".join(f"{key}: {label_counts.get(key, 0)}" for key in LABELS)
        return (
            f"已审核 {len(self.reviews)}/{len(self.candidates)}；"
            f"debug: {split_counts.get('debug', 0)}，test: {split_counts.get('test', 0)}  \n"
            f"类别统计：{labels}"
        )

    def save(self, label: str, split: str) -> tuple[str | None, str]:
        if self.index >= len(self.candidates):
            return None, self.status() + "  \n全部候选图片已审核。"
        if label not in CHOICES:
            return self.current(), self.status() + "  \n请先选择标签。"
        if split not in {"debug", "test"}:
            return self.current(), self.status() + "  \n请选择数据划分。"

        source = self.candidates[self.index]
        review = {"image": source.name, "decision": label, "split": split}
        self.reviews[source.name] = review

        if label != "reject":
            target_dir = self.data_root / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            shutil.copy2(source, target)
            self.annotations[source.name] = {
                "image": target.as_posix(),
                "label": label,
                "split": split,
            }

        write_jsonl(self.annotations_path, self.annotations.values())
        write_jsonl(self.review_path, self.reviews.values())
        self.index += 1
        self._advance_to_unreviewed()
        return self.current(), self.status()

    def skip(self) -> tuple[str | None, str]:
        if self.candidates:
            self.index = (self.index + 1) % len(self.candidates)
            self._advance_to_unreviewed()
        return self.current(), self.status()


def build_app(curator: Curator) -> gr.Blocks:
    with gr.Blocks(title="NYU Kitchen 数据筛选") as demo:
        gr.Markdown("# NYU Kitchen 数据筛选与标注")
        gr.Markdown(
            "删除严重模糊或损坏图片时选择 `reject`；其他图片选择一个主要标签和数据划分。"
        )
        image = gr.Image(value=curator.current(), interactive=False, label="候选厨房图片")
        with gr.Row():
            label = gr.Radio(CHOICES, label="人工标签")
            split = gr.Radio(["debug", "test"], value="debug", label="数据划分")
        with gr.Row():
            save_button = gr.Button("保存并查看下一张", variant="primary")
            skip_button = gr.Button("暂时跳过")
        status = gr.Markdown(curator.status())
        save_button.click(curator.save, [label, split], [image, status])
        skip_button.click(curator.skip, None, [image, status])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate and label exported kitchen images")
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    if not args.candidates.is_dir():
        raise FileNotFoundError("Run prepare_data.py before starting the curation tool")
    build_app(Curator(args.candidates, args.data_root)).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
