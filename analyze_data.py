from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from inspection.dataset import load_annotations, resolve_image
from inspection.schemas import LABELS


def select_representatives(annotations: list, count: int) -> list:
    buckets = {label: [] for label in LABELS}
    for annotation in annotations:
        buckets[annotation.label].append(annotation)
    selected = []
    while len(selected) < count:
        added = False
        for label in LABELS:
            if buckets[label] and len(selected) < count:
                selected.append(buckets[label].pop(0))
                added = True
        if not added:
            break
    return selected


def contact_sheet(project_root: Path, annotations: list, output: Path) -> None:
    tile_size = (320, 260)
    image_size = (320, 220)
    columns = 4
    rows = (len(annotations) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(canvas)
    for index, annotation in enumerate(annotations):
        image = Image.open(resolve_image(project_root, annotation)).convert("RGB")
        image = ImageOps.contain(image, image_size)
        x = (index % columns) * tile_size[0]
        y = (index // columns) * tile_size[1]
        canvas.paste(image, (x + (image_size[0] - image.width) // 2, y))
        caption = f"{Path(annotation.image).name} | {annotation.label}"
        draw.text((x + 5, y + 225), caption, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def analyze(annotations_path: Path, output_dir: Path, representative_count: int) -> dict:
    project_root = Path(__file__).resolve().parent
    annotations = load_annotations(annotations_path)
    summary = {
        "total": len(annotations),
        "split_counts": dict(Counter(item.split for item in annotations)),
        "label_counts": dict(Counter(item.label for item in annotations)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    representatives = select_representatives(annotations, representative_count)
    if representatives:
        contact_sheet(project_root, representatives, output_dir / "representative_images.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved dataset analysis to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize curated data and build a contact sheet")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/data_analysis"))
    parser.add_argument("--representatives", type=int, default=8)
    args = parser.parse_args()
    analyze(args.annotations, args.output, args.representatives)


if __name__ == "__main__":
    main()
