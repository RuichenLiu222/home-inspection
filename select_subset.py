from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from inspection.schemas import LABELS


DEFAULT_TARGETS = {
    "debug": {
        "floor_obstruction": 2,
        "countertop_clutter": 4,
        "unsafe_object_placement": 1,
        "normal": 6,
        "uncertain": 2,
    },
    "test": {
        "floor_obstruction": 4,
        "countertop_clutter": 8,
        "unsafe_object_placement": 3,
        "normal": 13,
        "uncertain": 2,
    },
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") not in LABELS:
            raise ValueError(f"Unsupported label at line {line_number}: {row.get('label')}")
        rows.append(row)
    return rows


def locate_image(project_root: Path, row: dict) -> Path:
    direct = project_root / str(row["image"])
    if direct.is_file():
        return direct
    filename = Path(str(row["image"])).name
    for split in ("debug", "test"):
        candidate = project_root / "data" / split / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found: {row['image']}")


def select_rows(rows: list[dict], seed: int) -> list[tuple[str, dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)

    required = {
        label: sum(DEFAULT_TARGETS[split][label] for split in ("debug", "test"))
        for label in LABELS
    }
    available = Counter(str(row["label"]) for row in rows)
    shortages = {
        label: (required[label], available[label])
        for label in LABELS
        if available[label] < required[label]
    }
    if shortages:
        raise ValueError(f"Not enough samples for target distribution: {shortages}")

    rng = random.Random(seed)
    for label in LABELS:
        grouped[label].sort(key=lambda item: Path(str(item["image"])).name)
        rng.shuffle(grouped[label])

    selected: list[tuple[str, dict]] = []
    offsets = Counter()
    for split in ("debug", "test"):
        for label in LABELS:
            count = DEFAULT_TARGETS[split][label]
            start = offsets[label]
            chosen = grouped[label][start : start + count]
            selected.extend((split, row) for row in chosen)
            offsets[label] += count
    return selected


def write_subset(
    source: Path,
    output: Path,
    selected_root: Path,
    seed: int,
) -> None:
    project_root = Path(__file__).resolve().parent
    rows = load_rows(source)
    selected = select_rows(rows, seed)

    output_rows = []
    for split, row in selected:
        source_image = locate_image(project_root, row)
        destination = selected_root / split / source_image.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination)
        output_rows.append(
            {
                "image": destination.relative_to(project_root).as_posix(),
                "label": row["label"],
                "split": split,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)

    print(f"Source annotations: {len(rows)}")
    print(f"Selected annotations: {len(output_rows)}")
    for split in ("debug", "test"):
        counts = Counter(row["label"] for row in output_rows if row["split"] == split)
        print(f"{split}: {sum(counts.values())} | {dict(counts)}")
    print(f"Saved annotations to: {output}")
    print(f"Copied selected images to: {selected_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducible 15-image debug and 30-image test subset"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/annotations_all_128.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations.jsonl"),
    )
    parser.add_argument(
        "--selected-root",
        type=Path,
        default=Path("data/selected"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(
            f"Source annotations not found: {args.source}. "
            "Back up data/annotations.jsonl before running this script."
        )
    write_subset(args.source, args.output, args.selected_root, args.seed)


if __name__ == "__main__":
    main()
