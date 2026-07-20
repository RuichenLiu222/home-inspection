from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .schemas import LABELS


@dataclass(frozen=True)
class Annotation:
    image: str
    label: str
    split: str = "test"

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Unsupported label {self.label!r}")
        if self.split not in {"debug", "test"}:
            raise ValueError(f"Unsupported split {self.split!r}")


def load_annotations(path: str | Path) -> list[Annotation]:
    path = Path(path)
    records: list[Annotation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(
                    Annotation(
                        image=str(value["image"]),
                        label=str(value["label"]),
                        split=str(value.get("split", "test")),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid annotation at {path}:{line_number}: {exc}") from exc
    return records


def resolve_image(project_root: Path, annotation: Annotation) -> Path:
    direct = project_root / annotation.image
    if direct.is_file():
        return direct
    fallback = project_root / "data" / annotation.split / Path(annotation.image).name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Image not found for annotation: {annotation.image}")


def iter_split(
    annotations: Iterable[Annotation],
    split: str,
) -> Iterator[Annotation]:
    for annotation in annotations:
        if annotation.split == split:
            yield annotation


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
