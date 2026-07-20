from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def _decode_matlab_string(handle: h5py.File, reference: object) -> str:
    value = handle[reference][()]
    flattened = np.asarray(value).reshape(-1)
    return "".join(chr(int(code)) for code in flattened if int(code) != 0)


def read_scene_types(handle: h5py.File) -> list[str]:
    dataset = handle["sceneTypes"]
    references = np.asarray(dataset).reshape(-1)
    return [_decode_matlab_string(handle, ref).strip().lower() for ref in references]


def rgb_from_hdf5(dataset: h5py.Dataset, index: int) -> np.ndarray:
    image = np.asarray(dataset[index])
    # MATLAB HxWx3xN is stored by HDF5 as Nx3xWxH for this dataset.
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(2, 1, 0)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = image
    else:
        raise ValueError(f"Unexpected RGB array shape at index {index}: {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def blur_score(image: np.ndarray) -> float:
    gray = image.astype(np.float32).mean(axis=2)
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def extract_kitchens(mat_path: Path, output_dir: Path, limit: int | None = None) -> int:
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {mat_path}\nRun download_data.py first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "candidates.jsonl"

    rows: list[dict] = []
    with h5py.File(mat_path, "r") as handle:
        scene_types = read_scene_types(handle)
        images = handle["images"]
        indices = [i for i, scene in enumerate(scene_types) if scene == "kitchen"]
        if limit is not None:
            indices = indices[:limit]

        for index in tqdm(indices, desc="Exporting kitchen RGB images"):
            image_array = rgb_from_hdf5(images, index)
            filename = f"kitchen_{index:04d}.jpg"
            destination = output_dir / filename
            if not destination.exists():
                Image.fromarray(image_array, mode="RGB").save(
                    destination,
                    quality=95,
                    subsampling=0,
                )
            rows.append(
                {
                    "image": destination.as_posix(),
                    "source_index": index,
                    "scene_type": scene_types[index],
                    "width": int(image_array.shape[1]),
                    "height": int(image_array.shape[0]),
                    "blur_score": round(blur_score(image_array), 3),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} kitchen images to {output_dir}")
    print(f"Candidate manifest: {manifest_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract kitchen RGB images from NYU Depth V2")
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("data/raw/nyu_depth_v2_labeled.mat"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/candidates"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    extract_kitchens(args.mat, args.output, args.limit)


if __name__ == "__main__":
    main()
