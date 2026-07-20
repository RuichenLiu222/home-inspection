from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = (
    "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/"
    "nyu_depth_v2_labeled.mat"
)


def download(url: str, output: Path, force: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"File already exists: {output}")
        print("Use --force to download it again.")
        return

    partial = output.with_suffix(output.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() and not force else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    with requests.get(url, stream=True, timeout=(20, 120), headers=headers) as response:
        response.raise_for_status()
        resumed = downloaded > 0 and response.status_code == 206
        if downloaded and not resumed:
            downloaded = 0
        mode = "ab" if resumed else "wb"
        remaining = int(response.headers.get("content-length", 0))
        total = downloaded + remaining if remaining else None
        with partial.open(mode) as handle, tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc="NYU Depth V2",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))

    partial.replace(output)
    print(f"Downloaded to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the NYU Depth V2 labeled .mat file")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/nyu_depth_v2_labeled.mat"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.url, args.output, args.force)


if __name__ == "__main__":
    main()
