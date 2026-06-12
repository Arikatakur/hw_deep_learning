"""Download an unlabeled COVID-19 rapid antigen test image dataset.

The script uses Bing Image Search API, SerpApi Google Images API, or the
official Roboflow SDK. It downloads valid images, converts them to RGB JPG,
removes perceptual duplicates, and creates an 80/20 train/test split.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import imagehash
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm


SEARCH_QUERIES = [
    "covid rapid antigen test cassette",
    "covid 19 rapid test positive negative",
    "covid antigen test C T line",
    "covid test kit C T result",
    "lateral flow covid test C T",
    "SARS-CoV-2 antigen rapid test cassette",
]

DEFAULT_TARGET_IMAGES = 500
DEFAULT_TRAIN_RATIO = 0.8
MIN_WIDTH = 220
MIN_HEIGHT = 220
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
REQUEST_TIMEOUT = 15

NEGATIVE_TERMS = {
    "logo",
    "icon",
    "vector",
    "illustration",
    "clipart",
    "cartoon",
    "banner",
    "poster",
    "infographic",
    "graph",
    "chart",
    "map",
    "mask",
    "vaccine",
    "syringe",
    "hospital",
    "doctor",
}

POSITIVE_TERMS = {
    "covid",
    "sars",
    "antigen",
    "rapid",
    "test",
    "cassette",
    "lateral",
    "flow",
    "result",
}


@dataclass(frozen=True)
class ImageCandidate:
    image_url: str
    page_url: str = ""
    title: str = ""


def read_env_file(path: Path) -> None:
    """Load KEY=VALUE entries from a .env file without overriding env vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_provider(provider: str | None) -> str:
    if provider:
        provider = provider.strip().lower()
    if provider in {"bing", "serpapi", "roboflow"}:
        return provider
    if os.getenv("BING_IMAGE_SEARCH_API_KEY"):
        return "bing"
    if os.getenv("SERPAPI_API_KEY"):
        return "serpapi"
    if os.getenv("ROBOFLOW_API_KEY"):
        return "roboflow"
    raise RuntimeError(
        "No API key found. Add BING_IMAGE_SEARCH_API_KEY, SERPAPI_API_KEY, "
        "or ROBOFLOW_API_KEY to .env."
    )


def search_bing(query: str, count: int, offset: int) -> list[ImageCandidate]:
    api_key = os.getenv("BING_IMAGE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BING_IMAGE_SEARCH_API_KEY is missing from .env.")

    endpoint = os.getenv(
        "BING_IMAGE_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/images/search"
    )
    params = {
        "q": query,
        "count": min(count, 150),
        "offset": offset,
        "imageType": "Photo",
        "safeSearch": "Moderate",
        "license": "Any",
    }
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    response = requests.get(
        endpoint, params=params, headers=headers, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    candidates = []
    for item in response.json().get("value", []):
        image_url = item.get("contentUrl")
        if not image_url:
            continue
        candidates.append(
            ImageCandidate(
                image_url=image_url,
                page_url=item.get("hostPageUrl", ""),
                title=item.get("name", ""),
            )
        )
    return candidates


def search_serpapi(query: str, count: int, start: int) -> list[ImageCandidate]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is missing from .env.")

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "ijn": start // max(count, 1),
        "safe": "active",
    }
    response = requests.get(
        "https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    candidates = []
    for item in response.json().get("images_results", []):
        image_url = item.get("original") or item.get("thumbnail")
        if not image_url:
            continue
        candidates.append(
            ImageCandidate(
                image_url=image_url,
                page_url=item.get("link", ""),
                title=item.get("title", ""),
            )
        )
    return candidates[:count]


def candidate_text(candidate: ImageCandidate) -> str:
    parsed = urlparse(candidate.image_url)
    text = " ".join(
        [
            candidate.title,
            candidate.page_url,
            parsed.netloc,
            parsed.path,
        ]
    )
    return text.lower()


def is_relevant_candidate(candidate: ImageCandidate) -> bool:
    text = candidate_text(candidate)
    if any(term in text for term in NEGATIVE_TERMS):
        return False
    return sum(term in text for term in POSITIVE_TERMS) >= 2


def is_probably_icon_or_logo(image: Image.Image) -> bool:
    width, height = image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return True
    aspect_ratio = width / height
    if aspect_ratio < 0.35 or aspect_ratio > 3.2:
        return True
    if image.mode in {"P", "LA"}:
        return True
    return False


def download_bytes(url: str) -> bytes | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        )
    }
    try:
        with requests.get(
            url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type:
                return None

            data = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    return None
            return bytes(data)
    except requests.RequestException:
        return None


def open_valid_image(image_bytes: bytes) -> Image.Image | None:
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if is_probably_icon_or_logo(image):
                return None
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None


def collect_candidates(provider: str, target: int) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    seen_urls: set[str] = set()
    per_query_goal = max(target // len(SEARCH_QUERIES) + 80, 120)

    for query in SEARCH_QUERIES:
        offset = 0
        empty_pages = 0
        added_for_query = 0
        while added_for_query < per_query_goal:
            try:
                if provider == "bing":
                    page = search_bing(query=query, count=100, offset=offset)
                    offset += 100
                else:
                    page = search_serpapi(query=query, count=100, start=offset)
                    offset += 100
            except requests.RequestException as exc:
                print(f"Search failed for query '{query}': {exc}")
                break

            if not page:
                empty_pages += 1
                if empty_pages >= 2:
                    break
                continue

            added_this_page = 0
            for candidate in page:
                url = candidate.image_url
                if url in seen_urls or not is_relevant_candidate(candidate):
                    continue
                seen_urls.add(url)
                candidates.append(candidate)
                added_this_page += 1
                added_for_query += 1

            if added_this_page == 0:
                empty_pages += 1
            if empty_pages >= 3 or offset >= 1000:
                break

    random.shuffle(candidates)
    return candidates


def image_is_duplicate(
    image: Image.Image, known_hashes: list[imagehash.ImageHash], threshold: int
) -> bool:
    image_hash = imagehash.phash(image)
    for known_hash in known_hashes:
        if image_hash - known_hash <= threshold:
            return True
    known_hashes.append(image_hash)
    return False


def reset_output_dirs(dataset_dir: Path) -> tuple[Path, Path, Path]:
    staging_dir = dataset_dir / "_downloaded"
    train_dir = dataset_dir / "train" / "images"
    test_dir = dataset_dir / "test" / "images"

    for directory in [staging_dir, train_dir, test_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    return staging_dir, train_dir, test_dir


def save_jpg(image: Image.Image, path: Path) -> None:
    image.save(path, "JPEG", quality=92, optimize=True)


def split_dataset(image_paths: Iterable[Path], train_dir: Path, test_dir: Path) -> tuple[int, int]:
    image_paths = sorted(image_paths)
    if len(image_paths) < 2:
        for path in image_paths:
            shutil.copy2(path, train_dir / path.name)
        return len(image_paths), 0

    train_paths, test_paths = train_test_split(
        image_paths,
        train_size=DEFAULT_TRAIN_RATIO,
        random_state=42,
        shuffle=True,
    )

    for path in train_paths:
        shutil.copy2(path, train_dir / path.name)
    for path in test_paths:
        shutil.copy2(path, test_dir / path.name)

    return len(train_paths), len(test_paths)


def download_roboflow_export(
    workspace: str, project: str, version: int, export_format: str, output_dir: Path
) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing from .env.")

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "The roboflow package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    downloaded = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(export_format, location=str(output_dir), overwrite=True)
    )
    return Path(downloaded.location)


def roboflow_image_paths(download_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(download_dir.rglob(suffix))
    return sorted(path for path in image_paths if path.is_file())


def process_local_images(
    source_paths: Iterable[Path],
    staging_dir: Path,
    target: int,
    hash_threshold: int,
) -> tuple[int, int, int]:
    known_hashes: list[imagehash.ImageHash] = []
    saved = 0
    duplicates_removed = 0
    skipped = 0

    paths = list(source_paths)
    random.shuffle(paths)
    progress = tqdm(paths, desc="Processing local images", unit="image")

    for source_path in progress:
        if saved >= target:
            break

        try:
            image_bytes = source_path.read_bytes()
        except OSError:
            skipped += 1
            continue

        image = open_valid_image(image_bytes)
        if image is None:
            skipped += 1
            continue

        if image_is_duplicate(image, known_hashes, hash_threshold):
            duplicates_removed += 1
            continue

        saved += 1
        filename = f"covid_test_{saved:06d}.jpg"
        save_jpg(image, staging_dir / filename)
        progress.set_postfix(saved=saved, dupes=duplicates_removed, skipped=skipped)

    return saved, duplicates_removed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and split COVID-19 rapid antigen test kit images."
    )
    parser.add_argument(
        "--provider",
        choices=["bing", "serpapi", "roboflow"],
        default=None,
        help="Dataset source. Defaults to the first matching API key found in .env.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_IMAGES,
        help="Target number of unique images to keep.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file.",
    )
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="Perceptual hash distance treated as a duplicate.",
    )
    parser.add_argument(
        "--roboflow-workspace",
        default="new-workspace-4ssos",
        help="Roboflow workspace slug.",
    )
    parser.add_argument(
        "--roboflow-project",
        default="atk_yolo",
        help="Roboflow project slug.",
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=None,
        help="Roboflow dataset version number.",
    )
    parser.add_argument(
        "--roboflow-format",
        default="yolov8",
        help="Roboflow export format to request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_env_file(args.env_file)
    provider = normalize_provider(args.provider)

    staging_dir, train_dir, test_dir = reset_output_dirs(args.dataset_dir)
    if provider == "roboflow":
        roboflow_version = args.roboflow_version or int(os.getenv("ROBOFLOW_VERSION", "1"))
        roboflow_dir = download_roboflow_export(
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=roboflow_version,
            export_format=args.roboflow_format,
            output_dir=args.dataset_dir / "_roboflow_export",
        )
        downloaded, duplicates_removed, skipped = process_local_images(
            source_paths=roboflow_image_paths(roboflow_dir),
            staging_dir=staging_dir,
            target=args.target,
            hash_threshold=args.hash_threshold,
        )
    else:
        candidates = collect_candidates(provider=provider, target=args.target * 2)

        known_hashes: list[imagehash.ImageHash] = []
        downloaded = 0
        duplicates_removed = 0
        skipped = 0

        progress = tqdm(candidates, desc="Downloading images", unit="image")
        for candidate in progress:
            if downloaded >= args.target:
                break

            image_bytes = download_bytes(candidate.image_url)
            if not image_bytes:
                skipped += 1
                continue

            image = open_valid_image(image_bytes)
            if image is None:
                skipped += 1
                continue

            if image_is_duplicate(image, known_hashes, args.hash_threshold):
                duplicates_removed += 1
                continue

            downloaded += 1
            filename = f"covid_test_{downloaded:06d}.jpg"
            save_jpg(image, staging_dir / filename)
            progress.set_postfix(saved=downloaded, dupes=duplicates_removed, skipped=skipped)

    train_count, test_count = split_dataset(staging_dir.glob("*.jpg"), train_dir, test_dir)

    print()
    print("Final summary")
    print(f"total downloaded: {downloaded}")
    print(f"duplicates removed: {duplicates_removed}")
    print(f"train count: {train_count}")
    print(f"test count: {test_count}")
    if downloaded < args.target:
        print(
            f"warning: target was {args.target}, but only {downloaded} unique valid images "
            "were found. Try the other API provider or rerun with a larger API quota."
        )


if __name__ == "__main__":
    main()
