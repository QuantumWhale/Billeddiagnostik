"""
Tile-based feature extraction for cytology triage (Case 2).

For each image, we:
1. Split it into a grid of fixed-size tiles.
2. For every tile, compute three "light AI" (classical CV, no black box) signals:
   - cellularity   : how much of the tile looks like cellular material vs empty background
   - blur_score    : how in-focus the tile is (Laplacian variance)
   - debris_score  : how much of the "non-cell" area looks like blood/debris noise
3. Save one row per tile to a CSV, which triage.py will later threshold/score.

Usage:
    python src/features.py
    python src/features.py --input data/raw --output data/processed/features.csv --tile-size 256
    python src/features.py --limit 5   # quick test on first 5 images
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_TILE_SIZE = 256


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as a BGR array (OpenCV's default color order)."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def tile_image(image: np.ndarray, tile_size: int = DEFAULT_TILE_SIZE):
    """
    Split an image into a grid of non-overlapping tile_size x tile_size tiles.

    Partial tiles at the right/bottom edge (when image dimensions aren't an
    exact multiple of tile_size) are dropped for simplicity - CRIC images are
    1376x1020, so with tile_size=256 we get a clean 5x3 grid (1280x768 used,
    edges discarded). Fine for a PoC; revisit if we need full coverage later.

    Returns a list of dicts: {"row": int, "col": int, "tile": np.ndarray}
    """
    height, width = image.shape[:2]
    tiles = []
    for row, y in enumerate(range(0, height - tile_size + 1, tile_size)):
        for col, x in enumerate(range(0, width - tile_size + 1, tile_size)):
            tile = image[y : y + tile_size, x : x + tile_size]
            tiles.append({"row": row, "col": col, "tile": tile})
    return tiles


def compute_blur_score(tile_gray: np.ndarray) -> float:
    """
    Focus/sharpness score using variance of the Laplacian.

    In-focus images have sharp edges -> high-frequency detail -> high variance
    after applying the Laplacian (2nd derivative) operator. Blurry images have
    smoothed-out edges -> low variance. Higher = sharper/more in-focus.
    """
    return cv2.Laplacian(tile_gray, cv2.CV_64F).var()


def compute_cellularity(tile_gray: np.ndarray):
    """
    Estimate how much of the tile is covered by cellular material (nuclei/
    cytoplasm) vs empty background, using Otsu thresholding.

    Papanicolaou-stained cells (nuclei especially) are darker than the mostly
    pale/empty background, so Otsu's automatic threshold on grayscale
    intensity gives a reasonable foreground/background split without us
    having to hand-pick a threshold value.

    Returns:
        cellularity_fraction: float in [0, 1], share of pixels classified as
            cellular material
        foreground_mask: boolean array, True where a pixel was classified as
            cellular material (reused by compute_debris_score)
    """
    # Otsu's method automatically picks the threshold that best separates
    # the image into two classes (foreground/background) from the pixel
    # intensity histogram.
    threshold, binary = cv2.threshold(
        tile_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    foreground_mask = binary > 0
    cellularity_fraction = float(foreground_mask.mean())
    return cellularity_fraction, foreground_mask


def compute_debris_score(tile_bgr: np.ndarray, foreground_mask: np.ndarray) -> float:
    """
    Proxy score for blood/debris contamination in the *non-cellular* area of
    the tile.

    Rationale: clean background should be fairly uniform (pale, low texture).
    Blood and debris introduce color variation and texture noise even in
    areas that aren't solid cellular material. We measure this as the
    standard deviation of hue in the background-classified region - higher
    variation suggests more contamination.

    Returns 0.0 if there's no background region to evaluate (e.g. tile is
    ~100% cellular material).
    """
    background_mask = ~foreground_mask
    if background_mask.sum() == 0:
        return 0.0

    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    background_hue = hue_channel[background_mask]
    return float(background_hue.std())


def extract_tile_features(tile_bgr: np.ndarray) -> dict:
    """Compute all three feature scores for a single tile."""
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    cellularity, foreground_mask = compute_cellularity(tile_gray)
    blur_score = compute_blur_score(tile_gray)
    debris_score = compute_debris_score(tile_bgr, foreground_mask)
    return {
        "cellularity": cellularity,
        "blur_score": blur_score,
        "debris_score": debris_score,
    }


def process_image(image_path: Path, tile_size: int) -> pd.DataFrame:
    """Extract tile-level features for one image, returning a DataFrame (one row per tile)."""
    image = load_image(image_path)
    tiles = tile_image(image, tile_size)

    rows = []
    for tile_info in tiles:
        features = extract_tile_features(tile_info["tile"])
        rows.append(
            {
                "image": image_path.name,
                "row": tile_info["row"],
                "col": tile_info["col"],
                **features,
            }
        )
    return pd.DataFrame(rows)


def process_dataset(input_dir: Path, tile_size: int, limit: int = None) -> pd.DataFrame:
    """Run feature extraction over every image in input_dir, returning one combined DataFrame."""
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir}. Did you run download_data.py first?"
        )

    print(f"Found {len(image_paths)} images. Extracting tile features (tile_size={tile_size})...")

    all_rows = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            df = process_image(image_path, tile_size)
        except ValueError as e:
            print(f"  [warn] skipping {image_path.name}: {e}")
            continue
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Extract tile-based CV features from CRIC images.")
    parser.add_argument("--input", type=str, default="data/raw", help="Directory of downloaded images")
    parser.add_argument("--output", type=str, default="data/processed/features.csv", help="Output CSV path")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size in pixels (square)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (for testing)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df = process_dataset(input_dir, args.tile_size, args.limit)
    features_df.to_csv(output_path, index=False)

    print(f"Done. {len(features_df)} tile rows from {features_df['image'].nunique()} images.")
    print(f"Saved to {output_path}")
    print("\nQuick preview:")
    print(features_df.describe())


if __name__ == "__main__":
    main()