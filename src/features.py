"""
Tile-based feature extraction for cytology triage (Case 2).
 
For each image, we:
1. Build ONE cellular-material mask for the whole image (see
   compute_foreground_mask - this is the foundation everything else rests on).
2. Split the image into a grid of tiles that covers every pixel.
3. For every tile, compute three "light AI" (classical CV, no black box) signals:
   - cellularity   : fraction of the tile covered by cellular material
   - blur_score    : how in-focus the tile is (variance of Laplacian)
   - debris_score  : how textured/noisy the non-cellular background is
4. Save one row per tile to a CSV, which triage.py then thresholds.
 
All thresholds live in triage.py. Constants below that affect *how features
are measured* (not how they're judged) are marked CALIBRATE where the right
value depends on the dataset and should be checked on real images.
 
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
 
# --- Measurement constants (CALIBRATE on real CRIC images) -------------------
# Saturation (HSV, 0-255) above which a pixel counts as stained material.
# Bare glass / mounting medium is near-grey (very low saturation); anything
# Pap-stained - nuclei, cytoplasm, most debris - carries colour. This is a
# FIXED threshold on purpose: an adaptive one (Otsu) would re-pick itself on
# every image, so an empty image gets an arbitrary split and cellularity
# stops being comparable across images. See compute_foreground_mask.
SATURATION_THRESHOLD = 25
 
# Kernel for the median filter applied to saturation before thresholding
# (kills single-pixel sensor/JPEG noise so it can't masquerade as material).
SATURATION_MEDIAN_KSIZE = 5
 
# Gaussian pre-smoothing before the Laplacian. Laplacian-based focus
# measures are the most noise-sensitive family (Pertuz et al. 2013), and
# raw pixel noise inflates the variance enough to make a blank tile look
# "sharp". A small blur removes that while keeping real cell edges.
BLUR_GAUSSIAN_KSIZE = 3
 
# How far (px) to pull the background region away from cell borders before
# measuring debris, so anti-aliased cell edges don't register as texture.
BACKGROUND_ERODE_PX = 5
 
 
def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as a BGR array (OpenCV's default color order)."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
 
 
# ---------------------------------------------------------------------------
# 1. Image-level foreground mask
# ---------------------------------------------------------------------------
 
def compute_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Boolean mask of cellular/stained material for the WHOLE image.
 
    Method: convert to HSV, median-filter the saturation channel, then apply
    a fixed saturation threshold. Saturation-based masking is the standard
    approach for separating stained material from glass in whole-slide
    analysis (e.g. CLAM, HistoQC, and most CAMELYON entries), because glass is
    colourless and stain is not - which holds regardless of how dark or pale
    the cells are.
 
    Why one mask per IMAGE and a FIXED threshold, rather than per-tile Otsu?
    The previous implementation ran Otsu on each tile's grayscale. Otsu
    assumes a bimodal histogram (material vs background). On an empty tile
    the histogram is unimodal (just background noise), and Otsu still forces
    a split - which classified ~50% of a blank tile as "cellular material".
    Any per-tile adaptive method has this failure mode by construction.
    A single fixed threshold means "cellular material" means the same thing
    in every tile of every image, which is what a triage comparison needs.
 
    Returns:
        boolean array, same height/width as the image, True = material.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    saturation = cv2.medianBlur(saturation, SATURATION_MEDIAN_KSIZE)
    return saturation > SATURATION_THRESHOLD
 
 
# ---------------------------------------------------------------------------
# 2. Tiling
# ---------------------------------------------------------------------------
 
def tile_bounds(height: int, width: int, tile_size: int = DEFAULT_TILE_SIZE):
    """
    Grid of (row, col, x0, y0, x1, y1) tile bounds that covers the ENTIRE
    image, with no pixels left out.
 
    tile_size is a *target*: we pick the number of rows/columns that gets
    tiles closest to tile_size x tile_size, then stretch the grid so the
    edges land exactly on the image borders. A 1376x1020 CRIC image with
    tile_size=256 becomes a 5x4 grid of ~275x255 tiles.
 
    Why not fixed 256x256 tiles and drop the leftover strip? Because that
    silently ignored ~30% of every CRIC image, so the triage verdict wasn't
    actually about the whole sample. All three feature scores are per-pixel
    measures (a fraction, a variance, a std), so a few percent variation in
    tile size doesn't bias them. Bounds are half-open like numpy slicing.
    """
    n_cols = max(1, round(width / tile_size))
    n_rows = max(1, round(height / tile_size))
    x_edges = [round(i * width / n_cols) for i in range(n_cols + 1)]
    y_edges = [round(j * height / n_rows) for j in range(n_rows + 1)]
    assert x_edges[0] == 0 and x_edges[-1] == width, "x grid does not span the image"
    assert y_edges[0] == 0 and y_edges[-1] == height, "y grid does not span the image"
 
    bounds = []
    for row in range(n_rows):
        for col in range(n_cols):
            bounds.append(
                {
                    "row": row,
                    "col": col,
                    "x0": x_edges[col],
                    "y0": y_edges[row],
                    "x1": x_edges[col + 1],
                    "y1": y_edges[row + 1],
                }
            )
    return bounds
 
 
# ---------------------------------------------------------------------------
# 3. Per-tile features
# ---------------------------------------------------------------------------
 
def compute_cellularity(tile_mask: np.ndarray) -> float:
    """Fraction of the tile covered by cellular material, from the image-level mask."""
    return float(tile_mask.mean())
 
 
def compute_blur_score(tile_gray: np.ndarray) -> float:
    """
    Focus/sharpness score: variance of the Laplacian (Pech-Pacheco et al.
    2000), the standard simple focus measure in microscopy autofocus.
 
    Sharp edges -> strong second-derivative response -> high variance.
    Blur smooths edges -> low variance. Higher = sharper.
 
    Two things to know when reading this number:
    - It is CONTENT-DEPENDENT: it measures how much edge detail exists, not
      focus in isolation. A blank tile scores near zero however perfectly
      focused it is. So blur is only meaningful on tiles that contain
      material - triage.py only judges focus on tiles that pass the
      cellularity check.
    - Absolute values depend on resolution, magnification and noise, so the
      threshold in triage.py must be calibrated on this dataset.
 
    We apply a small Gaussian blur first so pixel noise doesn't inflate the
    score (Laplacian operators are the most noise-sensitive focus measures).
    """
    smoothed = cv2.GaussianBlur(tile_gray, (BLUR_GAUSSIAN_KSIZE, BLUR_GAUSSIAN_KSIZE), 0)
    return float(cv2.Laplacian(smoothed, cv2.CV_64F).var())
 
 
def compute_debris_score(tile_gray: np.ndarray, tile_mask: np.ndarray) -> float:
    """
    Proxy for debris/contamination: intensity texture in the NON-cellular
    background, measured as the standard deviation of grayscale within the
    background region after pulling it back from cell borders.
 
    Rationale: clean glass is flat and uniform. Cell fragments, mucus,
    granular debris and haemolysed blood give the background texture even
    where it isn't solid stained material.
 
    Why this replaced the earlier "hue std of background" measure: hue is
    numerically undefined for near-grey pixels - a nearly white pixel flips
    between "reddish" and "bluish" from tiny noise, because there's almost
    no colour to measure. So clean pale background produced a large hue std
    and the cleanest tiles were flagged as debris-laden. Grayscale intensity
    has no such instability.
 
    Limitation (be honest about this when presenting): this detects
    *texture*, not blood specifically. Intact red blood cells in Pap stain
    are pinkish/orange, but so is the cytoplasm of superficial squamous
    cells, so a colour-based "blood" flag would fire on healthy cells.
    Reliable blood-vs-cell separation needs shape cues or labels; it's out
    of scope for this rule-based PoC.
 
    Returns 0.0 if there's no background left to evaluate.
    """
    background = ~tile_mask
    # Erode the background so the fuzzy edge around cells isn't counted.
    kernel = np.ones((BACKGROUND_ERODE_PX * 2 + 1,) * 2, np.uint8)
    background = cv2.erode(background.astype(np.uint8), kernel).astype(bool)
    if background.sum() == 0:
        return 0.0
    return float(tile_gray[background].std())
 
 
def extract_tile_features(tile_bgr: np.ndarray, tile_mask: np.ndarray) -> dict:
    """Compute all three feature scores for a single tile."""
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "cellularity": compute_cellularity(tile_mask),
        "blur_score": compute_blur_score(tile_gray),
        "debris_score": compute_debris_score(tile_gray, tile_mask),
    }
 
 
# ---------------------------------------------------------------------------
# 4. Orchestration
# ---------------------------------------------------------------------------
 
def process_image(image_path: Path, tile_size: int) -> pd.DataFrame:
    """Extract tile-level features for one image, returning a DataFrame (one row per tile)."""
    image = load_image(image_path)
    mask = compute_foreground_mask(image)
    height, width = image.shape[:2]
 
    rows = []
    for b in tile_bounds(height, width, tile_size):
        tile = image[b["y0"] : b["y1"], b["x0"] : b["x1"]]
        tile_mask = mask[b["y0"] : b["y1"], b["x0"] : b["x1"]]
        rows.append({"image": image_path.name, **b, **extract_tile_features(tile, tile_mask)})
    return pd.DataFrame(rows)
 
 
def process_dataset(input_dir: Path, tile_size: int, limit: int = None) -> pd.DataFrame:
    """Run feature extraction over every image in input_dir, returning one combined DataFrame."""
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}. Did you run download_data.py first?")
 
    print(f"Found {len(image_paths)} images. Extracting tile features (tile_size={tile_size})...")
 
    frames = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            frames.append(process_image(image_path, tile_size))
        except ValueError as e:
            print(f"  [warn] skipping {image_path.name}: {e}")
 
    if not frames:
        raise RuntimeError("Every image failed to load - nothing to write.")
    return pd.concat(frames, ignore_index=True)
 
 
def main():
    parser = argparse.ArgumentParser(description="Extract tile-based CV features from CRIC images.")
    parser.add_argument("--input", type=str, default="data/raw", help="Directory of downloaded images")
    parser.add_argument("--output", type=str, default="data/processed/features.csv", help="Output CSV path")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Target tile size in pixels")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (for testing)")
    args = parser.parse_args()
 
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    features_df = process_dataset(Path(args.input), args.tile_size, args.limit)
    features_df.to_csv(output_path, index=False)
 
    print(f"Done. {len(features_df)} tile rows from {features_df['image'].nunique()} images.")
    print(f"Saved to {output_path}")
    print("\nQuick preview:")
    print(features_df[["cellularity", "blur_score", "debris_score"]].describe())
 
 
if __name__ == "__main__":
    main()
 
