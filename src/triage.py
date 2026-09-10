"""
Rule-based triage scoring for Case 2 (sample suitability).

Takes the tile-level features produced by features.py and turns them into a
per-image triage decision: egnet (suitable) / suboptimal / uegnet (unsuitable).

This is deliberately NOT a trained classifier - just thresholds applied to
the CV features, so every decision can be explained in plain language
("this image was flagged uegnet because only 8% of tiles had usable cell
material"). That's the whole point of the "light AI" approach: transparent,
inspectable, no black box.

IMPORTANT: the threshold values below (TILE_*, IMAGE_*) are placeholder
starting points, not calibrated against real pathologist judgement yet.
Once we've eyeballed real CRIC tiles against their scores, we should come
back and tune these - see the note in main().

Two-step logic:
1. TILE level: is this individual tile "usable" - does it have enough
   cellular material, is it in focus, and is it clean enough?
2. IMAGE level: given the % of usable tiles in the image, bucket the whole
   image into egnet / suboptimal / uegnet.

Usage:
    python src/triage.py
    python src/triage.py --input data/processed/features.csv --output data/processed/triage_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd

# --- Tile-level thresholds -------------------------------------------------
# A tile counts as "usable" only if it passes all three checks.
TILE_CELLULARITY_MIN = 0.05   # at least 5% of the tile is cellular material
TILE_BLUR_MIN = 50.0          # Laplacian variance - below this is considered out of focus
TILE_DEBRIS_MAX = 30.0        # background hue std - above this is considered too noisy/contaminated

# --- Image-level thresholds -------------------------------------------------
# Based on the *fraction* of usable tiles in the image.
IMAGE_SUITABLE_MIN = 0.50     # >=50% usable tiles -> egnet (suitable)
IMAGE_UNSUITABLE_MAX = 0.15   # <15% usable tiles -> uegnet (unsuitable)
# anything in between -> suboptimal


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"image", "row", "col", "cellularity", "blur_score", "debris_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"features file is missing expected columns: {missing}")
    return df


def classify_tiles(
    df: pd.DataFrame,
    cellularity_min: float = TILE_CELLULARITY_MIN,
    blur_min: float = TILE_BLUR_MIN,
    debris_max: float = TILE_DEBRIS_MAX,
) -> pd.DataFrame:
    """Add a boolean 'tile_usable' column: True if the tile passes all three checks."""
    df = df.copy()
    df["tile_usable"] = (
        (df["cellularity"] >= cellularity_min)
        & (df["blur_score"] >= blur_min)
        & (df["debris_score"] <= debris_max)
    )
    return df


def aggregate_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse tile-level rows into one row per image with summary stats."""
    grouped = df.groupby("image").agg(
        total_tiles=("tile_usable", "size"),
        usable_tiles=("tile_usable", "sum"),
        mean_cellularity=("cellularity", "mean"),
        mean_blur_score=("blur_score", "mean"),
        mean_debris_score=("debris_score", "mean"),
    )
    grouped["pct_usable_tiles"] = grouped["usable_tiles"] / grouped["total_tiles"]
    return grouped.reset_index()


def classify_images(
    image_df: pd.DataFrame,
    suitable_min: float = IMAGE_SUITABLE_MIN,
    unsuitable_max: float = IMAGE_UNSUITABLE_MAX,
) -> pd.DataFrame:
    """Add a 'triage' column: 'egnet' / 'suboptimal' / 'uegnet', based on pct_usable_tiles."""
    df = image_df.copy()

    def _label(pct_usable: float) -> str:
        if pct_usable >= suitable_min:
            return "egnet"
        if pct_usable < unsuitable_max:
            return "uegnet"
        return "suboptimal"

    df["triage"] = df["pct_usable_tiles"].apply(_label)
    return df


def run_triage(features_path: Path) -> pd.DataFrame:
    tile_df = load_features(features_path)
    tile_df = classify_tiles(tile_df)
    image_df = aggregate_image_scores(tile_df)
    image_df = classify_images(image_df)
    return image_df


def main():
    parser = argparse.ArgumentParser(description="Triage CRIC images as egnet/suboptimal/uegnet.")
    parser.add_argument("--input", type=str, default="data/processed/features.csv", help="Path to tile features CSV")
    parser.add_argument("--output", type=str, default="data/processed/triage_results.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    results = run_triage(input_path)
    results.to_csv(output_path, index=False)

    print(f"Triaged {len(results)} images.")
    print(results["triage"].value_counts())
    print(f"\nSaved to {output_path}")
    print("\nNOTE: thresholds are placeholder starting points (see top of triage.py).")
    print("Compare a handful of egnet/suboptimal/uegnet images visually before trusting these.")


if __name__ == "__main__":
    main()