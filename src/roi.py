"""
Region-of-interest ranking for Case 1 ("Scopio-light": where is there
something to look at?).

Reuses the per-tile measurements from features.py. Nothing here classifies
cell types or diagnoses anything - a tile is either worth a look or it isn't,
and the ones worth a look are put in a review order.

Per tile:
  is_roi        - the tile contains stained material (cellularity above a
                  low floor). Everything else is background/glass the
                  pathologist can skip.
  in_focus      - focus check, so blurry regions are reviewed last
  priority_rank - 1 = look here first. ROI tiles are ranked by cellularity
                  (most material first); out-of-focus ROI tiles go to the end
                  of the list. Non-ROI tiles have no rank.

Per image:
  roi_tiles, total_tiles, pct_roi_area - how much of the field needs review.

Case 1 wants finer localisation than Case 2, so run features.py with a
smaller tile size into its own file first:
    python src/features.py --tile-size 128 --output data/processed/features_case1.csv
    python src/roi.py

Thresholds are placeholders - calibrate on real images via the app.
"""

import argparse
from pathlib import Path

import pandas as pd

ROI_CELLULARITY_MIN = 0.03   # lower than Case 2's usable-tile floor: "anything to look at"
ROI_BLUR_MIN = 50.0          # same focus measure as Case 2; below = reviewed last

DEFAULT_INPUT = Path("data/processed/features_case1.csv")
DEFAULT_OUTPUT = Path("data/processed/roi_results.csv")


def rank_tiles(
    df: pd.DataFrame,
    cellularity_min: float = ROI_CELLULARITY_MIN,
    blur_min: float = ROI_BLUR_MIN,
) -> pd.DataFrame:
    """Add is_roi, in_focus and priority_rank (per image) to the tile rows."""
    df = df.copy()
    df["is_roi"] = df["cellularity"] >= cellularity_min
    df["in_focus"] = df["blur_score"] >= blur_min

    # Sort so that, within each image: ROI tiles first, in-focus before
    # out-of-focus, then most cellular first. Then number them.
    ordered = df.sort_values(
        ["image", "is_roi", "in_focus", "cellularity"],
        ascending=[True, False, False, False],
    )
    ordered["priority_rank"] = ordered.groupby("image").cumcount() + 1
    ordered.loc[~ordered["is_roi"], "priority_rank"] = pd.NA
    return ordered.sort_values(["image", "row", "col"]).reset_index(drop=True)


def summarise_images(df: pd.DataFrame) -> pd.DataFrame:
    per_image = df.groupby("image").agg(
        total_tiles=("is_roi", "size"),
        roi_tiles=("is_roi", "sum"),
        roi_in_focus=("in_focus", lambda s: int((s & df.loc[s.index, "is_roi"]).sum())),
    )
    per_image["pct_roi_area"] = per_image["roi_tiles"] / per_image["total_tiles"]
    return per_image.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Rank tiles by review priority (Case 1).")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Tile features CSV (fine grid)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Create it with:\n"
            "  python src/features.py --tile-size 128 --output data/processed/features_case1.csv"
        )

    ranked = rank_tiles(pd.read_csv(input_path))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)

    summary = summarise_images(ranked)
    print(f"Ranked {len(ranked)} tiles across {len(summary)} images.")
    print(f"Mean ROI area per image: {summary['pct_roi_area'].mean():.1%} "
          f"(min {summary['pct_roi_area'].min():.1%}, max {summary['pct_roi_area'].max():.1%})")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()