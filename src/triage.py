"""

Rule-based triage scoring for Case 2 (sample suitability).
 
Takes the tile-level features produced by features.py and turns them into a
per-image triage decision: egnet (suitable) / suboptimal / uegnet (unsuitable).
 
This is deliberately NOT a trained classifier - just thresholds applied to
the CV features, so every decision can be explained in plain language
("flagged uegnet because only 15% of tiles had usable cell material").
That's the point of the "light AI" approach: transparent, inspectable.
 
Two-step logic:
1. TILE level - each tile is judged in order:
     a) enough cellular material?        (else: fails on C)
     b) if so, is it in focus?           (else: fails on B)
     c) if so, is the background clean?  (else: fails on D)
   Focus and debris are only assessed on tiles that have material. An empty
   tile is simply empty - calling it "out of focus" is meaningless (a blank
   tile always scores ~0 on any focus measure) and would mislead the review.
2. IMAGE level - the share of usable tiles decides the bucket.
 
Image-level thresholds follow the Bethesda System's adequacy logic:
  - a specimen with >75% of cells obscured is "unsatisfactory"
  - 50-75% obscured is reported as "satisfactory but partially obscured"
So: <25% usable -> uegnet, 25-50% -> suboptimal, >=50% -> egnet.
Caveat: Bethesda counts the fraction of *cells* obscured; we use the
fraction of *tile area* that is usable as a proxy. Also, CRIC images are
single microscope fields, not whole slides - in a real deployment these
per-field scores would be aggregated across the whole preparation.
 
The TILE_* thresholds are placeholders to calibrate on real CRIC images:
Laplacian variance depends on resolution and noise, and the debris scale
depends on staining. Use the review app to see what they flag and adjust.
 
Usage:
    python src/triage.py
    python src/triage.py --input data/processed/features.csv --output data/processed/triage_results.csv
"""
 
import argparse
from pathlib import Path
 
import pandas as pd
 
# --- Tile-level thresholds (CALIBRATE) --------------------------------------
TILE_CELLULARITY_MIN = 0.05   # >=5% of the tile is stained material
TILE_BLUR_MIN = 50.0          # Laplacian variance below this = out of focus (judged on cellular tiles only)
TILE_DEBRIS_MAX = 12.0        # background grayscale std above this = too much debris (cellular tiles only)
 
# --- Image-level thresholds (Bethesda-aligned) --------------------------------
IMAGE_SUITABLE_MIN = 0.50     # >=50% usable tiles -> egnet
IMAGE_UNSUITABLE_MAX = 0.25   # <25% usable tiles  -> uegnet  (i.e. >75% not usable)
# 25-50% -> suboptimal
 
REQUIRED_COLUMNS = {"image", "row", "col", "cellularity", "blur_score", "debris_score"}
 
 
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"features file is missing expected columns: {missing}")
    return df
 
 
def classify_tiles(
    df: pd.DataFrame,
    cellularity_min: float = TILE_CELLULARITY_MIN,
    blur_min: float = TILE_BLUR_MIN,
    debris_max: float = TILE_DEBRIS_MAX,
) -> pd.DataFrame:
    """
    Add per-tile judgement columns:
      has_cells   - passed the cellularity check
      in_focus    - passed the blur check   (only meaningful where has_cells)
      is_clean    - passed the debris check (only meaningful where has_cells)
      tile_usable - all three
      reasons     - which checks failed, as letters: C = cellularity,
                    B = blur, D = debris. B/D are only reported when the
                    tile has cells, since they're undefined otherwise.
    """
    df = df.copy()
    df["has_cells"] = df["cellularity"] >= cellularity_min
    df["in_focus"] = df["blur_score"] >= blur_min
    df["is_clean"] = df["debris_score"] <= debris_max
    df["tile_usable"] = df["has_cells"] & df["in_focus"] & df["is_clean"]
 
    def _reasons(row) -> str:
        if not row["has_cells"]:
            return "C"
        failed = []
        if not row["in_focus"]:
            failed.append("B")
        if not row["is_clean"]:
            failed.append("D")
        return ",".join(failed)
 
    df["reasons"] = df.apply(_reasons, axis=1)
    return df
 
 
def aggregate_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse tile rows into one row per image.
 
    Focus and debris means are taken over CELLULAR tiles only - averaging in
    empty tiles would drag focus down and say nothing about the sample.
    """
    cellular = df[df["has_cells"]]
    per_image = df.groupby("image").agg(
        total_tiles=("tile_usable", "size"),
        cellular_tiles=("has_cells", "sum"),
        usable_tiles=("tile_usable", "sum"),
        mean_cellularity=("cellularity", "mean"),
    )
    cellular_stats = cellular.groupby("image").agg(
        mean_blur_cellular=("blur_score", "mean"),
        mean_debris_cellular=("debris_score", "mean"),
    )
    per_image = per_image.join(cellular_stats)  # NaN where an image has no cellular tiles
    per_image["pct_cellular_tiles"] = per_image["cellular_tiles"] / per_image["total_tiles"]
    per_image["pct_usable_tiles"] = per_image["usable_tiles"] / per_image["total_tiles"]
    return per_image.reset_index()
 
 
def classify_images(
    image_df: pd.DataFrame,
    suitable_min: float = IMAGE_SUITABLE_MIN,
    unsuitable_max: float = IMAGE_UNSUITABLE_MAX,
) -> pd.DataFrame:
    """Add a 'triage' column: 'egnet' / 'suboptimal' / 'uegnet', from pct_usable_tiles."""
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
    tile_df = classify_tiles(load_features(features_path))
    return classify_images(aggregate_image_scores(tile_df))
 
 
DEFAULT_LABELS_PATH = Path("data/processed/manual_labels.csv")
 
 
def attach_human_eval(image_df: pd.DataFrame, labels_path: Path = DEFAULT_LABELS_PATH) -> pd.DataFrame:
    """
    Add 'human_eval' and 'human_eval_at' columns from the manual labels the
    review app saves (data/processed/manual_labels.csv), plus 'agrees'
    (True/False where both verdicts exist, blank otherwise).
 
    manual_labels.csv is the durable record: triage_results.csv is rebuilt
    from scratch on every run of this script, so human judgements are kept
    separately and re-attached here. Images not yet reviewed get blanks.
    """
    df = image_df.drop(columns=["human_eval", "human_eval_at", "agrees"], errors="ignore")
    if not labels_path.exists():
        df["human_eval"] = pd.NA
        df["human_eval_at"] = pd.NA
        df["agrees"] = pd.NA
        return df
 
    labels = pd.read_csv(labels_path).rename(
        columns={"manual_label": "human_eval", "labeled_at": "human_eval_at"}
    )[["image", "human_eval", "human_eval_at"]]
    df = df.merge(labels, on="image", how="left")
    both = df["human_eval"].notna()
    df["agrees"] = pd.NA
    df.loc[both, "agrees"] = df.loc[both, "triage"] == df.loc[both, "human_eval"]
    return df
 
 
def main():
    parser = argparse.ArgumentParser(description="Triage CRIC images as egnet/suboptimal/uegnet.")
    parser.add_argument("--input", type=str, default="data/processed/features.csv", help="Path to tile features CSV")
    parser.add_argument("--output", type=str, default="data/processed/triage_results.csv", help="Output CSV path")
    args = parser.parse_args()
 
    results = attach_human_eval(run_triage(Path(args.input)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
 
    print(f"Triaged {len(results)} images.")
    print(results["triage"].value_counts())
 
    reviewed = results["human_eval"].notna()
    if reviewed.any():
        n_agree = int(results.loc[reviewed, "agrees"].sum())
        print(f"\nHuman eval attached for {reviewed.sum()} images - agreement {n_agree}/{reviewed.sum()}.")
    else:
        print("\nNo human evaluations yet - label images in the app (streamlit run src/app.py).")
 
    print(f"\nSaved to {args.output}")
    print("\nNOTE: TILE_* thresholds are placeholders - review flagged tiles in the app before trusting them.")
 
 
if __name__ == "__main__":
    main()
 
