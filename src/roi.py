"""
Rangering af interesseområder (ROI) til Case 1 ("Scopio-light": hvor er der
noget at kigge på?).

Genbruger målingerne pr. felt fra features.py. Der bliver ikke klassificeret
celletyper eller stillet nogen diagnoser her - et felt er enten værd at kigge
på eller ikke, og dem, der er værd at kigge på, sættes i den rækkefølge, de
skal gennemgås i.

Pr. felt:
  is_roi        - feltet indeholder farvet materiale (cellularitet over en
                  lav bundgrænse). Alt andet er baggrund/glas, som patologen
                  kan springe over.
  in_focus      - fokustjek, så uskarpe områder bliver gennemgået til sidst
  priority_rank - 1 = kig her først. ROI-felter rangeres efter cellularitet
                  (mest materiale først), og ROI-felter, der er ude af fokus,
                  ryger bagerst i listen. Felter, der ikke er ROI, får ingen rang.

Pr. billede:
  roi_tiles, total_tiles, pct_roi_area - hvor meget af synsfeltet, der skal gennemgås.

Case 1 skal bruge en finere lokalisering end Case 2, så husk at køre
features.py med en mindre feltstørrelse og ud i sin egen fil først:
    python src/features.py --tile-size 128 --output data/processed/features_case1.csv
    python src/roi.py

Tærsklerne er pladsholdere og skal kalibreres på rigtige billeder via appen.
"""

import argparse
from pathlib import Path

import pandas as pd

ROI_CELLULARITY_MIN = 0.03   # lavere end Case 2's grænse for brugbare felter: "noget som helst at kigge på"
ROI_BLUR_MIN = 50.0          # samme fokusmål som i Case 2; under grænsen = gennemgås til sidst

DEFAULT_INPUT = Path("data/processed/features_case1.csv")
DEFAULT_OUTPUT = Path("data/processed/roi_results.csv")


def rank_tiles(
    df: pd.DataFrame,
    cellularity_min: float = ROI_CELLULARITY_MIN,
    blur_min: float = ROI_BLUR_MIN,
) -> pd.DataFrame:
    """Tilføjer is_roi, in_focus og priority_rank (pr. billede) til feltrækkerne."""
    df = df.copy()
    df["is_roi"] = df["cellularity"] >= cellularity_min
    df["in_focus"] = df["blur_score"] >= blur_min

    # Sortér, så det inden for hvert billede er ROI-felter først, felter i fokus
    # før dem ude af fokus, og så dem med mest materiale først. Til sidst
    # nummererer vi dem.
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