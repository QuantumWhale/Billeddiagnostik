"""

Regelbaseret triagescoring til Case 2 (er prøven egnet?).
 
Tager de feltvise features, som features.py laver, og omsætter dem til en
triage-beslutning pr. billede: egnet / suboptimal / uegnet.
 
Det er med vilje IKKE en trænet klassifikator - bare tærskler lagt på
CV-featurene, så hver eneste beslutning kan forklares i almindeligt sprog
("markeret uegnet, fordi kun 15 % af felterne havde brugbart cellemateriale").
Det er hele pointen med "light AI"-tilgangen: den er gennemsigtig, og man kan
selv gå ind og se efter.
 
To-trins logik:
1. FELT-niveau - hvert felt vurderes i rækkefølge:
     a) nok cellemateriale?          (ellers: fejler på C)
     b) hvis ja, er det i fokus?     (ellers: fejler på B)
     c) hvis ja, er baggrunden ren?  (ellers: fejler på D)
   Fokus og debris vurderes kun på felter, der har materiale. Et tomt felt er
   bare tomt, og at kalde det "ude af fokus" giver ingen mening (et blankt felt
   scorer altid ~0 på ethvert fokusmål) og ville bare forvirre gennemgangen.
2. BILLED-niveau - andelen af brugbare felter afgør, hvilken kategori billedet havner i.
 
Tærsklerne på billedniveau følger Bethesda-systemets logik for egnethed:
  - en prøve, hvor >75 % af cellerne er skjult, er "unsatisfactory"
  - 50-75 % skjult rapporteres som "satisfactory but partially obscured"
Altså: <25 % brugbart -> uegnet, 25-50 % -> suboptimal, >=50 % -> egnet.
Forbehold: Bethesda tæller andelen af *celler*, der er skjult, mens vi bruger
andelen af *feltareal*, der er brugbart, som proxy. Derudover er
CRIC-billederne enkelte synsfelter fra mikroskopet og ikke hele glas, så i en
rigtig implementering skulle de her scorer pr. synsfelt samles op over hele
præparatet.
 
TILE_*-tærsklerne er pladsholdere, der skal kalibreres på rigtige CRIC-billeder:
Laplace-variansen afhænger af opløsning og støj, og debris-skalaen afhænger af
farvningen. Brug gennemgangsappen til at se, hvad de markerer, og justér så.
 
Brug:
    python src/triage.py
    python src/triage.py --input data/processed/features.csv --output data/processed/triage_results.csv
"""
 
import argparse
from pathlib import Path
 
import pandas as pd
 
# --- Tærskler på feltniveau (CALIBRATE) -------------------------------------
TILE_CELLULARITY_MIN = 0.05   # >=5 % af feltet er farvet materiale
TILE_BLUR_MIN = 50.0          # Laplace-varians under det her = ude af fokus (vurderes kun på felter med celler)
TILE_DEBRIS_MAX = 12.0        # gråtone-std i baggrunden over det her = for meget debris (kun felter med celler)
 
# --- Tærskler på billedniveau (følger Bethesda) -------------------------------
IMAGE_SUITABLE_MIN = 0.50     # >=50 % brugbare felter -> egnet
IMAGE_UNSUITABLE_MAX = 0.25   # <25 % brugbare felter -> uegnet  (dvs. >75 % ikke brugbare)
# 25-50 % -> suboptimal
 
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
    Tilføjer kolonner med vurderingen af hvert felt:
      has_cells   - har bestået cellularitets-tjekket
      in_focus    - har bestået fokus-tjekket  (giver kun mening, hvor has_cells)
      is_clean    - har bestået debris-tjekket (giver kun mening, hvor has_cells)
      tile_usable - alle tre
      reasons     - hvilke tjek der fejlede, som bogstaver: C = cellularitet,
                    B = blur (uskarphed), D = debris. B/D rapporteres kun,
                    når feltet har celler, da de ellers ikke er defineret.
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
    Slår feltrækkerne sammen til én række pr. billede.
 
    Gennemsnittene for fokus og debris tages kun over felter MED CELLER - hvis
    man tog de tomme felter med, ville de trække fokus ned uden at sige noget
    om selve prøven.
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
    per_image = per_image.join(cellular_stats)  # NaN, hvis et billede ikke har nogen felter med celler
    per_image["pct_cellular_tiles"] = per_image["cellular_tiles"] / per_image["total_tiles"]
    per_image["pct_usable_tiles"] = per_image["usable_tiles"] / per_image["total_tiles"]
    return per_image.reset_index()
 
 
def classify_images(
    image_df: pd.DataFrame,
    suitable_min: float = IMAGE_SUITABLE_MIN,
    unsuitable_max: float = IMAGE_UNSUITABLE_MAX,
) -> pd.DataFrame:
    """Tilføjer en 'triage'-kolonne: 'egnet' / 'suboptimal' / 'uegnet', ud fra pct_usable_tiles."""
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
    Tilføjer kolonnerne 'human_eval' og 'human_eval_at' ud fra de manuelle
    labels, som gennemgangsappen gemmer (data/processed/manual_labels.csv), og
    derudover 'agrees' (True/False, hvor begge vurderinger findes, ellers tom).
 
    Husk at det er manual_labels.csv, der er den varige log: triage_results.csv
    bygges forfra, hver gang scriptet her køres, så de menneskelige vurderinger
    gemmes for sig og bliver sat på igen her. Billeder, der ikke er vurderet
    endnu, får tomme værdier.
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
 
