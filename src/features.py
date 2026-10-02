"""
Feltbaseret feature-udtræk til cytologi-triagering (Case 2).

For hvert billede:
1. Bygges ÉN maske over cellulært materiale for hele billedet (se
   compute_foreground_mask - alt andet hviler på den).
2. Deles billedet i et gitter af felter, der dækker hver eneste pixel.
3. Beregnes tre "light AI"-signaler pr. felt (klassisk billedanalyse, ingen
   black box):
   - cellularity   : andel af feltet, der er dækket af cellulært materiale
   - blur_score    : hvor skarpt feltet er (varians af Laplace-operatoren)
   - debris_score  : hvor teksturfyldt/støjende den ikke-cellulære baggrund er
4. Gemmes én række pr. felt i en CSV, som triage.py derefter tærskler.

Alle tærskler til *vurdering* ligger i triage.py. Konstanterne herunder styrer,
*hvordan* der måles, og er markeret KALIBRÉR, hvor den rigtige værdi afhænger
af datasættet og bør tjekkes på rigtige billeder.

Metoderne er ikke oversat direkte fra én artikel - det er standardbyggesten
fra klassisk billedanalyse, sat sammen med inspiration fra litteraturen:

Referencer
  [1] Pech-Pacheco et al. (2000). "Diatom autofocusing in brightfield
      microscopy: a comparative study." ICPR. - Varians af Laplace som fokusmål.
  [2] Pertuz, Puig & Garcia (2013). "Analysis of focus measure operators for
      shape-from-focus." Pattern Recognition 46(5). - Sammenligning af
      fokusmål; Laplace-baserede er de mest støjfølsomme.
  [3] Lu et al. (2021). "Data-efficient and weakly supervised computational
      pathology on whole-slide images" (CLAM). Nature Biomedical Engineering.
      - Mætningsbaseret maske til at finde væv/materiale på helglas.
  [4] Janowczyk et al. (2019). "HistoQC: An open-source quality control tool
      for digital pathology slides." JCO Clinical Cancer Informatics.
      - Kvalitetskontrol af digitale glas, bl.a. mætning og fokus.
  [5] Thyroidea-cytologistudie med Laplace-varians pr. felt (512x512) på
      ~100.000 felter fra 36 helglas - grundlag for at måle fokus pr. felt
      frem for pr. billede. (Præcis reference skal verificeres før brug i
      rapport.)
  [6] "Automated quality evaluation of cervical cytopathology WSIs based on
      content analysis" (2025) - kvalitet vurderet på flere dimensioner
      (celleantal, farvning, materiale-andel), ikke ét tal.
  [7] Nayar & Wilbur (red.) (2015). "The Bethesda System for Reporting
      Cervical Cytology", 3. udg. - Egnethedskriterier; bruges til
      tærsklerne i triage.py.

Brug:
    python src/features.py
    python src/features.py --input data/raw --output data/processed/features.csv --tile-size 256
    python src/features.py --limit 5   # hurtig test på de første 5 billeder
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_TILE_SIZE = 256

# --- Målekonstanter (KALIBRÉR på rigtige CRIC-billeder) -----------------------
# Mætning (HSV, 0-255), over hvilken en pixel regnes som farvet materiale.
# Rent glas / monteringsmedium er næsten gråt (meget lav mætning); alt
# Pap-farvet - kerner, cytoplasma, det meste debris - har farve. Tærsklen er
# med vilje FAST: en adaptiv (Otsu) ville vælge sig selv om på hvert billede,
# så et tomt billede får en vilkårlig opdeling, og cellularitet holder op med
# at kunne sammenlignes på tværs af billeder. Se compute_foreground_mask.
# Mætningsbaseret maskering efter [3] og [4].
SATURATION_THRESHOLD = 25

# Kernestørrelse til medianfilteret, der køres på mætningen før tærskling
# (fjerner enkeltpixel-støj fra sensor/JPEG, så den ikke tæller som materiale).
SATURATION_MEDIAN_KSIZE = 5

# Gaussisk udglatning før Laplace. Laplace-baserede fokusmål er den mest
# støjfølsomme familie [2], og rå pixelstøj puster variansen
# nok op til, at et tomt felt ser "skarpt" ud. En lille udglatning fjerner
# det, men bevarer rigtige cellekanter.
BLUR_GAUSSIAN_KSIZE = 3

# Hvor langt (px) baggrundsområdet trækkes væk fra cellekanterne, før debris
# måles, så udjævnede kanter omkring celler ikke registreres som tekstur.
BACKGROUND_ERODE_PX = 5


def load_image(path: Path) -> np.ndarray:
    """Indlæs et billede fra disk som BGR-array (OpenCV's standard farverækkefølge)."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


# ---------------------------------------------------------------------------
# 1. Forgrundsmaske for hele billedet
# ---------------------------------------------------------------------------

def compute_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Boolsk maske over cellulært/farvet materiale for HELE billedet.

    Metode: konvertér til HSV, medianfiltrér mætningskanalen og anvend en fast
    mætningstærskel. Mætningsbaseret maskering er standardtilgangen til at
    adskille farvet materiale fra glas i helglas-analyse (CLAM [3], HistoQC
    [4] og de fleste CAMELYON-bidrag), fordi glas er farveløst og farvning
    ikke er - uanset hvor mørke eller lyse cellerne er.

    Hvorfor én maske pr. BILLEDE med FAST tærskel frem for Otsu pr. felt?
    Den tidligere implementering kørte Otsu på hvert felts gråtoner. Otsu
    forudsætter et bimodalt histogram (materiale vs. baggrund). På et tomt
    felt er histogrammet unimodalt (kun baggrundsstøj), og Otsu tvinger
    alligevel en opdeling igennem - hvilket klassificerede ~50 % af et blankt
    felt som "cellulært materiale". Enhver adaptiv metode pr. felt har den
    fejl indbygget. Én fast tærskel betyder, at "cellulært materiale" betyder
    det samme i hvert felt i hvert billede - og det er, hvad en
    triage-sammenligning kræver.

    Returnerer:
        boolsk array med samme højde/bredde som billedet, True = materiale.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    saturation = cv2.medianBlur(saturation, SATURATION_MEDIAN_KSIZE)
    return saturation > SATURATION_THRESHOLD


# ---------------------------------------------------------------------------
# 2. Opdeling i felter
# ---------------------------------------------------------------------------

def tile_bounds(height: int, width: int, tile_size: int = DEFAULT_TILE_SIZE):
    """
    Gitter af feltgrænser (row, col, x0, y0, x1, y1), der dækker HELE
    billedet uden at udelade pixels.

    tile_size er et *mål*: vi vælger det antal rækker/kolonner, der giver
    felter tættest på tile_size x tile_size, og strækker derefter gitteret,
    så kanterne lander præcis på billedets kanter. Et CRIC-billede på
    1376x1020 med tile_size=256 bliver et 5x4-gitter af ~275x255-felter.

    Hvorfor måle pr. felt og ikke pr. billede? Fordi cytologipræparater er
    ujævne - celler, blod og fokus varierer hen over glasset - så ét tal for
    hele billedet udjævner præcis de lokale problemer, vi vil finde. Det er
    den gængse tilgang i kvalitetsvurdering af helglas, fx [5].

    Hvorfor ikke faste 256x256-felter og smide resten væk? Fordi det i
    stilhed ignorerede ~30 % af hvert CRIC-billede, så triage-vurderingen
    reelt ikke handlede om hele prøven. Alle tre feature-mål er pr.-pixel-mål
    (en andel, en varians, en standardafvigelse), så nogle få procents
    variation i feltstørrelse skævvrider dem ikke. Grænserne er halvåbne
    ligesom numpy-slicing.
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
# 3. Features pr. felt
# ---------------------------------------------------------------------------

def compute_cellularity(tile_mask: np.ndarray) -> float:
    """Andel af feltet dækket af cellulært materiale, ud fra billedmasken."""
    return float(tile_mask.mean())


def compute_blur_score(tile_gray: np.ndarray) -> float:
    """
    Fokus-/skarphedsmål: varians af Laplace-operatoren [1], det almindelige
    simple fokusmål i mikroskopi-autofokus, her beregnet pr. felt som i [5].

    Skarpe kanter -> kraftigt andenafledt-respons -> høj varians.
    Uskarphed udjævner kanter -> lav varians. Højere = skarpere.

    To ting at vide, når man læser tallet:
    - Det er INDHOLDSAFHÆNGIGT: det måler, hvor mange kantdetaljer der er,
      ikke fokus isoleret set. Et tomt felt scorer nær nul, uanset hvor
      perfekt fokuseret det er. Derfor giver fokus kun mening på felter med
      materiale - triage.py vurderer kun fokus på felter, der består
      cellularitets-tjekket.
    - Absolutte værdier afhænger af opløsning, forstørrelse og støj, så
      tærsklen i triage.py skal kalibreres på dette datasæt.

    Vi lægger en lille Gaussisk udglatning på først, så pixelstøj ikke puster
    scoren op (Laplace-operatorer er de mest støjfølsomme fokusmål [2]).
    """
    smoothed = cv2.GaussianBlur(tile_gray, (BLUR_GAUSSIAN_KSIZE, BLUR_GAUSSIAN_KSIZE), 0)
    return float(cv2.Laplacian(smoothed, cv2.CV_64F).var())


def compute_debris_score(tile_gray: np.ndarray, tile_mask: np.ndarray) -> float:
    """
    Proxy for debris/forurening: intensitetstekstur i den IKKE-cellulære
    baggrund, målt som standardafvigelsen af gråtonen i baggrundsområdet
    efter at det er trukket tilbage fra cellekanterne.

    Begrundelse: rent glas er fladt og ensartet. Cellefragmenter, slim,
    kornet debris og hæmolyseret blod giver baggrunden tekstur, også hvor
    der ikke er solidt farvet materiale. At baggrundsstøj indgår som sin egen
    kvalitetsdimension ved siden af cellemængde og fokus følger tanken i [6].
    Selve målet (gråtone-std i eroderet baggrund) er vores eget valg.

    Hvorfor dette erstattede det tidligere mål "hue-std i baggrunden": hue
    (farvetone) er numerisk udefineret for næsten grå pixels - en næsten hvid
    pixel skifter mellem "rødlig" og "blålig" ved den mindste støj, fordi der
    næsten ingen farve er at måle. Så ren lys baggrund gav en høj hue-std, og
    de reneste felter blev markeret som debris-fyldte. Gråtoneintensitet har
    ikke den ustabilitet.

    Begrænsning (vær ærlig om det ved fremlæggelse): dette fanger *tekstur*,
    ikke blod specifikt. Intakte røde blodlegemer i Pap-farvning er
    lyserøde/orange, men det er cytoplasmaet i superficielle pladeceller
    også, så et farvebaseret "blod"-flag ville slå ud på raske celler.
    Pålidelig adskillelse af blod og celler kræver formtræk eller labels; det
    er uden for scope for denne regelbaserede PoC.

    Returnerer 0.0, hvis der ikke er baggrund tilbage at vurdere.
    """
    background = ~tile_mask
    # Erodér baggrunden, så den udjævnede kant omkring celler ikke tælles med.
    kernel = np.ones((BACKGROUND_ERODE_PX * 2 + 1,) * 2, np.uint8)
    background = cv2.erode(background.astype(np.uint8), kernel).astype(bool)
    if background.sum() == 0:
        return 0.0
    return float(tile_gray[background].std())


def extract_tile_features(tile_bgr: np.ndarray, tile_mask: np.ndarray) -> dict:
    """Beregn alle tre feature-mål for ét felt."""
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "cellularity": compute_cellularity(tile_mask),
        "blur_score": compute_blur_score(tile_gray),
        "debris_score": compute_debris_score(tile_gray, tile_mask),
    }


# ---------------------------------------------------------------------------
# 4. Orkestrering
# ---------------------------------------------------------------------------

def process_image(image_path: Path, tile_size: int) -> pd.DataFrame:
    """Udtræk feltfeatures for ét billede; returnerer en DataFrame (én række pr. felt)."""
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
    """Kør feature-udtræk på alle billeder i input_dir; returnerer én samlet DataFrame."""
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