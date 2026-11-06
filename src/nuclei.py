"""
Kernebaseret lokalisering til Case 1 ("hvor er der noget at kigge på").
 
I stedet for at lægge et gitter ned over billedet finder vi her selve kernerne
og samler dem, der ligger tæt på hinanden, i klynger, fordi det er de områder,
en patolog rent faktisk ville gå hen og kigge på. Der bliver ikke klassificeret
noget: en kerne er bare en mørk, farvet og nogenlunde rund klat af en realistisk
størrelse, og hvert trin er en simpel regel, så man altid kan forklare, hvorfor
en given kerne er blevet fundet.
 
Detektion, per billede:
  1. Let Gaussisk blur og derefter en maske med "kernekandidater":
       mørk   (gråtone under NUCLEUS_INTENSITY_MAX)   - kerner optager hæmatoxylin
     OG farvet (mætning over NUCLEUS_SATURATION_MIN) - sorterer mørkt men ufarvet
                                                        skidt fra (støv, snavs)
  2. Morfologisk opening for at fjerne småprikker og closing for at lukke små huller.
  3. Sammenhængende komponenter. Hver komponent beholdes som én kerne, hvis
     arealet og rundheden er realistiske. Komponenter, der er for store,
     behandles som klumper af kerner, der rører hinanden, og deles med en
     watershed på distance-transformen - det er den klassiske standardmetode,
     og den kan godt skille kerner ad, der overlapper lidt, men den tæller
     stadig for få i tætte klumper (kendt begrænsning).
 
Klyngedannelse:
  Kerner, hvis centre ligger inden for CLUSTER_LINK_DIST px af hinanden,
  kobles sammen til én klynge (i praksis gør vi det ved at tegne en skive
  rundt om hvert centrum og så tage de sammenhængende komponenter). Hver
  klynge opsummeres som en cirkel rundt om dens kerner og rangeres efter
  antal kerner, med flest først.
 
Output:
  data/processed/nuclei.csv    - én række per fundet kerne
  data/processed/clusters.csv  - én række per klynge, med priority_rank
 
Husk at alle konstanter markeret CALIBRATE afhænger af forstørrelse/pixelstørrelse
og farvning, så de skal tjekkes op mod rigtige CRIC-billeder (og CRIC's
kerneannoteringer), før man stoler for meget på dem.
 
Brug:
    python src/nuclei.py
    python src/nuclei.py --limit 5
"""
 
import argparse
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
 
from features import IMAGE_EXTENSIONS, load_image
 
# --- Detektionskonstanter (CALIBRATE) ------------------------------------------
NUCLEUS_INTENSITY_MAX = 130     # gråtone (0-255), under hvilken en pixel er "mørk nok" til at være kerne
NUCLEUS_SATURATION_MIN = 40     # HSV-mætning, over hvilken pixlen er farvet (altså ikke støv/snavs)
BLUR_KSIZE = 3                  # udjævning før tærskling
OPEN_KSIZE = 3                  # fjerner prikker, der er mindre end det her
CLOSE_KSIZE = 5                 # lukker små huller inde i kernerne
 
NUCLEUS_MIN_AREA = 30           # px^2 - mindre end det er prikker og lymfocytstykker
NUCLEUS_MAX_AREA = 1500         # px^2 - større end det er en klump kerner -> prøv at dele den
NUCLEUS_MIN_CIRCULARITY = 0.35  # 4*pi*A/P^2, hvor 1 = perfekt cirkel. Lav grænse, fordi kerner kun er sådan nogenlunde runde
WATERSHED_MIN_DIST = 8          # px mellem toppe, når klumper deles (~ en kerneradius)
 
# --- Klyngekonstanter (CALIBRATE) ----------------------------------------------
CLUSTER_LINK_DIST = 60          # px - kerner, der ligger tættere end det, hører til samme klynge
 
DEFAULT_OUTPUT_DIR = Path("data/processed")
 
 
# ---------------------------------------------------------------------------
# Detektion
# ---------------------------------------------------------------------------
 
def nucleus_candidate_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Binær uint8-maske (0/255) med de mørke, farvede pixels, efter oprydning."""
    blurred = cv2.GaussianBlur(image_bgr, (BLUR_KSIZE, BLUR_KSIZE), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((gray < NUCLEUS_INTENSITY_MAX) & (sat > NUCLEUS_SATURATION_MIN)).astype(np.uint8) * 255
 
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE, OPEN_KSIZE))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    return mask
 
 
def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))
 
 
def _describe_component(component_mask: np.ndarray, x_off: int, y_off: int) -> dict | None:
    """Areal / centrum / radius / cirkularitet for én binær klat, eller None, hvis den ikke ligner en kerne."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < NUCLEUS_MIN_AREA:
        return None
    circ = _circularity(contour)
    if circ < NUCLEUS_MIN_CIRCULARITY:
        return None
    m = cv2.moments(contour)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return {
        "x": cx + x_off,
        "y": cy + y_off,
        "area": float(area),
        "radius": float(np.sqrt(area / np.pi)),
        "circularity": circ,
    }
 
 
def _split_clump(clump_mask: np.ndarray) -> list[np.ndarray]:
    """
    Deler en stor klat op i mindre klatter med en watershed på distance-transformen.
    Returnerer en liste af binære masker (samme form som clump_mask), én per segment.
    """
    dist = cv2.distanceTransform(clump_mask, cv2.DIST_L2, 5)
    # Toppene i distance-transformen er centrene af de enkelte kerner.
    kernel = np.ones((WATERSHED_MIN_DIST * 2 + 1,) * 2, np.uint8)
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist > WATERSHED_MIN_DIST * 0.5)
    n_markers, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if n_markers <= 2:  # 0 = baggrund, så <=2 betyder højst én top, og så er der ikke noget at dele
        return [clump_mask]
 
    markers = markers + 1               # baggrunden bliver 1, toppene 2..n
    markers[clump_mask == 0] = 0        # 0 = "ukendt" for watershed
    color = cv2.cvtColor(clump_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    segments = []
    for label in range(2, n_markers + 1):
        seg = ((markers == label) & (clump_mask > 0)).astype(np.uint8) * 255
        if seg.any():
            segments.append(seg)
    return segments or [clump_mask]
 
 
def detect_nuclei(image_bgr: np.ndarray) -> pd.DataFrame:
    """Én række per fundet kerne: x, y, radius, area, circularity, from_clump."""
    mask = nucleus_candidate_mask(image_bgr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
 
    rows = []
    for i in range(1, n):  # 0 er baggrund
        x, y, w, h, area = stats[i]
        if area < NUCLEUS_MIN_AREA:
            continue
        comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
 
        if area <= NUCLEUS_MAX_AREA:
            d = _describe_component(comp, x, y)
            if d:
                rows.append({**d, "from_clump": False})
        else:
            for seg in _split_clump(comp):
                d = _describe_component(seg, x, y)
                if d:
                    rows.append({**d, "from_clump": True})
 
    df = pd.DataFrame(rows, columns=["x", "y", "area", "radius", "circularity", "from_clump"])
    df.insert(0, "nucleus_id", range(1, len(df) + 1))
    return df
 
 
# ---------------------------------------------------------------------------
# Klyngedannelse
# ---------------------------------------------------------------------------
 
def cluster_nuclei(nuclei: pd.DataFrame, image_shape: tuple, link_dist: int = CLUSTER_LINK_DIST) -> pd.DataFrame:
    """
    Giver hver kerne et cluster_id og returnerer én række per klynge
    (cx, cy, radius, n_nuclei, priority_rank). Husk at `nuclei` bliver
    ændret in place, fordi der tilføjes en 'cluster_id'-kolonne til den.
    """
    height, width = image_shape[:2]
    if nuclei.empty:
        nuclei["cluster_id"] = pd.Series(dtype=int)
        return pd.DataFrame(columns=["cluster_id", "cx", "cy", "radius", "n_nuclei", "priority_rank"])
 
    canvas = np.zeros((height, width), np.uint8)
    r = max(1, link_dist // 2)
    for _, nuc in nuclei.iterrows():
        cv2.circle(canvas, (int(round(nuc["x"])), int(round(nuc["y"]))), r, 255, -1)
    _, labels = cv2.connectedComponents(canvas, connectivity=8)
 
    nuclei["cluster_id"] = [
        int(labels[min(height - 1, int(round(nuc["y"]))), min(width - 1, int(round(nuc["x"])))])
        for _, nuc in nuclei.iterrows()
    ]
 
    clusters = []
    for cid, members in nuclei.groupby("cluster_id"):
        pts = members[["x", "y"]].to_numpy(dtype=np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        clusters.append({
            "cluster_id": int(cid),
            "cx": float(cx),
            "cy": float(cy),
            "radius": float(radius + members["radius"].mean()),  # så cirklen omslutter hele kernerne og ikke kun deres centre
            "n_nuclei": int(len(members)),
        })
    out = pd.DataFrame(clusters).sort_values(["n_nuclei", "radius"], ascending=[False, True])
    out["priority_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Orkestrering
# ---------------------------------------------------------------------------
 
def process_image(image_path: Path):
    image = load_image(image_path)
    nuclei = detect_nuclei(image)
    clusters = cluster_nuclei(nuclei, image.shape)
    nuclei.insert(0, "image", image_path.name)
    clusters.insert(0, "image", image_path.name)
    return nuclei, clusters
 
 
def main():
    parser = argparse.ArgumentParser(description="Detect nuclei and group them into clusters (Case 1).")
    parser.add_argument("--input", type=str, default="data/raw", help="Directory of images")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Where to write nuclei.csv / clusters.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images")
    args = parser.parse_args()
 
    input_dir = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}.")
 
    all_nuclei, all_clusters = [], []
    for path in tqdm(image_paths, desc="Detecting nuclei"):
        try:
            nuclei, clusters = process_image(path)
        except ValueError as e:
            print(f"  [warn] skipping {path.name}: {e}")
            continue
        all_nuclei.append(nuclei)
        all_clusters.append(clusters)
 
    nuclei_df = pd.concat(all_nuclei, ignore_index=True)
    clusters_df = pd.concat(all_clusters, ignore_index=True)
    nuclei_df.to_csv(out_dir / "nuclei.csv", index=False)
    clusters_df.to_csv(out_dir / "clusters.csv", index=False)
 
    per_image = nuclei_df.groupby("image").size()
    print(f"\n{len(nuclei_df)} nuclei in {len(image_paths)} images "
          f"(median {per_image.median():.0f} per image, min {per_image.min() if len(per_image) else 0}, max {per_image.max() if len(per_image) else 0}); "
          f"{len(clusters_df)} clusters.")
    print(f"Saved to {out_dir / 'nuclei.csv'} and {out_dir / 'clusters.csv'}")
 
 
if __name__ == "__main__":
    main()
 
