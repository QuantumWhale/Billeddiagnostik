"""
Nucleus-based localisation for Case 1 ("where is there something to look at").
 
Instead of a grid, this finds the nuclei themselves and groups nearby ones
into clusters - the regions a pathologist would actually go to. Nothing is
classified: a nucleus is a dark, stained, roughly round blob of plausible
size. Every step is a simple rule so every detection can be explained.
 
Detection, per image:
  1. Light Gaussian blur, then a "nucleus candidate" mask:
       dark  (grayscale below NUCLEUS_INTENSITY_MAX)   - nuclei take up haematoxylin
     AND stained (saturation above NUCLEUS_SATURATION_MIN) - rejects unstained
                                                          dark junk (dust, dirt)
  2. Morphological opening to drop specks, closing to fill small holes.
  3. Connected components. Each is kept as one nucleus if its area and
     roundness are plausible. Oversized components are treated as clumps of
     touching nuclei and split with a distance-transform watershed - the
     standard classical approach; it separates lightly overlapping nuclei
     but will still under-count dense clumps (known limitation).
 
Clustering:
  Nuclei whose centres lie within CLUSTER_LINK_DIST px of each other are
  linked into one cluster (done as: draw a disc around every centre, take
  connected components). Each cluster is summarised as a circle around its
  members, ranked by number of nuclei - most first.
 
Outputs:
  data/processed/nuclei.csv    - one row per detected nucleus
  data/processed/clusters.csv  - one row per cluster, with priority_rank
 
All constants marked CALIBRATE depend on magnification/pixel size and stain,
so check them against real CRIC images (and the CRIC nucleus annotations).
 
Usage:
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
 
# --- Detection constants (CALIBRATE) -------------------------------------------
NUCLEUS_INTENSITY_MAX = 130     # grayscale (0-255) below which a pixel is "dark enough" to be nucleus
NUCLEUS_SATURATION_MIN = 40     # HSV saturation above which it is stained (not dust/dirt)
BLUR_KSIZE = 3                  # pre-smoothing before thresholding
OPEN_KSIZE = 3                  # removes specks smaller than this
CLOSE_KSIZE = 5                 # fills pinholes inside nuclei
 
NUCLEUS_MIN_AREA = 30           # px^2 - smaller = specks, lymphocyte fragments
NUCLEUS_MAX_AREA = 1500         # px^2 - larger = clump of nuclei -> try to split
NUCLEUS_MIN_CIRCULARITY = 0.35  # 4*pi*A/P^2; 1 = perfect circle. Low bar: nuclei are only roughly round
WATERSHED_MIN_DIST = 8          # px between peaks when splitting clumps (~ nucleus radius)
 
# --- Clustering constants (CALIBRATE) ------------------------------------------
CLUSTER_LINK_DIST = 60          # px - nuclei closer than this belong to the same cluster
 
DEFAULT_OUTPUT_DIR = Path("data/processed")
 
 
# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
 
def nucleus_candidate_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Binary uint8 mask (0/255) of dark, stained pixels after cleanup."""
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
    """Area / centre / radius / circularity for one binary blob, or None if it isn't nucleus-like."""
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
    Split a large blob into sub-blobs with a distance-transform watershed.
    Returns a list of binary masks (same shape as clump_mask), one per segment.
    """
    dist = cv2.distanceTransform(clump_mask, cv2.DIST_L2, 5)
    # Peaks of the distance transform = centres of the individual nuclei.
    kernel = np.ones((WATERSHED_MIN_DIST * 2 + 1,) * 2, np.uint8)
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist > WATERSHED_MIN_DIST * 0.5)
    n_markers, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if n_markers <= 2:  # 0 = background, so <=2 means at most one peak: nothing to split
        return [clump_mask]
 
    markers = markers + 1               # background becomes 1, peaks 2..n
    markers[clump_mask == 0] = 0        # 0 = "unknown" for watershed
    color = cv2.cvtColor(clump_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    segments = []
    for label in range(2, n_markers + 1):
        seg = ((markers == label) & (clump_mask > 0)).astype(np.uint8) * 255
        if seg.any():
            segments.append(seg)
    return segments or [clump_mask]
 
 
def detect_nuclei(image_bgr: np.ndarray) -> pd.DataFrame:
    """One row per detected nucleus: x, y, radius, area, circularity, from_clump."""
    mask = nucleus_candidate_mask(image_bgr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
 
    rows = []
    for i in range(1, n):  # 0 is background
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
# Clustering
# ---------------------------------------------------------------------------
 
def cluster_nuclei(nuclei: pd.DataFrame, image_shape: tuple, link_dist: int = CLUSTER_LINK_DIST) -> pd.DataFrame:
    """
    Assign a cluster_id to each nucleus and return one row per cluster
    (cx, cy, radius, n_nuclei, priority_rank). Modifies `nuclei` in place
    by adding 'cluster_id'.
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
            "radius": float(radius + members["radius"].mean()),  # enclose the nuclei, not just their centres
            "n_nuclei": int(len(members)),
        })
    out = pd.DataFrame(clusters).sort_values(["n_nuclei", "radius"], ascending=[False, True])
    out["priority_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Orchestration
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
 
