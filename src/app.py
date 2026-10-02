"""
Gennemgangs-app til Billeddiagnostik-PoC'en - Case 1 og Case 2.
 
Case 1 - Felter ("Scopio-light"): hvor er der noget at kigge på?
    Viser hvert billede med et fint gitter; baggrundsfelter dæmpes, felter
    med materiale nummereres i gennemgangsrækkefølge. Man kan klikke sig
    igennem dem zoomet ind.  Data: features_case1.csv + roi_results.csv (roi.py).
 
Case 1 - Kerner: samme spørgsmål, men uden gitter.
    Fundne kerner samles i klynger, tegnes som cirkler og nummereres i
    gennemgangsrækkefølge. Man kan skifte mellem klynger og enkelte kerner.
    Data: nuclei.csv + clusters.csv (nuclei.py).
 
Case 2 - Triagering: er prøven brugbar?
    Viser det grove gitter med hvert felts vurdering (grøn = brugbart, rød =
    ikke brugbart med årsag C/B/D), billedets samlede triage, og lader en
    gemme sin egen vurdering. Hvert klik gemmes i manual_labels.csv (den
    varige log) og spejles i kolonnen 'human_eval' i triage_results.csv.
    Data: features.csv + triage_results.csv (features.py + triage.py).
 
Kør fra projektets rodmappe:
    streamlit run src/app.py
 
Ved første opstart på Streamlit Cloud hentes billederne automatisk fra
Figshare (se ensure_raw_data), da de fylder ~800 MB og ikke ligger på GitHub.
"""
 
import subprocess
import sys
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
 
# Disse imports virker, fordi Streamlit lægger scriptets egen mappe (src/)
# på import-stien, når appen køres.
from roi import summarise_images
from triage import attach_human_eval, classify_tiles
 
# --- Stier (relative til projektets rodmappe, hvor `streamlit run` køres) ---
RAW_DIR = Path("data/raw")
CASE2_FEATURES = Path("data/processed/features.csv")
CASE2_TRIAGE = Path("data/processed/triage_results.csv")
LABELS_PATH = Path("data/processed/manual_labels.csv")
CASE1_ROI = Path("data/processed/roi_results.csv")
CASE1_NUCLEI = Path("data/processed/nuclei.csv")
CASE1_CLUSTERS = Path("data/processed/clusters.csv")
 
TRIAGE_CATEGORIES = ["egnet", "suboptimal", "uegnet"]
TILE_BOUND_COLS = ["x0", "y0", "x1", "y1"]
 
CASE_DESCRIPTIONS = {
    "Case 1": "Lokalisering: *hvor er der noget at kigge på?* Ingen klassifikation, ingen diagnose.",
    "Case 2": "Triagering: *er prøven brugbar?* Auto-vurdering pr. billede, din vurdering gemmes ved siden af.",
}
SUBVIEWS = {
    "Case 1": ["Felter", "Kerner"],
    "Case 2": ["Oversigt", "Enkelt billede"],
}
 
CASES = {
    "Case 1 - Felter": (
        "Finder områder med celler/materiale vs. tom baggrund og viser dem som "
        "prioriterede felter i et gitter. Ingen klassifikation, ingen diagnose - kun "
        "*hvor er der noget at kigge på*."
    ),
    "Case 1 - Kerner": (
        "Samme spørgsmål, men uden gitter: finder cellekerner (mørke, farvede, "
        "runde objekter) og samler dem i klynger, der vises som cirkler i "
        "gennemgangsrækkefølge. Ingen klassifikation af celletype."
    ),
    "Case 2 - Oversigt": (
        "Alle billeder med auto-vurdering og din vurdering. Klik på et billede "
        "for at åbne det."
    ),
    "Case 2 - Enkelt billede": (
        "Vurderer cellemængde, fokus og baggrundsstøj pr. felt og klassificerer "
        "prøven som egnet / suboptimal / uegnet. Din egen vurdering gemmes ved "
        "siden af maskinens."
    ),
}
 
# Farver (RGB)
COLOR_USABLE = (0, 200, 0)
COLOR_FAILED = (220, 40, 40)
COLOR_ROI = (0, 200, 0)
COLOR_ROI_BLURRY = (240, 160, 0)
COLOR_CURRENT = (255, 230, 0)
COLOR_NUCLEUS = (0, 220, 0)
COLOR_CLUSTER = (255, 140, 0)
 
 
# ---------------------------------------------------------------------------
# Datahentning ved opstart
# ---------------------------------------------------------------------------
 
@st.cache_resource(show_spinner=False)
def ensure_raw_data():
    # Billederne fylder ca. 800 MB og ligger derfor ikke på GitHub.
    # Når appen kører på Streamlit Cloud, henter vi dem fra Figshare
    # første gang, så serveren har de samme billeder som lokalt.
    # cache_resource sørger for, at det kun sker én gang per opstart.
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.png")):
        return
    with st.spinner("Henter billeder fra Figshare – første opstart tager et par minutter..."):
        # download_data.py ligger i samme mappe som denne fil (src/), så vi
        # finder den derfra i stedet for at regne med, hvad arbejdsmappen er.
        subprocess.run([sys.executable, str(Path(__file__).parent / "download_data.py")], check=True)
 
 
# ---------------------------------------------------------------------------
# Fælles hjælpefunktioner (ingen Streamlit indeni - kan testes for sig selv)
# ---------------------------------------------------------------------------
 
def check_bounds(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in TILE_BOUND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} has no tile bounds (x0,y0,x1,y1) - regenerate it with the current features.py.")
    return df
 
 
def grid_info(tiles: pd.DataFrame) -> str:
    """Læsbar beskrivelse af det gitter, der faktisk er brugt, ud fra de gemte feltgrænser."""
    n_rows, n_cols = int(tiles["row"].max()) + 1, int(tiles["col"].max()) + 1
    w = int((tiles["x1"] - tiles["x0"]).median())
    h = int((tiles["y1"] - tiles["y0"]).median())
    return f"Grid: {n_cols} × {n_rows} felter = {n_cols * n_rows} pr. billede · feltstørrelse ≈ {w} × {h} px"
 
 
def assert_tiles_span_image(tiles: pd.DataFrame, image: Image.Image) -> None:
    if tiles.empty:
        return
    assert tiles["x0"].min() == 0 and tiles["y0"].min() == 0, "tiles don't start at image origin"
    assert tiles["x1"].max() == image.width and tiles["y1"].max() == image.height, (
        f"tiles span {tiles['x1'].max()}x{tiles['y1'].max()} but image is "
        f"{image.width}x{image.height} - the features file was computed on a different image size"
    )
 
 
# ---- Case 2 - data ----------------------------------------------------------
 
def load_case2():
    tiles = classify_tiles(check_bounds(pd.read_csv(CASE2_FEATURES), "features.csv"))
    triage = pd.read_csv(CASE2_TRIAGE)
    return tiles, triage
 
 
def load_manual_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["image", "manual_label", "labeled_at"])
 
 
def save_manual_label(image_name: str, label: str) -> None:
    """Opdatér/indsæt i manual_labels.csv og genopfrisk human_eval-kolonnerne i triage_results.csv."""
    labels = load_manual_labels()
    labels = labels[labels["image"] != image_name]
    new_row = pd.DataFrame(
        [{"image": image_name, "manual_label": label, "labeled_at": datetime.now().isoformat(timespec="seconds")}]
    )
    labels = pd.concat([labels, new_row], ignore_index=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_PATH, index=False)
 
    results = attach_human_eval(pd.read_csv(CASE2_TRIAGE), LABELS_PATH)
    results.to_csv(CASE2_TRIAGE, index=False)
 
 
def draw_triage_overlay(image: Image.Image, tiles: pd.DataFrame) -> Image.Image:
    """Case 2: grøn = brugbart, rød = ikke brugbart, med de fejlede tjek (C/B/D) i hjørnet."""
    assert_tiles_span_image(tiles, image)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for _, t in tiles.iterrows():
        x0, y0, x1, y1 = int(t["x0"]), int(t["y0"]), int(t["x1"]), int(t["y1"])
        color = COLOR_USABLE if t["tile_usable"] else COLOR_FAILED
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=3)
        if not t["tile_usable"]:
            label = t["reasons"]
            draw.rectangle([x0, y0, x0 + 14 * len(label) + 8, y0 + 22], fill=color)
            draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255))
    return overlay
 
 
# ---- Case 1 - felter --------------------------------------------------------
 
def load_case1() -> pd.DataFrame:
    return check_bounds(pd.read_csv(CASE1_ROI), "roi_results.csv")
 
 
def draw_roi_overlay(image: Image.Image, tiles: pd.DataFrame, current_rank=None) -> Image.Image:
    """
    Case 1: baggrundsfelter dæmpes, ROI-felter indrammes (grøn = i fokus,
    orange = uskarpt) med deres rækkefølge; det aktuelle felt i gult.
    """
    assert_tiles_span_image(tiles, image)
    base = image.convert("RGBA")
    dim = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dim_draw = ImageDraw.Draw(dim)
    for _, t in tiles.iterrows():
        if not t["is_roi"]:
            dim_draw.rectangle([int(t["x0"]), int(t["y0"]), int(t["x1"]) - 1, int(t["y1"]) - 1], fill=(40, 40, 40, 150))
    out = Image.alpha_composite(base, dim).convert("RGB")
 
    draw = ImageDraw.Draw(out)
    for _, t in tiles.iterrows():
        x0, y0, x1, y1 = int(t["x0"]), int(t["y0"]), int(t["x1"]), int(t["y1"])
        if not t["is_roi"]:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(90, 90, 90), width=1)
            continue
        is_current = current_rank is not None and int(t["priority_rank"]) == int(current_rank)
        color = COLOR_CURRENT if is_current else (COLOR_ROI if t["in_focus"] else COLOR_ROI_BLURRY)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=5 if is_current else 2)
        label = str(int(t["priority_rank"]))
        draw.rectangle([x0, y0, x0 + 9 * len(label) + 8, y0 + 18], fill=color)
        draw.text((x0 + 4, y0 + 3), label, fill=(0, 0, 0) if is_current else (255, 255, 255))
    return out
 
 
def crop_tile(image: Image.Image, tile: pd.Series, margin: int = 24, scale: int = 3) -> Image.Image:
    """Zoomet visning af ét felt med lidt kontekst omkring."""
    x0 = max(0, int(tile["x0"]) - margin)
    y0 = max(0, int(tile["y0"]) - margin)
    x1 = min(image.width, int(tile["x1"]) + margin)
    y1 = min(image.height, int(tile["y1"]) + margin)
    crop = image.crop((x0, y0, x1, y1))
    return crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
 
 
# ---- Case 1 - kerner --------------------------------------------------------
 
def load_case1_nuclei():
    return pd.read_csv(CASE1_NUCLEI), pd.read_csv(CASE1_CLUSTERS)
 
 
def draw_nuclei_overlay(image: Image.Image, nuclei: pd.DataFrame, clusters: pd.DataFrame,
                        show_clusters: bool, show_nuclei: bool, current_rank=None) -> Image.Image:
    """Cirkler om klynger (nummereret efter rækkefølge) og/eller om enkelte kerner."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if show_nuclei:
        for _, n in nuclei.iterrows():
            r = n["radius"] + 3
            draw.ellipse([n["x"] - r, n["y"] - r, n["x"] + r, n["y"] + r], outline=COLOR_NUCLEUS, width=2)
    if show_clusters:
        for _, c in clusters.iterrows():
            is_current = current_rank is not None and int(c["priority_rank"]) == int(current_rank)
            color = COLOR_CURRENT if is_current else COLOR_CLUSTER
            r = c["radius"]
            draw.ellipse([c["cx"] - r, c["cy"] - r, c["cx"] + r, c["cy"] + r],
                         outline=color, width=6 if is_current else 3)
            label = str(int(c["priority_rank"]))
            lx, ly = int(c["cx"] - r), int(c["cy"] - r) - 20
            ly = max(0, ly)
            draw.rectangle([lx, ly, lx + 10 * len(label) + 8, ly + 18], fill=color)
            draw.text((lx + 4, ly + 3), label, fill=(0, 0, 0) if is_current else (255, 255, 255))
    return out
 
 
def crop_circle(image: Image.Image, cx: float, cy: float, radius: float, margin: int = 30, scale: int = 2) -> Image.Image:
    x0 = max(0, int(cx - radius) - margin); y0 = max(0, int(cy - radius) - margin)
    x1 = min(image.width, int(cx + radius) + margin); y1 = min(image.height, int(cy + radius) + margin)
    crop = image.crop((x0, y0, x1, y1))
    return crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
 
 
# ---------------------------------------------------------------------------
# UI-dele
# ---------------------------------------------------------------------------
 
def pick_image(image_names: list, key: str) -> str:
    """Forrige/vælg/næste-navigation i sidepanelet; husker positionen pr. case.
    Selectboxens session-state er den eneste sandhed, så forrige/næste og
    spring sætter bare den, før widgetten tegnes."""
    sel_key = f"{key}_select"
    if st.session_state.get(sel_key) not in image_names:
        st.session_state[sel_key] = image_names[0]
    jump = st.session_state.pop(f"{key}_jump_to", None)  # fx 'Åbn' i oversigten
    if jump in image_names:
        st.session_state[sel_key] = jump
 
    idx = image_names.index(st.session_state[sel_key])
    col_prev, col_select, col_next = st.sidebar.columns([1, 3, 1])
    if col_prev.button("◀", key=f"{key}_prev"):
        st.session_state[sel_key] = image_names[max(0, idx - 1)]
    if col_next.button("▶", key=f"{key}_next"):
        st.session_state[sel_key] = image_names[min(len(image_names) - 1, idx + 1)]
    chosen = col_select.selectbox("Image", image_names, key=sel_key, label_visibility="collapsed")
    st.sidebar.caption(f"Billede {image_names.index(chosen) + 1} / {len(image_names)}")
    return chosen
 
 
def render_case1():
    if not CASE1_ROI.exists():
        st.error("roi_results.csv not found. Run:\n"
                 "  python3 src/features.py --tile-size 128 --output data/processed/features_case1.csv\n"
                 "  python3 src/roi.py")
        st.stop()
 
    tiles_df = load_case1()
    summary = summarise_images(tiles_df).set_index("image")
    st.sidebar.info(grid_info(tiles_df))
 
    st.sidebar.header("Billede")
    image_name = pick_image(sorted(tiles_df["image"].unique().tolist()), key="c1")
    image_tiles = tiles_df[tiles_df["image"] == image_name]
    roi_tiles = image_tiles[image_tiles["is_roi"]].sort_values("priority_rank")
    n_roi = len(roi_tiles)
 
    # Klik igennem ROI-felter i prioriteret rækkefølge (0 = oversigt, intet felt valgt)
    step_key = f"c1_step_{image_name}"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    st.sidebar.markdown("---")
    st.sidebar.header("Gennemgang")
    c_a, c_b, c_c = st.sidebar.columns(3)
    if c_a.button("Oversigt", key="c1_overview"):
        st.session_state[step_key] = 0
    if c_b.button("◀ Forrige", key="c1_prevroi"):
        st.session_state[step_key] = max(1, st.session_state[step_key] - 1) if n_roi else 0
    if c_c.button("Næste ▶", key="c1_nextroi"):
        st.session_state[step_key] = min(n_roi, st.session_state[step_key] + 1)
    current = st.session_state[step_key]
    st.sidebar.caption(f"Felt {current} / {n_roi}" if current else f"{n_roi} felter at gennemgå")
 
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Forklaring**  \n"
        "🟩 grøn = materiale, i fokus  \n"
        "🟧 orange = materiale, uskarpt (gennemgås sidst)  \n"
        "⬛ mørk = baggrund, kan springes over  \n"
        "Tal = gennemgangsrækkefølge"
    )
 
    image_path = RAW_DIR / image_name
    if not image_path.exists():
        st.error(f"Image file not found: {image_path}")
        st.stop()
    image = Image.open(image_path).convert("RGB")
 
    col_image, col_info = st.columns([3, 1])
    with col_image:
        st.image(draw_roi_overlay(image, image_tiles, current_rank=current or None),
                 caption=image_name, width="stretch")
 
    with col_info:
        s = summary.loc[image_name]
        st.subheader("Oversigt")
        st.metric("Felter med materiale", f"{int(s['roi_tiles'])} / {int(s['total_tiles'])}")
        st.metric("Andel af billedet", f"{s['pct_roi_area']:.0%}")
        st.metric("Heraf i fokus", f"{int(s['roi_in_focus'])} / {int(s['roi_tiles'])}")
 
        if current:
            t = roi_tiles.iloc[current - 1]
            st.markdown("---")
            st.subheader(f"Felt #{current}")
            st.image(crop_tile(image, t), width="stretch")
            st.metric("Cellularitet", f"{t['cellularity']:.3f}")
            st.metric("Fokus", f"{t['blur_score']:.1f}" + ("" if t["in_focus"] else "  ⚠ uskarpt"))
            st.caption(f"Række {int(t['row'])}, kolonne {int(t['col'])} · px {int(t['x0'])}-{int(t['x1'])}, {int(t['y0'])}-{int(t['y1'])}")
 
    with st.expander("Alle felter"):
        st.dataframe(
            image_tiles[["priority_rank", "row", "col", "cellularity", "blur_score", "is_roi", "in_focus"]]
            .sort_values(["row", "col"]).reset_index(drop=True),
            width="stretch",
        )
 
 
def render_case1_nuclei():
    if not CASE1_NUCLEI.exists() or not CASE1_CLUSTERS.exists():
        st.error("nuclei.csv / clusters.csv not found. Run:\n  python3 src/nuclei.py")
        st.stop()
 
    nuclei_df, clusters_df = load_case1_nuclei()
    st.sidebar.info("Kerner = mørke, farvede, runde objekter · klynge = kerner < 60 px fra hinanden")
 
    st.sidebar.header("Visning")
    mode = st.sidebar.radio("Vis", ["Klynger", "Enkelte kerner", "Begge"], index=0, label_visibility="collapsed")
    show_clusters = mode in ("Klynger", "Begge")
    show_nuclei = mode in ("Enkelte kerner", "Begge")
    min_size = st.sidebar.slider("Vis klynger med mindst … kerner", 1, 10, 1,
                                 help="Skjuler små klynger, så et tæt felt kan koges ned til de vigtigste steder.")
 
    st.sidebar.header("Billede")
    image_names = sorted(set(nuclei_df["image"]) | set(clusters_df["image"]))
    if not image_names:
        st.warning("Ingen kerner fundet i nogen billeder.")
        st.stop()
    image_name = pick_image(image_names, key="c1n")
    nuclei = nuclei_df[nuclei_df["image"] == image_name]
    all_clusters = clusters_df[clusters_df["image"] == image_name]
    clusters = all_clusters[all_clusters["n_nuclei"] >= min_size].sort_values("priority_rank").copy()
    clusters["priority_rank"] = range(1, len(clusters) + 1)  # nummerér om efter filtrering
    n_clusters = len(clusters)
 
    step_key = f"c1n_step_{image_name}"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    st.sidebar.markdown("---")
    st.sidebar.header("Gennemgang (klynger)")
    c_a, c_b, c_c = st.sidebar.columns(3)
    if c_a.button("Oversigt", key="c1n_overview"):
        st.session_state[step_key] = 0
    if c_b.button("◀ Forrige", key="c1n_prevcluster"):
        st.session_state[step_key] = max(1, st.session_state[step_key] - 1) if n_clusters else 0
    if c_c.button("Næste ▶", key="c1n_nextcluster"):
        st.session_state[step_key] = min(n_clusters, st.session_state[step_key] + 1)
    current = st.session_state[step_key]
    st.sidebar.caption(f"Klynge {current} / {n_clusters}" if current else f"{n_clusters} klynger at gennemgå")
 
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Forklaring**  \n"
        "🟠 orange cirkel = klynge af kerner, tal = rækkefølge (flest kerner først)  \n"
        "🟢 grøn cirkel = enkelt kerne  \n"
        "🟡 gul = aktuel klynge"
    )
 
    image_path = RAW_DIR / image_name
    if not image_path.exists():
        st.error(f"Image file not found: {image_path}")
        st.stop()
    image = Image.open(image_path).convert("RGB")
 
    col_image, col_info = st.columns([3, 1])
    with col_image:
        st.image(draw_nuclei_overlay(image, nuclei, clusters, show_clusters, show_nuclei,
                                     current_rank=current or None),
                 caption=image_name, width="stretch")
 
    with col_info:
        st.subheader("Oversigt")
        st.metric("Kerner fundet", len(nuclei))
        st.metric("Klynger", f"{n_clusters}" + (f" (af {len(all_clusters)})" if min_size > 1 else ""))
        if n_clusters:
            st.metric("Største klynge", f"{int(clusters['n_nuclei'].max())} kerner")
        if current:
            c = clusters.iloc[current - 1]
            st.markdown("---")
            st.subheader(f"Klynge #{current}")
            st.image(crop_circle(image, c["cx"], c["cy"], c["radius"]), width="stretch")
            st.metric("Kerner i klyngen", int(c["n_nuclei"]))
            st.caption(f"Centrum ({int(c['cx'])}, {int(c['cy'])}) · radius {int(c['radius'])} px")
 
    with st.expander("Alle klynger"):
        st.dataframe(clusters[["priority_rank", "n_nuclei", "cx", "cy", "radius"]].round(1)
                     .reset_index(drop=True), width="stretch")
    with st.expander("Alle kerner"):
        st.dataframe(nuclei[["nucleus_id", "x", "y", "radius", "area", "circularity", "from_clump", "cluster_id"]]
                     .round(2).reset_index(drop=True), width="stretch")
 
 
def render_case2_overview():
    if not CASE2_TRIAGE.exists():
        st.error("Run features.py and triage.py first - triage_results.csv not found.")
        st.stop()
    triage_df = pd.read_csv(CASE2_TRIAGE)
    if "human_eval" not in triage_df:
        triage_df["human_eval"] = pd.NA
 
    st.sidebar.header("Filter")
    scope = st.sidebar.radio("Vis", ["Alle", "Kun uden human evaluering", "Kun uenige"], key="ov_scope")
    auto_filter = st.sidebar.multiselect("Auto evaluering", TRIAGE_CATEGORIES, default=TRIAGE_CATEGORIES, key="ov_auto")
    per_row = st.sidebar.slider("Billeder pr. række", 2, 6, 4, key="ov_cols")
 
    df = triage_df[triage_df["triage"].isin(auto_filter)]
    if scope == "Kun uden human evaluering":
        df = df[df["human_eval"].isna()]
    elif scope == "Kun uenige":
        df = df[df["human_eval"].notna() & (df["human_eval"] != df["triage"])]
    df = df.sort_values("image").reset_index(drop=True)
 
    n_reviewed = int(triage_df["human_eval"].notna().sum())
    n_agree = int((triage_df["human_eval"].notna() & (triage_df["human_eval"] == triage_df["triage"])).sum())
    st.markdown(f"**{len(df)} billeder vist** · vurderet {n_reviewed} / {len(triage_df)}"
                + (f" · enighed {n_agree} / {n_reviewed}" if n_reviewed else ""))
 
    if df.empty:
        st.info("Ingen billeder matcher filteret.")
        return
 
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for col, (_, row) in zip(cols, df.iloc[start : start + per_row].iterrows()):
            path = RAW_DIR / row["image"]
            with col:
                if path.exists():
                    thumb = Image.open(path).convert("RGB")
                    thumb.thumbnail((400, 400))
                    st.image(thumb, width="stretch")
                else:
                    st.warning("mangler fil")
                human = row["human_eval"] if pd.notna(row["human_eval"]) else "Ingen evaluering"
                st.markdown(f"**Auto:** {row['triage']}  \n**Human:** {human}")
                st.caption(row["image"][:16] + "…")
                if st.button("Åbn", key=f"open_{row['image']}", width="stretch"):
                    st.session_state["pending_nav"] = {"case": "Case 2", "sub": "Enkelt billede", "image": row["image"]}
                    st.rerun()
 
 
def render_case2():
    if not CASE2_FEATURES.exists() or not CASE2_TRIAGE.exists():
        st.error("Run features.py and triage.py first - features.csv / triage_results.csv not found.")
        st.stop()
 
    tiles_df, triage_df = load_case2()
    labels_df = load_manual_labels()
    st.sidebar.info(grid_info(tiles_df))
 
    st.sidebar.header("Filter")
    st.session_state.setdefault("c2_filter_auto", TRIAGE_CATEGORIES)
    st.session_state.setdefault("c2_only_unlabeled", False)
    selected = st.sidebar.multiselect("Auto evaluering", TRIAGE_CATEGORIES, key="c2_filter_auto")
    only_unlabeled = st.sidebar.checkbox("Kun billeder jeg ikke har vurderet", key="c2_only_unlabeled")
 
    filtered = triage_df[triage_df["triage"].isin(selected)]
    if only_unlabeled:
        filtered = filtered[~filtered["image"].isin(labels_df["image"])]
    image_names = sorted(filtered["image"].tolist())
    if not image_names:
        st.warning("Ingen billeder matcher filteret.")
        st.stop()
 
    st.sidebar.header("Billede")
    image_name = pick_image(image_names, key="c2")
 
    n_reviewed = int(triage_df["human_eval"].notna().sum()) if "human_eval" in triage_df else 0
    n_agree = int(triage_df["agrees"].fillna(False).astype(bool).sum()) if "agrees" in triage_df else 0
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Vurderet:** {n_reviewed} / {len(triage_df)} billeder")
    if n_reviewed:
        st.sidebar.markdown(f"**Enighed indtil nu:** {n_agree} / {n_reviewed}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Forklaring**  \n"
        "🟩 grøn = brugbart felt  \n"
        "🟥 rød = ikke brugbart:  \n"
        "&nbsp;&nbsp;C = for lidt cellemateriale  \n"
        "&nbsp;&nbsp;B = uskarpt (kun vurderet hvor der er celler)  \n"
        "&nbsp;&nbsp;D = for meget debris (kun vurderet hvor der er celler)"
    )
 
    image_row = triage_df[triage_df["image"] == image_name].iloc[0]
    image_tiles = tiles_df[tiles_df["image"] == image_name]
 
    image_path = RAW_DIR / image_name
    if not image_path.exists():
        st.error(f"Image file not found: {image_path}")
        st.stop()
    image = Image.open(image_path).convert("RGB")
 
    col_image, col_info = st.columns([3, 1])
    with col_image:
        st.image(draw_triage_overlay(image, image_tiles), caption=image_name, width="stretch")
 
    with col_info:
        st.subheader("Auto evaluering")
        st.markdown(f"## `{image_row['triage']}`")
        existing = image_row.get("human_eval")
        st.subheader("Human evaluering")
        if pd.notna(existing):
            verdict = "enig" if existing == image_row["triage"] else "uenig"
            st.markdown(f"## `{existing}`")
            st.caption(f"{verdict} med auto evaluering · {image_row.get('human_eval_at', '')}")
        else:
            st.markdown("## `ingen evaluering`")
 
        msg_key = f"saved_msg_{image_name}"
        if st.session_state.get(msg_key):
            st.success(f"Din evaluering ({st.session_state.pop(msg_key)}) er gemt")
 
        st.markdown("---")
        st.metric("Brugbare felter", f"{int(image_row['usable_tiles'])} / {int(image_row['total_tiles'])}")
        st.metric("Felter med celler", f"{int(image_row['cellular_tiles'])} / {int(image_row['total_tiles'])}")
        st.metric("Gns. cellularitet", f"{image_row['mean_cellularity']:.3f}")
        blur, debris = image_row["mean_blur_cellular"], image_row["mean_debris_cellular"]
        st.metric("Fokus (felter m. celler)", "n/a" if pd.isna(blur) else f"{blur:.1f}")
        st.metric("Debris (felter m. celler)", "n/a" if pd.isna(debris) else f"{debris:.1f}")
 
        st.markdown("---")
        st.subheader("Din vurdering")
        for category in TRIAGE_CATEGORIES:
            if st.button(category, key=f"label_{category}", width="stretch"):
                save_manual_label(image_name, category)
                st.session_state[msg_key] = category
                st.rerun()
 
    with st.expander("Alle felter"):
        st.dataframe(
            image_tiles[["row", "col", "cellularity", "blur_score", "debris_score",
                         "has_cells", "in_focus", "is_clean", "tile_usable", "reasons"]]
            .sort_values(["row", "col"]).reset_index(drop=True),
            width="stretch",
        )
 
 
def apply_pending_navigation():
    """Widget-state kan kun sættes, før widgetten tegnes, så spring bestilt i
    forrige kørsel (fx 'Åbn' i oversigten) udføres her, som det første."""
    nav = st.session_state.pop("pending_nav", None)
    if not nav:
        return
    st.session_state["case"] = nav["case"]
    st.session_state[f"sub_{nav['case']}"] = nav["sub"]
    if "image" in nav:
        st.session_state["c2_jump_to"] = nav["image"]
        # Nulstil enkeltbillede-filtrene, så det ønskede billede med sikkerhed er i listen.
        st.session_state["c2_filter_auto"] = TRIAGE_CATEGORIES
        st.session_state["c2_only_unlabeled"] = False
 
 
def main():
    st.set_page_config(page_title="Billeddiagnostik PoC", layout="wide")
    ensure_raw_data()  # efter set_page_config, som skal være første Streamlit-kald
    apply_pending_navigation()
 
    case = st.sidebar.radio("Vælg case", list(SUBVIEWS.keys()), key="case")
    st.sidebar.caption(CASE_DESCRIPTIONS[case])
    sub = st.sidebar.radio("Visning", SUBVIEWS[case], key=f"sub_{case}", horizontal=True)
    st.sidebar.caption(CASES[f"{case} - {sub}"])
    st.sidebar.markdown("---")
    st.title(f"{case} - {sub}")
 
    if case == "Case 1" and sub == "Felter":
        render_case1()
    elif case == "Case 1" and sub == "Kerner":
        render_case1_nuclei()
    elif case == "Case 2" and sub == "Oversigt":
        render_case2_overview()
    else:
        render_case2()
 
 
if __name__ == "__main__":
    main()