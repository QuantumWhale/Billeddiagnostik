"""
Review interface for the Billeddiagnostik PoC - Case 1 and Case 2.
 
Case 1 - Lokalisering ("Scopio-light"): where is there something to look at?
    Shows each image with a fine tile grid; background tiles are dimmed,
    tiles with material are numbered in review order. Step through them
    with a zoomed view.  Data: features_case1.csv + roi_results.csv (roi.py).
 
Case 1 - Kerner (nucleus view): the same question answered without a grid.
    Detected nuclei are grouped into clusters, drawn as circles and numbered
    in review order. Toggle between clusters and individual nuclei.
    Data: nuclei.csv + clusters.csv (nuclei.py).
 
Case 2 - Triagering: is the sample usable?
    Shows the coarse tile grid with each tile's verdict (green = usable, red =
    failed with reason C/B/D), the image-level triage, and lets you record a
    human verdict. Each click is saved to manual_labels.csv (durable record)
    and mirrored into triage_results.csv's 'human_eval' column.
    Data: features.csv + triage_results.csv (features.py + triage.py).
 
Run from the project root:
    streamlit run src/app.py
"""
 
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
 
# These imports work because Streamlit adds the script's own folder (src/)
# to the import path when it runs the app.
from roi import summarise_images
from triage import attach_human_eval, classify_tiles
import theme
 
# --- Paths (relative to the project root, where you run `streamlit run`) -----
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
 
# Colours (RGB)
COLOR_USABLE = (0, 200, 0)
COLOR_FAILED = (220, 40, 40)
COLOR_ROI = (0, 200, 0)
COLOR_ROI_BLURRY = (240, 160, 0)
COLOR_CURRENT = (255, 230, 0)
COLOR_NUCLEUS = (0, 220, 0)
COLOR_CLUSTER = (255, 140, 0)
 
 
# ---------------------------------------------------------------------------
# Shared helpers (no Streamlit inside - testable on their own)
# ---------------------------------------------------------------------------
 
def check_bounds(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in TILE_BOUND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} has no tile bounds (x0,y0,x1,y1) - regenerate it with the current features.py.")
    return df
 
 
def grid_info(tiles: pd.DataFrame) -> str:
    """Human-readable description of the grid actually used, from the stored bounds."""
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
 
 
# ---- Case 2 data ------------------------------------------------------------
 
def load_case2():
    tiles = classify_tiles(check_bounds(pd.read_csv(CASE2_FEATURES), "features.csv"))
    triage = pd.read_csv(CASE2_TRIAGE)
    return tiles, triage
 
 
def load_manual_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["image", "manual_label", "labeled_at"])
 
 
def save_manual_label(image_name: str, label: str) -> None:
    """Upsert into manual_labels.csv, then refresh human_eval columns in triage_results.csv."""
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
    """Case 2: green = usable, red = failed with the failed checks (C/B/D) in the corner."""
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
 
 
# ---- Case 1 data ------------------------------------------------------------
 
def load_case1() -> pd.DataFrame:
    return check_bounds(pd.read_csv(CASE1_ROI), "roi_results.csv")
 
 
def draw_roi_overlay(image: Image.Image, tiles: pd.DataFrame, current_rank=None) -> Image.Image:
    """
    Case 1: background tiles dimmed, ROI tiles outlined (green = in focus,
    orange = blurry) with their review rank; the current tile in yellow.
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
    """Zoomed view of one tile with a little context around it."""
    x0 = max(0, int(tile["x0"]) - margin)
    y0 = max(0, int(tile["y0"]) - margin)
    x1 = min(image.width, int(tile["x1"]) + margin)
    y1 = min(image.height, int(tile["y1"]) + margin)
    crop = image.crop((x0, y0, x1, y1))
    return crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
 
 
# ---- Case 1 (nuclei) data ------------------------------------------------
 
def load_case1_nuclei():
    return pd.read_csv(CASE1_NUCLEI), pd.read_csv(CASE1_CLUSTERS)
 
 
def draw_nuclei_overlay(image: Image.Image, nuclei: pd.DataFrame, clusters: pd.DataFrame,
                        show_clusters: bool, show_nuclei: bool, current_rank=None) -> Image.Image:
    """Circles around clusters (numbered by review order) and/or around individual nuclei."""
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
# UI pieces
# ---------------------------------------------------------------------------

def rgb(color: tuple) -> str:
    return "rgb({},{},{})".format(*color)


def pick_image(image_names: list, key: str) -> str:
    """Sidebar prev/select/next navigation; remembers position per case.
    The selectbox's session-state entry is the single source of truth, so
    prev/next and jumps just set it before the widget is drawn."""
    sel_key = f"{key}_select"
    if st.session_state.get(sel_key) not in image_names:
        st.session_state[sel_key] = image_names[0]
    jump = st.session_state.pop(f"{key}_jump_to", None)  # e.g. 'Åbn' in the overview
    if jump in image_names:
        st.session_state[sel_key] = jump

    idx = image_names.index(st.session_state[sel_key])
    col_prev, col_select, col_next = st.sidebar.columns([1, 3, 1])
    if col_prev.button("◀", key=f"{key}_prev"):
        st.session_state[sel_key] = image_names[max(0, idx - 1)]
    if col_next.button("▶", key=f"{key}_next"):
        st.session_state[sel_key] = image_names[min(len(image_names) - 1, idx + 1)]
    return col_select.selectbox("Image", image_names, key=sel_key, label_visibility="collapsed")


def position(image_names: list, image_name: str) -> str:
    return f"Billede {image_names.index(image_name) + 1} / {len(image_names)}"


def sample_id(image_name: str) -> str:
    return Path(image_name).stem


def step_controls(step_key: str, n_items: int, key: str, noun: str) -> int:
    """◀ Oversigt ▶ stepping through items in priority order (0 = overview)."""
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    c_prev, c_overview, c_next = st.sidebar.columns([1, 2, 1])
    if c_prev.button("◀", key=f"{key}_prev{noun}", width="stretch"):
        st.session_state[step_key] = max(1, st.session_state[step_key] - 1) if n_items else 0
    if c_overview.button("Oversigt", key=f"{key}_overview", width="stretch"):
        st.session_state[step_key] = 0
    if c_next.button("▶", key=f"{key}_next{noun}", width="stretch"):
        st.session_state[step_key] = min(n_items, st.session_state[step_key] + 1)
    current = st.session_state[step_key]
    st.sidebar.caption(f"{noun.capitalize()} {current} / {n_items}" if current
                       else f"{n_items} {noun}er at gennemgå · tryk ▶ for at starte")
    return current


def load_image_or_stop(image_name: str) -> Image.Image:
    image_path = RAW_DIR / image_name
    if not image_path.exists():
        st.error(f"Image file not found: {image_path}")
        st.stop()
    return Image.open(image_path).convert("RGB")


def hint(text: str) -> None:
    st.markdown(f'<div class="bd-card bd-hint">{text}</div>', unsafe_allow_html=True)


def render_case1():
    if not CASE1_ROI.exists():
        st.error("roi_results.csv not found. Run:\n"
                 "  python3 src/features.py --tile-size 128 --output data/processed/features_case1.csv\n"
                 "  python3 src/roi.py")
        st.stop()

    tiles_df = load_case1()
    summary = summarise_images(tiles_df).set_index("image")

    theme.side_heading("Billede")
    image_names = sorted(tiles_df["image"].unique().tolist())
    image_name = pick_image(image_names, key="c1")
    image_tiles = tiles_df[tiles_df["image"] == image_name]
    roi_tiles = image_tiles[image_tiles["is_roi"]].sort_values("priority_rank")
    n_roi = len(roi_tiles)

    theme.side_heading("Gennemgang")
    current = step_controls(f"c1_step_{image_name}", n_roi, key="c1", noun="felt")

    theme.legend("Forklaring", [
        (rgb(COLOR_ROI), False, "<b>Materiale</b>, i fokus"),
        (rgb(COLOR_ROI_BLURRY), False, "<b>Materiale</b>, uskarpt · gennemgås sidst"),
        ("rgb(70,70,70)", False, "<b>Baggrund</b> · kan springes over"),
        (rgb(COLOR_CURRENT), False, "<b>Aktuelt felt</b> · tal = rækkefølge"),
    ])
    st.sidebar.info(grid_info(tiles_df))

    image = load_image_or_stop(image_name)
    theme.topbar("Case 1 · Felter", "Hvor er der noget at kigge på?",
                 sample_id(image_name), position(image_names, image_name))

    col_image, col_info = st.columns([3, 1.1])
    with col_image:
        st.image(draw_roi_overlay(image, image_tiles, current_rank=current or None), width="stretch")

    with col_info:
        s = summary.loc[image_name]
        theme.stats_card("Billedet", [
            ("Felter med materiale", f"{int(s['roi_tiles'])} / {int(s['total_tiles'])}", "", True),
            ("Andel af billedet", f"{s['pct_roi_area']:.0%}"),
            ("Heraf i fokus", f"{int(s['roi_in_focus'])} / {int(s['roi_tiles'])}"),
        ])

        if current:
            t = roi_tiles.iloc[current - 1]
            focus = f"{t['blur_score']:.1f}" + ("" if t["in_focus"] else " · uskarpt")
            theme.stats_card(f"Felt #{current} af {n_roi}", [
                ("Cellularitet", f"{t['cellularity']:.3f}"),
                ("Fokus", focus, "" if t["in_focus"] else "warn"),
                ("Position", f"række {int(t['row'])}, kol. {int(t['col'])}"),
            ])
            with st.container(key="bd_inset"):
                st.image(crop_tile(image, t), width="stretch")
            st.caption(f"px {int(t['x0'])}–{int(t['x1'])} × {int(t['y0'])}–{int(t['y1'])}")
        else:
            hint("Tryk <b>▶</b> i sidepanelet for at gå felterne igennem i prioriteret "
                 "rækkefølge. Det aktuelle felt vises forstørret her.")

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

    theme.side_heading("Visning")
    mode = st.sidebar.radio("Vis", ["Klynger", "Enkelte kerner", "Begge"], index=0,
                            horizontal=True, label_visibility="collapsed")
    show_clusters = mode in ("Klynger", "Begge")
    show_nuclei = mode in ("Enkelte kerner", "Begge")
    min_size = st.sidebar.slider("Vis klynger med mindst … kerner", 1, 10, 1,
                                 help="Skjuler små klynger, så et tæt felt kan koges ned til de vigtigste steder.")

    theme.side_heading("Billede")
    image_names = sorted(set(nuclei_df["image"]) | set(clusters_df["image"]))
    if not image_names:
        st.warning("Ingen kerner fundet i nogen billeder.")
        st.stop()
    image_name = pick_image(image_names, key="c1n")
    nuclei = nuclei_df[nuclei_df["image"] == image_name]
    all_clusters = clusters_df[clusters_df["image"] == image_name]
    clusters = all_clusters[all_clusters["n_nuclei"] >= min_size].sort_values("priority_rank").copy()
    clusters["priority_rank"] = range(1, len(clusters) + 1)  # renumber after filtering
    n_clusters = len(clusters)

    theme.side_heading("Gennemgang (klynger)")
    current = step_controls(f"c1n_step_{image_name}", n_clusters, key="c1n", noun="klynge")

    theme.legend("Forklaring", [
        (rgb(COLOR_CLUSTER), True, "<b>Klynge</b> af kerner · tal = rækkefølge, flest kerner først"),
        (rgb(COLOR_NUCLEUS), True, "<b>Enkelt kerne</b>"),
        (rgb(COLOR_CURRENT), True, "<b>Aktuel klynge</b>"),
    ])
    st.sidebar.info("Kerner = mørke, farvede, runde objekter · klynge = kerner < 60 px fra hinanden")

    image = load_image_or_stop(image_name)
    theme.topbar("Case 1 · Kerner", "Hvor er der noget at kigge på?",
                 sample_id(image_name), position(image_names, image_name))

    col_image, col_info = st.columns([3, 1.1])
    with col_image:
        st.image(draw_nuclei_overlay(image, nuclei, clusters, show_clusters, show_nuclei,
                                     current_rank=current or None), width="stretch")

    with col_info:
        rows = [
            ("Kerner fundet", f"{len(nuclei)}", "", True),
            ("Klynger", f"{n_clusters}" + (f" af {len(all_clusters)}" if min_size > 1 else "")),
        ]
        if n_clusters:
            rows.append(("Største klynge", f"{int(clusters['n_nuclei'].max())} kerner"))
        theme.stats_card("Billedet", rows)

        if current:
            c = clusters.iloc[current - 1]
            theme.stats_card(f"Klynge #{current} af {n_clusters}", [
                ("Kerner i klyngen", f"{int(c['n_nuclei'])}"),
                ("Centrum", f"({int(c['cx'])}, {int(c['cy'])})"),
                ("Radius", f"{int(c['radius'])} px"),
            ])
            with st.container(key="bd_inset"):
                st.image(crop_circle(image, c["cx"], c["cy"], c["radius"]), width="stretch")
        else:
            hint("Tryk <b>▶</b> i sidepanelet for at gå klyngerne igennem, største først. "
                 "Den aktuelle klynge vises forstørret her.")

    with st.expander("Alle klynger"):
        st.dataframe(clusters[["priority_rank", "n_nuclei", "cx", "cy", "radius"]].round(1)
                     .reset_index(drop=True), width="stretch")
    with st.expander("Alle kerner"):
        st.dataframe(nuclei[["nucleus_id", "x", "y", "radius", "area", "circularity", "from_clump", "cluster_id"]]
                     .round(2).reset_index(drop=True), width="stretch")


def verdict_dot(label) -> str:
    color = theme.VERDICT_COLORS.get(label, "transparent")
    border = "" if label in theme.VERDICT_COLORS else f"border:1px dashed {theme.NEUTRAL};"
    return f'<span class="dot" style="background:{color};{border}"></span>'


def render_case2_overview():
    if not CASE2_TRIAGE.exists():
        st.error("Run features.py and triage.py first - triage_results.csv not found.")
        st.stop()
    triage_df = pd.read_csv(CASE2_TRIAGE)
    if "human_eval" not in triage_df:
        triage_df["human_eval"] = pd.NA

    theme.side_heading("Filter")
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
    theme.side_heading("Status")
    theme.progress_card(n_reviewed, len(triage_df), n_agree)
    theme.legend("Forklaring", [
        (theme.OK, False, "<b>egnet</b>"),
        (theme.WARN, False, "<b>suboptimal</b>"),
        (theme.BAD, False, "<b>uegnet</b>"),
    ])

    theme.topbar("Case 2 · Oversigt", "Alle prøver med auto-vurdering og din vurdering",
                 counter=f"{len(df)} af {len(triage_df)} billeder vist")

    if df.empty:
        hint("Ingen billeder matcher filteret.")
        return

    with st.container(key="bd_overview"):
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
                    human = row["human_eval"] if pd.notna(row["human_eval"]) else None
                    st.markdown(
                        f'<div class="bd-thumb-meta">'
                        f'<span class="pair">{verdict_dot(row["triage"])}<span class="who">Auto</span>'
                        f'<span class="val">{row["triage"]}</span></span>'
                        f'<span class="pair">{verdict_dot(human)}<span class="who">Din</span>'
                        f'<span class="val">{human or "–"}</span></span>'
                        f'</div><div class="bd-thumb-id">{row["image"][:16]}…</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Åbn", key=f"open_{row['image']}", width="stretch"):
                        st.session_state["pending_nav"] = {"case": "Case 2", "sub": "Enkelt billede", "image": row["image"]}
                        st.rerun()


def render_case2():
    if not CASE2_FEATURES.exists() or not CASE2_TRIAGE.exists():
        st.error("Run features.py and triage.py first - features.csv / triage_results.csv not found.")
        st.stop()

    tiles_df, triage_df = load_case2()
    labels_df = load_manual_labels()

    theme.side_heading("Filter")
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

    theme.side_heading("Billede")
    image_name = pick_image(image_names, key="c2")

    n_reviewed = int(triage_df["human_eval"].notna().sum()) if "human_eval" in triage_df else 0
    n_agree = int(triage_df["agrees"].fillna(False).astype(bool).sum()) if "agrees" in triage_df else 0
    theme.side_heading("Status")
    theme.progress_card(n_reviewed, len(triage_df), n_agree)
    theme.legend("Forklaring", [
        (rgb(COLOR_USABLE), False, "<b>Brugbart felt</b>"),
        (rgb(COLOR_FAILED), False, "<b>Ikke brugbart</b>, med årsag:"),
        (rgb(COLOR_FAILED), "C", "for lidt cellemateriale"),
        (rgb(COLOR_FAILED), "B", "uskarpt <i>(kun felter med celler)</i>"),
        (rgb(COLOR_FAILED), "D", "for meget debris <i>(kun felter med celler)</i>"),
    ])
    st.sidebar.info(grid_info(tiles_df))

    image_row = triage_df[triage_df["image"] == image_name].iloc[0]
    image_tiles = tiles_df[tiles_df["image"] == image_name]
    image = load_image_or_stop(image_name)

    theme.topbar("Case 2 · Triagering", "Er prøven egnet til vurdering?",
                 sample_id(image_name), position(image_names, image_name))

    col_image, col_info = st.columns([3, 1.1])
    with col_image:
        st.image(draw_triage_overlay(image, image_tiles), width="stretch")

    with col_info:
        msg_key = f"saved_msg_{image_name}"
        if st.session_state.get(msg_key):
            st.markdown(f'<div class="bd-saved">✓ Din vurdering (<b>{st.session_state.pop(msg_key)}</b>) er gemt</div>',
                        unsafe_allow_html=True)

        existing = image_row.get("human_eval")
        human = existing if pd.notna(existing) else None
        when = image_row.get("human_eval_at", "")
        theme.verdict_card(image_row["triage"], human, human == image_row["triage"],
                           when if isinstance(when, str) else "")

        with st.container(key="bd_verdict_btns"):
            st.markdown('<p class="bd-h" style="margin:0">Sæt din vurdering</p>', unsafe_allow_html=True)
            for col, category in zip(st.columns([1, 1.4, 1]), TRIAGE_CATEGORIES):  # 'suboptimal' is the long word
                if col.button(category, key=f"label_{category}", width="stretch"):
                    save_manual_label(image_name, category)
                    st.session_state[msg_key] = category
                    st.rerun()

        blur, debris = image_row["mean_blur_cellular"], image_row["mean_debris_cellular"]
        theme.stats_card("Nøgletal", [
            ("Brugbare felter", f"{int(image_row['usable_tiles'])} / {int(image_row['total_tiles'])}", "", True),
            ("Felter med celler", f"{int(image_row['cellular_tiles'])} / {int(image_row['total_tiles'])}"),
            ("Gns. cellularitet", f"{image_row['mean_cellularity']:.3f}"),
            ("Fokus (felter m. celler)", "n/a" if pd.isna(blur) else f"{blur:.1f}"),
            ("Debris (felter m. celler)", "n/a" if pd.isna(debris) else f"{debris:.1f}"),
        ])

    with st.expander("Alle felter"):
        st.dataframe(
            image_tiles[["row", "col", "cellularity", "blur_score", "debris_score",
                         "has_cells", "in_focus", "is_clean", "tile_usable", "reasons"]]
            .sort_values(["row", "col"]).reset_index(drop=True),
            width="stretch",
        )


def apply_pending_navigation():
    """Widget state can only be set before the widget is drawn, so jumps
    requested during the previous run (e.g. 'Åbn' in the overview) are
    applied here, first thing."""
    nav = st.session_state.pop("pending_nav", None)
    if not nav:
        return
    st.session_state["case"] = nav["case"]
    st.session_state[f"sub_{nav['case']}"] = nav["sub"]
    if "image" in nav:
        st.session_state["c2_jump_to"] = nav["image"]
        # Reset single-view filters so the requested image is certainly in the list.
        st.session_state["c2_filter_auto"] = TRIAGE_CATEGORIES
        st.session_state["c2_only_unlabeled"] = False


def main():
    st.set_page_config(page_title="Billeddiagnostik · Cytologi", page_icon="🔬", layout="wide")
    theme.inject()
    apply_pending_navigation()

    st.sidebar.markdown('<div class="bd-brand">Billeddiagnostik<span>Cytologi · Patologi, AUH Skejby</span></div>',
                        unsafe_allow_html=True)
    case = st.sidebar.radio("Vælg case", list(SUBVIEWS.keys()), key="case")
    theme.side_note(CASE_DESCRIPTIONS[case])
    sub = st.sidebar.radio("Visning", SUBVIEWS[case], key=f"sub_{case}", horizontal=True)

    if case == "Case 1" and sub == "Felter":
        render_case1()
    elif case == "Case 1" and sub == "Kerner":
        render_case1_nuclei()
    elif case == "Case 2" and sub == "Oversigt":
        render_case2_overview()
    else:
        render_case2()

    theme.side_heading("Om visningen")
    theme.side_note(CASES[f"{case} - {sub}"])


if __name__ == "__main__":
    main()
