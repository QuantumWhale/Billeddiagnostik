"""
Det visuelle lag til Billeddiagnostiks gennemgangsinterface.

Designmålet er en rolig klinisk arbejdsstation og ikke et demo-dashboard.
Tre principper styrer alt herinde:

1. Præparatet er det eneste mættede på skærmen. Cytologifarvninger er
   lyserøde/lilla, så interfacet er bygget op af afmættede, kølige grå nuancer,
   og det konkurrerer derfor aldrig med - eller ændrer opfattelsen af - det
   materiale, der bliver gennemgået.
2. Skærmen er et fast budget. En patolog skal kunne nå frem til en vurdering
   uden at scrolle, så den lodrette plads bruges med omtanke: billederne har en
   maksimal højde i forhold til viewporten, og tal vises som kompakte
   label/værdi-rækker i stedet for Streamlits meget høje standard-metrics.
3. Status vises med farve *og* tekst, aldrig kun med farve.

Det hele er almindelig CSS, der sprøjtes ind én gang pr. kørsel, så appen
virker præcis som før, hvis man fjerner modulet.
"""

import re

import streamlit as st

# --- Klinisk palet -----------------------------------------------------------
# Kølige, neutrale farver med lav kroma. Den eneste accentfarve er en
# petroleumsblå, der minder om et instrument.
INK = "#132434"          # primær tekst
INK_SOFT = "#5c6f80"     # sekundær tekst
LINE = "#d5dde5"         # tynde streger og kortkanter
SURFACE = "#ffffff"      # kort
GROUND = "#eaeef3"       # appens baggrund
ACCENT = "#0d6a8a"       # navigation, fokus, links

# Farver til vurderingerne: de er forskellige i både farvetone *og* lyshed,
# så de stadig kan skelnes med rød-grøn farveblindhed, som ~8 % af de
# mandlige klinikere har.
OK = "#1a7f4b"           # egnet
WARN = "#a86a08"         # suboptimal
BAD = "#b8372b"          # uegnet
NEUTRAL = "#7b8a99"      # endnu ikke vurderet

VERDICT_COLORS = {
    "egnet": OK,
    "suboptimal": WARN,
    "uegnet": BAD,
}

FONT_STACK = (
    'system-ui, -apple-system, "Segoe UI", "Source Sans", '
    "Roboto, Helvetica, Arial, sans-serif"
)

CSS = f"""
<style>
:root {{
  --ink: {INK};
  --ink-soft: {INK_SOFT};
  --line: {LINE};
  --surface: {SURFACE};
  --ground: {GROUND};
  --accent: {ACCENT};
  --ok: {OK};
  --warn: {WARN};
  --bad: {BAD};
  --neutral: {NEUTRAL};
  /* Højdebudget for hovedbilledet: viewporten minus topbjælken, de
     sammenklappede tabel-expanders og padding. Billedet bliver større på en
     stor diagnostikskærm og mindre på en laptop, men skubber aldrig siden ud i
     scroll. */
  --img-h: calc(100vh - 265px);
  --img-h-inset: 21vh;   /* det zoomede udsnit i højre kolonne */
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  font-family: {FONT_STACK};
  color: var(--ink);
}}
[data-testid="stApp"] {{ background: var(--ground); }}

/* --- Genvind lodret plads ---------------------------------------------------
   Streamlit kommer med 96px padding i toppen og 160px i bunden. På en 900px
   laptopskærm er det over en fjerdedel af den brugbare højde, der går til
   ingenting. */
[data-testid="stMainBlockContainer"] {{
  padding-top: 1.1rem;
  padding-bottom: 1.6rem;
  padding-left: 1.8rem;
  padding-right: 1.8rem;
  max-width: 1800px;
}}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
/* Streamlit giver markdown-blokke en bundmargen på -1rem for at modvirke
   afstanden mellem afsnit. Vores HTML-blokke har deres egen afstand, så den
   negative margen fik overskrifter og kort til at glide ind under den næste
   kontrol. */
[data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
[data-testid="stMarkdownContainer"] > p:last-child {{ margin-bottom: 0; }}
[data-testid="stMain"] .stElementContainer {{ margin-bottom: 0; }}
[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {{ gap: 1rem; }}

/* --- Topbjælke -------------------------------------------------------------
   Erstatter den alt for store sidetitel. Den viser det, man skal bruge for at
   holde styr på, hvor man er: hvilken visning, hvilken prøve, og hvor langt man
   er nået i serien. */
.bd-topbar {{
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.55rem 1.1rem;
  padding: 0.3rem 0 0.7rem 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 0.85rem;
}}
.bd-topbar .t-title {{
  font-size: 1.12rem; font-weight: 650; letter-spacing: -0.01em;
  color: var(--ink); margin-right: 0.2rem;
}}
.bd-topbar .t-sub {{ font-size: 0.82rem; color: var(--ink-soft); }}
.bd-topbar .t-right {{ margin-left: auto; display: flex; align-items: baseline; gap: 1.1rem; }}
.bd-topbar .t-id {{
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 0.78rem; color: var(--ink-soft);
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 3px; padding: 0.16rem 0.45rem;
}}
.bd-topbar .t-count {{
  font-size: 0.8rem; color: var(--ink-soft); font-variant-numeric: tabular-nums;
}}

/* --- Kort ------------------------------------------------------------------ */
.bd-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 0.6rem 0.75rem;
  margin-bottom: 0.55rem;
}}
/* Kvalificeret med p, så Streamlits egne afsnitsstile ikke vinder. */
.bd-h, [data-testid="stMarkdownContainer"] p.bd-h {{
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--ink-soft);
  margin: 0 0 0.45rem 0;
}}

/* --- Visning af vurderinger -------------------------------------------------
   Auto- og menneskevurderingen står side om side: sammenligningen er hele
   pointen med Case 2, så den skal kunne ses med ét blik og ikke to. */
.bd-verdicts {{ display: flex; gap: 0.5rem; }}
.bd-verdicts > div {{ flex: 1; min-width: 0; }}
.bd-vlabel {{
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--ink-soft); margin-bottom: 0.2rem;
}}
.bd-chip {{
  display: block; text-align: center;
  font-size: 0.95rem; font-weight: 650; letter-spacing: -0.01em;
  padding: 0.34rem 0.4rem; border-radius: 4px;
  color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.bd-chip.egnet {{ background: var(--ok); }}
.bd-chip.suboptimal {{ background: var(--warn); }}
.bd-chip.uegnet {{ background: var(--bad); }}
.bd-chip.tom {{
  background: transparent; color: var(--neutral);
  border: 1px dashed var(--neutral); font-weight: 550;
}}
.bd-agree {{
  font-size: 0.72rem; margin-top: 0.35rem; color: var(--ink-soft);
  display: flex; align-items: center; gap: 0.3rem;
}}
.bd-agree b {{ font-weight: 650; }}
.bd-agree.enig b {{ color: var(--ok); }}
.bd-agree.uenig b {{ color: var(--bad); }}

/* --- Kompakt statistik ------------------------------------------------------
   st.metric tegner en ~90px høj blok per tal. Fem af dem skubbede
   vurderingsknapperne helt ud af skærmen; de her rækker er ~24px hver. */
.bd-stats {{ display: flex; flex-direction: column; }}
.bd-stat {{
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 0.6rem; padding: 0.2rem 0;
  border-bottom: 1px solid #eef2f6;
  font-size: 0.84rem;
}}
.bd-stat:last-child {{ border-bottom: 0; }}
.bd-stat .k {{ color: var(--ink-soft); }}
.bd-stat .v {{
  font-weight: 650; font-variant-numeric: tabular-nums;
  color: var(--ink); white-space: nowrap;
}}
.bd-stat.lead {{ font-size: 0.98rem; padding: 0.1rem 0 0.3rem 0; }}
.bd-stat.lead .v {{ font-size: 1.15rem; font-weight: 700; }}
.bd-stat .v.warn {{ color: var(--warn); }}
.bd-stat .v.bad {{ color: var(--bad); }}

/* --- Billeder --------------------------------------------------------------
   Begræns højden frem for bredden: et præparat på 1376x1020 i fuld
   kolonnebredde er højere end skærmen. */
[data-testid="stMain"] [data-testid="stImage"] img {{
  max-height: var(--img-h);
  width: auto !important;
  max-width: 100%;
  display: block; margin: 0 auto;
  border-radius: 4px;
  border: 1px solid var(--line);
  background: var(--surface);
}}
[data-testid="stMain"] [data-testid="stImageContainer"],
[data-testid="stMain"] [data-testid="stFullScreenFrame"] > div {{
  display: flex; justify-content: center;
}}
/* Det indsatte udsnit i højre kolonne får et meget mindre budget. */
.st-key-bd_inset [data-testid="stImage"] img {{
  max-height: var(--img-h-inset);
  border-color: #c8d2dc;
}}
[data-testid="stImageCaption"] {{
  font-size: 0.72rem; color: var(--ink-soft); padding-top: 0.25rem;
}}

/* --- Sidepanel -------------------------------------------------------------- */
.bd-brand {{
  font-size: 0.98rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ink);
  padding: 0 0 0.55rem 0; border-bottom: 1px solid var(--line); margin-bottom: 0.3rem;
}}
.bd-brand span {{
  display: block; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.02em;
  color: var(--ink-soft); margin-top: 0.1rem;
}}
.bd-hint {{ font-size: 0.8rem; color: var(--ink-soft); line-height: 1.4; }}
[data-testid="stSidebar"] {{
  background: #f6f8fa;
  border-right: 1px solid var(--line);
}}
[data-testid="stSidebarUserContent"] {{ padding: 1rem 1rem 2rem 1rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.4rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  font-size: 0.7rem !important; font-weight: 700 !important;
  letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--ink-soft) !important;
  padding: 0.55rem 0 0.1rem 0 !important; margin: 0 !important;
}}
[data-testid="stSidebar"] hr {{ margin: 0.5rem 0; border-color: var(--line); }}

/* Primær case-navigation - med vilje den største tekst i sidepanelet, fordi
   det er den eneste kontrol, der ændrer, hvad hele skærmen betyder. */
.st-key-case [data-testid="stWidgetLabel"] {{ display: none; }}
.st-key-case, .st-key-case [data-testid="stRadio"],
.st-key-case [role="radiogroup"] {{ width: 100% !important; }}
.st-key-case [role="radiogroup"] {{ gap: 0.15rem; }}
.st-key-case [role="radiogroup"] > label {{
  padding: 0.4rem 0.5rem; border-radius: 4px; width: 100%;
  border: 1px solid transparent;
}}
.st-key-case [role="radiogroup"] > label:hover {{ background: #e9eff4; }}
.st-key-case label p {{
  font-size: 1.14rem !important;
  font-weight: 650 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}}
.st-key-case [role="radiogroup"] > label:has(input:checked) {{
  background: #e3edf2; border-color: #bcd4de;
}}
.st-key-case [role="radiogroup"] > label:has(input:checked) p {{ color: var(--accent) !important; }}

/* Skiftet mellem delvisninger ser ud som en segmenteret kontrol under casen. */
[class*="st-key-sub_"] [data-testid="stWidgetLabel"] {{ display: none; }}
[class*="st-key-sub_"] label p {{ font-size: 0.9rem !important; font-weight: 550 !important; }}

.bd-side-note {{
  font-size: 0.78rem; line-height: 1.35; color: var(--ink-soft);
  padding: 0.1rem 0 0.5rem 0;
}}
.bd-side-note em {{ color: var(--ink); font-style: italic; }}

/* Signaturforklaring: swatches med fast bredde, så rækkerne står på linje. */
.bd-legend {{ font-size: 0.78rem; line-height: 1.5; color: var(--ink-soft); }}
.bd-legend .row {{ display: flex; align-items: flex-start; gap: 0.45rem; margin-bottom: 0.18rem; }}
.bd-legend .sw {{
  flex: 0 0 auto; width: 11px; height: 11px; border-radius: 2px;
  margin-top: 0.28rem; border: 1px solid rgba(0,0,0,0.15);
}}
.bd-legend .sw.ring {{ background: transparent; border-width: 2px; border-radius: 50%; }}
.bd-legend b {{ color: var(--ink); font-weight: 600; }}
/* Årsagskoderne tegnes som de samme røde hjørnemærker, der bruges på billedet. */
.bd-legend .code {{
  flex: 0 0 auto; min-width: 15px; text-align: center;
  font-size: 0.64rem; font-weight: 700; line-height: 1.45; color: #fff;
  border-radius: 2px; padding: 0 0.2rem; margin-top: 0.08rem;
}}

/* Visning af fremskridt i sidepanelet. */
.bd-progress {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 4px; padding: 0.45rem 0.55rem; font-size: 0.8rem;
}}
.bd-progress .row {{ display: flex; justify-content: space-between; gap: 0.5rem; }}
.bd-progress .v {{ font-weight: 650; font-variant-numeric: tabular-nums; }}
.bd-bar {{
  height: 4px; border-radius: 2px; background: #dde4ea;
  margin-top: 0.35rem; overflow: hidden;
}}
.bd-bar > span {{ display: block; height: 100%; background: var(--accent); }}

/* --- Widgets ---------------------------------------------------------------- */
[data-testid="stMain"] .stButton button, [data-testid="stSidebar"] .stButton button {{
  border-radius: 4px; border: 1px solid var(--line);
  font-weight: 550; font-size: 0.84rem;
  padding: 0.28rem 0.6rem; min-height: 0; color: var(--ink);
  background: var(--surface); transition: none;
}}
[data-testid="stMain"] .stButton button:hover, [data-testid="stSidebar"] .stButton button:hover {{
  border-color: var(--accent); color: var(--accent); background: #f2f8fa;
}}

/* De tre vurderingsknapper er hovedhandlingen i Case 2: farvekodede, i fuld
   bredde og placeret lige under den vurdering, de ændrer. */
.st-key-bd_verdict_btns [data-testid="stHorizontalBlock"] {{ gap: 0.35rem; }}
.st-key-bd_verdict_btns .stButton button {{
  width: 100%; font-weight: 650; font-size: 0.86rem;
  padding: 0.34rem 0.15rem; color: #fff;
}}
.st-key-bd_verdict_btns .stButton button p {{ font-size: inherit; white-space: nowrap; }}
.st-key-label_egnet button {{ background: var(--ok) !important; border-color: var(--ok) !important; }}
.st-key-label_suboptimal button {{ background: var(--warn) !important; border-color: var(--warn) !important; }}
.st-key-label_uegnet button {{ background: var(--bad) !important; border-color: var(--bad) !important; }}
.st-key-bd_verdict_btns .stButton button:hover {{ filter: brightness(1.1); color: #fff !important; }}
.st-key-bd_verdict_btns .stButton button p {{ color: #fff !important; }}

[data-testid="stExpander"] {{
  border: 1px solid var(--line); border-radius: 5px; background: var(--surface);
}}
[data-testid="stExpander"] summary {{ font-size: 0.82rem; font-weight: 600; color: var(--ink-soft); }}
[data-testid="stExpander"] summary:hover {{ color: var(--accent); }}

[data-testid="stSidebar"] [data-testid="stAlertContainer"] {{
  background: #eaf1f5; border: 1px solid #cfdde5; border-radius: 4px;
  padding: 0.45rem 0.55rem; font-size: 0.76rem; color: var(--ink-soft);
}}
[data-testid="stSidebar"] [data-testid="stAlertContainer"] p {{ font-size: 0.76rem; }}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] {{
  border-radius: 4px; border-color: var(--line); font-size: 0.84rem;
}}
[data-testid="stWidgetLabel"] p {{ font-size: 0.78rem; color: var(--ink-soft); }}
[data-testid="stCaptionContainer"] p {{ font-size: 0.76rem; color: var(--ink-soft); }}

[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

/* Oversigtsgitter: hvert thumbnail er et kort, hvis venstre kant viser vurderingen. */
.bd-thumb-meta {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.4rem; font-size: 0.75rem; padding: 0.1rem 0 0.2rem 0;
}}
.bd-thumb-meta .pair {{ display: flex; align-items: center; gap: 0.3rem; }}
.bd-thumb-meta .dot {{
  width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto;
  border: 1px solid rgba(0,0,0,0.12);
}}
.bd-thumb-meta .who {{ color: var(--ink-soft); }}
.bd-thumb-meta .val {{ color: var(--ink); font-weight: 600; }}
.bd-thumb-id {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.68rem; color: var(--neutral);
  line-height: 1.4; padding-bottom: 0.25rem;
}}
.st-key-bd_overview [data-testid="stImage"] img {{ max-height: none; }}

/* Bekræftelsesstribe, når noget er gemt. */
.bd-saved {{
  background: #e7f3ec; border: 1px solid #b9dcc7; color: #14603a;
  border-radius: 4px; padding: 0.3rem 0.5rem; font-size: 0.79rem;
  margin-bottom: 0.45rem;
}}
</style>
"""


def inject() -> None:
    """Lægger stylesheetet på. Kan roligt kaldes ved hver eneste rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Små hjælpefunktioner til rendering. De returnerer ingenting og skriver hver
# én markdown-blok, så DOM'en holdes flad og den lodrette rytme forudsigelig.
# ---------------------------------------------------------------------------

def topbar(title: str, subtitle: str = "", sample_id: str = "", counter: str = "") -> None:
    right = ""
    if sample_id:
        right += f'<span class="t-id">{sample_id}</span>'
    if counter:
        right += f'<span class="t-count">{counter}</span>'
    st.markdown(
        f'<div class="bd-topbar">'
        f'<span class="t-title">{title}</span>'
        f'<span class="t-sub">{subtitle}</span>'
        f'<span class="t-right">{right}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _stat_row(label: str, value: str, tone: str = "", lead: bool = False) -> str:
    cls = "bd-stat lead" if lead else "bd-stat"
    return f'<div class="{cls}"><span class="k">{label}</span><span class="v {tone}">{value}</span></div>'


def stats_card(heading: str, rows: list) -> None:
    """rows: liste af (label, value) eller (label, value, tone) eller (label, value, tone, lead)."""
    body = "".join(_stat_row(*r) for r in rows)
    head = f'<p class="bd-h">{heading}</p>' if heading else ""
    st.markdown(
        f'<div class="bd-card">{head}<div class="bd-stats">{body}</div></div>',
        unsafe_allow_html=True,
    )


def verdict_card(auto: str, human: str | None, agrees: bool | None, when: str = "") -> None:
    """Auto-vurderingen og brugerens egen vurdering side om side."""
    auto_cls = auto if auto in VERDICT_COLORS else "tom"
    if human:
        human_html = f'<span class="bd-chip {human}">{human}</span>'
    else:
        human_html = '<span class="bd-chip tom">ikke vurderet</span>'

    note = ""
    if human:
        word = "enig" if agrees else "uenig"
        stamp = f" · {when[:16].replace('T', ' ')}" if when else ""
        note = f'<div class="bd-agree {word}"><b>{word}</b> med auto{stamp}</div>'

    st.markdown(
        f'<div class="bd-card">'
        f'<div class="bd-verdicts">'
        f'<div><div class="bd-vlabel">Auto-vurdering</div>'
        f'<span class="bd-chip {auto_cls}">{auto}</span></div>'
        f'<div><div class="bd-vlabel">Din vurdering</div>{human_html}</div>'
        f"</div>{note}</div>",
        unsafe_allow_html=True,
    )


def legend(title: str, items: list) -> None:
    """items: liste af (colour, marker, html_text). marker er False for en udfyldt
    swatch, True for en ring eller en kort streng som fx "C" for et kodemærke."""
    rows = ""
    for color, marker, text in items:
        if isinstance(marker, str):
            mark = f'<span class="code" style="background:{color}">{marker}</span>'
        elif marker:
            mark = f'<span class="sw ring" style="border-color:{color}"></span>'
        else:
            mark = f'<span class="sw" style="background:{color}"></span>'
        rows += f'<div class="row">{mark}<span>{text}</span></div>'
    st.sidebar.markdown(
        f'<p class="bd-h" style="margin-bottom:.3rem">{title}</p>'
        f'<div class="bd-legend">{rows}</div>',
        unsafe_allow_html=True,
    )


def side_heading(text: str) -> None:
    st.sidebar.markdown(f'<p class="bd-h" style="margin:.5rem 0 0 0">{text}</p>', unsafe_allow_html=True)


def side_note(text: str) -> None:
    # Markdown bliver ikke parset inde i rå HTML, så vi laver selv *fremhævning* om.
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    st.sidebar.markdown(f'<div class="bd-side-note">{text}</div>', unsafe_allow_html=True)


def progress_card(reviewed: int, total: int, agree: int | None = None) -> None:
    pct = (reviewed / total * 100) if total else 0
    agree_row = ""
    if agree is not None and reviewed:
        agree_row = (
            f'<div class="row"><span>Enighed</span>'
            f'<span class="v">{agree} / {reviewed}</span></div>'
        )
    st.sidebar.markdown(
        f'<div class="bd-progress">'
        f'<div class="row"><span>Vurderet</span><span class="v">{reviewed} / {total}</span></div>'
        f'{agree_row}<div class="bd-bar"><span style="width:{pct:.0f}%"></span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
