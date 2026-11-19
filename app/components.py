"""HTML sections and the palette for the RUL Predictor interface.

Everything visual lives here: the stylesheet is loaded from ``styles.css``
and the page is assembled from small HTML builders. ``app.py`` only decides
*which* section to show and supplies the values.

Streamlit's own theme is read with ``st.context.theme`` (light/dark) so the
palette follows the appearance picked in the app menu.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
_STYLES = (_HERE / "styles.css").read_text(encoding="utf-8")

# --------------------------------------------------------------------------
# Palette: the same token names, two moods. The light values also sit in
# styles.css so the page renders correctly before the palette is injected.
# --------------------------------------------------------------------------
LIGHT = {
    "bg": "#f4f4f3",
    "surface": "#ffffff",
    "surface-2": "#fbfbfa",
    "surface-3": "#ededec",
    "ink": "#0a0a0a",
    "ink-2": "#28282b",
    "muted": "#6d6d74",
    "faint": "#9a9aa1",
    "line": "#e3e3e1",
    "line-strong": "#cfcfcd",
    "accent": "#1d4ed8",
    "ok": "#15803d",
    "warn": "#b45309",
    "danger": "#b42318",
    "on-ink": "#ffffff",
    "invert-bg": "#0a0a0a",
    "invert-fg": "#ffffff",
    "shadow-color": "rgba(10, 10, 10, 0.08)",
}

# OLED black rather than a lifted grey: cards sit a hair above the page so
# borders still read on pure-black screens.
DARK = {
    "bg": "#000000",
    "surface": "#080808",
    "surface-2": "#0c0c0c",
    "surface-3": "#141414",
    "ink": "#f5f5f5",
    "ink-2": "#d6d6d9",
    "muted": "#a0a0a6",
    "faint": "#6e6e75",
    "line": "#1e1e20",
    "line-strong": "#313135",
    "accent": "#7aa2ff",
    "ok": "#5bd68c",
    "warn": "#f5c451",
    "danger": "#ff7b72",
    "on-ink": "#000000",
    "invert-bg": "#0d0d0d",
    "invert-fg": "#f5f5f5",
    "shadow-color": "rgba(0, 0, 0, 0.65)",
}


def active_theme() -> str:
    """'dark' or 'light', following the Streamlit appearance setting."""
    try:
        theme = st.context.theme
        value = theme.get("type") if hasattr(theme, "get") else None
        # Older/alternate Streamlit builds expose the same value as an attribute.
        if value is None:
            value = getattr(theme, "type", None) or getattr(theme, "base", None)
    except Exception:
        value = None
    return "dark" if value == "dark" else "light"


def palette() -> dict[str, str]:
    """The colour tokens for the active appearance."""
    return DARK if active_theme() == "dark" else LIGHT


# Altair draws into a canvas, so its colours are fixed when the script runs.
# Keep the chart on theme-neutral tones that stay legible on both backgrounds;
# that way it does not need a rerun after the appearance switch.
CHART = {
    "ink": "#4f6ed8",
    "muted": "#7a7a83",
    "faint": "#7a7a83",
    "line": "rgba(128, 128, 128, 0.28)",
}


def chart_palette() -> dict[str, str]:
    """Colours for the canvas chart, readable in light and dark alike."""
    return CHART


def apply_theme() -> None:
    """Inject the stylesheet plus both palettes.

    Streamlit paints the active light/dark setting as the ``color-scheme`` of
    its root ``.stApp`` element and updates that value locally when the user
    flips the appearance switch. Pairing each token with ``light-dark()`` on
    that element means the custom UI follows the switch immediately, without
    waiting for a rerun. The active palette is still written to :root as a
    fallback for older Streamlit builds and for portalled popovers.
    """
    scheme = active_theme()
    variables = "\n".join(f"  --{name}: {value};" for name, value in palette().items())
    live_variables = "\n".join(
        f"  --{name}: light-dark({LIGHT[name]}, {DARK[name]});" for name in LIGHT
    )
    live_variables += (
        "\n  --shadow-sm: 0 1px 2px var(--shadow-color);"
        "\n  --shadow-md: 0 28px 60px -38px var(--shadow-color);"
    )
    _html(
        f"<style>\n{_STYLES}\n"
        f":root {{\n{variables}\n  color-scheme: {scheme};\n}}\n"
        f"@supports (color: light-dark(#fff, #000)) {{\n"
        f".stApp {{\n{live_variables}\n}}\n"
        f"}}\n</style>"
    )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _html(markup: str) -> None:
    """Render raw HTML without Streamlit's sanitizer.

    ``st.html`` strips <style> tags and inline SVG, so the sections go through
    ``st.markdown(unsafe_allow_html=True)`` instead. The wrapper lets the
    stylesheet cancel Streamlit's negative markdown margin, and blank lines
    are removed so Markdown never ends the HTML block early.
    """
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    st.markdown(
        '<div class="rul-block">' + "\n".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


_BRAND_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M4.6 16.5a8.5 8.5 0 1 1 14.8 0"></path>
  <path d="M12 12.6 16.2 8.4"></path>
  <circle cx="12" cy="13.4" r="1.25" fill="currentColor" stroke="none"></circle>
</svg>
"""


def header_and_hero(dataset: str, compact: bool = False, show_eyebrow: bool = True) -> None:
    """The large title block."""
    hero_class = "hero is-compact" if compact else "hero"
    if not show_eyebrow:
        hero_class += " no-eyebrow"
    eyebrow = (
        f'<p class="eyebrow"><span class="dot"></span> Remaining useful life · {_e(dataset)}</p>'
        if show_eyebrow else ""
    )
    _html(
        f"""
<section class="{hero_class}" id="top">
  {eyebrow}
  <h1>RUL Predictor</h1>
  <p class="lede">Remaining useful life of a turbofan engine, read from its last cycles of
  sensor history. Pick a fleet and a model — all preloaded — then drop in the engine data.</p>
</section>
"""
    )


def field_head(title: str, note: str = "", meta: str = "", anchor: str = "") -> None:
    """A label that sits above a Streamlit widget."""
    extra = f'<p class="field-meta">{_e(meta)}</p>' if meta else ""
    note_html = f"<p>{_e(note)}</p>" if note else ""
    id_attr = f' id="{_e(anchor)}"' if anchor else ""
    _html(
        f'<div class="field-head"{id_attr}><h3>{_e(title)}</h3>{note_html}{extra}</div>'
    )


def fleet_head(dataset: str, title: str, note: str, anchor: str = "") -> None:
    """Top of the fleet card, in two separate blocks so the card can lay them out on its own:
    the status line stays at the top, and the title with its short note is centred in the space
    between the status line and the fleet buttons (see ``.fleet-head`` / ``.fleet-copy`` in
    styles.css)."""
    id_attr = f' id="{_e(anchor)}"' if anchor else ""
    _html(
        f"""
<div class="fleet-head"{id_attr}>
<p class="eyebrow"><span class="dot"></span> Remaining useful life · {_e(dataset)}</p>
</div>
"""
    )
    _html(
        f"""
<div class="fleet-copy">
<h3>{_e(title)}</h3>
<p>{_e(note)}</p>
</div>
"""
    )


def section_head(title: str, note: str = "", anchor: str = "") -> None:
    anchor_attr = f' id="{_e(anchor)}"' if anchor else ""
    _html(
        f'<div class="section-head"{anchor_attr}><h2>{_e(title)}</h2><p>{_e(note)}</p></div>'
    )


# --------------------------------------------------------------------------
# Right-hand card: a waiting state before a model is loaded, the model summary
# after. The upload guide lives in the info button next to the model source.
# --------------------------------------------------------------------------
def upload_guide_popover() -> None:
    """The model-file guide, folded into an info button so it does not crowd the page."""
    with st.popover("Which files do I upload?", icon=":material/info:", type="tertiary"):
        st.markdown(
            """
**Random Forest**  
`random_forest_fd00X.joblib`  
Add the preprocessor file as well if the app asks for it.

**XGBoost**  
`xgboost_fd00X.json` + `xgboost_fd00X_meta.joblib`

**LSTM · CNN-LSTM · TCN · Transformer**  
`*.safetensors` + `final_preprocessor_fd00X.joblib`
"""
        )
        st.divider()
        st.markdown(
            "Drop every file in at once — the app works out which is the model, "
            "the preprocessor and the metadata."
        )


def preprocessor_note(filename: str) -> None:
    """A small line under the model buttons: the preprocessor is shared and always loaded."""
    _html(
        f'<p class="pre-note"><span class="chip">Preprocessor</span> '
        f'<code class="k">{_e(filename)}</code> loads with every model.</p>'
    )


def empty_model_card() -> None:
    _html(
        """
<aside class="card" id="guide">
  <div class="card-head">
    <h3>No model loaded</h3>
    <span class="chip">Waiting</span>
  </div>
  <p class="card-lede">Upload your model files on the left, or switch back to
  “Fleet models” to use the preloaded ones.</p>
  <p class="card-note">Not sure which files to use? The
  “Which files do I upload?” button has the full list.</p>
</aside>
"""
    )


def fleet_specs(dataset: str, n_conditions: int) -> None:
    """Operating condition + Fault modes: the two boxes that sit under the FD001-FD004 buttons."""
    conditions = EXPECTED_LABEL.get(int(n_conditions), f"{int(n_conditions)} conditions")
    _html(
        f"""
<dl class="spec is-row is-fleet">
  <div><dt>Operating condition</dt><dd>{_e(conditions)}</dd></div>
  <div><dt>Fault modes</dt><dd>{_e(FAULT_MODES.get(dataset, "—"))}</dd></div>
</dl>
"""
    )


# Test-set scores per (fleet, model), shown in the model card. Fill these in from the notebook, e.g.
#   ("FD001", "XGBoost"): {"rmse": 15.2, "nasa": 410},
# Until a pair is listed (or the loaded model carries a ``metrics`` dict with the same keys), the
# box shows a dash.
MODEL_METRICS: dict[tuple[str, str], dict[str, float]] = {}


def _load_metrics_file() -> None:
    """Fill MODEL_METRICS from artifacts/metrics.json, written by compute_metrics.py."""
    import json
    for folder in (_HERE / "artifacts", _HERE):
        path = folder / "metrics.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            for fleet, models in data.items():
                for model, values in models.items():
                    MODEL_METRICS.setdefault((fleet, model), values)
            return


_load_metrics_file()


def _metric_text(art: dict, dataset: str, name: str, key: str, digits: int) -> str:
    metrics = art.get("metrics") or MODEL_METRICS.get((dataset, name)) or {}
    value = metrics.get(key)
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def model_card(art: dict, dataset: str, summary: str, source: str = "Preloaded") -> None:
    """Compact model summary: one title line, one row of specs."""
    pre = art["pre"]
    name = art.get("model_name") or "Trained model"
    rmse = _metric_text(art, dataset, name, "rmse", 2)
    nasa = _metric_text(art, dataset, name, "nasa", 0)
    if summary.startswith(f"{name} · "):  # the title already says which model it is
        summary = summary[len(name) + 3:]
    body = f"""
<aside class="card is-compact" id="guide">
  <div class="card-head">
    <span class="model-line">
      <span class="status" title="Model ready"><i></i></span>
      <b>{_e(name)}</b>
      <span>{_e(summary)}</span>
    </span>
    <span class="card-tags">
      <span class="chip">{_e(source)}</span>
      <span class="chip">{_e(dataset)}</span>
    </span>
  </div>
  <dl class="spec is-row">
    <div><dt>RMSE</dt><dd>{_e(rmse)}</dd></div>
    <div><dt>NASA score</dt><dd>{_e(nasa)}</dd></div>
    <div><dt>Window</dt><dd>{int(pre["window_size"])} cycles</dd></div>
    <div><dt>RUL cap</dt><dd>{pre["rul_cap"]:g}</dd></div>
  </dl>
</aside>
"""
    _html(body)


# Fault modes per C-MAPSS fleet (benchmark summary table).
FAULT_MODES = {
    "FD001": "1 (HPC)",
    "FD002": "1 (HPC)",
    "FD003": "2 (HPC and Fan)",
    "FD004": "2 (HPC and Fan)",
}

EXPECTED_LABEL = {
    1: "1 condition",
    6: "6 conditions",
}


# --------------------------------------------------------------------------
# Workspace pieces
# --------------------------------------------------------------------------
def steps_card() -> None:
    _html(
        """
<section class="steps" aria-label="How the app works">
  <div class="step">
    <span>01</span>
    <h4>Choose the fleet</h4>
    <p>FD001 to FD004. The model the training notebook picked for that fleet, with its
    sensor list, window length and scalers, is already loaded.</p>
  </div>
  <div class="step">
    <span>02</span>
    <h4>Drop the engine data</h4>
    <p>Upload the official C-MAPSS test file with one row per engine and cycle, or a CSV
    with a header. You can also type the most recent cycles by hand.</p>
  </div>
  <div class="step">
    <span>03</span>
    <h4>Read the life left</h4>
    <p>Every engine gets a predicted RUL in cycles, computed the same way as the notebook:
    windowed features, per-condition scaling, capped target.</p>
  </div>
</section>
"""
    )


def result_card(predicted: float, latest: int, cap: float) -> None:
    share = max(0.0, min(1.0, predicted / cap if cap else 0.0)) * 100.0
    healthy = predicted >= 0.6 * cap
    eyebrow = "Engine looks healthy" if healthy else "Wear is visible"
    state = "" if healthy else " is-low"
    _html(
        f"""
<section class="result{state}" aria-label="Predicted remaining useful life">
  <p class="eyebrow"><span class="dot"></span> {_e(eyebrow)}</p>
  <div class="result-row">
    <div class="result-value">{predicted:,.1f}<span>cycles left</span></div>
    <div class="result-aside">
      <div><span>Latest cycle</span><b>{latest}</b></div>
      <div><span>Training cap</span><b>{cap:g}</b></div>
    </div>
  </div>
  <div class="meter" role="img" aria-label="Predicted life as a share of the training cap">
    <i style="width:{share:.1f}%"></i>
  </div>
  <p class="result-note">The model was trained with RUL capped at {cap:g}, so a value close to
  {cap:g} means the engine looks healthy. The bar shows where this prediction sits on that scale.</p>
</section>
"""
    )


def stats(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="stat"><span>{_e(label)}</span><b>{_e(value)}</b></div>'
        for label, value in items
    )
    _html(f'<section class="stats" aria-label="Model summary">{cells}</section>')


# --------------------------------------------------------------------------
# Prescriptive maintenance: label, palette token, plain-language meaning.
# --------------------------------------------------------------------------
CONTINUE = "Continue Operation"
SERVICE = "Schedule Servicing"
GROUND = "Immediate Grounding"

_PRESCRIPTION_TOKEN = {CONTINUE: "ok", SERVICE: "warn", GROUND: "danger"}


_PRESCRIPTION_CLASS = {CONTINUE: "rx-ok", SERVICE: "rx-warn", GROUND: "rx-danger"}


def predictions_table(out, plan: float, replace: float, safety: float) -> None:
    """The per-engine table. Plain HTML so the label can be a pill hugging its text
    (a Streamlit dataframe can only colour a whole cell)."""
    rows = "".join(
        f'<tr><td class="num">{int(r.unit)}</td><td class="num">{int(r.last_cycle)}</td>'
        f'<td class="num">{r.predicted_RUL:,.1f}</td>'
        f'<td><span class="rx {_PRESCRIPTION_CLASS.get(r.prescriptive, "")}">{_e(r.prescriptive)}</span></td></tr>'
        for r in out.rename(columns={"Prescriptive maintenance": "prescriptive"}).itertuples()
    )
    _html(
        f"""
<div class="rx-shell">
<div class="rx-table-wrap" role="region" aria-label="Predicted remaining useful life for each engine" tabindex="0">
<table class="rx-table">
<thead><tr><th class="num">Engine</th><th class="num">Last cycle</th><th class="num">Predicted RUL (cycles)</th><th>Prescriptive maintenance</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<span class="rx-info">
<button type="button" class="rx-info-btn" aria-label="How the prescriptive maintenance label is decided">i</button>
<span class="rx-pop" role="tooltip"><span class="rx-pop-card">
<b>Prescriptive maintenance</b>
{_legend_markup(plan, replace, safety)}
</span></span>
</span>
</div>
"""
    )


def _legend_markup(plan: float, replace: float, safety: float) -> str:
    """What each prescriptive-maintenance label means and the RUL range behind it."""
    note = (
        f"Judged on the predicted RUL minus a {safety:g}-cycle safety margin."
        if safety else "Judged on the predicted RUL."
    )
    return f"""
<div class="rx-legend">
  <span class="rx-item"><span class="rx rx-ok">{CONTINUE}</span><small>above {plan:g} cycles · engine is fine</small></span>
  <span class="rx-item"><span class="rx rx-warn">{SERVICE}</span><small>{replace:g} to {plan:g} cycles · service or repair</small></span>
  <span class="rx-item"><span class="rx rx-danger">{GROUND}</span><small>below {replace:g} cycles · replace the engine</small></span>
  <span class="rx-note">{_e(note)}</span>
</div>
"""


def panel_head_info(title: str, note_html: str, info_title: str, info_code: str, info_note: str = "") -> None:
    """A panel heading with an "i" button at the top right; the extra detail lives in its popover.

    Uses the same hover/focus popover as the prescriptive-maintenance key (``rx-info``).
    ``note_html`` is trusted HTML written in app.py; ``info_code`` and ``info_note`` are escaped.
    """
    _html(
        f"""
<div class="panel-head has-info">
<h3>{_e(title)}</h3>
<p>{note_html}</p>
<span class="rx-info">
<button type="button" class="rx-info-btn" aria-label="{_e(info_title)}">i</button>
<span class="rx-pop" role="tooltip"><span class="rx-pop-card">
<b>{_e(info_title)}</b>
<code class="cols">{_e(info_code)}</code>
<span class="rx-pop-text">{_e(info_note)}</span>
</span></span>
</span>
</div>
"""
    )


def workspace_head(note: str) -> None:
    section_head("Predict", note, anchor="predict")


def footer() -> None:
    _html(
        """
<footer class="footer" id="about">
  <p>Predictions follow the training notebook: the last cycles of each engine are turned into a
  sliding window, scaled per operating condition, and passed to the bundled model. Nothing is
  typed by hand about feature names.</p>
  <span>Light and dark appearance · ⋮ menu</span>
</footer>
"""
    )

