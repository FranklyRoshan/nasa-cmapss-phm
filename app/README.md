# RUL Predictor

A Streamlit app that predicts the **remaining useful life (RUL)** of jet engines from the NASA
C-MAPSS dataset (fleets FD001 to FD004). Pick a fleet, pick one of its preloaded models, upload an
engine file, and the app returns a predicted RUL and a maintenance recommendation for every engine.

The interface is custom HTML and CSS built on top of Streamlit widgets. Streamlit's default chrome
(header band, deploy button, sidebar, default metric cards) is removed.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run it from this folder, because everything is looked up next to `app.py`. Streamlit **1.50 or
newer** is recommended (the app uses `st.context.theme`, keyed containers and `st.segmented_control`).

On Windows, `requirements.txt` also installs `msvc-runtime`. `torch` needs the Microsoft C++
runtime, and `app.py` looks for this per-user copy if it is not installed system-wide.

## Folder layout

| Path | Purpose |
| --- | --- |
| `app.py` | The application: data processing, model loading, prediction and page structure. |
| `components.py` | HTML builders, light and dark palette, the fleet and model-card pieces, metrics lookup. |
| `styles.css` | The whole visual layer: colours, spacing, typography and responsive rules. |
| `compute_metrics.py` | One-off script that computes RMSE and NASA score for every model. |
| `requirements.txt` | Python dependencies. |
| `artifacts/` | Bundled models and preprocessors, plus `metrics.json` once it has been generated. |
| `cmapss_data/` | Official NASA files (`train_`, `test_` and `RUL_FD00X.txt`). Only used by `compute_metrics.py`. |
| `samples/` | Sample engine files to try the app with. |
| `.streamlit/config.toml` | Theme for the Streamlit widgets that are not custom-drawn. |

## Using the app

1. **Select the fleet.** FD001 to FD004. Under the buttons the app shows the fleet's operating
   conditions and fault modes (FD001: 1 condition, HPC fault; FD002: 6 conditions, HPC; FD003: 1
   condition, HPC and Fan; FD004: 6 conditions, HPC and Fan).
2. **Choose the model.** XGBoost, Random Forest, LSTM, CNN-LSTM, TCN or Transformer. All six are
   preloaded for every fleet. The card also names the notebook's best model for that fleet.
   Every model shares the fleet's preprocessor file (`final_preprocessor_fd00X.joblib`).
3. **Upload engine data.** The official `test_FD00X.txt` (no header, 26 columns) or a CSV with a
   header row. The model card on the right shows the loaded model with its RMSE, NASA score,
   window size and RUL cap.
4. **Read the predictions.** One row per engine, using its last cycle. Download them as CSV.
5. **Predict one engine by hand.** Enter the last cycles' sensor readings in the readings table
   under **Predict**. The last row is the cycle that gets scored.

The **Test bench** tab lets you load your own model files instead of the bundled ones.

### Maintenance recommendation

Each predicted RUL (minus a 10-cycle safety margin) is mapped to one of three actions:

| Adjusted RUL | Recommendation |
| --- | --- |
| above 100 cycles | Continue Operation |
| 30 to 100 cycles | Schedule Servicing |
| below 30 cycles | Immediate Grounding |

These thresholds are `T_PLAN`, `T_REPLACE` and `SAFETY_CYCLES` near the top of `app.py`.

## RMSE and NASA score

The two numbers in the model card are not computed when you upload a file, because an uploaded
engine file has no true RUL. They are computed once, against the official test answers, and saved.

```bash
python compute_metrics.py
```

- Needs `test_FD001.txt` to `test_FD004.txt` and `RUL_FD001.txt` to `RUL_FD004.txt` in `cmapss_data/`.
- Runs all 24 models (6 per fleet, 4 fleets) with the app's own loading and prediction code.
- Writes `artifacts/metrics.json`, which the app reads at startup. Restart the app afterwards.
- True RUL is capped at the model's RUL cap (125) before scoring. Set `CLIP_TRUE_RUL_TO_CAP = False`
  in `compute_metrics.py` if your notebook scored against uncapped values.

The two formulas:

- **RMSE:** square root of the mean of (predicted minus actual) squared.
- **NASA score:** with `d = predicted - actual`, each engine adds `exp(-d/13) - 1` when `d < 0`
  (early prediction) and `exp(d/10) - 1` otherwise (late prediction, penalised harder). Lower is better.

A box shows a dash until `metrics.json` has a value for that fleet and model.

## Sharing the app

Send the whole folder, including `artifacts/metrics.json`. Others do not need to run
`compute_metrics.py`; they only run the two commands in Quick start. `cmapss_data/` can be left out
to keep the folder small, because the app itself never reads it.

Saved models can fail to load if the library versions differ a lot from the ones used for
training (scikit-learn is the most sensitive). To lock the versions, run `pip freeze` on the
training machine and copy the versions into `requirements.txt`.

## Design notes

- **Light and dark.** `components.py` reads `st.context.theme` and injects the matching palette.
  Switch the appearance from the **⋮** menu, top right.
- **One stylesheet.** `styles.css` holds the colours and spacing. The palette is injected as CSS
  variables (`--ink`, `--surface`, `--line` and so on), so changing `LIGHT` or `DARK` in
  `components.py` restyles the page.
- **Fixed card size.** The three setup cards keep their size. In the fleet card the status line,
  heading, buttons and the operating-condition and fault-mode boxes are spaced evenly.
- **Responsive buttons.** The model buttons measure their card, not the screen. They stay on one
  row when there is room and become two rows of three on a narrow card. The FD001 to FD004
  buttons always stay on one row and shrink their text to fit. Both rely on CSS container queries,
  so use a current browser (Chrome or Edge 105+, Firefox 110+, Safari 16+).
- **No sanitizer surprises.** `st.html` strips `<style>` and inline SVG, so sections are rendered
  through `st.markdown(..., unsafe_allow_html=True)` by the `_html()` helper in `components.py`.

## Where to change things

- Colours, radii, shadows: top of `styles.css` and `LIGHT` / `DARK` in `components.py`.
- Copy in the guide and cards: `components.py`.
- Fault modes per fleet and the metric boxes: `components.py`.
- Models, file names and the notebook's best-model pick: `MODEL_SPECS` and `BEST_MODEL` in `app.py`.
- Maintenance thresholds: `T_PLAN`, `T_REPLACE`, `SAFETY_CYCLES` in `app.py`.
- Page structure and which panel is shown: the `main()` section of `app.py`.
