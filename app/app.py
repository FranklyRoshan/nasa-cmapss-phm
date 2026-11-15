"""RUL Prediction app for NASA C-MAPSS (FD001-FD004).

Pick a fleet and one of its preloaded models (XGBoost, Random Forest, LSTM, CNN-LSTM, TCN,
Transformer) from artifacts/ — no model upload needed. The sensor list, window length, scalers and operating-condition model all come from
the saved preprocessor, and the processing below repeats notebook section 17.7 exactly. Uploading
your own model files is still possible on the "Test bench" tab.

The interface is custom HTML/CSS (components.py + styles.css) on top of Streamlit widgets, so the
default Streamlit chrome and skin are removed.
"""

# TO RUN THIS CODE
# pip install -r requirements.txt
# python -m streamlit run app.py

import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view

import components as ui

# ------------------------------------------------------------------
# Constants copied from the notebook (cell 1.2 "Project Configuration")
# ------------------------------------------------------------------
SETTING_COLS = ["alt_ft", "Mach", "TRA"]
SENSOR_COLS = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30",
               "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
RAW_COLS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS
EXPECTED_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
PRE_KEYS = ["dataset_id", "measurement_cols", "imputer", "n_conditions", "settings_scaler", "kmeans",
            "live_sensors", "condition_scalers", "fallback_scaler", "window_size", "rul_cap", "base_cols"]

# Prescriptive maintenance (Prescriptive Analysis doc). RUL_eff = predicted RUL - K_SIGMA * sigma
# - SAFETY_CYCLES. The model gives no per-engine uncertainty (sigma), so K_SIGMA is 0 here; raise
# it only if you add a sigma estimate. SAFETY_CYCLES is the doc's 10-15 cycle floor (set 0 to use
# the raw prediction).
T_PLAN, T_REPLACE = 100.0, 30.0
SAFETY_CYCLES = 10.0
PRESCRIPTIVE_COL = "Prescriptive maintenance"

# Where the model comes from: the library shipped in artifacts/, or the user's own files.
FLEET_MODELS, TEST_BENCH = "Fleet models", "Test bench"
SOURCE_LABELS = {
    FLEET_MODELS: ":material/flight: Fleet models",
    TEST_BENCH: ":material/science: Test bench",
}

# Every model the notebook saved per fleet, shipped in artifacts/ next to this script. The user picks
# one with the buttons under "Choose the model"; the preprocessor is shared by all of them, so it is
# always loaded alongside whichever model is chosen. File names are looked up as
#   final_<slug>_fd00X.<ext>   (or <slug>_fd00X.<ext>)
# and the first slug that exists on disk wins, so small naming differences do not break the app.
MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_SPECS = {
    "XGBoost":       {"slugs": ("xgboost", "xgb"), "ext": ".json", "meta": True},
    "Random Forest": {"slugs": ("random_forest", "randomforest", "rf"), "ext": ".joblib"},
    "LSTM":          {"slugs": ("lstm",), "ext": ".safetensors"},
    "CNN-LSTM":      {"slugs": ("cnn_lstm", "cnnlstm", "cnn-lstm"), "ext": ".safetensors"},
    "TCN":           {"slugs": ("tcn",), "ext": ".safetensors"},
    # FD001's file is final_transformer_fd001; FD002-FD004 were saved as temporal_transformer_fd00X.
    "Transformer":   {"slugs": ("transformer", "temporal_transformer"), "ext": ".safetensors"},
}
# The notebook's best model per fleet, mentioned under "Choose the model".
# XGBoost is light, so it is what loads first; the heavier networks load when picked.
DEFAULT_MODEL = "XGBoost"
BEST_MODEL = {"FD001": "Transformer", "FD002": "LSTM", "FD003": "XGBoost", "FD004": "LSTM"}


# ------------------------------------------------------------------
# Notebook helpers (cell 1.5), unchanged
# ------------------------------------------------------------------
def assign_conditions(frame, settings_scaler, kmeans_model):
    if kmeans_model is None:
        return np.zeros(len(frame), dtype="int32")
    return kmeans_model.predict(settings_scaler.transform(frame[SETTING_COLS])).astype("int32")


def normalize_sensors(frame, live_sensors, condition_scalers, fallback_scaler):
    normalized = frame.copy()
    normalized[live_sensors] = normalized[live_sensors].astype("float64")
    condition_values = frame["condition"].to_numpy()
    for condition in np.unique(condition_values):
        mask = condition_values == condition
        scaler = condition_scalers.get(int(condition), fallback_scaler)
        normalized.loc[mask, live_sensors] = scaler.transform(frame.loc[mask, live_sensors])
    return normalized


def build_windows(frame, feature_cols, window_size, last_only=False):
    window_list, unit_list, cycle_list = [], [], []
    for unit_id, engine in frame.groupby("unit", sort=True):
        values = engine[feature_cols].to_numpy(dtype=np.float32)
        padding = np.repeat(values[:1], window_size - 1, axis=0)
        padded = np.concatenate([padding, values], axis=0)
        engine_windows = sliding_window_view(padded, window_size, axis=0).transpose(0, 2, 1)
        engine_cycles = engine["cycle"].to_numpy()
        if last_only:
            engine_windows, engine_cycles = engine_windows[-1:], engine_cycles[-1:]
        window_list.append(np.ascontiguousarray(engine_windows))
        unit_list.append(np.full(len(engine_cycles), unit_id))
        cycle_list.append(engine_cycles)
    return np.concatenate(window_list), np.concatenate(unit_list), np.concatenate(cycle_list)


def extract_window_features(windows, extra_values, extra_names, sensor_names):
    steps = windows.shape[1]
    time_centered = np.arange(steps, dtype=np.float64) - (steps - 1) / 2.0
    slope = (windows * time_centered[None, :, None]).sum(axis=1) / (time_centered ** 2).sum()
    blocks = {"last": windows[:, -1, :], "mean": windows.mean(axis=1), "std": windows.std(axis=1),
              "min": windows.min(axis=1), "max": windows.max(axis=1),
              "change": windows[:, -1, :] - windows[:, 0, :], "slope": slope}
    names = [f"{sensor}_{stat}" for stat in blocks for sensor in sensor_names] + list(extra_names)
    matrix = np.concatenate(list(blocks.values()) + [np.asarray(extra_values, dtype=np.float64)], axis=1)
    return matrix.astype(np.float32), names


# ------------------------------------------------------------------
# Neural-network architectures (notebook sections 12-15, unchanged).
# Defined lazily so tree-model users do not need PyTorch.
# ------------------------------------------------------------------
_RUNTIME_HANDLES = []


def _register_torch_runtime() -> None:
    """Windows: torch needs the MSVC runtime; a per-user copy of it is not on the DLL path."""
    import os
    import site
    import sys

    if os.name != "nt":
        return
    folders = [Path(sys.executable).parent]
    try:
        user_site = Path(site.getusersitepackages())
        folders += [user_site.parent.parent, user_site.parent.parent / "Scripts"]
    except Exception:  # noqa: BLE001 - a best-effort lookup, never fatal
        pass
    for folder in folders:
        if (folder / "msvcp140.dll").is_file():
            try:
                _RUNTIME_HANDLES.append(os.add_dll_directory(str(folder)))
            except (AttributeError, OSError):
                pass


@st.cache_resource
def get_network_builder():
    import math

    _register_torch_runtime()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim, hidden, layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            sequence_out, _ = self.lstm(x)
            return self.head(sequence_out[:, -1, :]).squeeze(-1)

    class CNNLSTMRegressor(nn.Module):
        def __init__(self, input_dim, filters, kernel, hidden, layers, dropout):
            super().__init__()
            assert kernel % 2 == 1, "use an odd kernel so the window length is preserved"
            self.conv = nn.Conv1d(input_dim, filters, kernel, padding=kernel // 2)
            self.norm = nn.BatchNorm1d(filters)
            self.lstm = nn.LSTM(filters, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            z = x.transpose(1, 2)
            z = F.relu(self.norm(self.conv(z)))
            z = z.transpose(1, 2)
            sequence_out, _ = self.lstm(z)
            return self.head(sequence_out[:, -1, :]).squeeze(-1)

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel, dilation):
            super().__init__()
            self.left_padding = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)

        def forward(self, x):
            return self.conv(F.pad(x, (self.left_padding, 0)))

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel, dilation, dropout):
            super().__init__()
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel, dilation)
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel, dilation)
            self.dropout = nn.Dropout(dropout)
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            z = self.dropout(F.relu(self.conv1(x)))
            z = self.dropout(F.relu(self.conv2(z)))
            return F.relu(z + self.shortcut(x))

    class TCNRegressor(nn.Module):
        def __init__(self, input_dim, channels, kernel, dropout):
            super().__init__()
            blocks, in_channels = [], input_dim
            for level, out_channels in enumerate(channels):
                blocks.append(TemporalBlock(in_channels, out_channels, kernel, 2 ** level, dropout))
                in_channels = out_channels
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.Linear(channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def sequence_features(self, x):
            return self.blocks(x.transpose(1, 2))

        def forward(self, x):
            return self.head(self.sequence_features(x)[:, :, -1]).squeeze(-1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
            table = torch.zeros(1, max_len, d_model)
            table[0, :, 0::2] = torch.sin(position * frequencies)
            table[0, :, 1::2] = torch.cos(position * frequencies)
            self.register_buffer("table", table)

        def forward(self, x):
            return x + self.table[:, :x.size(1)]

    class TransformerRegressor(nn.Module):
        def __init__(self, input_dim, d_model, nhead, layers, ff, dropout, max_len):
            super().__init__()
            self.projection = nn.Linear(input_dim, d_model)
            self.positional = PositionalEncoding(d_model, max_len)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=dropout,
                                                       batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x):
            z = self.positional(self.projection(x))
            z = self.encoder(z)
            return self.head(z[:, -1, :]).squeeze(-1)

    def build_model(model_name, input_dim, config, window):
        # Saved names vary ("Temporal Transformer", "cnn_lstm", ...): match on the key words.
        key = "".join(ch for ch in str(model_name).lower() if ch.isalnum())
        if "transformer" in key:
            model_name = "Transformer"
        elif "cnn" in key and "lstm" in key:
            model_name = "CNN-LSTM"
        elif key == "lstm":
            model_name = "LSTM"
        elif key == "tcn":
            model_name = "TCN"
        if model_name == "LSTM":
            return LSTMRegressor(input_dim, **config)
        if model_name == "CNN-LSTM":
            return CNNLSTMRegressor(input_dim, **config)
        if model_name == "TCN":
            return TCNRegressor(input_dim, **config)
        if model_name == "Transformer":
            return TransformerRegressor(input_dim, max_len=window, **config)
        raise ValueError(f"Unknown model name in the file: {model_name!r}")

    return torch, build_model


# ------------------------------------------------------------------
# Loading the uploaded files
# ------------------------------------------------------------------
def _check_preprocessor(pre):
    missing = [k for k in PRE_KEYS if k not in pre]
    if missing:
        raise ValueError(f"The preprocessor file is missing {missing}. "
                         "Use the preprocessor_*.joblib or final_preprocessor_*.joblib saved by the notebook.")


@st.cache_resource(show_spinner="Loading model files...")
def load_artifacts(_sources, key):
    """Sort the files (name, bytes) into model / preprocessor / metadata by what is inside them."""
    _register_torch_runtime()
    joblibs, weights = [], []
    for name, data in _sources:
        ext = Path(name).suffix.lower()
        if ext == ".joblib":
            joblibs.append(joblib.load(io.BytesIO(data)))
        elif ext in (".json", ".safetensors"):
            weights.append((ext, data))
        else:
            raise ValueError(f"Unsupported file type: {name}")

    dicts = [o for o in joblibs if isinstance(o, dict)]
    bundles = [o for o in dicts if "model" in o]
    pres = [o for o in dicts if "live_sensors" in o and "model" not in o]
    metas = [o for o in dicts if "feature_names" in o and "model" not in o and "live_sensors" not in o]
    bare = [o for o in joblibs if not isinstance(o, dict) and hasattr(o, "predict")]

    n_models = len(bundles) + len(bare) + len(weights)
    if n_models == 0:
        raise ValueError("No model found. Upload the model file (.joblib, .json or .safetensors).")
    if n_models > 1:
        raise ValueError("More than one model file was uploaded. Upload one model at a time.")

    art = {"feature_names": None, "model_name": None}
    embedded_pre = None
    if bundles or bare:                                  # Random Forest
        bundle = bundles[0] if bundles else {"model": bare[0]}
        art.update(kind="tree", model=bundle["model"], feature_names=bundle.get("feature_names"),
                   model_name=bundle.get("model_name", "Random Forest"))
        embedded_pre = bundle.get("preprocessor")
        if art["feature_names"] is None and hasattr(bundle["model"], "feature_names_in_"):
            art["feature_names"] = list(bundle["model"].feature_names_in_)
    elif weights[0][0] == ".json":                       # XGBoost
        import xgboost as xgb
        model = xgb.XGBRegressor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_bytes(weights[0][1])
            model.load_model(str(path))
        meta = metas[0] if metas else {}
        names = meta.get("feature_names") or model.get_booster().feature_names
        art.update(kind="tree", model=model, feature_names=list(names) if names else None,
                   model_name=meta.get("model_name", "XGBoost"))
        embedded_pre = meta.get("preprocessor")
    else:                                                # neural network (.safetensors)
        from safetensors import safe_open
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(weights[0][1])
            with safe_open(str(path), framework="pt") as f:
                meta = f.metadata() or {}
            state = load_file(str(path))
        for needed in ("model_name", "config", "input_dim"):
            if needed not in meta:
                raise ValueError(f"The .safetensors file has no '{needed}' metadata. "
                                 "Re-save it with the notebook's save cell.")
        art.update(kind="nn", model_name=meta["model_name"], config=json.loads(meta["config"]),
                   input_dim=int(meta["input_dim"]), state=state,
                   feature_names=json.loads(meta["feature_names"]) if "feature_names" in meta else None,
                   window=int(meta["window_size"]) if "window_size" in meta else None,
                   pre_hint=meta.get("preprocessor_file"))

    pre = embedded_pre if embedded_pre is not None else (pres[0] if pres else None)
    if pre is None:
        hint = art.get("pre_hint") or "preprocessor_<dataset>.joblib"
        raise ValueError(f"This model needs its preprocessor file ({hint}). Upload it together with the model.")
    _check_preprocessor(pre)
    art["pre"] = pre
    art["dataset_id"] = pre["dataset_id"]

    if art["kind"] == "nn":
        live = pre["live_sensors"]
        if art["feature_names"] is not None and art["feature_names"] != live:
            raise ValueError("The sensors stored in the model file differ from the preprocessor's sensors. "
                             "They do not belong together (different run or dataset).")
        if art["input_dim"] != len(live):
            raise ValueError(f"The network expects {art['input_dim']} inputs but the preprocessor has {len(live)} sensors.")
        if art["window"] is not None and art["window"] != pre["window_size"]:
            raise ValueError("Window size in the model file and the preprocessor differ.")
        torch, build_model = get_network_builder()
        model = build_model(art["model_name"], art["input_dim"], art["config"], pre["window_size"])
        model.load_state_dict(art.pop("state"))
        model.eval()
        art["model"] = model
    return art


# ------------------------------------------------------------------
# Inputs and prediction (notebook section 17.7)
# ------------------------------------------------------------------
def required_inputs(pre):
    """Columns the user must provide per cycle. Settings only matter when the dataset has several conditions."""
    return (SETTING_COLS if pre["n_conditions"] > 1 else []) + list(pre["live_sensors"])


def predict_engines(art, raw):
    """raw: one row per engine and cycle. Returns (table with one row per engine, engines shorter than the window)."""
    pre = art["pre"]
    live, window = pre["live_sensors"], pre["window_size"]
    need = required_inputs(pre)
    df = raw.copy()

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "unit" not in df.columns:
        df["unit"] = 1
    if "cycle" not in df.columns:
        if art["kind"] == "tree":
            raise ValueError("A 'cycle' column is required (engine age is a feature of this model).")
        df["cycle"] = df.groupby("unit").cumcount() + 1

    check = need + ["cycle"]
    df[check] = df[check].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in check if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
    if bad:
        raise ValueError(f"Empty or non-numeric values in: {bad}")

    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    if df.duplicated(["unit", "cycle"]).any():
        raise ValueError("Some engine has the same cycle number twice.")

    # Columns the model never reads are filled by the imputer with its training medians (as in training).
    for col in pre["measurement_cols"]:
        if col not in df.columns:
            df[col] = np.nan
    # Same result as imputer.transform for a median imputer, but it only reads the stored medians, so it
    # does not break when scikit-learn differs between the notebook and this app.
    medians = pd.Series(pre["imputer"].statistics_, index=pre["measurement_cols"])
    df[pre["measurement_cols"]] = df[pre["measurement_cols"]].astype("float64").fillna(medians)
    df["condition"] = assign_conditions(df, pre["settings_scaler"], pre["kmeans"])
    prepared = normalize_sensors(df[pre["base_cols"]], live, pre["condition_scalers"], pre["fallback_scaler"])

    windows, units, cycles = build_windows(prepared, live, window, last_only=True)
    sizes = df.groupby("unit").size()
    short = [str(u) for u in sizes[sizes < window].index]

    if art["kind"] == "tree":
        extra_names = ["cycle"] + (["condition"] if pre["n_conditions"] > 1 else [])
        extra = prepared.groupby("unit").tail(1)[extra_names].to_numpy()
        features, names = extract_window_features(windows, extra, extra_names, live)
        if art["feature_names"] is not None and names != list(art["feature_names"]):
            raise ValueError("The model's feature list does not match this preprocessor "
                             "(different run or dataset).")
        preds = np.clip(np.ravel(art["model"].predict(pd.DataFrame(features, columns=names))), 0.0, None)
    else:
        torch, _ = get_network_builder()
        with torch.no_grad():
            out = art["model"](torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))).numpy()
        preds = np.clip(np.ravel(out) * pre["rul_cap"], 0.0, None)  # network predicts RUL / RUL_CAP

    return pd.DataFrame({"unit": units, "last_cycle": cycles, "predicted_RUL": preds.round(1)}), short


def prescribe(predicted_rul: pd.Series) -> pd.Series:
    """Map each predicted RUL to Continue Operation / Schedule Servicing / Immediate Grounding."""
    effective = predicted_rul.astype(float) - SAFETY_CYCLES
    return pd.Series(
        np.select(
            [effective > T_PLAN, effective >= T_REPLACE],
            [ui.CONTINUE, ui.SERVICE],
            default=ui.GROUND,
        ),
        index=predicted_rul.index,
    )


def read_engine_file(file):
    """Official headerless C-MAPSS txt (26 columns) or any CSV with column names."""
    text = file.getvalue().decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("The file is empty.")

    def is_number(token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    first = [t for t in text.strip().splitlines()[0].replace(",", " ").replace(";", " ").split() if t]
    has_header = not all(is_number(t) for t in first)
    df = pd.read_csv(io.StringIO(text), sep=r"[\s,;]+", engine="python", header=0 if has_header else None)
    df = df.dropna(axis=1, how="all")
    if has_header:
        canon = {c.lower(): c for c in RAW_COLS}
        df.columns = [canon.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    elif df.shape[1] == len(RAW_COLS):
        df.columns = RAW_COLS
    else:
        raise ValueError(f"The file has no header row and {df.shape[1]} columns. The official C-MAPSS layout has "
                         f"{len(RAW_COLS)}: unit, cycle, 3 settings, 21 sensors. Add a header row otherwise.")
    return df


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _resolve_artifact(sources, signature):
    """Load the selected files, re-using the cached artifact while the selection is unchanged."""
    if not sources:
        st.session_state.pop("artifact", None)
        st.session_state.pop("artifact_sig", None)
        st.session_state.pop("manual_result", None)
        return None, None

    if st.session_state.get("artifact_sig") == signature and st.session_state.get("artifact") is not None:
        return st.session_state["artifact"], None

    try:
        artifact = load_artifacts(sources, signature)
    except Exception as exc:  # noqa: BLE001 - shown to the user as-is
        return None, str(exc)

    st.session_state["artifact"] = artifact
    st.session_state["artifact_sig"] = signature
    st.session_state.pop("manual_result", None)  # a new model invalidates the old prediction
    return artifact, None


def _bundled_path(name: str) -> Path:
    """A bundled file from artifacts/, or from the app folder if the folder was flattened."""
    for folder in (MODEL_DIR, Path(__file__).resolve().parent):
        if (folder / name).is_file():
            return folder / name
    return MODEL_DIR / name


def _bundled_files(dataset: str, model: str) -> tuple[str, ...]:
    """File names (in artifacts/) that make up one preloaded model: weights, optional meta, preprocessor."""
    spec, fd = MODEL_SPECS[model], dataset.lower()

    main = next(
        (name for slug in spec["slugs"] for prefix in ("final_", "")
         if _bundled_path(name := f"{prefix}{slug}_{fd}{spec['ext']}").is_file()),
        None,
    )
    pre = next(
        (name for name in (f"final_preprocessor_{fd}.joblib", f"preprocessor_{fd}.joblib")
         if _bundled_path(name).is_file()),
        None,
    )

    missing = []
    if main is None:
        missing.append(f"final_{spec['slugs'][0]}_{fd}{spec['ext']}")
    if pre is None:
        missing.append(f"final_preprocessor_{fd}.joblib")
    if missing:
        raise FileNotFoundError(
            f"{model} for {dataset} is not available. Missing from artifacts/: " + ", ".join(missing)
        )

    names = [main]
    if spec.get("meta"):  # XGBoost keeps its feature list in a small side file
        meta = main[: -len(spec["ext"])] + "_meta.joblib"
        if _bundled_path(meta).is_file():
            names.append(meta)
    names.append(pre)
    return tuple(names)


def _bundled_sources(dataset: str, model: str):
    """The preloaded files of one model, read from artifacts/ next to this script."""
    names = _bundled_files(dataset, model)
    return names, tuple((name, _bundled_path(name).read_bytes()) for name in names)


def _setup_columns(dataset_guess: str):
    """Fleet + model source on the left, file guide / model summary on the right."""
    left, right = st.columns([1.06, 0.94], gap="large")

    files, sources, signature = [], (), None
    summary, source, bundle_error = "", "Preloaded", ""

    with left:
        with st.container(key="dataset-field"):
            ui.field_head(
                "Select the fleet",
                "FD001 to FD004. Every fleet comes with its own set of preloaded models.",
                anchor="setup",
            )
            dataset = st.segmented_control(
                "Dataset",
                list(EXPECTED_CONDITIONS),
                default=dataset_guess,
                required=True,
                key="dataset",
                label_visibility="collapsed",
                wrap=True,
            ) or dataset_guess

        with st.container(key="model-source-field"):
            with st.container(horizontal=True, vertical_alignment="center"):
                mode = st.segmented_control(
                    "Model source",
                    [FLEET_MODELS, TEST_BENCH],
                    default=FLEET_MODELS,
                    required=True,
                    key="model_source",
                    format_func=SOURCE_LABELS.get,
                    label_visibility="collapsed",
                    help="Fleet models: the preloaded models for each fleet. "
                         "Test bench: score with model files you trained yourself.",
                ) or FLEET_MODELS
                if mode == TEST_BENCH:
                    ui.upload_guide_popover()
        use_own = mode == TEST_BENCH

        model_choice, preprocessor_file = None, ""
        if use_own:
            source = "Uploaded"
            with st.container(key="upload-field"):
                ui.field_head(
                    "Upload the trained model files",
                    "One model plus its preprocessor. Several files can go in together.",
                    meta="200 MB per file · JOBLIB · JSON · SAFETENSORS",
                )
                files = st.file_uploader(
                    "Trained model files",
                    type=["joblib", "json", "safetensors"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key="model_files",
                ) or []
            if files:
                signature = ("upload", tuple(
                    (f.name, f.size, getattr(f, "file_id", None)) for f in files))
                sources = tuple((f.name, f.getvalue()) for f in files)
                total = sum(f.size for f in files)
                summary = f"{len(files)} uploaded file{'s' if len(files) != 1 else ''} · {_human_size(total)}"
        else:
            with st.container(key="model-pick-field"):
                ui.field_head(
                    "Choose the model",
                    f"All six models are preloaded for {dataset}. The notebook's pick is "
                    f"{BEST_MODEL[dataset]}.",
                )
                # One key per fleet, so each fleet remembers its own choice.
                model_choice = st.segmented_control(
                    "Model",
                    list(MODEL_SPECS),
                    default=DEFAULT_MODEL,
                    required=True,
                    key=f"model_choice_{dataset}",
                    label_visibility="collapsed",
                    wrap=True,
                ) or DEFAULT_MODEL
                ui.preprocessor_note(f"final_preprocessor_{dataset.lower()}.joblib")

            try:
                names, sources = _bundled_sources(dataset, model_choice)
            except Exception as exc:  # noqa: BLE001 - shown to the user as-is
                bundle_error = str(exc)
            else:
                preprocessor_file = names[-1]
                # The mtime makes a model replaced on disk reload without restarting the app.
                signature = ("bundled", dataset, model_choice, tuple(
                    _bundled_path(name).stat().st_mtime_ns for name in names))
                total = sum(len(data) for _name, data in sources)
                summary = f"{model_choice} · {len(sources)} bundled files · {_human_size(total)}"

        if signature is None:
            artifact, _ = _resolve_artifact((), None)  # clears any model from the other source
            load_error = bundle_error or None
        else:
            artifact, load_error = _resolve_artifact(sources, signature)

    engine_file = None
    with right:
        if artifact is not None:
            ui.model_card(artifact, dataset, summary, source=source)
            if artifact["dataset_id"] == dataset:
                # The engine-file uploader fills the space under the model card.
                engine_file = _upload_card(
                    required_inputs(artifact["pre"]), int(artifact["pre"]["window_size"])
                )
        else:
            ui.empty_model_card()

    return dataset, files, artifact, load_error, engine_file


def _formula_chart(frame: pd.DataFrame, cap: float):
    """A small Altair bar chart that follows the interface palette."""
    import altair as alt

    colours = ui.chart_palette()
    top = max(cap, float(frame["predicted_RUL"].max()) * 1.08, 1.0)

    x = alt.X(
        "unit:O",
        title="Engine",
        axis=alt.Axis(
            labelAngle=0, labelColor=colours["muted"], titleColor=colours["faint"],
            labelFontSize=11, titleFontSize=11, labelPadding=8, titlePadding=10,
        ),
        scale=alt.Scale(paddingInner=0.72, paddingOuter=0.2),
    )
    y = alt.Y(
        "predicted_RUL:Q",
        title="Predicted RUL (cycles)",
        scale=alt.Scale(domain=[0, top]),
        axis=alt.Axis(
            labelColor=colours["muted"], titleColor=colours["faint"], labelFontSize=11,
            titleFontSize=11, labelPadding=6, titlePadding=10, tickCount=4,
            gridColor=colours["line"], gridOpacity=0.9, domain=False, ticks=False,
        ),
    )

    base = alt.Chart(frame)
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=x,
        y=y,
        color=alt.value(colours["ink"]),
        tooltip=[
            alt.Tooltip("unit:O", title="Engine"),
            alt.Tooltip("predicted_RUL:Q", title="RUL (cycles)", format=".1f"),
        ],
    )

    chart = bars
    if len(frame) <= 14:
        labels = base.mark_text(dy=-9, color=colours["muted"], fontSize=11).encode(
            x=x, y=y, text=alt.Text("predicted_RUL:Q", format=".1f")
        )
        chart = bars + labels

    return (
        chart.properties(height=260, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axisX(grid=False)
    )


def _manual_panel(artifact: dict, inputs: list[str], window: int, cap: float) -> None:
    pre = artifact["pre"]

    with st.container(key="manual-panel"):
        st.html(
            '<div class="panel-head"><h3>Cycle readings</h3>'
            '<p>One row per cycle, oldest first. The last row is the cycle you want to score.</p></div>'
        )

        first, second, third = st.columns([0.9, 0.9, 2.2], vertical_alignment="bottom")
        n = int(first.number_input(
            "Cycles to enter", min_value=1, max_value=int(window), value=1, step=1,
            key=f"cycles-{window}",
        ))
        latest = int(second.number_input(
            "Latest cycle", min_value=n, value=n, step=1, key=f"latest-{n}",
        ))
        with third:
            if n < window:
                st.caption(
                    f"With fewer than {window} cycles, the first row is repeated backwards to fill the window, "
                    "the same way the notebook treats young engines."
                )
            else:
                st.caption(f"Enter {n} consecutive cycles; the last one is cycle {latest}.")

        with st.form("manual", border=False, clear_on_submit=False):
            table = st.data_editor(
                pd.DataFrame(np.nan, index=range(n), columns=inputs),
                hide_index=True,
                num_rows="fixed",
                key=f"table_{n}_{hash(tuple(inputs))}",
                column_config={c: st.column_config.NumberColumn(c, format="%g") for c in inputs},
                height=min(420, 52 + 35 * n),
                alt="Sensor readings for recent engine cycles",
            )
            submitted = st.form_submit_button("Predict RUL", type="primary", width="content")

        if submitted:
            raw = table.copy()
            raw.insert(0, "cycle", np.arange(latest - n + 1, latest + 1))
            try:
                out, _short = predict_engines(artifact, raw)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}", icon=":material/error:")
            else:
                st.session_state["manual_result"] = {
                    "rul": float(out["predicted_RUL"].iloc[0]),
                    "latest": latest,
                    "n": n,
                }

        result = st.session_state.get("manual_result")
        if result and result.get("n") == n and result.get("latest") == latest:
            ui.result_card(result["rul"], result["latest"], cap)
        elif result:
            st.caption("The last prediction belongs to a different number of cycles; press Predict RUL again.")


def _upload_card(inputs: list[str], window: int):
    """The engine-file uploader. It sits in the right column, under the model card."""
    cols = ["unit", "cycle"] + inputs

    with st.container(key="file-panel"):
        st.html(
            '<div class="panel-head"><h3>Engine data</h3>'
            '<p>Upload the official <code class="k">test_FD00X.txt</code> (no header, 26 columns) '
            'or a CSV with a header row.</p></div>'
        )

        with st.expander("Columns the file must contain"):
            st.code(", ".join(cols))
            st.caption(
                "`unit` is the engine id; leave it out if the file holds a single engine. "
                f"Each engine's last {window} rows are used."
            )

        return st.file_uploader("Data file", type=["csv", "txt"], key="engine_file", label_visibility="collapsed")


def _file_results(artifact: dict, dataset: str, window: int, cap: float, file) -> None:
    """Predictions for the uploaded engine file: summary, chart and table."""
    try:
        out, short = predict_engines(artifact, read_engine_file(file))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}", icon=":material/error:")
        return

    if short:
        st.warning(
            f"{len(short)} engine(s) have fewer than {window} rows and were padded with their "
            f"first row: {', '.join(short[:10])}{'...' if len(short) > 10 else ''}",
            icon=":material/warning:",
        )

    out[PRESCRIPTIVE_COL] = prescribe(out["predicted_RUL"])

    with st.container(key="table-panel"):
        head, action = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.html(
                f'<div class="panel-head"><h3>{len(out)} prediction{"s" if len(out) != 1 else ""}</h3>'
                '<p>One row per engine, using its last cycle in the file.</p></div>'
            )
        action.download_button(
            "Download CSV",
            out.to_csv(index=False).encode(),
            file_name=f"rul_predictions_{dataset.lower()}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
        ui.stats(
            [
                ("Engines", f"{len(out):,}"),
                ("Lowest RUL", f"{out['predicted_RUL'].min():.1f}"),
                ("Median RUL", f"{out['predicted_RUL'].median():.1f}"),
                ("Highest RUL", f"{out['predicted_RUL'].max():.1f}"),
            ]
        )
        ui.predictions_table(out, T_PLAN, T_REPLACE, SAFETY_CYCLES)

    with st.container(key="chart-panel"):
        st.html(
            '<div class="panel-head"><h3>Predicted life by engine</h3>'
            '<p>Remaining useful life in cycles, per engine.</p></div>'
        )
        st.altair_chart(
            _formula_chart(out, cap),
            width="stretch",
            theme=None,
            alt="Predicted remaining useful life for each engine",
        )


def main():
    st.set_page_config(
        page_title="RUL Predictor",
        page_icon=":material/precision_manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    ui.apply_theme()
    dataset_guess = st.session_state.get("dataset", "FD001")
    uses_own = st.session_state.get("model_source") == TEST_BENCH
    ui.header_and_hero(dataset_guess, compact=True)  # same compact title in every mode

    dataset, files, artifact, load_error, engine_file = _setup_columns(dataset_guess)

    if load_error:
        st.error(f"Could not load the files: {load_error}", icon=":material/error:")
        st.stop()

    if artifact is not None and artifact["dataset_id"] != dataset:
        st.error(
            f"These files were trained on {artifact['dataset_id']}, but {dataset} is selected. "
            f"Select {artifact['dataset_id']} above or upload the {dataset} files.",
            icon=":material/error:",
        )
        st.stop()

    if artifact is None:
        note = ("Add your model files above, or switch back to Fleet models."
                if uses_own else "Waiting for a model.")
        ui.section_head("Predict", note, anchor="predict")
        ui.steps_card()
        ui.footer()
        return

    pre = artifact["pre"]
    window, cap = int(pre["window_size"]), float(pre["rul_cap"])
    inputs = required_inputs(pre)
    ignored = [s for s in SENSOR_COLS if s not in pre["live_sensors"]]
    n_cond = int(pre["n_conditions"])

    ui.workspace_head(
        f"{artifact['model_name']} · {dataset} · last {window} cycles"
    )

    # Once an engine file is in the Engine data box (beside the model card) its results replace the
    # manual entry form, so nothing sits between the page top and the results. Remove the file
    # (hover it and press X) to bring manual entry back.
    if engine_file is not None:
        _file_results(artifact, dataset, window, cap, engine_file)
    else:
        with st.expander("Inputs this model reads", key="inputs-expander"):
            st.caption(
                ("Operating settings: " + ", ".join(SETTING_COLS) + ". " if n_cond > 1 else "")
                + "Sensors: " + ", ".join(pre["live_sensors"])
                + (f". Not used (flat in the training data): {', '.join(ignored)}." if ignored else ".")
            )
            st.caption(
                f"{len(inputs)} values per cycle, over the last {window} cycles of an engine. "
                f"Training capped RUL at {cap:g}."
            )

        _manual_panel(artifact, inputs, window, cap)

    ui.footer()


if __name__ == "__main__":
    main()
