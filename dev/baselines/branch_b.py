#!/usr/bin/env python3
"""
CARDIO-CODE -- classical-DSP arrhythmia screening for motion-corrupted PPG / ECG.

    python pipeline.py --input ecg.csv --output predictions.csv \
                       [--annotation annotation.txt] [--plots]

Stages (every one is a deterministic, explainable signal-processing step; there
is NO machine learning, no trained model and no learned classifier anywhere):

  0. load + identify channels (PPG / ECG / accelerometer / time) by column name,
     falling back to a spectral heuristic; estimate fs from the Timestamp column
  1. preprocessing: NaN interpolation, uniform resampling if the time grid is
     irregular, median removal
  2. baseline wander removal (zero-phase Butterworth high-pass, chosen by a
     measured comparison against moving-median and spline baselines)
  3. motion artefact suppression (NLMS with the accelerometer as noise reference
     when one exists; STFT HR-ridge tracking + spectral exclusion of motion peaks)
  4. beat detection: Pan-Tompkins-style envelope, adaptive dual thresholds,
     refractory period, secondary-wave rejection, search-back, fiducial refinement
  5. inter-beat intervals (IBI)
  6. rule-based anomaly detection (premature/compensatory, pause/dropped beat,
     marked bradycardia, AF-like irregularity), gated by signal quality
  7. per-sample confidence score from signal-quality indices
  8. predictions.csv  (+ optional validation against annotations, plots)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import resource
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal, stats

LOG = logging.getLogger("cardio")

# --------------------------------------------------------------------------- #
# Output column names -- change here only.  The PDF text extraction shows
# "Is peak", "Ibi ms", "ano flag", "confidence score"; we assume the original
# identifiers are underscored.
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS = {
    "time": "Timestamp",
    "peak": "Is_peak",
    "ibi": "Ibi_ms",
    "anomaly": "ano_flag",
    "confidence": "confidence_score",
}

# --------------------------------------------------------------------------- #
# Parameter dictionary.  Every value is either a physiological bound, a value
# from the cited literature, or was chosen on the FIRST HALF of the provided
# record only (see report, "Tuning protocol").  Nothing here is fitted per file:
# all thresholds used at run time are adaptive (running estimates / local
# medians) and these numbers are only their rates, ratios and bounds.
# --------------------------------------------------------------------------- #
PARAMS = {
    # ---- input / sampling -------------------------------------------------
    "default_fs": 360.0,          # Hz, used ONLY if no time column, no --fs and
                                  # no annotation-derived fs (MIT-BIH standard)
    "nonuniform_tol": 0.05,       # |dt - median dt| > 5 % of median counts as jitter
    "nonuniform_frac": 0.01,      # resample if > 1 % of intervals are jittered/gaps
    "max_interp_gap_s": 1.0,      # gaps longer than this are interpolated but get
                                  # confidence 0 and cannot contain beats
    # ---- baseline wander --------------------------------------------------
    "baseline_method": "butter",  # butter | median | spline (see compare_baselines)
    "hp_cutoff": {"ecg": 0.5, "ppg": 0.3},   # Hz; ECG 0.5 Hz = AHA monitoring
                                  # band; PPG 0.3 Hz keeps a 36 bpm (0.6 Hz) pulse
    "hp_order": 2,                # x2 by sosfiltfilt -> 4th-order, zero phase
    "median_win_s": {"ecg": (0.2, 0.6), "ppg": (1.5, 3.0)},  # two-stage median
    "spline_knot_s": 1.0,         # spline baseline through 1-s segment medians
    "lowpass": {"ecg": 40.0, "ppg": 8.0},    # Hz upper edge of the clean signal
    "lp_order": 4,
    "line_noise_db": 10.0,        # notch 50/60 Hz only if the peak is > 10 dB
    "notch_q": 30.0,              # above its neighbourhood (ECG only)
    # ---- motion suppression ----------------------------------------------
    "nlms_fs": 25.0,              # Hz, NLMS runs at a decimated rate (PPG < 8 Hz)
    "nlms_taps": 8,               # per accelerometer axis (0.32 s at 25 Hz)
    "nlms_mu": 0.01,              # normalised step size, 0 < mu < 2
    "nlms_min_coherence": 0.10,   # apply NLMS only if acc/PPG coherence in the
                                  # cardiac band exceeds this (else it cannot help)
    "stft_win_s": 8.0,            # STFT window (resolution 0.125 Hz = 7.5 bpm)
    "stft_hop_s": 1.0,
    "stft_fs": 25.0,              # spectral analysis at 25 Hz
    "hr_range_bpm": (30.0, 220.0),
    "hr_jump_cost": 0.08,         # Viterbi cost per bpm of HR change per hop (nats)
    "hr_harmonic_weight": 0.5,    # emission = P(f) + 0.5 P(2f): favours fundamental
    "motion_peak_ratio": 4.0,     # PPG spectral peak > 4x frame median = interferer
    "motion_guard_hz": 0.15,      # never attenuate within 0.15 Hz of k*HR
    "motion_gain": 0.1,           # attenuation applied to motion bins (-20 dB)
    "motion_harmonics": 3,
    "motion_mask": True,          # spectral exclusion of motion peaks (PPG only)
    # ---- beat detection ---------------------------------------------------
    "det_band": {"ecg": (5.0, 15.0), "ppg": (0.5, 8.0)},  # Pan-Tompkins QRS band;
                                                          # PPG cardiac band
    "mwi_s": {"ecg": 0.15, "ppg": 0.15},   # moving-window integration width
    "refractory_s": 0.25,         # no two beats closer than 250 ms (240 bpm)
    "secondary_wave_s": 0.36,     # T-wave / dicrotic-wave check window
    "secondary_slope_ratio": 0.5, # ... reject if slope < 50 % of previous beat
    "spk_rate": 0.125,            # running signal-peak estimate update rate
    "npk_rate": 0.125,            # running noise-peak estimate update rate
    "thr_frac": 0.25,             # TH1 = NPK + 0.25 (SPK - NPK)   (Pan & Tompkins)
    "thr2_frac": 0.5,             # TH2 = 0.5 TH1 (search-back threshold)
    "searchback_factor": 1.66,    # search back if gap > 1.66 x mean of last 8 IBIs
    "rr_avg_n": 8,
    "lost_signal_s": 3.0,         # no beat for 3 s -> halve SPK (re-acquire)
    "refine_win_s": {"ecg": (-0.10, 0.10), "ppg": (-0.05, 0.30)},
    "ibi_min_ms": 300.0,          # physiological IBI bounds (200 .. 30 bpm)
    "ibi_max_ms": 2000.0,
    # ---- anomaly rules ----------------------------------------------------
    "local_median_beats": 9,      # centred local median IBI (+-4 beats)
    "long_ref_beats": 30,         # preceding-beat reference for pauses
    "premature_ratio": 0.80,      # IBI < 80 % of local median  (>20 % early)
    "compensatory_ratio": 1.10,   # followed by IBI > 110 % of local median
    "compensatory_sum_tol": 0.20, # short+long within 20 % of 2 x median (full comp.)
    "pause_ratio": 1.60,          # IBI > 1.6 x preceding reference = dropped beat
    "brady_ibi_ms": 1500.0,       # HR < 40 bpm = marked bradycardia
    "irr_win_beats": 10,          # sliding irregularity window (beats)
    "nrmssd_th": 0.12,            # RMSSD / mean IBI (AF literature 0.1 .. 0.15)
    "tpr_z": 1.96,                # turning-point count consistent with randomness
    "sd_ratio_th": 0.60,          # Poincare SD1/SD2 (AF ~ 1, sinus ~ 0.2-0.5)
    "irr_votes": 2,               # >= 2 of {nRMSSD, TPR, SD1/SD2}, nRMSSD mandatory
    "gate_template_corr": 0.86,   # morphology gate: window-median template corr
                                  # (Orphanidou et al. 2015 acceptability limit)
    "gate_shape": 0.50,           # and shape SQI (kurtosis/skewness) ...
    "gate_motion": 0.50,          # ... and motion SQI must reach these to flag
    "merge_gap_s": 1.0,           # merge flagged segments closer than 1 s
    # ---- confidence -------------------------------------------------------
    "template_block_beats": 30,   # median beat template per 30-beat block
    "template_len": 64,           # beats resampled to 64 points for correlation
    "sqi_block_s": 2.0,           # block length for kurtosis/skewness/motion SQI
    "sqi_weights": {"template": 0.35, "purity": 0.15, "amplitude": 0.15,
                    "shape": 0.15, "motion": 0.20},
    "purity_bw_hz": 0.15,         # half-width around k*HR counted as "cardiac"
    "purity_harmonics": {"ecg": 5, "ppg": 3},
    "purity_norm": 0.60,          # purity 0.6 -> SQI 1 (clean record level)
    "shape_center": {"ecg": 5.0, "ppg": 0.0},  # ECG kurtosis (clean >> 5, noise ~3)
    "shape_scale": {"ecg": 1.0, "ppg": 0.15},  # PPG skewness (clean > 0, noise ~0)
    "amp_dev_scale": 0.30,        # exp(-|a - local median| / (0.3 median))
    "conf_smooth_s": 1.0,
    "no_beat_s": 3.0,             # > 3 s without a beat -> morphology SQIs = 0
    # ---- evaluation -------------------------------------------------------
    "tolerances_ms": (50.0, 100.0),
    "max_lag_s": 0.5,             # PPG-vs-ECG reference lag search range
}

# WFDB beat codes (annotations that denote a heartbeat) and those treated as a
# normal *rhythm* beat (conduction morphology such as RBBB is not a rhythm
# anomaly and cannot be seen in an IBI series anyway).
WFDB_BEATS = set("NLRBAaJSVrFejnE/fQ?")
WFDB_NORMAL = set("NLRB/f")
ANOMALY_EVENT_AUX = {"MISSB"}          # blocked / dropped beat comments
ANOMALY_EVENT_TYPES = {"x"}            # non-conducted P wave


# =========================================================================== #
# 0. Loading and channel identification
# =========================================================================== #
@dataclass
class Record:
    t_orig: np.ndarray                 # per-row time in seconds (output rows)
    fs: float
    primary: np.ndarray                # uniform-grid primary channel
    primary_name: str
    modality: str                      # "ecg" | "ppg"
    acc: np.ndarray | None             # (n, k) uniform-grid accelerometer
    ecg_ref: np.ndarray | None         # ECG reference when primary is PPG
    valid: np.ndarray                  # bool, False inside long gaps / NaN runs
    grid_index: np.ndarray             # output row -> uniform-grid index
    resampled: bool
    t0: float
    info: dict = field(default_factory=dict)


_RX = {
    "time": re.compile(r"(time|timestamp|^t$|^t_|sec|^ms$|date)", re.I),
    "sample": re.compile(r"(sample|^idx$|^index$|^n$)", re.I),
    "acc": re.compile(r"(acc|accel|^a_?[xyz]$|^[xyz]$|^[xyz]_|gyr|imu|motion)", re.I),
    "ppg": re.compile(r"(ppg|pleth|bvp|pulse|^ir$|infra|^red$|green|optic|^pg)", re.I),
    "ecg": re.compile(r"(ecg|ekg|mlii|^ml|lead|^v[1-6]$|^i{1,3}$|^av[rlf]$)", re.I),
}


def _clean_name(c) -> str:
    return re.sub(r"^[\s'\"]+|[\s'\"]+$", "", str(c))


def read_table(path: str) -> pd.DataFrame:
    """Read a delimited text file; the delimiter is sniffed from the header."""
    with open(path, "r", newline="") as fh:
        head = fh.read(8192)
    try:
        delim = csv.Sniffer().sniff(head, delimiters=",;\t| ").delimiter
    except csv.Error:
        delim = ","
    df = pd.read_csv(path, sep=delim, engine="c", skipinitialspace=True)
    df.columns = [_clean_name(c) for c in df.columns]
    return df


def spectral_modality(x: np.ndarray, fs: float) -> str:
    """ECG has sharp QRS complexes -> substantial power 10-40 Hz; PPG has almost
    none above 8 Hz.  Ratio > 0.1 of the 0.5-40 Hz power => ECG."""
    x = x[np.isfinite(x)]
    if len(x) < fs * 4 or fs < 60:
        return "ppg"          # cannot resolve 10-40 Hz; wearables' PPG is low-rate
    f, p = signal.welch(x - np.median(x), fs, nperseg=int(min(len(x), 4 * fs)))
    band = (f >= 0.5) & (f <= 40)
    hf = (f >= 10) & (f <= 40)
    return "ecg" if p[hf].sum() / max(p[band].sum(), 1e-20) > 0.10 else "ppg"


def identify_columns(df: pd.DataFrame) -> dict:
    """Classify columns by name.  Returns dict of lists."""
    out = {k: [] for k in ("time", "sample", "ppg", "ecg", "acc", "unknown")}
    for c in df.columns:
        name = c.strip()
        for key in ("time", "sample", "acc", "ppg", "ecg"):
            if _RX[key].search(name):
                out[key].append(c)
                break
        else:
            if pd.api.types.is_numeric_dtype(df[c]) or \
               pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9:
                out["unknown"].append(c)
    return out


def parse_time_column(s: pd.Series) -> np.ndarray:
    """Return time in seconds.  Handles numeric seconds / ms / us / ns, datetime
    strings and 'm:ss.mmm' strings.  The unit is chosen as the one giving a
    plausible sampling rate (8 Hz .. 5 kHz), preferring seconds."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        v = num.to_numpy(float)
        d = np.diff(v[np.isfinite(v)])
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            if 8.0 <= 1.0 / (med * scale) <= 5000.0:
                return v * scale
        return v
    txt = s.astype(str).str.strip()
    if txt.str.match(r"^\d+:\d+(:\d+)?(\.\d+)?$").mean() > 0.9:
        parts = txt.str.split(":", expand=True).astype(float)
        secs = parts.iloc[:, -1]
        for i in range(2, parts.shape[1] + 1):
            secs = secs + parts.iloc[:, -i] * 60 ** (i - 1)
        return secs.to_numpy(float)
    dt = pd.to_datetime(txt, errors="coerce")
    return (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)


def fill_nans(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across NaNs (vectorised).  Returns (filled, nan_mask)."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x), bad
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x, bad


def long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs of length >= min_len (vectorised run-length)."""
    if not mask.any():
        return mask
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_len
    out = np.zeros_like(mask)
    for s, e in zip(starts[keep], ends[keep]):
        out[s:e] = True
    return out


def load_record(path: str, fs_override: float | None = None,
                ann_fs: float | None = None, channel: str | None = None,
                modality: str | None = None, p: dict = PARAMS) -> Record:
    df = read_table(path)
    cols = identify_columns(df)
    LOG.info("columns: %s", {k: v for k, v in cols.items() if v})

    # ---- time axis -------------------------------------------------------
    n = len(df)
    fs_source = None
    if cols["time"]:
        t = parse_time_column(df[cols["time"][0]])
        t, _ = fill_nans(t)
        d = np.diff(t)
        fs = 1.0 / np.median(d[d > 0])
        fs_source = f"median diff of '{cols['time'][0]}'"
    else:
        if fs_override:
            fs, fs_source = float(fs_override), "--fs"
        elif ann_fs:
            fs, fs_source = float(ann_fs), "annotation time/sample regression"
        else:
            fs, fs_source = p["default_fs"], "DEFAULT (no time column!)"
            LOG.warning("no time column and no --fs: assuming fs = %.1f Hz", fs)
        if cols["sample"]:
            idx = pd.to_numeric(df[cols["sample"][0]], errors="coerce").to_numpy(float)
            idx, _ = fill_nans(idx)
        else:
            idx = np.arange(n, dtype=float)
        t = idx / fs
    if fs_override and cols["time"]:
        LOG.info("--fs given but a time column exists; fs from time column is used")

    # sort / de-duplicate if needed (keeps output row order: we map back)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    d = np.diff(ts)
    med = np.median(d[d > 0])
    jitter = np.abs(d - med) > p["nonuniform_tol"] * med
    # any real gap (> 1.5 median intervals), duplicate or out-of-order stamp, or
    # widespread jitter forces resampling onto a uniform grid
    nonuniform = (jitter.mean() > p["nonuniform_frac"]) or np.any(d <= 0) \
        or np.any(d > 1.5 * med) or np.any(order != np.arange(n))
    info = {"n_rows": n, "fs": fs, "fs_source": fs_source,
            "duration_s": float(ts[-1] - ts[0] + med),
            "jitter_frac": float(jitter.mean()),
            "max_gap_s": float(d.max()) if len(d) else 0.0}

    # ---- channel selection ----------------------------------------------
    def numeric(c):
        return pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    if channel:
        if channel not in df.columns:
            raise SystemExit(f"--channel {channel!r} not in {list(df.columns)}")
        prim_name = channel
    elif cols["ppg"]:
        prim_name = cols["ppg"][0]
    elif cols["ecg"]:
        prim_name = cols["ecg"][0]
    elif cols["unknown"]:
        prim_name = cols["unknown"][0]
    else:
        raise SystemExit("no usable signal column found")
    if modality is None:
        if prim_name in cols["ppg"]:
            modality = "ppg"
        elif prim_name in cols["ecg"]:
            modality = "ecg"
        else:
            modality = spectral_modality(numeric(prim_name), fs)
            LOG.info("modality of %r from spectral heuristic: %s", prim_name, modality)
    ecg_ref_name = cols["ecg"][0] if (modality == "ppg" and cols["ecg"]) else None

    # ---- uniform grid ------------------------------------------------------
    t0 = ts[0]
    if nonuniform:
        m = int(np.floor((ts[-1] - t0) * fs)) + 1
        tg = t0 + np.arange(m) / fs
        LOG.info("non-uniform sampling (%.1f %% jitter, max gap %.3f s): linear "
                 "resampling to %d-point uniform grid at %.2f Hz",
                 100 * info["jitter_frac"], info["max_gap_s"], m, fs)
    else:
        tg = None

    def to_grid(x):
        x = x[order]
        good = np.isfinite(x)
        if tg is None:
            xf, bad = fill_nans(x)
            return xf, bad
        if good.sum() < 2:
            return np.zeros(len(tg)), np.ones(len(tg), bool)
        xg = np.interp(tg, ts[good], x[good])
        # grid points far from any valid sample are "invalid"
        tv = ts[good]
        k = np.clip(np.searchsorted(tv, tg), 1, len(tv) - 1)
        dist = np.minimum(np.abs(tg - tv[k - 1]), np.abs(tv[k] - tg))
        return xg, dist > 1.5 / fs

    prim, bad = to_grid(numeric(prim_name))
    nan_frac = float(bad.mean())
    valid = ~long_runs(bad, int(p["max_interp_gap_s"] * fs))
    if tg is not None:          # gaps in the time column itself
        gap_mask = np.zeros(len(tg), bool)
        big = np.flatnonzero(d > p["max_interp_gap_s"])
        for g in big:
            gap_mask[(tg > ts[g]) & (tg < ts[g + 1])] = True
        valid &= ~gap_mask
    acc = None
    if cols["acc"]:
        acc = np.column_stack([to_grid(numeric(c))[0] for c in cols["acc"]])
    ecg_ref = to_grid(numeric(ecg_ref_name))[0] if ecg_ref_name else None

    if tg is None:
        grid_index = np.empty(n, int)
        grid_index[order] = np.arange(n)
    else:
        grid_index = np.clip(np.rint((t - t0) * fs).astype(int), 0, len(tg) - 1)

    info.update({"primary": prim_name, "modality": modality, "acc": cols["acc"],
                 "ecg_ref": ecg_ref_name, "nan_frac": nan_frac,
                 "resampled": tg is not None, "columns": cols})
    LOG.info("primary=%r (%s)  fs=%.3f Hz [%s]  duration=%.1f s  NaN=%.2f %%  "
             "acc=%s  ecg_ref=%s", prim_name, modality, fs, fs_source,
             info["duration_s"], 100 * nan_frac, cols["acc"] or None, ecg_ref_name)
    return Record(t_orig=t, fs=fs, primary=prim, primary_name=prim_name,
                  modality=modality, acc=acc, ecg_ref=ecg_ref, valid=valid,
                  grid_index=grid_index, resampled=tg is not None, t0=t0, info=info)


def make_record(x, fs, modality, acc=None, t0=0.0) -> Record:
    """Build a Record from in-memory, uniformly sampled arrays (used by the
    robustness / tuning tools)."""
    x, bad = fill_nans(np.asarray(x, float))
    n = len(x)
    return Record(t_orig=t0 + np.arange(n) / fs, fs=fs, primary=x,
                  primary_name="signal", modality=modality, acc=acc, ecg_ref=None,
                  valid=~long_runs(bad, int(PARAMS["max_interp_gap_s"] * fs)),
                  grid_index=np.arange(n), resampled=False, t0=t0,
                  info={"duration_s": n / fs})


# =========================================================================== #
# Annotation parsing
# =========================================================================== #
def _hms_to_s(txt: str) -> float:
    parts = [float(v) for v in txt.split(":")]
    s = 0.0
    for v in parts:
        s = s * 60 + v
    return s


def parse_annotation(path: str) -> dict:
    """Robust annotation parser.

    Supported formats
      * WFDB `rdann` text dump (MIT-BIH):  Time  Sample#  Type  Sub Chan Num [Aux]
        -> beats (by WFDB beat code), rhythm changes '(XXX' in Aux, events such as
           MISSB comments / 'x' non-conducted P waves
      * one number per line (peak times in seconds, or sample indices if all
        integers and values exceed 1e4) with optional header
      * CSV with a column matching time/sample/peak
    Returns dict(times, samples, types, aux, fs, beat_mask, format)."""
    raw = Path(path).read_text().splitlines()
    rows = [r for r in raw if r.strip()]
    wfdb = re.compile(r"^\s*(\d+:\d+(?::\d+)?\.\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+"
                      r"(-?\d+)\s+(-?\d+)\s*(.*)$")
    parsed = [wfdb.match(r) for r in rows]
    hits = [m for m in parsed if m]
    if len(hits) >= 0.5 * max(len(rows) - 1, 1):
        times = np.array([_hms_to_s(m.group(1)) for m in hits])
        samples = np.array([int(m.group(2)) for m in hits])
        types = np.array([m.group(3) for m in hits])
        aux = np.array([m.group(7).strip() for m in hits])
        fs = None
        if len(times) > 2 and np.ptp(times) > 0:
            # annotation times have 1-ms resolution, so the regression is only
            # meaningful to ~1e-3 Hz; rounding removes least-squares noise
            # (360.0000054 -> 360.0)
            fs = round(float(np.polyfit(times, samples, 1)[0]), 3)
        return {"format": "wfdb", "times": times, "samples": samples,
                "types": types, "aux": aux, "fs": fs,
                "beat_mask": np.isin(types, list(WFDB_BEATS))}
    # generic: numbers in the first numeric-looking column
    try:
        df = pd.read_csv(io.StringIO("\n".join(rows)), sep=None, engine="python")
        df.columns = [_clean_name(c) for c in df.columns]
        pick = None
        for c in df.columns:
            if re.search(r"time|sec|peak|sample|idx", c, re.I):
                pick = c
                break
        if pick is None:
            raise ValueError
        v = pd.to_numeric(df[pick], errors="coerce").dropna().to_numpy(float)
        is_sample = bool(re.search(r"sample|idx", pick, re.I))
    except Exception:
        v = pd.to_numeric(pd.Series(rows), errors="coerce").dropna().to_numpy(float)
        is_sample = bool(np.all(v == np.round(v)) and v.max() > 1e4)
    n = len(v)
    return {"format": "list", "times": None if is_sample else v,
            "samples": v.astype(int) if is_sample else None,
            "types": np.array(["N"] * n), "aux": np.array([""] * n), "fs": None,
            "beat_mask": np.ones(n, bool)}


def reference_from_annotation(ann: dict, fs: float, t0: float = 0.0) -> dict:
    """Reference beat times (s), types, and reference anomaly intervals (s).

    Anomaly definition used for validation (documented in the report):
      * any rhythm episode whose label is not '(N' (e.g. '(BII' 2:1 AV block,
        '(AFIB', '(B' bigeminy ...), from its marker to the next rhythm marker
      * each non-normal beat (V, A, S, F, J, E, a, e, j, n, ...): interval from the
        previous beat to the next beat (premature + compensatory interval)
      * each dropped beat (MISSB comment, 'x' non-conducted P): previous beat ->
        next beat (the pause)."""
    if ann["format"] == "wfdb" and ann.get("fs"):
        # sample numbers are at the ANNOTATION's rate (time/sample regression),
        # which is more precise than the ms-resolution time column
        tt = ann["samples"] / ann["fs"] + t0
    elif ann["times"] is not None:
        tt = ann["times"]
    else:
        tt = ann["samples"] / fs + t0
    bm = ann["beat_mask"]
    beat_t, beat_ty = tt[bm], ann["types"][bm]
    intervals = []
    # rhythm episodes
    rh = [(t, a) for t, a in zip(tt, ann["aux"]) if a.startswith("(")]
    end_t = beat_t[-1] if len(beat_t) else (tt[-1] if len(tt) else 0.0)
    for i, (t, a) in enumerate(rh):
        stop = rh[i + 1][0] if i + 1 < len(rh) else end_t
        if a.strip() not in ("(N",):
            intervals.append((t, stop, a.strip()))
    # ectopic beats & dropped beats
    def around(t):
        k = np.searchsorted(beat_t, t)
        lo = beat_t[max(k - 1, 0)]
        hi = beat_t[min(k + 1, len(beat_t) - 1)] if k < len(beat_t) and \
            beat_t[k] == t else beat_t[min(k, len(beat_t) - 1)]
        return lo, hi
    for t, ty in zip(beat_t, beat_ty):
        if ty not in WFDB_NORMAL and ty != "Q":
            lo, hi = around(t)
            intervals.append((lo, hi, ty))
    ev = np.isin(ann["aux"], list(ANOMALY_EVENT_AUX)) | \
        np.isin(ann["types"], list(ANOMALY_EVENT_TYPES))
    for t in tt[ev]:
        k = np.searchsorted(beat_t, t)
        if 0 < k < len(beat_t):
            intervals.append((beat_t[k - 1], beat_t[k], "dropped"))
    return {"beat_t": beat_t, "beat_types": beat_ty, "intervals": intervals}


# =========================================================================== #
# Filtering helpers
# =========================================================================== #
def butter_sos(kind: str, fc, fs: float, order: int):
    nyq = fs / 2.0
    if kind == "bandpass":
        lo, hi = fc
        hi = min(hi, 0.45 * fs)
        if lo <= 0:
            return signal.butter(order, hi, "lowpass", fs=fs, output="sos")
        return signal.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")
    if kind == "lowpass" and fc >= 0.45 * fs:
        return None
    return signal.butter(order, min(fc, 0.95 * nyq), kind, fs=fs, output="sos")


def zfilt(sos, x):
    """Zero-phase filtering (forward-backward) -> no peak-timing bias."""
    return x if sos is None else signal.sosfiltfilt(sos, x)


def decimate_to(x: np.ndarray, fs: float, target: float):
    """Anti-aliased decimation by an integer factor to ~target Hz."""
    q = max(1, int(fs // target))
    if q == 1:
        return x, fs
    y = zfilt(butter_sos("lowpass", 0.4 * fs / q, fs, 4), x)[::q]
    return y, fs / q


# =========================================================================== #
# 2. Baseline wander removal
# =========================================================================== #
def baseline_butter(x, fs, modality, p=PARAMS):
    sos = butter_sos("highpass", p["hp_cutoff"][modality], fs, p["hp_order"])
    return x - zfilt(sos, x)          # returns the *baseline* estimate


def baseline_median(x, fs, modality, p=PARAMS):
    """Two-stage moving median (de Chazal et al. 2004 for ECG: 200 ms removes
    QRS/P, 600 ms removes T).  Computed on a ~50 Hz subsample and linearly
    interpolated back: the baseline is < 1 Hz so this loses nothing and makes
    the median filter ~7x cheaper."""
    xd, fsd = x[:: max(1, int(fs // 50))], fs / max(1, int(fs // 50))
    b = xd
    for w in p["median_win_s"][modality]:
        k = int(round(w * fsd)) | 1
        b = ndimage.median_filter(b, size=k, mode="nearest")
    return np.interp(np.arange(len(x)) / fs, np.arange(len(xd)) / fsd, b)


def baseline_spline(x, fs, modality, p=PARAMS):
    """Cubic spline through the median of each 1-s segment (knot at centre)."""
    L = int(p["spline_knot_s"] * fs)
    m = len(x) // L
    if m < 4:
        return np.full_like(x, np.median(x))
    med = np.median(x[: m * L].reshape(m, L), axis=1)
    tk = (np.arange(m) + 0.5) * L
    cs = interpolate.CubicSpline(tk, med, bc_type="natural")
    return cs(np.clip(np.arange(len(x)), tk[0], tk[-1]))


BASELINES = {"butter": baseline_butter, "median": baseline_median,
             "spline": baseline_spline}


def band_power(x, fs, lo, hi, nperseg=None):
    f, P = signal.welch(x, fs, nperseg=nperseg or int(min(len(x), 16 * fs)))
    m = (f >= lo) & (f < hi)
    return P[m].sum() * (f[1] - f[0]), f, P


def compare_baselines(x, fs, modality, p=PARAMS) -> dict:
    """Score each baseline method on the same signal:
      lf_db      residual power < 0.5 Hz relative to raw (more negative = better)
      inband_db  change of power in the cardiac band 0.7 .. f_hi Hz (0 = ideal;
                 measures distortion of the waveform we want to keep)
      runtime_s  wall-clock
    score = -lf_db - 10 * |inband_db|  (1 dB of in-band distortion costs as much
    as 10 dB of extra low-frequency suppression)."""
    hi = p["lowpass"][modality]
    x = x - np.median(x)
    lf_raw, f, P_raw = band_power(x, fs, 0.0, 0.5)
    ib_raw, _, _ = band_power(x, fs, 0.7, hi)
    res = {"_psd_raw": (f, P_raw)}
    for name, fn in BASELINES.items():
        t0 = time.perf_counter()
        y = x - fn(x, fs, modality, p)
        dt = time.perf_counter() - t0
        lf, f, P = band_power(y, fs, 0.0, 0.5)
        ib, _, _ = band_power(y, fs, 0.7, hi)
        lf_db = 10 * np.log10(lf / lf_raw)
        ib_db = 10 * np.log10(ib / ib_raw)
        res[name] = {"lf_db": lf_db, "inband_db": ib_db, "runtime_s": dt,
                     "score": -lf_db - 10 * abs(ib_db), "psd": (f, P)}
    return res


# =========================================================================== #
# 3. Motion artefact suppression and HR ridge tracking
# =========================================================================== #
def notch_line_noise(x, fs, p=PARAMS):
    """Adaptive mains notch: applied only if a 50/60 Hz peak stands > 10 dB above
    its 2-5 Hz neighbourhood (so clean files are not touched)."""
    applied = []
    f, P = signal.welch(x, fs, nperseg=int(min(len(x), 4 * fs)))
    for f0 in (50.0, 60.0):
        if f0 > 0.45 * fs:
            continue
        pk = P[np.abs(f - f0) <= 0.5].max()
        nb = np.median(P[(np.abs(f - f0) >= 2) & (np.abs(f - f0) <= 5)])
        if 10 * np.log10(pk / max(nb, 1e-20)) > p["line_noise_db"]:
            b, a = signal.iirnotch(f0, p["notch_q"], fs)
            x = signal.filtfilt(b, a, x)
            applied.append(f0)
    return x, applied


def stft_power(x, fs, p=PARAMS):
    """|STFT|^2 with an 8-s Hann window and 1-s hop, zero-padded to ~0.012 Hz."""
    nper = int(round(p["stft_win_s"] * fs))
    hop = int(round(p["stft_hop_s"] * fs))
    nfft = int(2 ** np.ceil(np.log2(max(nper, fs / 0.0125))))
    f, t, Z = signal.stft(x, fs, window="hann", nperseg=nper, noverlap=nper - hop,
                          nfft=nfft, boundary="even", padded=True)
    return f, t, Z


def track_hr(P, f, p=PARAMS, penalty_map=None):
    """Heart-rate ridge through the spectrogram by Viterbi dynamic programming.

    emission  E(f) = P(f) + w * P(2f)   (harmonic support favours the fundamental)
              normalised per frame, cost = -log E
    transition cost = hr_jump_cost * |delta bpm|  (soft continuity constraint:
              small HR changes are cheap, large jumps need strong evidence)
    penalty_map (optional, 0..1 per bin) down-weights bins where motion
              (accelerometer) power is high.
    Returns HR in Hz per frame."""
    lo, hi = np.array(p["hr_range_bpm"]) / 60.0
    sel = (f >= lo) & (f <= hi)
    fb = f[sel]
    P1 = P[sel]
    k2 = np.searchsorted(f, 2 * fb).clip(0, len(f) - 1)
    E = P1 + p["hr_harmonic_weight"] * P[k2]
    if penalty_map is not None:
        E = E * np.clip(1.0 - penalty_map[sel], 0.1, 1.0)
    E = E / (E.sum(axis=0, keepdims=True) + 1e-20)
    cost = -np.log(E + 1e-12)                        # (bins, frames)
    bpm = fb * 60.0
    lam = p["hr_jump_cost"]
    nb, nt = cost.shape
    ar = np.arange(nb)
    acc = cost[:, 0].copy()
    back = np.empty((nb, nt), np.int32)
    # min-plus step with an L1 transition cost, exact in O(bins) per frame:
    #   best(i) = min_j acc(j) + lam |bpm_i - bpm_j|
    # = min( lam bpm_i + cummin_{j<=i}(acc_j - lam bpm_j),
    #       -lam bpm_i + revcummin_{j>=i}(acc_j + lam bpm_j) )
    for j in range(1, nt):                          # loop over frames only
        g = acc - lam * bpm
        cf = np.minimum.accumulate(g)
        af = np.maximum.accumulate(np.where(g <= cf, ar, 0))
        h = (acc + lam * bpm)[::-1]
        cb = np.minimum.accumulate(h)
        ab = (nb - 1 - np.maximum.accumulate(np.where(h <= cb, ar, 0)))[::-1]
        fwd = lam * bpm + cf
        bwd = cb[::-1] - lam * bpm
        use_f = fwd <= bwd
        back[:, j] = np.where(use_f, af, ab)
        acc = np.where(use_f, fwd, bwd) + cost[:, j]
    path = np.empty(nt, np.int32)
    path[-1] = int(np.argmin(acc))
    for j in range(nt - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    return fb[path]


def nlms_cancel(d, U, taps, mu, delta=None):
    """Regularised normalised LMS: d = PPG (primary), U = (n, k) accelerometer
    references.  y is the motion estimate; e = d - y is the cleaned PPG.

        w <- w + mu e x / (delta + |x|^2),   delta = L * mean regressor power

    The regulariser matters: when the wrist is still, the accelerometer holds
    only sensor noise, |x|^2 -> 0 and plain NLMS takes huge steps that fit the
    cardiac pulse onto noise (measured: residual motion x8 WORSE).  With delta
    the step shrinks automatically when there is no motion to cancel.
    The loop runs at the decimated rate (25 Hz => 45 k iterations per 30 min)."""
    n, k = U.shape
    Upad = np.vstack([np.zeros((taps - 1, k)), U])
    X = np.lib.stride_tricks.sliding_window_view(Upad, taps, axis=0)[:, :, ::-1]
    X = np.ascontiguousarray(X.reshape(n, k * taps))
    nrm = np.einsum("ij,ij->i", X, X)
    nrm += np.mean(nrm) if delta is None else delta
    w = np.zeros(k * taps)
    y = np.empty(n)
    for i in range(n):
        xi = X[i]
        yi = w @ xi
        w += (mu * (d[i] - yi) / nrm[i]) * xi
        y[i] = yi
    return y


def upsample_to(x_d, q, n):
    if q == 1:
        return x_d[:n]
    y = signal.resample_poly(x_d, q, 1)
    return np.pad(y, (0, max(0, n - len(y))), mode="edge")[:n]


def motion_suppress_ppg(x, fs, acc, p=PARAMS):
    """PPG motion suppression (x = baseline-free, band-limited PPG).
    1. NLMS with accelerometer references (if present and coherent)
    2. STFT HR-ridge tracking
    3. spectral exclusion: narrow-band peaks that are NOT near k*HR (and, with an
       accelerometer, bins where the accelerometer has power) are attenuated by
       -20 dB; broadband content (incl. irregular-rhythm energy) is kept.
    All spectral work is done at ~25 Hz (PPG content < 8 Hz); the removed
    component is interpolated back to fs (polyphase) and subtracted."""
    info = {}
    xd, fsd = decimate_to(x, fs, p["stft_fs"])
    q = int(round(fs / fsd))
    removed_d = np.zeros_like(xd)
    Ad = None
    if acc is not None and acc.shape[1] > 0:
        band = butter_sos("bandpass", p["det_band"]["ppg"], fs, 2)
        Ad = np.column_stack([decimate_to(zfilt(band, acc[:, j]), fs, p["stft_fs"])[0]
                              for j in range(acc.shape[1])])
        Ad = (Ad - Ad.mean(0)) / (Ad.std(0) + 1e-12)
        nper = int(8 * fsd)
        coh = []
        for j in range(Ad.shape[1]):
            fc, C = signal.coherence(xd, Ad[:, j], fsd, nperseg=min(nper, len(xd)))
            coh.append(C[(fc >= 0.5) & (fc <= 5)].mean())
        info["acc_coherence"] = [round(float(c), 3) for c in coh]
        if max(coh) > p["nlms_min_coherence"]:
            y = nlms_cancel(xd, Ad, p["nlms_taps"], p["nlms_mu"])
            removed_d += y
            info["nlms"] = True
        else:
            info["nlms"] = False
        LOG.info("accelerometer coherence %s -> NLMS %s", info["acc_coherence"],
                 "applied" if info["nlms"] else "skipped")
    xd_n = xd - removed_d
    f, tt, Z = stft_power(xd_n, fsd, p)
    P = np.abs(Z) ** 2
    pen = None
    if Ad is not None:
        _, _, Za = stft_power(Ad.sum(1), fsd, p)
        Pa = np.abs(Za) ** 2
        pen = Pa / (Pa.max(axis=0, keepdims=True) + 1e-20)
    hr = track_hr(P, f, p, pen)
    # spectral exclusion only when no coherent reference was cancelled: with
    # NLMS applied, the extra mask removed cardiac energy of irregular rhythms
    # (anomaly Se 0.79 -> 0.45 at 0 dB on the synthetic test) for no peak gain
    if p["motion_mask"] and not info.get("nlms", False):
        guard = np.zeros(P.shape, bool)
        for k in range(1, p["motion_harmonics"] + 1):
            guard |= np.abs(f[:, None] - k * hr[None, :]) <= p["motion_guard_hz"]
        med = np.median(P[(f >= 0.3) & (f <= 8)], axis=0, keepdims=True)
        loc = P >= ndimage.maximum_filter1d(P, 9, axis=0)     # local spectral max
        motion = loc & (P > p["motion_peak_ratio"] * med) & ~guard
        if pen is not None:
            motion |= (pen > 0.3) & ~guard
        motion = ndimage.binary_dilation(motion, np.ones((9, 1), bool))  # +-0.05 Hz
        motion &= ~guard
        gain = np.where(motion, p["motion_gain"], 1.0)
        _, xm = signal.istft(Z * gain, fsd, window="hann",
                             nperseg=int(round(p["stft_win_s"] * fsd)),
                             noverlap=int(round(p["stft_win_s"] * fsd))
                             - int(round(p["stft_hop_s"] * fsd)),
                             nfft=Z.shape[0] * 2 - 2, boundary=True)
        xm = xm[: len(xd_n)]
        removed_d += xd_n - np.pad(xm, (0, len(xd_n) - len(xm)))
        info["motion_bins_frac"] = float(motion[(f >= 0.5) & (f <= 8)].mean())
    removed = upsample_to(removed_d, q, len(x))
    info.update({"hr_t": tt, "hr_hz": hr, "stft": (f, tt, P)})
    return x - removed, removed, info


def beat_snr_db(x, fs, beat_t, pre=0.25, post=0.45):
    """Beat-synchronous SNR using reference beat times: the ensemble median beat
    is 'signal', the per-beat deviation from it is 'noise' (baseline wander,
    motion, EMG all count as noise).  SNR = 10 log10(var(template)/var(resid))."""
    idx = np.rint(np.asarray(beat_t) * fs).astype(int)
    off = np.arange(-int(pre * fs), int(post * fs))
    ok = (idx + off[0] >= 0) & (idx + off[-1] < len(x))
    B = x[idx[ok, None] + off[None, :]]
    B = B - np.median(x)
    tmpl = np.median(B, axis=0)
    return float(10 * np.log10(tmpl.var() / np.mean((B - tmpl) ** 2)))


def preprocess(rec: Record, p=PARAMS) -> dict:
    """Stages 1-3.  Returns the cleaned signal and intermediates."""
    fs, mod = rec.fs, rec.modality
    x0 = rec.primary - np.median(rec.primary)                  # 1. centring
    base = BASELINES[p["baseline_method"]](x0, fs, mod, p)     # 2. baseline
    xb = x0 - base
    out = {"x0": x0, "baseline": base, "xb": xb}
    if mod == "ecg":
        xn, notched = notch_line_noise(xb, fs, p)
        clean = zfilt(butter_sos("lowpass", p["lowpass"]["ecg"], fs, p["lp_order"]), xn)
        # "removed" = in-band artefact only (mains + > 40 Hz EMG).  Baseline
        # wander is removed almost perfectly and must not lower confidence.
        out.update(clean=clean, removed=xb - clean, notch=notched, motion_info={})
    else:
        xl = zfilt(butter_sos("lowpass", p["lowpass"]["ppg"], fs, p["lp_order"]), xb)
        clean, removed, info = motion_suppress_ppg(xl, fs, rec.acc, p)
        out.update(clean=clean, removed=removed, notch=[],
                   motion_info=info, pre_motion=xl)
    return out


# =========================================================================== #
# 4. Beat (R-peak / systolic-peak) detection -- Pan-Tompkins style
# =========================================================================== #
def detection_envelope(x0, clean, fs, mod, p=PARAMS):
    """ECG: 5-15 Hz band-pass of the *centred raw* signal (QRS band, rejects P/T,
    baseline and EMG), derivative, squaring, moving-window integration (MWI).
    PPG: 0.5-8 Hz cleaned pulse, derivative clipped to positive values (only
    the systolic upstroke), squaring, MWI.  The MWI is centred (symmetric
    window) so its maxima are not delayed."""
    src = x0 if mod == "ecg" else clean
    bp = zfilt(butter_sos("bandpass", p["det_band"][mod], fs, 2), src)
    d = np.gradient(bp) * fs
    if mod == "ppg":
        d = np.clip(d, 0, None)
    w = max(3, int(round(p["mwi_s"][mod] * fs)))
    mwi = ndimage.uniform_filter1d(d * d, size=w, mode="nearest")
    return bp, np.abs(d), mwi


def detect_beats(mwi, slope_abs, fs, valid, p=PARAMS) -> dict:
    """Adaptive dual-threshold beat detection on MWI candidates.

    Candidates : local maxima of the MWI at least one refractory period apart
                 (scipy find_peaks, vectorised).
    Classify   : candidate > TH1 -> beat (SPK <- 0.125 a + 0.875 SPK)
                 else noise (NPK <- 0.125 a + 0.875 NPK)
                 TH1 = NPK + 0.25 (SPK - NPK),  TH2 = 0.5 TH1
    Secondary  : within 360 ms of the last beat and max slope < 50 % of the last
    wave         beat's slope -> T wave (ECG) / dicrotic wave (PPG) -> noise
    Search-back: if no beat for 1.66 x mean(last 8 IBI), the largest skipped
                 candidate above TH2 is accepted (SPK <- 0.25 a + 0.75 SPK)
    Re-acquire : if no beat for 3 s, SPK is halved (recovers from an artefact
                 that inflated SPK)
    The loop runs over candidates (~2 per beat), not over samples."""
    dist = max(1, int(p["refractory_s"] * fs))
    cand, _ = signal.find_peaks(mwi, distance=dist)
    cand = cand[valid[cand]]
    if len(cand) == 0:
        return {"peaks": np.array([], int), "cand": cand, "th1": np.array([])}
    amp = mwi[cand]
    slope = ndimage.maximum_filter1d(slope_abs, size=max(3, int(0.15 * fs)))[cand]
    n_init = int(10 * fs)
    SPK = mwi[:n_init].max() / 3.0
    NPK = mwi[:n_init].mean() / 2.0
    rs, rn = p["spk_rate"], p["npk_rate"]
    sw = int(p["secondary_wave_s"] * fs)
    lost = int(p["lost_signal_s"] * fs)
    beats, bslope, rr = [], [], []
    last_pos = -1                     # candidate index of last accepted beat
    ref_t = cand[0]
    th1_log = np.empty(len(cand))

    def accept(j, sb=False):
        nonlocal SPK, last_pos, ref_t
        r = rs * 2 if sb else rs
        SPK = r * amp[j] + (1 - r) * SPK
        if beats:
            rr.append(cand[j] - beats[-1])
            if len(rr) > p["rr_avg_n"]:
                rr.pop(0)
        beats.append(cand[j])
        bslope.append(slope[j])
        last_pos = j
        ref_t = cand[j]

    for i in range(len(cand)):
        c, a = cand[i], amp[i]
        th1 = NPK + p["thr_frac"] * (SPK - NPK)
        th1_log[i] = th1
        if beats and rr:
            rr_avg = np.mean(rr)
            if c - beats[-1] > p["searchback_factor"] * rr_avg and i - last_pos > 1:
                js = np.arange(last_pos + 1, i)
                ok = (cand[js] - beats[-1] >= dist) & (c - cand[js] >= dist) & \
                     (amp[js] > p["thr2_frac"] * th1)
                if ok.any():
                    j = js[ok][np.argmax(amp[js[ok]])]
                    accept(j, sb=True)
                    th1 = NPK + p["thr_frac"] * (SPK - NPK)
        if a > th1:
            if beats and c - beats[-1] < sw and \
                    slope[i] < p["secondary_slope_ratio"] * bslope[-1]:
                NPK = rn * a + (1 - rn) * NPK
                continue
            accept(i)
        else:
            NPK = rn * a + (1 - rn) * NPK
            if c - ref_t > lost:
                SPK *= 0.5
                ref_t = c
    return {"peaks": np.array(beats, int), "cand": cand, "th1": th1_log}


def refine_peaks(env_peaks, clean, mwi, fs, mod, p=PARAMS):
    """Move each envelope maximum to the fiducial point on the cleaned signal:
    ECG -> largest |deflection| within +-100 ms (handles inverted/ectopic QRS);
    PPG -> maximum (systolic peak) within [-50, +300] ms of the upstroke energy.
    Then enforce the physiological minimum IBI (300 ms) by dropping the weaker
    (lower MWI) beat of any too-close pair."""
    if len(env_peaks) == 0:
        return env_peaks
    a, b = p["refine_win_s"][mod]
    off = np.arange(int(a * fs), int(b * fs) + 1)
    idx = np.clip(env_peaks[:, None] + off[None, :], 0, len(clean) - 1)
    seg = clean[idx]
    k = np.argmax(np.abs(seg) if mod == "ecg" else seg, axis=1)
    pk = np.unique(idx[np.arange(len(idx)), k])
    min_d = p["ibi_min_ms"] / 1000 * fs
    while len(pk) > 1:
        close = np.flatnonzero(np.diff(pk) < min_d)
        if len(close) == 0:
            break
        j = close[0]
        drop = j if mwi[pk[j]] < mwi[pk[j + 1]] else j + 1
        pk = np.delete(pk, drop)
    return pk


def compute_ibi(peaks, fs):
    """IBI (ms) assigned to the LATER peak of each pair; first peak -> NaN."""
    ibi = np.full(len(peaks), np.nan)
    if len(peaks) > 1:
        ibi[1:] = np.diff(peaks) / fs * 1000.0
    return ibi


def hr_track_for(mod, clean, mwi, fs, pre, p=PARAMS):
    """HR ridge: PPG -> already tracked during motion suppression; ECG -> track
    on the MWI envelope (a pulse train whose fundamental is the heart rate)."""
    if mod == "ppg" and "hr_hz" in pre["motion_info"]:
        mi = pre["motion_info"]
        return mi["hr_t"], mi["hr_hz"], mi["stft"]
    env = mwi - mwi.mean()
    xd, fsd = decimate_to(env, fs, p["stft_fs"])
    f, t, Z = stft_power(xd, fsd, p)
    P = np.abs(Z) ** 2
    return t, track_hr(P, f, p), (f, t, P)


# =========================================================================== #
# 6/7. Signal-quality indices, confidence score
# =========================================================================== #
def rolling(x, w, fn=np.nanmedian, centred=True):
    """Rolling statistic over a 1-D array with NaN padding (vectorised)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    if centred:
        pad = np.r_[np.full(w // 2, np.nan), x, np.full(w - 1 - w // 2, np.nan)]
    else:                                   # window i = x[i-w .. i-1]
        pad = np.r_[np.full(w, np.nan), x[:-1]]
    V = np.lib.stride_tricks.sliding_window_view(pad, w)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(V, axis=1)


def beat_sqis(clean, peaks, fs, mod, p=PARAMS) -> dict:
    """Per-beat template correlation and amplitude stability."""
    n = len(peaks)
    if n < 3:
        return {"template": np.zeros(n), "amplitude": np.zeros(n), "amp": np.zeros(n)}
    med_ibi = np.median(np.diff(peaks)) / fs
    pre = np.clip(0.25 * med_ibi, 0.15, 0.40)
    post = np.clip(0.45 * med_ibi, 0.25, 0.60)
    L = p["template_len"]
    off = np.rint(np.linspace(-pre * fs, post * fs, L)).astype(int)
    idx = np.clip(peaks[:, None] + off[None, :], 0, len(clean) - 1)
    Braw = clean[idx]
    c0 = int(np.argmin(np.abs(off)))
    amp = np.ptp(Braw, axis=1) if mod == "ecg" else Braw[:, c0] - Braw[:, :c0 + 1].min(1)
    Bz = Braw - Braw.mean(1, keepdims=True)
    Bz /= Bz.std(1, keepdims=True) + 1e-12
    r = np.empty(n)
    blk = p["template_block_beats"]
    for s in range(0, n, blk):               # one median template per block
        e = min(n, s + blk)
        s0 = max(0, e - blk)                 # last block borrows earlier beats
        T = np.median(Bz[s0:e], axis=0)
        T = (T - T.mean()) / (T.std() + 1e-12)
        r[s:e] = Bz[s:e] @ T / L
    lm = rolling(amp, 9)
    rel = np.abs(amp - lm) / (np.abs(lm) + 1e-12)
    return {"template": np.clip(r, 0, 1),
            "amplitude": np.exp(-rel / p["amp_dev_scale"]), "amp": amp}


def block_sqis(clean, removed, fs, mod, p=PARAMS) -> dict:
    """Per 2-s block: shape SQI (ECG kurtosis / PPG skewness, logistic-mapped)
    and motion SQI = 1 - var(removed) / (var(removed) + var(clean)), i.e. the
    fraction of the raw power that was NOT artefact."""
    L = max(8, int(p["sqi_block_s"] * fs))
    m = max(1, len(clean) // L)
    C = clean[: m * L].reshape(m, L)
    R = removed[: m * L].reshape(m, L)
    if mod == "ecg":
        v = stats.kurtosis(C, axis=1, fisher=False)
    else:
        v = stats.skew(C, axis=1)
    shape = 1 / (1 + np.exp(-(v - p["shape_center"][mod]) / p["shape_scale"][mod]))
    vr, vc = R.var(1), C.var(1)
    motion = 1 - vr / (vr + vc + 1e-20)
    tc = (np.arange(m) + 0.5) * L
    return {"t_idx": tc, "shape": shape, "motion": motion, "raw_shape": v}


def spectral_purity(f, P, hr_hz, mod, p=PARAMS):
    fmax = p["lowpass"]["ppg"] if mod == "ppg" else 8.0
    band = (f >= 0.5) & (f <= fmax)
    near = np.zeros(P.shape, bool)
    for k in range(1, p["purity_harmonics"][mod] + 1):
        near |= np.abs(f[:, None] - k * hr_hz[None, :]) <= p["purity_bw_hz"]
    num = (P * (near & band[:, None])).sum(0)
    den = P[band].sum(0) + 1e-20
    return num / den


def confidence_score(n, fs, peaks, bsq, blk, hr_t, purity, valid, p=PARAMS):
    """Per-sample confidence in [0, 1]:
        C = 0.35 template + 0.15 purity + 0.15 amplitude + 0.15 shape + 0.20 motion
    beat-level SQIs are linearly interpolated between beats, block/frame SQIs
    between block/frame centres; morphology SQIs are 0 where no beat was found
    for > 3 s; C is smoothed over 1 s and forced to 0 in invalid gaps."""
    w = p["sqi_weights"]
    xi = np.arange(n)
    comp = {}
    if len(peaks) >= 2:
        comp["template"] = np.interp(xi, peaks, bsq["template"])
        comp["amplitude"] = np.interp(xi, peaks, bsq["amplitude"])
        k = np.searchsorted(peaks, xi).clip(1, len(peaks) - 1)
        dist = np.minimum(np.abs(xi - peaks[k - 1]), np.abs(peaks[k] - xi))
        far = dist > p["no_beat_s"] * fs
        comp["template"][far] = 0
        comp["amplitude"][far] = 0
    else:
        comp["template"] = comp["amplitude"] = np.zeros(n)
    comp["purity"] = np.interp(xi, hr_t * fs, np.clip(purity / p["purity_norm"], 0, 1))
    comp["shape"] = np.interp(xi, blk["t_idx"], blk["shape"])
    comp["motion"] = np.interp(xi, blk["t_idx"], blk["motion"])
    C = sum(w[k] * comp[k] for k in w) / sum(w.values())
    C = ndimage.uniform_filter1d(C, max(1, int(p["conf_smooth_s"] * fs)))
    C[~valid] = 0.0
    return np.clip(C, 0, 1), comp


# =========================================================================== #
# 6. Rule-based anomaly detection
# =========================================================================== #
def detect_anomalies(peaks, ibi, fs, n, beat_template, beat_motion, beat_shape,
                     p=PARAMS) -> dict:
    """Beat- and window-level rules on the RAW IBI series (ms).

    R1 premature + compensatory (PVC signature): IBI_i < 0.80 x local median AND
       IBI_{i+1} > 1.10 x local median                -> flag [peak_{i-1}, peak_{i+1}]
    R2 pause / dropped beat: IBI_i > 1.6 x median of the preceding 30 IBIs
                                                      -> flag [peak_{i-1}, peak_i]
    R3 marked bradycardia: IBI_i > 1500 ms (HR < 40)  -> flag [peak_{i-1}, peak_i]
    R4 AF-like irregularity in a 10-beat sliding window, computed on the
       ectopic-corrected IBIs (IBIs explained by R1/R2 are removed first):
         v1 nRMSSD = RMSSD / mean IBI > 0.12           (irregular magnitude)
         v2 turning-point count within +-1.96 sigma of the random expectation
         v3 Poincare SD1/SD2 > 0.6                     (no short-term structure)
       fires if v1 AND (v1 + v2 + v3 >= 2)              -> flag whole window
    Quality gate: a rule can only fire where the local (+-5 beat) median template
    correlation >= 0.86 AND shape SQI >= 0.5 AND motion SQI >= 0.5.  Irregularity in low-quality
    signal is logged as "noise-irregular" and NOT flagged: morphology SQIs do
    not depend on rhythm, so a clean irregular rhythm keeps a high SQI while
    motion-induced false beats lower it."""
    nb = len(peaks)
    mask = np.zeros(n + 1, np.int32)
    reasons = np.zeros(nb, np.int32)                     # bit field per beat
    out = {"mask": np.zeros(n, bool), "reasons": reasons, "segments": [],
           "counts": {}, "noise_irregular": 0}
    if nb < 4:
        return out
    I = ibi.copy()
    I[(I < p["ibi_min_ms"])] = np.nan
    locmed = rolling(I, p["local_median_beats"])
    longref = rolling(I, p["long_ref_beats"], centred=False)
    longref = np.where(np.isfinite(longref), longref, locmed)
    tq = rolling(beat_template, 11)
    mq = rolling(beat_motion, 11)
    sq = rolling(beat_shape, 11)
    ok = (tq >= p["gate_template_corr"]) & (mq >= p["gate_motion"]) & \
        (sq >= p["gate_shape"])
    nxt = np.r_[I[1:], np.nan]
    with np.errstate(invalid="ignore"):
        r1 = (I < p["premature_ratio"] * locmed) & (nxt > p["compensatory_ratio"] * locmed)
        r2 = I > p["pause_ratio"] * longref
        r3 = I > p["brady_ibi_ms"]
    raw = {"R1": r1, "R2": r2, "R3": r3}
    r1, r2, r3 = r1 & ok, r2 & ok, r3 & ok
    segs = []
    for i in np.flatnonzero(r1):
        segs.append((peaks[i - 1], peaks[min(i + 1, nb - 1)]))
    for i in np.flatnonzero(r2 | r3):
        segs.append((peaks[i - 1], peaks[i]))
    reasons |= r1 * 1 | r2 * 2 | r3 * 4
    # ---- R4 irregularity on ectopic-corrected IBIs -----------------------
    Ic = I.copy()
    expl = r1 | np.r_[False, r1[:-1]] | raw["R2"] | raw["R3"]
    Ic[expl] = np.nan
    W = p["irr_win_beats"]
    r4_win = np.zeros(nb, bool)
    fire = np.zeros(0, bool)
    if nb > W:
        V = np.lib.stride_tricks.sliding_window_view(Ic, W)     # (nw, W)
        cnt = np.isfinite(V).sum(1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(V, 1)
            dV = np.diff(V, axis=1)
            rmssd = np.sqrt(np.nanmean(dV ** 2, 1))
            nrmssd = rmssd / mean
            a, b, c = V[:, :-2], V[:, 1:-1], V[:, 2:]
            tp = np.nansum(((b - a) * (b - c)) > 0, 1)
            nv = np.isfinite(a + b + c).sum(1) + 2
            mu = 2 * (nv - 2) / 3
            sig = np.sqrt(np.maximum(16 * nv - 29, 1) / 90)
            sd1 = np.nanstd(dV, 1) / np.sqrt(2)
            sdnn = np.nanstd(V, 1)
            sd2 = np.sqrt(np.maximum(2 * sdnn ** 2 - sd1 ** 2, 1e-12))
            v1 = nrmssd > p["nrmssd_th"]
            v2 = np.abs(tp - mu) <= p["tpr_z"] * sig
            v3 = (sd1 / sd2) > p["sd_ratio_th"]
        fire = v1 & ((v1.astype(int) + v2 + v3) >= p["irr_votes"]) & (cnt >= 6)
        # window j covers IBIs j..j+W-1 (beats j-1 .. j+W-1)
        wq = np.lib.stride_tricks.sliding_window_view(ok, W).all(1)
        out["noise_irregular"] = int((fire & ~wq).sum())
        fire &= wq
        for j in np.flatnonzero(fire):
            segs.append((peaks[max(j - 1, 0)], peaks[j + W - 1]))
            r4_win[j:j + W] = True
    reasons |= r4_win * 8
    # ---- propagate to samples (difference array) and merge ----------------
    for s, e in segs:
        mask[s] += 1
        mask[e + 1] -= 1
    m = np.cumsum(mask[:n]) > 0
    gap = int(p["merge_gap_s"] * fs)
    if gap > 0 and m.any():
        m = ~long_runs(~m, gap + 1) | m
        # keep leading/trailing unflagged regions untouched
        d = np.flatnonzero(m)
        m[: d[0]] = False
        m[d[-1] + 1:] = False
    out["mask"] = m
    out["counts"] = {"R1_premature": int(r1.sum()), "R2_pause": int(r2.sum()),
                     "R3_brady": int(r3.sum()), "R4_irregular_windows":
                     int(fire.sum()),
                     "gated_out": int(sum((raw[k] & ~ok).sum() for k in raw))}
    return out


# =========================================================================== #
# Orchestration
# =========================================================================== #
class Timer:
    def __init__(self):
        self.t = {}
        self._t0 = time.perf_counter()
        self._last = self._t0

    def lap(self, name):
        now = time.perf_counter()
        self.t[name] = now - self._last
        self._last = now

    def total(self):
        return time.perf_counter() - self._t0


def run(rec: Record, p=PARAMS, timer: Timer | None = None) -> dict:
    timer = timer or Timer()
    fs, mod = rec.fs, rec.modality
    pre = preprocess(rec, p)
    timer.lap("baseline+motion")
    bp, slope, mwi = detection_envelope(pre["x0"], pre["clean"], fs, mod, p)
    det = detect_beats(mwi, slope, fs, rec.valid, p)
    peaks = refine_peaks(det["peaks"], pre["clean"], mwi, fs, mod, p)
    peaks = peaks[rec.valid[peaks]]
    ibi = compute_ibi(peaks, fs)
    timer.lap("peak_detection")
    hr_t, hr_hz, (f, ft, P) = hr_track_for(mod, pre["clean"], mwi, fs, pre, p)
    purity = spectral_purity(f, P, hr_hz, mod, p)
    bsq = beat_sqis(pre["clean"], peaks, fs, mod, p)
    blk = block_sqis(pre["clean"], pre["removed"], fs, mod, p)
    n = len(pre["clean"])
    conf, comp = confidence_score(n, fs, peaks, bsq, blk, hr_t, purity, rec.valid, p)
    timer.lap("confidence")
    bmot = np.interp(peaks, blk["t_idx"], blk["motion"])
    bshape = np.interp(peaks, blk["t_idx"], blk["shape"])
    ano = detect_anomalies(peaks, ibi, fs, n, bsq["template"], bmot, bshape, p)
    ano["mask"] &= rec.valid
    timer.lap("anomaly")
    n_long = int(np.sum(ibi > p["ibi_max_ms"]))
    LOG.info("beats=%d  median IBI=%.0f ms  IBI>%.0f ms (gaps)=%d  mean conf=%.3f",
             len(peaks), np.nanmedian(ibi) if len(peaks) > 1 else np.nan,
             p["ibi_max_ms"], n_long, conf.mean())
    LOG.info("anomaly rules: %s  noise-irregular windows suppressed=%d  "
             "flagged=%.1f %% of samples", ano["counts"], ano["noise_irregular"],
             100 * ano["mask"].mean())
    return {"pre": pre, "bp": bp, "mwi": mwi, "det": det, "peaks": peaks,
            "ibi": ibi, "hr_t": hr_t, "hr_hz": hr_hz, "stft": (f, ft, P),
            "purity": purity, "bsq": bsq, "blk": blk, "conf": conf, "comp": comp,
            "ano": ano}


def to_output_rows(rec: Record, res: dict, ibi_fill="zero") -> pd.DataFrame:
    """Map uniform-grid results back to the input rows (one row per sample)."""
    n = len(rec.t_orig)
    t = rec.t_orig
    is_peak = np.zeros(n, np.int8)
    ibi_col = np.zeros(n)
    peaks = res["peaks"]
    if len(peaks):
        tp = rec.t0 + peaks / rec.fs
        order = np.argsort(t, kind="stable")
        ts = t[order]
        k = np.clip(np.searchsorted(ts, tp), 1, n - 1)
        near = np.where(np.abs(ts[k] - tp) < np.abs(ts[k - 1] - tp), k, k - 1)
        rows = order[near]
        is_peak[rows] = 1
        ibi_vals = np.nan_to_num(res["ibi"], nan=0.0)
        ibi_col[rows] = ibi_vals
        if ibi_fill == "hold":        # sample-and-hold of the most recent IBI
            held = np.zeros(n)
            held[order[near]] = ibi_vals
            idx = np.where(np.isin(np.arange(n), order[near]), np.arange(n), 0)
            ibi_col = held[np.maximum.accumulate(idx)] if n else held
    gi = rec.grid_index
    c = OUTPUT_COLUMNS
    return pd.DataFrame({
        c["time"]: np.round(t, 6),
        c["peak"]: is_peak,
        c["ibi"]: np.round(ibi_col, 1),
        c["anomaly"]: res["ano"]["mask"][gi].astype(np.int8),
        c["confidence"]: np.round(res["conf"][gi], 4),
    })


# =========================================================================== #
# Validation
# =========================================================================== #
def match_events(det, ref, tol):
    """One-to-one matching of detected to reference times (s) within +-tol,
    closest pairs first.  Returns (det_idx, ref_idx)."""
    if len(det) == 0 or len(ref) == 0:
        return np.array([], int), np.array([], int)
    k = np.clip(np.searchsorted(ref, det), 1, len(ref) - 1)
    di = np.r_[np.arange(len(det)), np.arange(len(det))]
    ri = np.r_[k - 1, k]
    dist = np.abs(det[di] - ref[ri])
    keep = dist <= tol
    di, ri, dist = di[keep], ri[keep], dist[keep]
    o = np.argsort(dist, kind="stable")
    di, ri = di[o], ri[o]
    _, u = np.unique(ri, return_index=True)
    di, ri = di[u], ri[u]
    _, u = np.unique(di, return_index=True)
    return di[u], ri[u]


def confusion(pred, ref):
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref)); tn = int(np.sum(~pred & ~ref))
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * ppv * se / max(ppv + se, 1e-12)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "sensitivity": se,
            "specificity": sp, "precision": ppv, "F1": f1}


def runs_of(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def evaluate(rec: Record, res: dict, ref: dict, p=PARAMS, t_range=None) -> dict:
    fs = rec.fs
    det_t = res["peaks"] / fs
    ref_t = ref["beat_t"] - rec.t0
    n = len(res["conf"])
    if t_range is not None:
        a, b = t_range
        det_sel = (det_t >= a) & (det_t < b)
        det_t, ref_t = det_t[det_sel], ref_t[(ref_t >= a) & (ref_t < b)]
    out = {"n_det": len(det_t), "n_ref": len(ref_t)}
    # lag between detected fiducials and reference (PPG systolic peak lags R)
    di, ri = match_events(det_t, ref_t, p["max_lag_s"])
    lag = float(np.median(det_t[di] - ref_t[ri])) if len(di) else 0.0
    out["lag_ms"] = 1000 * lag
    shift = lag if abs(lag) > 0.02 else 0.0
    out["lag_compensated"] = bool(shift)
    for tol in p["tolerances_ms"]:
        di, ri = match_events(det_t - shift, ref_t, tol / 1000)
        tp = len(di)
        pr, rc = tp / max(len(det_t), 1), tp / max(len(ref_t), 1)
        out[f"peak@{int(tol)}ms"] = {"TP": tp, "FP": len(det_t) - tp,
                                    "FN": len(ref_t) - tp, "precision": pr,
                                    "recall": rc, "F1": 2 * pr * rc / max(pr + rc, 1e-12)}
    di, ri = match_events(det_t - shift, ref_t, max(p["tolerances_ms"]) / 1000)
    o = np.argsort(di)
    di, ri = di[o], ri[o]
    cons = (np.diff(di) == 1) & (np.diff(ri) == 1)
    d_ibi = np.diff(det_t[di])[cons] * 1000
    r_ibi = np.diff(ref_t[ri])[cons] * 1000
    err = d_ibi - r_ibi
    out["ibi"] = {"n_pairs": int(len(err)),
                  "MAE_ms": float(np.mean(np.abs(err))) if len(err) else np.nan,
                  "RMSE_ms": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                  "bias_ms": float(np.mean(err)) if len(err) else np.nan,
                  "LoA_ms": float(1.96 * np.std(err)) if len(err) else np.nan}
    out["_ibi_pairs"] = (r_ibi, d_ibi, ref_t[ri][1:][cons])
    # anomaly, sample level
    ref_mask = np.zeros(n, bool)
    for s, e, _ in ref["intervals"]:
        a_, b_ = int(round((s - rec.t0) * fs)), int(round((e - rec.t0) * fs))
        ref_mask[max(a_, 0):min(b_ + 1, n)] = True
    pred = res["ano"]["mask"].copy()
    sel = np.ones(n, bool)
    if t_range is not None:
        sel[:] = False
        sel[int(t_range[0] * fs):int(t_range[1] * fs)] = True
    out["anomaly_sample"] = confusion(pred[sel], ref_mask[sel])
    # anomaly, event level (any overlap counts)
    rr = [(s, e) for s, e in runs_of(ref_mask & sel)]
    pr_ = [(s, e) for s, e in runs_of(pred & sel)]
    hit_ref = sum(bool(pred[s:e].any()) for s, e in rr)
    hit_pred = sum(bool(ref_mask[s:e].any()) for s, e in pr_)
    out["anomaly_event"] = {"ref_events": len(rr), "detected": hit_ref,
                            "event_sensitivity": hit_ref / max(len(rr), 1),
                            "pred_events": len(pr_), "true_pred": hit_pred,
                            "event_precision": hit_pred / max(len(pr_), 1)}
    out["_ref_mask"] = ref_mask
    # HR track vs reference HR
    if len(ref_t) > 2:
        rhr = 60 / np.diff(ref_t)
        rh_t = ref_t[1:]
        tt = res["hr_t"]
        win = np.array([np.median(rhr[(rh_t >= t - 4) & (rh_t < t + 4)])
                        if np.any((rh_t >= t - 4) & (rh_t < t + 4)) else np.nan
                        for t in tt])
        ok = np.isfinite(win)
        if t_range is not None:
            ok &= (tt >= t_range[0]) & (tt < t_range[1])
        out["hr_track_MAE_bpm"] = float(np.mean(np.abs(res["hr_hz"][ok] * 60 - win[ok])))
        out["_ref_hr"] = (tt, win)
    return out


def public(d):
    """Drop private (underscore) keys and make JSON-serialisable."""
    if isinstance(d, dict):
        return {k: public(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, (list, tuple)):
        return [public(v) for v in d]
    return d


# =========================================================================== #
# main
# =========================================================================== #
def build_argparser():
    ap = argparse.ArgumentParser(
        description="CARDIO-CODE classical-DSP beat / IBI / anomaly pipeline")
    ap.add_argument("--input", required=True, help="input CSV (PPG and/or ECG)")
    ap.add_argument("--output", default="predictions.csv", help="output CSV path")
    ap.add_argument("--annotation", help="reference annotations for validation")
    ap.add_argument("--plots", action="store_true", help="write figures")
    ap.add_argument("--plot-dir", default="plots", help="figure directory")
    ap.add_argument("--metrics", default=None,
                    help="write metrics JSON here (default: next to output when "
                         "--annotation is given)")
    ap.add_argument("--fs", type=float, help="sampling rate if there is no time column")
    ap.add_argument("--channel", help="force the primary signal column")
    ap.add_argument("--modality", choices=["ppg", "ecg"], help="force signal type")
    ap.add_argument("--ibi-fill", choices=["zero", "hold"], default="zero",
                    help="Ibi_ms on non-peak rows: 0 (default) or last IBI held")
    ap.add_argument("--no-motion-mask", action="store_true",
                    help="disable STFT spectral exclusion of motion peaks")
    ap.add_argument("--log-level", default="INFO")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    p = dict(PARAMS)
    if a.no_motion_mask:
        p["motion_mask"] = False
    timer = Timer()
    ann = parse_annotation(a.annotation) if a.annotation else None
    rec = load_record(a.input, a.fs, ann["fs"] if ann else None, a.channel,
                      a.modality, p)
    timer.lap("load")
    res = run(rec, p, timer)
    df = to_output_rows(rec, res, a.ibi_fill)
    df.to_csv(a.output, index=False)      # values pre-rounded in to_output_rows
    timer.lap("write_csv")
    total = timer.total()
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    eff = {"wall_s": total, "stages_s": timer.t,
           "samples": int(len(rec.t_orig)),
           "samples_per_s": len(rec.t_orig) / total,
           "realtime_factor": rec.info["duration_s"] / total,
           "peak_rss_MB": mem}
    LOG.info("wrote %s (%d rows)  total %.2f s  (%.2f M samples/s, %.0fx real "
             "time)  peak RSS %.0f MB", a.output, len(df), total,
             eff["samples_per_s"] / 1e6, eff["realtime_factor"], mem)
    metrics = {"input": rec.info | {"columns": None}, "efficiency": eff,
               "params": {k: v for k, v in p.items()}}
    if ann is not None:
        ref = reference_from_annotation(ann, rec.fs, 0.0)
        full = evaluate(rec, res, ref, p)
        ref_shift = ref["beat_t"] - rec.t0 + full["lag_ms"] / 1000
        half = rec.info["duration_s"] / 2
        metrics["validation"] = {
            "full": full,
            "first_half_tuning": evaluate(rec, res, ref, p, (0, half)),
            "second_half_test": evaluate(rec, res, ref, p, (half, 1e12))}
        metrics["validation"]["snr_db"] = {
            "raw": beat_snr_db(res["pre"]["x0"], rec.fs, ref_shift),
            "clean": beat_snr_db(res["pre"]["clean"], rec.fs, ref_shift)}
        v = full
        LOG.info("peaks @50ms: P=%.4f R=%.4f F1=%.4f | @100ms F1=%.4f | lag %.1f ms",
                 v["peak@50ms"]["precision"], v["peak@50ms"]["recall"],
                 v["peak@50ms"]["F1"], v["peak@100ms"]["F1"], v["lag_ms"])
        LOG.info("IBI: MAE=%.2f ms RMSE=%.2f ms bias=%.2f LoA=+-%.2f (n=%d)",
                 v["ibi"]["MAE_ms"], v["ibi"]["RMSE_ms"], v["ibi"]["bias_ms"],
                 v["ibi"]["LoA_ms"], v["ibi"]["n_pairs"])
        s = v["anomaly_sample"]
        LOG.info("anomaly (samples): Se=%.3f Sp=%.3f P=%.3f F1=%.3f | events "
                 "detected %d/%d, pred-event precision %.3f", s["sensitivity"],
                 s["specificity"], s["precision"], s["F1"],
                 v["anomaly_event"]["detected"], v["anomaly_event"]["ref_events"],
                 v["anomaly_event"]["event_precision"])
        LOG.info("beat-synchronous SNR: raw %.2f dB -> clean %.2f dB",
                 metrics["validation"]["snr_db"]["raw"],
                 metrics["validation"]["snr_db"]["clean"])
    else:
        ref = None
    mpath = a.metrics or (str(Path(a.output).with_suffix("")) + "_metrics.json"
                          if ann is not None else None)
    if mpath:
        Path(mpath).write_text(json.dumps(public(metrics), indent=2, default=str))
        LOG.info("metrics -> %s", mpath)
    if a.plots:
        try:                                      # plotting kept out of the hot path
            from tools.figures import make_figures
        except ImportError:
            LOG.warning("--plots needs the tools/ folder next to pipeline.py")
        else:
            make_figures(rec, res, ref, metrics, a.plot_dir, p)
    return metrics


if __name__ == "__main__":
    main()
