#!/usr/bin/env python3
"""
CARDIO-CODE — DSP-only arrhythmia detection pipeline for noisy ECG / PPG.

    python pipeline.py --input ecg.csv --output predictions.csv [--fs 360] [--mode auto|ecg|ppg] [--channel MLII]

Stages (all classical signal processing; rationale and measurements in report.pdf):
  1. load + sanitize          (NaN interpolation, clipping / flat-line mask)
  2. baseline-wander removal  (zero-phase Butterworth high-pass / band-pass)
  3. beat enhancement         (ECG: Pan-Tompkins energy; PPG: Elgendi two moving averages)
  4. adaptive peak picking    (dual thresholds, refractory period, RR search-back)
  5. peak refinement          (dominant-polarity extremum + parabolic sub-sample interpolation)
  6. IBI series               (physiological gating)
  7. signal quality           (beat-to-template correlation, amplitude consistency, clipping) -> confidence 0..1
  8. anomaly rules            (pause / dropped beat, bradycardia, premature beat, AF-type irregularity)
  9. predictions.csv          (Timestamp, Is_peak, Ibi_ms, ano_flag, confidence_score)
"""
import argparse
import csv
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

# --------------------------------------------------------------------------------------
# Parameters (every value is justified in report.pdf)
# --------------------------------------------------------------------------------------
P = dict(
    default_fs=360.0,           # Hz, used only without a time column and without --fs (MIT-BIH rate)
    nonuniform_tol=0.05,        # |dt - median dt| > 5 % of median counts as jitter ...
    nonuniform_frac=0.01,       # ... resample to a uniform grid if > 1 % of intervals are jittered / gaps
    max_interp_gap_s=1.0,       # s, longer NaN runs / time gaps are invalid (no beats, confidence 0)
    line_noise_db=10.0,         # dB, mains peak prominence needed before notching (ECG)
    nlms=True,                  # accelerometer NLMS motion cancellation (PPG with ACC columns)
    nlms_taps=8, nlms_mu=0.01,  # per axis taps (0.32 s at 25 Hz), normalised step
    nlms_min_coherence=0.10,    # PPG/ACC coherence 0.5-5 Hz required to apply NLMS
    ms_after_nlms=False,        # spectral line removal after NLMS (off: NLMS alone is better)
    # ECG
    ecg_hp=0.5,                 # Hz, baseline-wander high-pass (AHA monitoring cut-off)
    ecg_qrs_band=(5.0, 15.0),   # Hz, Pan-Tompkins QRS band
    ecg_mwi=0.150,              # s, moving-window integration length
    ecg_refractory=0.25,        # s, no two QRS closer than this
    ecg_refine=0.10,            # s, +/- window for R-peak refinement
    ecg_lp=40.0,                # Hz, upper edge of the morphology band
    ecg_detector='hybrid',      # 'pt' (Pan-Tompkins), 'dp', or 'hybrid' (window-wise fusion)
    fuse_win=10.0,              # s, fusion window
    fuse_ratio=1.25,            # ECG fusion: choose DP if PT's median |dIBI| is this much larger
    ecg_noise_gate=0.45,        # between-beat/at-beat envelope ratio below which a window counts as clean
    ppg_noise_gate=0.2,             # choose DP if PT's median |dIBI| is this much larger
    # PPG
    ppg_band=(0.5, 8.0),        # Hz, MSPTDfast / Elgendi pass band
    ppg_w1=0.111,               # s, Elgendi systolic-peak window
    ppg_w2=0.667,               # s, Elgendi beat window
    ppg_beta=0.02,              # Elgendi offset
    ppg_refractory=0.30,        # s
    # PPG motion suppression (short-time spectral)
    motion_suppress=True,       # short-time spectral motion line removal (PPG)
    ms_fs=25.0,                 # Hz, working rate of the spectral stage
    ms_win=8.0, ms_hop=1.0,     # s, STFT window / hop
    ms_protect=0.15,            # Hz, protected half-width around each cardiac harmonic
    ms_rel=0.5,                 # line counts as motion if |X| > rel * |X(f0)|
    ms_purity=0.75,             # frames with spectral purity below this are motion-corrupted
    hr_track_lambda=100.0,        # Viterbi penalty per Hz of HR change between frames
    hr_octave_cost=4.0,         # cost of an HR halving/doubling (AV block onset)
    hr_jump_cost=10.0,          # cap on any HR change cost (sudden rhythm change, e.g. block ends)
    ppg_detector='dp',          # 'dp' (fused Elgendi + HR-guided DP) or 'elgendi'
    ppg_fuse_ratio=2.2,        # PPG fusion bias towards Elgendi
    dp_scope='fuse',            # 'fuse' (noise-gated window fusion), 'all' (DP everywhere) or 'motion'
    dp_weight=3.0,              # IBI deviation penalty per unit |log(IBI/expected)|
    dp_octave=0.3,              # extra penalty for an IBI of ~2x expected (dropped beat)
    dp_start=1.0,               # cost of starting a new beat chain
    dp_min_rise=0.3,            # candidate must rise >= 30% of the local typical beat
    dp_pur_ref=0.8,             # purity at which the HR prior gets full weight
    dp_wmin=1.0,                # minimum prior weight (very clean signal)
    # Search-back
    searchback_factor=1.66,     # x RR average (Pan-Tompkins)
    # IBI gate
    ibi_min=0.25, ibi_max=3.0,  # s
    # Quality
    sqi_template_half=5.0,      # s, template built from beats within +/- this
    sqi_smooth=2.0,             # s, smoothing of per-sample confidence
    quality_gate=0.5,           # rules are only trusted above this beat quality
    amp_sigma=2.0,              # octaves; amplitude tolerance of beat quality used by the anomaly rules
    conf_fusion=False,          # add purity / shape / motion SQIs to confidence (measured: no gain, off)
    conf_amp_sigma=0.5,         # octaves; amplitude tolerance inside the confidence score
    conf_w=(0.5, 0.15, 0.15, 0.2),  # weights: beat quality, purity, shape, motion
    sqi_block=2.0,              # s, block length of shape / motion SQIs
    tmpl_refine_ecg=False,      # template cross-correlation peak refinement
    tmpl_refine_ppg=False,      # tried: worsens IBI MAE
    tmpl_lag=0.06,              # s, max shift
    tmpl_pre=0.20, tmpl_post=0.05,  # s, PPG correlation window around the apex (upstroke dominated)
    tmpl_min_corr=0.5,          # only refine beats that resemble the template
    # Anomaly rules
    ref_beats=8,                # beats in the running reference IBI
    pause_ratio=1.5,            # IBI > ratio * reference  -> pause / dropped beat
    pause_abs=2.0,              # s, IBI above this is always a pause
    pause_gap_check=False,       # veto pauses that contain an undetected pulse-sized peak
    pause_gap_frac=0.5,         # ... peak higher than this fraction of the median beat height
    brady_bpm=45.0,             # median HR of 5 beats below this -> bradycardia
    premature_ratio=0.80,       # IBI < ratio * reference ...
    compensatory_ratio=1.60,    # ... and IBI + next IBI > ratio * reference
    morph_corr=0.80,            # ECG ectopic morphology: corr below this with good neighbours
    af_win=16,                  # beats in the irregularity window
    af_masd=0.10,               # median |dIBI| / median IBI above this -> irregular
    af_qmin=0.5,                # median beat quality required in an AF window
    af_sd_ratio=0.6,            # Poincare SD1/SD2 above this = no short-term structure (AF ~1, sinus 0.2-0.5)
    af_vote='either',           # second AF condition: 'tpr', 'sd', 'either' or 'both'
    af_nrmssd=0.15,             # RMSSD / mean IBI must also exceed this (rejects sinus arrhythmia)
    event_qmin=0.5,             # beat quality required for pause / premature events
    flag_dilate=0.0,            # s, dilate flagged spans
    flag_close=6.0,             # s, close gaps shorter than this
)


# --------------------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------------------
# Input layer: delimiter sniffing, column identification by name, time-unit detection, uniform resampling of
# jittered / gappy time axes, invalid-gap masking.
_RX = {
    'time': re.compile(r'(time|timestamp|^t$|^t_|sec|^ms$|date)', re.I),
    'sample': re.compile(r'(sample|^idx$|^index$|^n$)', re.I),
    'acc': re.compile(r'(acc|accel|^a_?[xyz]$|^[xyz]$|^[xyz]_|gyr|imu|motion)', re.I),
    'ppg': re.compile(r'(ppg|pleth|bvp|pulse|^ir$|infra|^red$|green|optic|^pg)', re.I),
    'ecg': re.compile(r'(ecg|ekg|mlii|^ml|lead|^v[1-6]$|^i{1,3}$|^av[rlf]$)', re.I),
}


def read_table(path):
    with open(path, 'r', newline='') as fh:
        head = fh.read(8192)
    try:
        delim = csv.Sniffer().sniff(head, delimiters=',;\t| ').delimiter
    except csv.Error:
        delim = ','
    df = pd.read_csv(path, sep=delim, engine='c', skipinitialspace=True)
    df.columns = [re.sub(r'^[\s\'"]+|[\s\'"]+$', '', str(c)) for c in df.columns]
    return df


def identify_columns(df):
    out = {k: [] for k in ('time', 'sample', 'ppg', 'ecg', 'acc', 'unknown')}
    for c in df.columns:
        for key in ('time', 'sample', 'acc', 'ppg', 'ecg'):
            if _RX[key].search(c):
                out[key].append(c)
                break
        else:
            if pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.9:
                out['unknown'].append(c)
    return out


def parse_time_column(col):
    """Seconds from numeric s / ms / us / ns (unit giving a plausible 8 Hz..5 kHz rate), 'm:ss.sss' or datetimes."""
    num = pd.to_numeric(col, errors='coerce')
    if num.notna().mean() > 0.9:
        v = num.to_numpy(float)
        d = np.diff(v[np.isfinite(v)])
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            if 8.0 <= 1.0 / (med * scale) <= 5000.0:
                return v * scale
        return v
    txt = col.astype(str).str.strip()
    if txt.str.match(r'^\d+:\d+(:\d+)?(\.\d+)?$').mean() > 0.9:
        parts = txt.str.split(':', expand=True).astype(float)
        secs = parts.iloc[:, -1]
        for i in range(2, parts.shape[1] + 1):
            secs = secs + parts.iloc[:, -i] * 60 ** (i - 1)
        return secs.to_numpy(float)
    dt = pd.to_datetime(txt, errors='coerce')
    return (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)


def fill_nans(x):
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x), bad
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x, bad


def long_runs(mask, min_len):
    """Keep only True-runs of length >= min_len."""
    if not mask.any():
        return mask
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros_like(mask)
    for a, b in zip(st, en):
        if b - a >= min_len:
            out[a:b] = True
    return out


def load_record(path, fs=None, channel=None):
    """Read a CSV and return the analysis signal on a uniform time grid plus everything needed to map results
    back to the input rows. Keys: t (per input row, s), x (grid), fs, channel, name, acc (grid, n x k or None),
    valid (grid, False inside long NaN runs / time gaps), grid_index (input row -> grid index), resampled."""
    df = read_table(path)
    cols = identify_columns(df)
    n = len(df)
    if cols['time']:
        t, _ = fill_nans(parse_time_column(df[cols['time'][0]]))
        d = np.diff(t)
        fs = 1.0 / np.median(d[d > 0])
    else:
        fs = fs or P['default_fs']
        base = fill_nans(pd.to_numeric(df[cols['sample'][0]], errors='coerce').to_numpy(float))[0] \
            if cols['sample'] else np.arange(n, dtype=float)
        t = base / fs
    order = np.argsort(t, kind='stable')
    ts = t[order]
    d = np.diff(ts)
    med = np.median(d[d > 0])
    jitter = np.abs(d - med) > P['nonuniform_tol'] * med
    nonuniform = (jitter.mean() > P['nonuniform_frac']) or np.any(d <= 0) or np.any(d > 1.5 * med) \
        or np.any(order != np.arange(n))
    if channel is None:
        cand = cols['ppg'] or cols['ecg'] or cols['unknown']
        if not cand:
            raise ValueError('no usable signal column in %s' % path)
        channel = cand[0]
    t0 = ts[0]
    tg = t0 + np.arange(int(np.floor((ts[-1] - t0) * fs)) + 1) / fs if nonuniform else None

    def to_grid(v):
        v = v[order]
        good = np.isfinite(v)
        if tg is None:
            return fill_nans(v)
        if good.sum() < 2:
            return np.zeros(len(tg)), np.ones(len(tg), bool)
        tv = ts[good]
        k = np.clip(np.searchsorted(tv, tg), 1, len(tv) - 1)
        dist = np.minimum(np.abs(tg - tv[k - 1]), np.abs(tv[k] - tg))
        return np.interp(tg, tv, v[good]), dist > 1.5 / fs

    num = lambda c: pd.to_numeric(df[c], errors='coerce').to_numpy(float)
    x, bad = to_grid(num(channel))
    valid = ~long_runs(bad, int(P['max_interp_gap_s'] * fs))
    if tg is not None:
        for g in np.flatnonzero(d > P['max_interp_gap_s']):
            valid &= ~((tg > ts[g]) & (tg < ts[g + 1]))
    acc = np.column_stack([to_grid(num(c))[0] for c in cols['acc']]) if cols['acc'] else None
    if tg is None:
        gi = np.empty(n, int)
        gi[order] = np.arange(n)
    else:
        gi = np.clip(np.rint((t - t0) * fs).astype(int), 0, len(tg) - 1)
    name = channel.lower() + (' ppg' if channel in cols['ppg'] else ' ecg' if channel in cols['ecg'] else '')
    return dict(t=t, x=x, fs=float(fs), channel=channel, name=name, acc=acc, valid=valid, grid_index=gi,
                resampled=tg is not None, t0=t0)


def load_signal(path, fs=None, channel=None):
    """Backward-compatible wrapper used by the dev tools: (t_grid, x_grid, fs, channel, name)."""
    r = load_record(path, fs, channel)
    return r['t0'] + np.arange(len(r['x'])) / r['fs'], r['x'], r['fs'], r['channel'], r['name']


def notch_line_noise(x, fs):
    """Adaptive 50/60 Hz notch, applied only if the mains peak stands > 10 dB above its 2-5 Hz neighbourhood."""
    f, pxx = signal.welch(x, fs, nperseg=int(min(len(x), 4 * fs)))
    for f0 in (50.0, 60.0):
        if f0 > 0.45 * fs:
            continue
        pk = pxx[np.abs(f - f0) <= 0.5].max()
        nb = np.median(pxx[(np.abs(f - f0) >= 2) & (np.abs(f - f0) <= 5)])
        if 10 * np.log10(pk / max(nb, 1e-20)) > P['line_noise_db']:
            b, a = signal.iirnotch(f0, 30.0, fs)
            x = signal.filtfilt(b, a, x)
    return x


def nlms_cancel(d, U, taps, mu):
    """Regularised NLMS: d = PPG, U = (n, k) accelerometer references; returns the motion
    estimate. delta = mean regressor power keeps the step small when the wrist is still."""
    n, k = U.shape
    Upad = np.vstack([np.zeros((taps - 1, k)), U])
    X = np.lib.stride_tricks.sliding_window_view(Upad, taps, axis=0)[:, :, ::-1]
    X = np.ascontiguousarray(X.reshape(n, k * taps))
    nrm = np.einsum('ij,ij->i', X, X)
    nrm += np.mean(nrm)
    w = np.zeros(k * taps)
    y = np.empty(n)
    for i in range(n):
        yi = w @ X[i]
        w += (mu * (d[i] - yi) / nrm[i]) * X[i]
        y[i] = yi
    return y


def acc_cancel(y, fs, acc):
    """Accelerometer-referenced motion cancellation at ~25 Hz; applied only when PPG/ACC coherence in the
    cardiac band exceeds nlms_min_coherence. Returns (cleaned y, applied?)."""
    q = max(1, int(fs // P['ms_fs']))
    lp = lambda v: signal.decimate(v, q, ftype='fir', zero_phase=True) if q > 1 else v
    yd = lp(y)
    fsd = fs / q
    A = np.column_stack([lp(bandpass(acc[:, j], fs, *P['ppg_band'])) for j in range(acc.shape[1])])[:len(yd)]
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    coh = []
    for j in range(A.shape[1]):
        fc, C = signal.coherence(yd, A[:, j], fsd, nperseg=min(int(8 * fsd), len(yd)))
        coh.append(C[(fc >= 0.5) & (fc <= 5)].mean())
    if max(coh) <= P['nlms_min_coherence']:
        return y, False
    m = nlms_cancel(yd, A, P['nlms_taps'], P['nlms_mu'])
    m = np.interp(np.arange(len(y)) / q, np.arange(len(m)), m) if q > 1 else m
    return y - m, True


def sanitize(x, fs):
    """Interpolate NaNs; build a mask of clipped / flat-line samples (unusable signal)."""
    x = x.astype(float).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad]) if (~bad).any() else 0.0
    # clipping: runs of >= 3 samples at the global extremes
    clip = np.zeros(len(x), bool)
    for v in (x.max(), x.min()):
        at = x == v
        run = uniform_filter1d(at.astype(float), 3) > 0.99
        if run.sum() > 3:
            clip |= at
    # flat line: rolling std ~ 0 over 0.5 s
    w = max(3, int(0.5 * fs))
    m = uniform_filter1d(x, w)
    var = uniform_filter1d(x * x, w) - m * m
    flat = var < 1e-6 * (np.var(x) + 1e-12)
    return x, clip | flat | bad


def detect_mode(x, fs, name=''):
    if any(k in name for k in ('ppg', 'pleth', 'bvp', 'pulse')):
        return 'ppg'
    if any(k in name for k in ('ecg', 'mlii', 'lead', 'v1', 'v5')):
        return 'ecg'
    if fs < 50:
        return 'ppg'
    f, pxx = signal.welch(x - np.median(x), fs, nperseg=int(min(len(x), 8 * fs)))
    hi = pxx[(f >= 10) & (f <= min(40, fs / 2 - 1))].sum()
    lo = pxx[(f >= 0.5) & (f <= 5)].sum() + 1e-12
    return 'ecg' if hi / lo > 0.05 else 'ppg'


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def bandpass(x, fs, lo, hi, order=2):
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(order, [lo, hi], 'bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def highpass(x, fs, fc, order=2):
    sos = signal.butter(order, fc, 'highpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def parabolic(y, k):
    """Sub-sample location of the extremum at integer indices k."""
    k = np.clip(k, 1, len(y) - 2)
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return k + np.clip(d, -0.5, 0.5)


def merge_intervals(iv, dilate=0, close=0, n=None):
    if len(iv) == 0:
        return np.zeros((0, 2), int)
    iv = np.asarray(iv, float).copy()
    iv[:, 0] -= dilate
    iv[:, 1] += dilate
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1] + close:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    out = np.array(out)
    if n is not None:
        out = np.clip(out, 0, n - 1)
    return out.round().astype(int)


# --------------------------------------------------------------------------------------
# 3-4. Beat detection
# --------------------------------------------------------------------------------------
def adaptive_pick(env, fs, refractory, cand_dist):
    """Pan-Tompkins style adaptive dual-threshold peak picking with RR search-back on an
    enhancement signal `env` (>= 0). Returns accepted peak indices into env."""
    cand, _ = signal.find_peaks(env, distance=max(1, int(cand_dist * fs)))
    if len(cand) < 2:
        return cand
    h = env[cand]
    init = cand < cand[0] + int(8 * fs)
    spki = np.percentile(h[init], 90) * 0.6 if init.sum() > 2 else h.max() * 0.5
    npki = np.median(h[init]) * 0.5 if init.sum() > 2 else 0.0
    refr = int(refractory * fs)
    acc = []
    rr = []
    last_noise = []                      # rejected candidates since last accepted beat
    for c, hc in zip(cand, h):
        th1 = npki + 0.25 * (spki - npki)
        if acc and c - acc[-1] < refr:
            if hc > env[acc[-1]] and hc > th1:   # a larger peak inside the refractory: replace
                acc[-1] = c
            continue
        if hc > th1:
            # search-back: long gap since last beat -> best rejected candidate above th1/2
            if acc and rr:
                rr_avg = np.median(rr[-8:])
                if c - acc[-1] > P['searchback_factor'] * rr_avg and last_noise:
                    ln = np.array(last_noise)
                    ok = (ln - acc[-1] >= refr) & (c - ln >= refr) & (env[ln] > 0.5 * th1)
                    if ok.any():
                        b = ln[ok][np.argmax(env[ln[ok]])]
                        spki = 0.25 * env[b] + 0.75 * spki
                        rr.append(b - acc[-1])
                        acc.append(b)
            if acc:
                rr.append(c - acc[-1])
            acc.append(c)
            spki = 0.125 * hc + 0.875 * spki
            last_noise = []
        else:
            npki = 0.125 * hc + 0.875 * npki
            last_noise.append(c)
    return np.array(acc, int)


def spectral_track(z, fs):
    """HR track of any beat-periodic signal z (short-time harmonic-sum spectrum + Viterbi)."""
    q = max(1, int(fs // P['ms_fs']))
    zd = signal.decimate(z, q, ftype='fir', zero_phase=True) if q > 1 else z
    fsd = fs / q
    nper, hop = int(P['ms_win'] * fsd), int(P['ms_hop'] * fsd)
    f, _, Z = signal.stft(zd - zd.mean(), fsd, window='hann', nperseg=nper, noverlap=nper - hop, boundary='even')
    return track_hr(np.abs(Z), f), hop / fsd


def ecg_dp(env, fs):
    """ECG beat selection by DP on the Pan-Tompkins envelope (robust to electrode-motion spikes)."""
    a = np.sqrt(env)
    cand, _ = signal.find_peaks(a, distance=max(1, int(P['ecg_refractory'] * fs)))
    if len(cand) < 3:
        return cand
    f0, hop = spectral_track(bandpass(a, fs, 0.5, 8.0), fs)
    scale = pd.Series(a[cand]).rolling(15, center=True, min_periods=1).quantile(0.75).to_numpy()
    sc = np.clip(a[cand] / (scale + 1e-12), 0, 1.5)
    keep = sc >= P['dp_min_rise']
    return dp_select(cand[keep], sc[keep], fs, f0, hop)


def fuse_detectors(a, b, fs, n, ratio=None, e=None, gate=None):
    """Window-wise fusion of two beat sequences (a = precise threshold detector, b = robust DP).
    Where they agree (>= 90 % of beats matched within 100 ms) keep a. Where they disagree, keep the one whose
    IBI series is steadier (median |dIBI|): false detections inflate it, a single missed beat does not."""
    w = int(P['fuse_win'] * fs)
    tol = int(0.1 * fs)
    out = []
    for s0 in range(0, n, w):
        pa = a[(a >= s0) & (a < s0 + w)]
        pb = b[(b >= s0) & (b < s0 + w)]
        if len(pa) == 0 or len(pb) == 0:
            out.append(pa if len(pa) else pb)
            continue
        da = np.abs(pb[:, None] - pa[None, :]).min(1) <= tol
        db = np.abs(pa[:, None] - pb[None, :]).min(1) <= tol
        if da.mean() >= 0.9 and db.mean() >= 0.9:
            out.append(pa)
            continue
        if e is not None and gate is not None and len(pa) >= 2:
            # noise gate: 95th pct of the enhancement signal between beats vs. its median at beats.
            # A clean window (even if irregular, e.g. AF) keeps the threshold detector.
            seg = e[s0:s0 + w]
            keep = np.ones(len(seg), bool)
            h = int(0.15 * fs)
            for p in pa - s0:
                keep[max(0, p - h):p + h] = False
            if keep.any() and np.percentile(seg[keep], 95) < gate * np.median(e[pa]):
                out.append(pa)
                continue
        ctx_a = a[(a >= s0 - w) & (a < s0 + 2 * w)]
        ctx_b = b[(b >= s0 - w) & (b < s0 + 2 * w)]
        ma = np.median(np.abs(np.diff(np.diff(ctx_a)))) if len(ctx_a) > 3 else np.inf
        mb = np.median(np.abs(np.diff(np.diff(ctx_b)))) if len(ctx_b) > 3 else np.inf
        out.append(pb if ma > (ratio or P['fuse_ratio']) * mb else pa)
    return np.concatenate(out).astype(int) if out else a


def detect_ecg(x, fs):
    base = highpass(x, fs, P['ecg_hp'])
    morph = bandpass(x, fs, P['ecg_hp'], P['ecg_lp'])           # morphology band for refinement / SQI
    y = bandpass(x, fs, *P['ecg_qrs_band'], order=3)
    env = uniform_filter1d(np.gradient(y) ** 2, max(1, int(P['ecg_mwi'] * fs)))
    if P['ecg_detector'] == 'dp':
        pk = ecg_dp(env, fs)
    elif P['ecg_detector'] == 'hybrid':
        pk = fuse_detectors(adaptive_pick(env, fs, P['ecg_refractory'], 0.2), ecg_dp(env, fs), fs, len(x),
                            e=np.sqrt(env), gate=P['ecg_noise_gate'])
    else:
        pk = adaptive_pick(env, fs, P['ecg_refractory'], 0.2)
    # dominant polarity of the QRS in the morphology band
    half = int(P['ecg_refine'] * fs)
    off = np.arange(-half, half + 1)
    idx = np.clip(pk[:, None] + off[None, :], 0, len(x) - 1)
    seg = morph[idx]
    pol = 1.0 if np.median(seg.max(1)) >= np.median(-seg.min(1)) else -1.0
    k = idx[np.arange(len(pk)), np.argmax(pol * seg, 1)]
    k = np.unique(k)
    # enforce refractory after refinement (keep larger)
    keep = np.ones(len(k), bool)
    for i in range(1, len(k)):
        if k[i] - k[i - 1] < int(P['ecg_refractory'] * fs):
            j = i if pol * morph[k[i]] < pol * morph[k[i - 1]] else i - 1
            keep[j] = False
    k = k[keep]
    tk = parabolic(pol * morph, k)
    return k, tk, pol * morph, base


def track_hr(S, f, fmin=0.5, fmax=3.5):
    """Viterbi tracking of the cardiac fundamental on a harmonic-sum spectrogram.
    S: |STFT| (freq x frames). Returns f0 per frame (Hz)."""
    band = np.flatnonzero((f >= fmin) & (f <= fmax))
    df = f[1] - f[0]

    def at(mult):
        j = np.clip(np.round(f[band] * mult / df).astype(int), 0, len(f) - 1)
        return S[j]
    hs = S[band] + 0.5 * at(2) + 0.33 * at(3)           # harmonic sum (fundamental wins over harmonics)
    hs = hs / (hs.max(0, keepdims=True) + 1e-12)
    emis = np.log(hs + 1e-3)
    fb = f[band]
    dist = np.abs(fb[:, None] - fb[None, :])             # Hz
    trans = -P['hr_track_lambda'] * dist                 # penalise jumps; octave jumps allowed at a fixed cost
    ratio = fb[:, None] / fb[None, :]
    trans = np.maximum(trans, np.where((np.abs(ratio - 2) < 0.1) | (np.abs(ratio - 0.5) < 0.05),
                                       -P['hr_octave_cost'], -np.inf))
    trans = np.maximum(trans, -P['hr_jump_cost'])        # truncated penalty: abrupt rhythm change possible
    nf = S.shape[1]
    score = emis[:, 0].copy()
    back = np.zeros((len(band), nf), np.int32)
    for k in range(1, nf):
        m = score[None, :] + trans                       # to x from
        back[:, k] = np.argmax(m, 1)
        score = m[np.arange(len(band)), back[:, k]] + emis[:, k]
    path = np.empty(nf, int)
    path[-1] = np.argmax(score)
    for k in range(nf - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    return fb[path]


def motion_suppress(y, fs, acc=None):
    """Short-time spectral motion cancellation (no ML): strong narrow spectral lines that are not
    harmonics of the tracked cardiac fundamental (or that coincide with accelerometer lines) are
    re-synthesised as a motion estimate and subtracted from the band-passed PPG."""
    q = max(1, int(fs // P['ms_fs']))                    # work at ~25 Hz: 15x cheaper at 360 Hz
    yd = signal.decimate(y, q, ftype='fir', zero_phase=True) if q > 1 else y
    fsd = fs / q
    nper = int(P['ms_win'] * fsd)
    hop = int(P['ms_hop'] * fsd)
    f, _, Z = signal.stft(yd, fsd, window='hann', nperseg=nper, noverlap=nper - hop, boundary='even')
    S = np.abs(Z)
    f0 = track_hr(S, f)
    df = f[1] - f[0]
    inb = (f >= P['ppg_band'][0]) & (f <= P['ppg_band'][1])
    # protect bins around cardiac harmonics
    h = np.arange(1, 6)
    near = np.abs(f[:, None, None] - h[None, :, None] * f0[None, None, :]).min(1) <= P['ms_protect'] + df
    # candidate motion lines: local spectral maxima, strong relative to the cardiac fundamental
    j0 = np.clip(np.round(f0 / df).astype(int), 0, len(f) - 1)
    ref = S[j0, np.arange(S.shape[1])]
    locmax = np.zeros_like(S, bool)
    locmax[1:-1] = (S[1:-1] >= S[:-2]) & (S[1:-1] >= S[2:])
    line = locmax & (S > P['ms_rel'] * ref[None, :]) & inb[:, None] & ~near
    if acc is not None:   # accelerometer present: only lines also present in the ACC spectrum
        ad = signal.decimate(acc, q, ftype='fir', zero_phase=True) if q > 1 else acc
        _, _, A = signal.stft(ad[:len(yd)], fsd, window='hann', nperseg=nper, noverlap=nper - hop,
                              boundary='even')
        A = np.abs(A)[:, :S.shape[1]]
        line &= A > 0.3 * A.max(0, keepdims=True)
    # widen each line to the Hann main lobe (+/-2 bins)
    mask = line.copy()
    for s in (1, 2):
        mask[s:] |= line[:-s]
        mask[:-s] |= line[s:]
    mask &= ~near
    _, m = signal.istft(Z * mask, fsd, window='hann', nperseg=nper, noverlap=nper - hop, boundary=True)
    m = m[:len(yd)]
    if len(m) < len(yd):
        m = np.pad(m, (0, len(yd) - len(m)))
    if q > 1:
        m = np.interp(np.arange(len(y)) / q, np.arange(len(m)), m)
    # spectral purity: share of in-band power on the cardiac harmonic comb (1.0 = pure pulse)
    S2 = S ** 2
    pur = (S2 * (near & inb[:, None])).sum(0) / ((S2 * inb[:, None]).sum(0) + 1e-12)
    moving = mask.any(0) | (pur < P['ms_purity'])        # frames treated as motion-corrupted
    return y - m, f0, moving, pur, hop / fsd          # true frame hop in seconds


def elgendi_peaks(y, fs):
    """Elgendi et al. 2013: blocks of interest from two moving averages of the squared, clipped PPG."""
    z = np.clip(y, 0, None) ** 2
    ma_peak = uniform_filter1d(z, max(1, int(P['ppg_w1'] * fs)))
    ma_beat = uniform_filter1d(z, max(1, int(P['ppg_w2'] * fs)))
    alpha = P['ppg_beta'] * uniform_filter1d(z, int(10 * fs))      # local (10 s) mean -> robust to bursts
    boi = ma_peak > ma_beat + alpha
    d = np.diff(boi.astype(np.int8), prepend=0, append=0)
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    w1 = int(P['ppg_w1'] * fs)
    pk = np.array([s + np.argmax(y[s:e]) for s, e in zip(st, en) if e - s >= w1], int)
    if len(pk):   # refractory: keep the larger
        keep = np.ones(len(pk), bool)
        last = 0
        for i in range(1, len(pk)):
            if pk[i] - pk[last] < int(P['ppg_refractory'] * fs):
                if y[pk[i]] > y[pk[last]]:
                    keep[last] = False
                    last = i
                else:
                    keep[i] = False
            else:
                last = i
        pk = pk[keep]
    return pk


def dp_peaks(y, fs, f0, hop, pur=None):
    """Dynamic-programming beat selection. Candidates = local maxima of the cleaned PPG. Each chosen
    candidate earns its normalised rise amplitude; each IBI pays w * |log(IBI / E(t))| where E is the
    expected IBI from the spectral HR track. Octave IBIs (dropped beat, 2:1 block) pay a reduced
    penalty so genuine pauses survive. Solved exactly by Viterbi-style DP over candidates."""
    cand, _ = signal.find_peaks(y, distance=max(1, int(P['ppg_refractory'] * fs)))
    if len(cand) < 3:
        return cand
    # rise amplitude: peak minus the minimum in the preceding 0.35 s
    w = int(0.35 * fs)
    mins = np.array([y[max(0, c - w):c + 1].min() for c in cand])
    rise = y[cand] - mins
    t = cand / fs
    scale = pd.Series(rise).rolling(15, center=True, min_periods=1).quantile(0.75).to_numpy()
    sc = np.clip(rise / (scale + 1e-12), 0, 1.5)
    keep = sc >= P['dp_min_rise']                      # tiny bumps (dicrotic notch, noise) are never beats
    return dp_select(cand[keep], sc[keep], fs, f0, hop, pur)


def dp_select(cand, sc, fs, f0, hop, pur=None):
    """Generic DP over scored candidate beats (see dp_peaks)."""
    if len(cand) < 3:
        return cand
    t = cand / fs
    E = 1.0 / np.interp(t, np.arange(len(f0)) * hop, f0)
    # prior strength follows the noise level: clean (pure) spectrum -> trust the waveform, not the tracker
    if pur is not None:
        pu = np.interp(t, np.arange(len(pur)) * hop, pur)
        wt = P['dp_weight'] * np.clip((1 - pu) / (1 - P['dp_pur_ref']), P['dp_wmin'], 1.0)
    else:
        wt = np.full(len(t), P['dp_weight'])
    n = len(cand)
    best = np.full(n, -np.inf)
    prev = np.full(n, -1)
    octv = P['dp_octave']
    maxgap = P['ibi_max']
    j0 = 0
    run_best, run_arg = 0.0, -1                    # best chain that ended more than maxgap ago
    for i in range(n):
        while t[i] - t[j0] > maxgap:
            if best[j0] > run_best:
                run_best, run_arg = best[j0], j0
            j0 += 1
        best[i] = run_best + sc[i] - P['dp_start']   # start a new chain after a gap (keeps earlier chains)
        prev[i] = run_arg
        js = np.arange(j0, i)
        if len(js):
            ibi = t[i] - t[js]
            r = np.abs(np.log(ibi / E[i]))
            r = np.minimum(r, octv + np.abs(np.log(ibi / (2 * E[i]))))
            v = best[js] + sc[i] - wt[i] * r
            k = np.argmax(v)
            if v[k] > best[i]:
                best[i] = v[k]
                prev[i] = js[k]
    # backtrack from the best end point in the last maxgap seconds
    tail = np.flatnonzero(t >= t[-1] - maxgap)
    i = tail[np.argmax(best[tail])]
    path = []
    while i >= 0:
        path.append(i)
        i = prev[i]
    return cand[np.array(path[::-1])]


def detect_ppg(x, fs, acc=None):
    base = bandpass(x, fs, *P['ppg_band'])
    raw_bp = base
    y = base
    f0 = None
    nlms_on = False
    if acc is not None and P['nlms']:
        base, nlms_on = acc_cancel(base, fs, acc)
        y = base
    if P['motion_suppress'] and (P['ms_after_nlms'] or not nlms_on):
        y, f0, moving, pur, hop = motion_suppress(base, fs, None)
    elif P['motion_suppress']:                 # HR track only (no line removal) after NLMS
        _, f0, moving, pur, hop = motion_suppress(base, fs, None)
    pk = elgendi_peaks(y, fs)
    if P['ppg_detector'] == 'dp' and f0 is not None and P['dp_scope'] == 'fuse':
        pk = fuse_detectors(pk, dp_peaks(y, fs, f0, hop, pur), fs, len(y), P['ppg_fuse_ratio'],
                            e=np.clip(y, 0, None), gate=P['ppg_noise_gate'])
    elif P['ppg_detector'] == 'dp' and f0 is not None and P['dp_scope'] == 'all':
        pk = dp_peaks(y, fs, f0, hop, pur)
    elif P['ppg_detector'] == 'dp' and f0 is not None and moving.any():
        # hybrid: HR-guided DP only inside motion-corrupted stretches (+/- margin), Elgendi elsewhere
        fr = np.flatnonzero(moving)
        iv = merge_intervals(np.stack([fr * hop - P['ms_win'] / 2, fr * hop + P['ms_win'] / 2], 1) * fs,
                             0, P['ms_win'] * fs, len(y))
        inm = np.zeros(len(y), bool)
        for a, b in iv:
            inm[a:b + 1] = True
        dp = dp_peaks(y, fs, f0, hop, pur)
        pk = np.sort(np.concatenate([pk[~inm[pk]], dp[inm[dp]]]))
    tk = parabolic(y, pk)
    return pk, tk, y, raw_bp


# --------------------------------------------------------------------------------------
# 7. Signal quality
# --------------------------------------------------------------------------------------
def beat_quality(morph, pk, fs, mode, mask):
    n = len(pk)
    if n < 3:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    if mode == 'ecg':
        a, b = int(0.10 * fs), int(0.20 * fs)
    else:
        ibi = np.median(np.diff(pk))
        a, b = int(0.35 * ibi), int(0.55 * ibi)
    off = np.arange(-a, b)
    idx = np.clip(pk[:, None] + off[None, :], 0, len(morph) - 1)
    seg = morph[idx]
    amp = seg.max(1) - seg.min(1)
    zs = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-12)
    t = pk / fs
    lo = np.searchsorted(t, t - P['sqi_template_half'])
    hi = np.searchsorted(t, t + P['sqi_template_half'], 'right')
    corr = np.empty(n)
    amp_ratio = np.empty(n)
    for i in range(n):
        tpl = np.median(zs[lo[i]:hi[i]], 0)
        tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
        corr[i] = np.mean(zs[i] * tpl)
        amp_ratio[i] = amp[i] / (np.median(amp[lo[i]:hi[i]]) + 1e-12)
    # amplitude consistency: tolerant (sigma = amp_sigma octaves) — pulse amplitude legitimately varies with IBI in AF
    la = np.log2(np.clip(amp_ratio, 1e-3, None))
    good = 1 - mask[idx].mean(1)
    q = np.clip(corr, 0, 1) * np.exp(-0.5 * (la / P['amp_sigma']) ** 2) * good
    # confidence uses a stricter amplitude tolerance (better calibrated); anomaly rules keep the AF-tolerant q
    q_conf = np.clip(corr, 0, 1) * np.exp(-0.5 * (la / P['conf_amp_sigma']) ** 2) * good
    return q, corr, q_conf


def confidence_series(q, pk, n, fs, mask):
    if len(pk) == 0:
        return np.zeros(n)
    conf = np.interp(np.arange(n), pk, q)
    # decay confidence inside implausibly long gaps between detections
    gaps = np.diff(pk)
    for s, g in zip(pk[:-1], gaps):
        if g > P['ibi_max'] * fs:
            conf[s:s + g] = 0.0
    conf[:pk[0]] = q[0] if pk[0] < P['ibi_max'] * fs else 0.0
    conf[pk[-1]:] = q[-1] if n - pk[-1] < P['ibi_max'] * fs else 0.0
    conf = uniform_filter1d(conf, max(1, int(P['sqi_smooth'] * fs)))
    conf[mask] = 0.0
    return np.clip(conf, 0, 1)


# --------------------------------------------------------------------------------------
# 8. Anomaly detection
# --------------------------------------------------------------------------------------
def turning_point_ok(v):
    """Turning point ratio in the band expected for a random series (Dash et al. 2009)."""
    m = len(v)
    if m < 5:
        return False
    tp = np.sum((v[1:-1] - v[:-2]) * (v[1:-1] - v[2:]) > 0)
    mu, sd = (2 * m - 4) / 3.0, np.sqrt((16 * m - 29) / 90.0)
    return abs(tp - mu) <= 2 * sd


def anomaly_rules(tk, q, corr, fs, mode):
    """Per-beat anomaly labels. tk: sub-sample peak positions. Returns dict of boolean arrays
    (index k refers to the IBI ending at beat k) and the reference IBI."""
    n = len(tk)
    ibi = np.diff(tk) / fs                                   # ibi[k-1] ends at beat k
    out = {r: np.zeros(n, bool) for r in ('pause', 'brady', 'premature', 'morph', 'af')}
    if n < 4:
        return out, ibi
    good = (q[1:] >= P['quality_gate']) & (q[:-1] >= P['quality_gate'])
    valid = good & (ibi >= P['ibi_min']) & (ibi <= P['ibi_max'])
    ref = np.empty(len(ibi))
    hist = list(ibi[valid][:P['ref_beats']]) or [np.median(ibi)]
    for j in range(len(ibi)):
        ref[j] = np.median(hist[-P['ref_beats']:])
        if valid[j]:
            hist.append(ibi[j])
    nxt = np.append(ibi[1:], np.nan)
    strong = (q[1:] >= P['event_qmin']) & (q[:-1] >= P['event_qmin'])   # single-beat events need clean beats
    pause = valid & strong & ((ibi > P['pause_ratio'] * ref) | (ibi > P['pause_abs']))
    prem = valid & strong & (ibi < P['premature_ratio'] * ref) & (ibi + nxt > P['compensatory_ratio'] * ref)
    med5 = pd.Series(ibi).where(valid).rolling(5, center=True, min_periods=3).median().to_numpy()
    brady = valid & (med5 > 60.0 / P['brady_bpm'])
    # AF-type irregularity: robust successive-difference statistic + turning points
    af = np.zeros(len(ibi), bool)
    clean = valid                       # median |dIBI| is already robust to isolated ectopics
    w = P['af_win']
    for j in range(0, len(ibi) - w + 1, max(1, w // 4)):
        sl = slice(j, j + w)
        v = ibi[sl][clean[sl]]
        if len(v) < w * 0.75 or np.median(q[1:][sl]) < P['af_qmin']:   # irregularity only trusted on clean beats
            continue
        masd = np.median(np.abs(np.diff(v))) / np.median(v)
        dv = np.diff(v)
        sd1 = np.std(dv) / np.sqrt(2)
        sd2 = np.sqrt(max(2 * np.var(v) - sd1 ** 2, 1e-12))
        tpr, sdr = turning_point_ok(v), sd1 / sd2 > P['af_sd_ratio']   # Poincare SD1/SD2
        vote = {'tpr': tpr, 'either': tpr or sdr, 'both': tpr and sdr, 'sd': sdr}[P['af_vote']]
        nrmssd = np.sqrt(np.mean(dv ** 2)) / np.mean(v)
        if masd > P['af_masd'] and nrmssd > P['af_nrmssd'] and vote:
            af[sl] |= clean[sl]
    out['pause'][1:] = pause
    out['brady'][1:] = brady
    out['premature'][1:] = prem
    out['af'][1:] = af
    if mode == 'ecg':   # ectopic morphology with good neighbours and early timing
        nb = np.minimum(np.roll(corr, 1), np.roll(corr, -1))
        early = np.ones(n, bool)
        early[1:] = ibi < 0.92 * ref
        out['morph'] = (corr < P['morph_corr']) & (nb > 0.9) & early
        out['morph'][[0, -1]] = False
    return out, ibi


def anomaly_intervals(pk, rules, n, fs):
    iv = []
    for r, lab in rules.items():
        for k in np.flatnonzero(lab):
            s = pk[k - 1] if k > 0 else pk[k]
            e = pk[k + 1] if (r in ('premature', 'morph') and k + 1 < len(pk)) else pk[k]
            iv.append((s, e))
    return merge_intervals(iv, P['flag_dilate'] * fs, P['flag_close'] * fs, n)


RULE_TEXT = {'pause': 'pause / dropped beat (IBI > 1.5 x reference or > 2 s)',
             'brady': 'bradycardia (5-beat median HR < 45 bpm)',
             'premature': 'premature beat with compensatory pause (PVC / APB signature)',
             'morph': 'ectopic QRS morphology (template correlation < 0.8)',
             'af': 'AF-type irregularity (median |dIBI|, nRMSSD, turning points / Poincare SD1/SD2)'}


def anomaly_events(pk, rules, fs, t0, ibi):
    """Explainable episode list: consecutive flagged beats of one rule (gap < flag_close) form an episode."""
    ev = []
    for r, lab in rules.items():
        ks = np.flatnonzero(lab)
        if not len(ks):
            continue
        start = prev = ks[0]
        for k in list(ks[1:]) + [None]:
            if k is not None and (pk[k] - pk[prev]) / fs <= P['flag_close'] + 3.0:
                prev = k
                continue
            a = pk[start - 1] if start > 0 else pk[start]
            b = pk[min(prev + 1, len(pk) - 1)] if r in ('premature', 'morph') else pk[prev]
            seg = ibi[max(start - 1, 0):prev] * 1000
            ev.append(dict(rule=r, reason=RULE_TEXT[r], t_start=round(t0 + a / fs, 3), t_end=round(t0 + b / fs, 3),
                           beats=int(prev - start + 1),
                           median_ibi_ms=round(float(np.median(seg)), 1) if len(seg) else None))
            if k is not None:
                start = prev = k
    return sorted(ev, key=lambda e: e['t_start'])


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def template_refine(morph, pk, tk, fs, mode):
    """Sub-sample peak refinement by cross-correlation with the local median beat template (+/- 5 s).
    The whole beat shape votes for the timing, so a motion-distorted apex moves less than with peak picking.
    Shift limited to +/- tmpl_lag; parabolic interpolation of the correlation curve."""
    n = len(pk)
    if n < 5:
        return tk
    if mode == 'ecg':
        a, b = int(0.08 * fs), int(0.12 * fs)
    else:
        med = np.median(np.diff(pk))
        a, b = int(P['tmpl_pre'] * fs), int(P['tmpl_post'] * fs)   # systolic upstroke + apex
    L = max(1, int(P['tmpl_lag'] * fs))
    off = np.arange(-a, b)
    lags = np.arange(-L, L + 1)
    idx0 = np.clip(pk[:, None] + off[None, :], 0, len(morph) - 1)
    seg = morph[idx0]
    zs = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-12)
    t = pk / fs
    lo = np.searchsorted(t, t - P['sqi_template_half'])
    hi = np.searchsorted(t, t + P['sqi_template_half'], 'right')
    tpl = np.empty_like(zs)
    for i in range(n):
        m = np.median(zs[lo[i]:hi[i]], 0)
        tpl[i] = (m - m.mean()) / (m.std() + 1e-12)
    cc = np.empty((n, len(lags)))
    for j, lg in enumerate(lags):
        sg = morph[np.clip(idx0 + lg, 0, len(morph) - 1)]
        sg = (sg - sg.mean(1, keepdims=True)) / (sg.std(1, keepdims=True) + 1e-12)
        cc[:, j] = (sg * tpl).mean(1)
    k = np.argmax(cc, 1)
    kk = np.clip(k, 1, len(lags) - 2)
    y0, y1, y2 = cc[np.arange(n), kk - 1], cc[np.arange(n), kk], cc[np.arange(n), kk + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    shift = lags[kk] + np.clip(d, -0.5, 0.5)
    ok = (k > 0) & (k < len(lags) - 1) & (cc[np.arange(n), k] > P['tmpl_min_corr'])
    # template alignment gives the offset of the beat relative to the template; the template's own apex sits at
    # the median peak position, so the refined time is pk + shift (keep the parabolic estimate where unreliable)
    out = tk.copy()
    out[ok] = pk[ok] + shift[ok] + (tk[ok] - pk[ok]).mean() * 0
    return out


def block_sqis(morph, removed, fs, mode, n):
    """Block SQIs per 2-s block: shape (ECG kurtosis / PPG skewness, logistic-mapped) and motion
    (fraction of power that was not removed as artefact). Returns per-sample arrays."""
    from scipy import stats
    L = max(8, int(P['sqi_block'] * fs))
    m = max(1, n // L)
    C = morph[:m * L].reshape(m, L)
    R = removed[:m * L].reshape(m, L)
    if mode == 'ecg':
        v = stats.kurtosis(C, axis=1, fisher=False)
        shape = 1 / (1 + np.exp(-(v - 5.0) / 1.0))
    else:
        v = stats.skew(C, axis=1)
        shape = 1 / (1 + np.exp(-v / 0.15))
    vr, vc = R.var(1), C.var(1)
    motion = 1 - vr / (vr + vc + 1e-20)
    tc = (np.arange(m) + 0.5) * L
    xi = np.arange(n)
    return np.interp(xi, tc, shape), np.interp(xi, tc, motion)


def purity_series(z, fs, n):
    """Spectral purity per frame (share of 0.5-8 Hz power within 0.15 Hz of the tracked cardiac harmonics),
    normalised so that 0.6 -> 1, interpolated to samples."""
    q = max(1, int(fs // P['ms_fs']))
    zd = signal.decimate(z, q, ftype='fir', zero_phase=True) if q > 1 else z
    fsd = fs / q
    nper, hop = int(P['ms_win'] * fsd), int(P['ms_hop'] * fsd)
    f, _, Z = signal.stft(zd - zd.mean(), fsd, window='hann', nperseg=nper, noverlap=nper - hop, boundary='even')
    S = np.abs(Z)
    f0 = track_hr(S, f)
    S2 = S ** 2
    inb = (f >= 0.5) & (f <= 8.0)
    near = np.abs(f[:, None, None] - np.arange(1, 6)[None, :, None] * f0[None, None, :]).min(1) <= 0.15
    pur = (S2 * (near & inb[:, None])).sum(0) / ((S2 * inb[:, None]).sum(0) + 1e-12)
    return np.interp(np.arange(n), np.arange(len(pur)) * hop * q, np.clip(pur / 0.6, 0, 1))


def run(x, fs, mode='auto', name='', return_debug=False, acc=None, valid=None):
    n = len(x)
    x, mask = sanitize(x, fs)
    if valid is not None:
        mask |= ~valid
    if mode == 'auto':
        mode = detect_mode(x, fs, name)
    if mode == 'ecg':
        if fs > 110:
            x = notch_line_noise(x, fs)
        pk, tk, morph, _ = detect_ecg(x, fs)
        removed = highpass(x, fs, P['ecg_hp']) - bandpass(x, fs, P['ecg_hp'], P['ecg_lp'])
        pur_src = bandpass(np.abs(morph), fs, 0.5, 8.0)
    else:
        pk, tk, morph, raw_bp = detect_ppg(x, fs, acc)
        removed = raw_bp - morph
        pur_src = morph
    if valid is not None and len(pk):          # no beats inside invalid gaps
        keep = valid[pk]
        pk, tk = pk[keep], tk[keep]
    if P['tmpl_refine_' + mode]:
        tk = template_refine(morph, pk, tk, fs, mode)
    q, corr, q_conf = beat_quality(morph, pk, fs, mode, mask)
    rules, ibi = anomaly_rules(tk, q, corr, fs, mode)
    if P['pause_gap_check'] and len(pk) > 2:
        # a "pause" whose gap contains a pulse-sized peak is a missed beat, not a real pause
        ref_h = np.median(morph[pk])
        refr = int(P['ecg_refractory' if mode == 'ecg' else 'ppg_refractory'] * fs)
        for k in np.flatnonzero(rules['pause']):
            a, b = pk[k - 1] + refr, pk[k] - refr
            if b > a and morph[a:b].max() > P['pause_gap_frac'] * ref_h:
                rules['pause'][k] = False
    conf = confidence_series(q_conf, pk, n, fs, mask)
    if P['conf_fusion']:
        # fused confidence: beat quality + block / spectral SQIs
        shape, motion = block_sqis(morph, removed, fs, mode, n)
        pur = purity_series(pur_src, fs, n)
        w = P['conf_w']
        fused = (w[0] * conf + w[1] * pur + w[2] * shape + w[3] * motion) / sum(w)
        fused = uniform_filter1d(fused, max(1, int(fs)))
        fused[conf == 0] = 0.0                 # keep hard zeros (no beats / invalid / clipped)
        conf = np.clip(fused, 0, 1)
    iv = anomaly_intervals(pk, rules, n, fs)
    is_peak = np.zeros(n, np.int8)
    is_peak[pk] = 1
    ibi_ms = np.zeros(n)
    if len(pk) > 1:
        ibi_ms[pk[1:]] = ibi * 1000.0
        # forward-hold the latest IBI until the next beat
        holder = np.zeros(n)
        holder[pk[1:]] = 1
        idx = np.maximum.accumulate(np.where(holder > 0, np.arange(n), 0))
        ibi_ms = np.where(idx >= pk[1], ibi_ms[idx], 0.0)
    ano = np.zeros(n, np.int8)
    for s, e in iv:
        ano[s:e + 1] = 1
    ano[mask] = 0
    res = dict(is_peak=is_peak, ibi_ms=ibi_ms, ano=ano, conf=conf, mode=mode, pk=pk, tk=tk, rules=rules, ibi=ibi)
    if return_debug:
        res.update(q=q, corr=corr, morph=morph, intervals=iv)
    return res


def read_annotation(path):
    """WFDB rdann text (Time, Sample #, Type, ..., Aux) or one beat time (s) per line.
    Returns beat times (s), beat types, rhythm changes [(t, label)], MISSB / x event times."""
    beats, types, rh, ev = [], [], [], []
    lines = open(path).read().splitlines()
    if lines and re.match(r'^\s*[\d.]+\s*$', lines[0]):
        return np.array([float(v) for v in lines if v.strip()]), None, [], []
    for L in lines[1:]:
        head, _, aux = L.partition('\t')
        p = head.split()
        if len(p) < 3:
            continue
        mm, _, ss = p[0].rpartition(':')
        t = (float(mm.split(':')[-1]) * 60 + (float(mm.split(':')[0]) * 3600 if mm.count(':') else 0) if mm else 0) + float(ss)
        ty, aux = p[2], aux.strip()
        if ty in 'NLRBAaJSVrFejnE/fQ?':
            beats.append(t)
            types.append(ty)
        elif ty == '+' and aux.startswith('('):
            rh.append((t, aux))
        elif (ty == '"' and 'MISSB' in aux) or ty == 'x':
            ev.append(t)
    return np.array(beats), np.array(types), rh, ev


def validate(out, ann_path, mode):
    """Beat Se/PPV/F1 (+-150 ms; PPG peaks are lag-compensated against ECG R annotations), IBI MAE and
    sample-level anomaly Se/PPV/F1 (GT = non-sinus rhythm + spans around ectopic beats and dropped beats)."""
    tb, ty, rh, ev = read_annotation(ann_path)
    ts = out['Timestamp'].to_numpy(float)
    det = ts[out['Is_peak'].to_numpy() == 1]
    lag = 0.0
    if mode == 'ppg' and len(det) and len(tb):
        j = np.clip(np.searchsorted(tb, det), 1, len(tb) - 1)
        off = det - np.where(np.abs(det - tb[j - 1]) < np.abs(det - tb[j]), tb[j - 1], tb[j])
        lag = float(np.median(off[np.abs(off) < 0.5])) if np.any(np.abs(off) < 0.5) else 0.0
    d = det - lag
    k = np.clip(np.searchsorted(d, tb), 1, max(len(d) - 1, 1))
    near = np.where(np.abs(d[k - 1] - tb) < np.abs(d[k] - tb), k - 1, k)
    ok = np.abs(d[near] - tb) <= 0.15
    used = np.zeros(len(d), bool)
    used[near[ok]] = True
    tp = int(ok.sum())
    se, ppv = tp / max(len(tb), 1), tp / max(len(d), 1)
    pair = ok[1:] & ok[:-1] & (near[1:] - near[:-1] == 1)
    ibi_err = np.abs(np.diff(tb)[pair] - (d[near[1:][pair]] - d[near[:-1][pair]])) * 1000
    m = dict(lag_ms=round(lag * 1000, 1), beats_ref=int(len(tb)), beats_det=int(len(d)), Se=se, PPV=ppv,
             F1=2 * se * ppv / max(se + ppv, 1e-12), ibi_mae_ms=float(ibi_err.mean()) if len(ibi_err) else None)
    if ty is not None:
        g = np.zeros(len(ts), bool)
        bounds = [t for t, _ in rh] + [ts[-1] + 1]
        for (a, lab), b in zip(rh, bounds[1:]):
            if lab not in ('(N', '(SBR'):
                g |= (ts >= a) & (ts <= b)
        for i, x in enumerate(ty):
            if x not in 'NLRej':
                g |= (ts >= tb[max(i - 1, 0)]) & (ts <= tb[min(i + 1, len(tb) - 1)])
        for t in ev:
            i = np.searchsorted(tb, t)
            if 0 < i < len(tb):
                g |= (ts >= tb[i - 1]) & (ts <= tb[i])
        a = out['ano_flag'].to_numpy().astype(bool)
        tp2, fp2, fn2 = (a & g).sum(), (a & ~g).sum(), (~a & g).sum()
        ase, appv = tp2 / max(tp2 + fn2, 1), tp2 / max(tp2 + fp2, 1)
        m.update(ano_Se=ase, ano_PPV=appv, ano_F1=2 * ase * appv / max(ase + appv, 1e-12))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', '-i', default='ecg.csv')
    ap.add_argument('--output', '-o', default='predictions.csv')
    ap.add_argument('--fs', type=float, default=None, help='sampling rate (Hz) if no time column')
    ap.add_argument('--mode', choices=('auto', 'ecg', 'ppg'), default='auto')
    ap.add_argument('--channel', default=None, help='signal column to analyse')
    ap.add_argument('--annotation', default=None, help='reference annotations (WFDB rdann text) -> print metrics')
    ap.add_argument('--events', default=None, help='write explainable anomaly episodes (JSON) to this path')
    ap.add_argument('--ibi-fill', choices=('hold', 'zero'), default='hold',
                    help='Ibi_ms on non-peak rows: hold the latest IBI or write 0')
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rec = load_record(a.input, a.fs, a.channel)
    fs = rec['fs']
    r = run(rec['x'], fs, a.mode, rec['name'], acc=rec['acc'], valid=rec['valid'])
    out = to_rows(rec, r, a.ibi_fill)
    out.to_csv(a.output, index=False)
    if a.events:
        import json
        ev = anomaly_events(r['pk'], r['rules'], fs, rec['t0'], r['ibi'])
        json.dump(dict(input=a.input, channel=rec['channel'], mode=r['mode'], fs=fs, episodes=ev), open(a.events, 'w'),
                  indent=1)
    dt = time.perf_counter() - t0
    print('channel=%s fs=%.2f mode=%s resampled=%s acc=%s beats=%d median_HR=%.1f bpm anomalous=%.1f%% '
          'mean_conf=%.2f time=%.2fs -> %s'
          % (rec['channel'], fs, r['mode'], rec['resampled'], rec['acc'] is not None, int(out['Is_peak'].sum()),
             60.0 / np.median(np.diff(r['tk']) / fs) if len(r['tk']) > 1 else 0,
             100 * out['ano_flag'].mean(), out['confidence_score'].mean(), dt, a.output), file=sys.stderr)
    if a.annotation:
        import json
        print(json.dumps(validate(out, a.annotation, r['mode'])), file=sys.stderr)


def to_rows(rec, r, ibi_fill='hold'):
    """Map uniform-grid results back to one output row per input sample."""
    t = rec['t']
    n = len(t)
    gi = rec['grid_index']
    is_peak = np.zeros(n, np.int8)
    ibi_col = np.zeros(n)
    pk, tk = r['pk'], r['tk']
    if len(pk):
        tp = rec['t0'] + pk / rec['fs']
        order = np.argsort(t, kind='stable')
        ts = t[order]
        k = np.clip(np.searchsorted(ts, tp), 1, n - 1)
        rows = order[np.where(np.abs(ts[k] - tp) < np.abs(ts[k - 1] - tp), k, k - 1)]
        is_peak[rows] = 1
        vals = np.r_[0.0, np.diff(tk) / rec['fs'] * 1000.0]
        ibi_col[rows] = vals
        if ibi_fill == 'hold':
            pos = np.zeros(n, bool)
            pos[rows] = True
            src = np.maximum.accumulate(np.where(pos, np.arange(n), 0))
            ibi_col = np.where(np.arange(n) >= rows.min(), ibi_col[src], 0.0)
    return pd.DataFrame({'Timestamp': np.round(t, 6), 'Is_peak': is_peak, 'Ibi_ms': np.round(ibi_col, 2),
                         'ano_flag': r['ano'][gi], 'confidence_score': np.round(r['conf'][gi], 3)})


if __name__ == '__main__':
    main()
