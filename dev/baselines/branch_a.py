#!/usr/bin/env python3
"""
CARDIO-CODE — DSP-only arrhythmia detection pipeline for noisy ECG / PPG.

    python pipeline.py --input ecg.csv --output predictions.csv [--fs 360] [--mode auto|ecg|ppg] [--channel MLII]

Stages (all classical signal processing, rationale in report.pdf):
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
    default_fs=360.0,
    # ECG
    ecg_hp=0.5,                 # Hz, baseline-wander high-pass (AHA monitoring cut-off)
    ecg_qrs_band=(5.0, 15.0),   # Hz, Pan-Tompkins QRS band
    ecg_mwi=0.150,              # s, moving-window integration length
    ecg_refractory=0.25,        # s, no two QRS closer than this
    ecg_refine=0.10,            # s, +/- window for R-peak refinement
    ecg_lp=40.0,
    ecg_detector='hybrid',      # 'pt' (Pan-Tompkins), 'dp', or 'hybrid' (window-wise fusion)
    fuse_win=10.0,              # s, fusion window
    fuse_ratio=1.25,
    ecg_noise_gate=0.45,        # between-beat/at-beat envelope ratio below which a window counts as clean
    ppg_noise_gate=0.2,             # choose DP if PT's median |dIBI| is this much larger          # 'pt' (Pan-Tompkins adaptive thresholds) or 'dp'                # Hz, upper edge of the morphology band
    # PPG
    ppg_band=(0.5, 8.0),        # Hz, MSPTDfast / Elgendi pass band
    ppg_w1=0.111,               # s, Elgendi systolic-peak window
    ppg_w2=0.667,               # s, Elgendi beat window
    ppg_beta=0.02,              # Elgendi offset
    ppg_refractory=0.30,        # s
    # PPG motion suppression (short-time spectral)
    motion_suppress=True,
    ms_fs=25.0,                 # Hz, working rate of the spectral stage
    ms_win=8.0, ms_hop=1.0,     # s, STFT window / hop
    ms_protect=0.15,            # Hz, protected half-width around each cardiac harmonic
    ms_rel=0.5,
    ms_purity=0.75,             # frames with spectral purity below this are motion-corrupted                 # line counts as motion if |X| > rel * |X(f0)|
    hr_track_lambda=100.0,        # Viterbi penalty per Hz of HR change between frames
    hr_octave_cost=4.0,
    hr_jump_cost=10.0,          # cap on any HR change cost (sudden rhythm change, e.g. block ends)
    ppg_detector='dp',
    ppg_fuse_ratio=2.2,        # PPG fusion bias towards Elgendi
    dp_scope='fuse',            # 'all': DP everywhere (best, see log); 'motion': DP only in motion frames          # 'dp' (HR-guided dynamic programming) or 'elgendi'
    dp_weight=3.0,              # IBI deviation penalty per unit |log(IBI/expected)|
    dp_octave=0.3,              # extra penalty for an IBI of ~2x expected (dropped beat)
    dp_start=1.0,
    dp_min_rise=0.3,
    dp_pur_ref=0.8,             # purity at which the HR prior gets full weight
    dp_wmin=1.0,                # minimum prior weight (very clean signal)            # candidate must rise >= 30% of the local typical beat               # cost of starting a new beat chain         # cost of an HR halving/doubling (AV block onset)
    # Search-back
    searchback_factor=1.66,     # x RR average (Pan-Tompkins)
    # IBI gate
    ibi_min=0.25, ibi_max=3.0,  # s
    # Quality
    sqi_template_half=5.0,      # s, template built from beats within +/- this
    sqi_smooth=2.0,             # s, smoothing of per-sample confidence
    quality_gate=0.5,           # rules are only trusted above this beat quality
    amp_sigma=2.0,
    tmpl_refine_ecg=False,      # template cross-correlation peak refinement
    tmpl_refine_ppg=False,      # tried: worsens IBI MAE
    tmpl_lag=0.06,              # s, max shift
    tmpl_pre=0.20, tmpl_post=0.05,  # s, PPG correlation window around the apex (upstroke dominated)
    tmpl_min_corr=0.5,          # only refine beats that resemble the template              # octaves; tolerance of the amplitude-consistency term in beat quality
    # Anomaly rules
    ref_beats=8,                # beats in the running reference IBI
    pause_ratio=1.5,            # IBI > ratio * reference  -> pause / dropped beat
    pause_abs=2.0,
    pause_gap_check=False,       # veto pauses that contain an undetected pulse-sized peak
    pause_gap_frac=0.5,         # ... peak higher than this fraction of the median beat height              # s, IBI above this is always a pause
    brady_bpm=45.0,             # median HR of 5 beats below this -> bradycardia
    premature_ratio=0.80,       # IBI < ratio * reference ...
    compensatory_ratio=1.60,    # ... and IBI + next IBI > ratio * reference
    morph_corr=0.80,            # ECG ectopic morphology: corr below this with good neighbours
    af_win=16,                  # beats in the irregularity window
    af_masd=0.10,
    af_qmin=0.5,
    event_qmin=0.5,             # beat quality required for pause / premature events                # median beat quality required in an AF window               # median |dIBI| / median IBI above this -> irregular
    flag_dilate=0.5,            # s, dilate flagged spans
    flag_close=2.0,             # s, close gaps shorter than this
)


# --------------------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------------------
def load_signal(path, fs=None, channel=None):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().strip("'\"").strip() for c in df.columns]
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}
    tcol = next((c for c in cols if 'time' in low[c] or low[c] in ('t', 'sec', 'seconds')), None)
    scol = next((c for c in cols if 'sample' in low[c] or low[c] in ('index', 'idx')), None)
    n = len(df)
    if tcol is not None:
        t = pd.to_numeric(df[tcol], errors='coerce').to_numpy(float)
        dt = np.nanmedian(np.diff(t))
        if 'ms' in low[tcol] or dt > 0.5:   # timestamps in milliseconds
            t = t / 1000.0
            dt /= 1000.0
        fs = fs or 1.0 / dt
    else:
        fs = fs or P['default_fs']
        base = pd.to_numeric(df[scol], errors='coerce').to_numpy(float) if scol else np.arange(n, dtype=float)
        t = base / fs
    sig_cols = [c for c in cols if c not in (tcol, scol) and pd.api.types.is_numeric_dtype(df[c])]
    if not sig_cols:
        raise ValueError('no numeric signal column found in %s' % path)
    acc_cols = [c for c in sig_cols if any(k in low[c] for k in ('acc', 'ax', 'ay', 'az', 'imu'))]
    cand = [c for c in sig_cols if c not in acc_cols] or sig_cols
    if channel is None:
        prio = [c for c in cand if any(k in low[c] for k in ('ppg', 'pleth', 'bvp', 'pulse'))]
        prio += [c for c in cand if any(k in low[c] for k in ('mlii', 'ii', 'ecg', 'lead'))]
        channel = (prio or cand)[0]
    x = df[channel].to_numpy(float)
    return t, x, float(fs), channel, low[channel]


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
    y = base
    f0 = None
    if P['motion_suppress']:
        y, f0, moving, pur, hop = motion_suppress(base, fs, acc)
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
    return pk, tk, y, base


# --------------------------------------------------------------------------------------
# 7. Signal quality
# --------------------------------------------------------------------------------------
def beat_quality(morph, pk, fs, mode, mask):
    n = len(pk)
    if n < 3:
        return np.zeros(n), np.zeros(n), np.ones(n)
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
    q = np.clip(corr, 0, 1) * np.exp(-0.5 * (np.log2(np.clip(amp_ratio, 1e-3, None)) / P['amp_sigma']) ** 2)
    bad = mask[idx].mean(1)
    q *= 1 - bad
    return q, corr, amp_ratio


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
        if masd > P['af_masd'] and turning_point_ok(v):
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


def run(x, fs, mode='auto', name='', return_debug=False):
    n = len(x)
    x, mask = sanitize(x, fs)
    if mode == 'auto':
        mode = detect_mode(x, fs, name)
    if mode == 'ecg':
        pk, tk, morph, _ = detect_ecg(x, fs)
    else:
        pk, tk, morph, _ = detect_ppg(x, fs)
    if P['tmpl_refine_' + mode]:
        tk = template_refine(morph, pk, tk, fs, mode)
    q, corr, _ = beat_quality(morph, pk, fs, mode, mask)
    rules, ibi = anomaly_rules(tk, q, corr, fs, mode)
    if P['pause_gap_check'] and len(pk) > 2:
        # a "pause" whose gap contains a pulse-sized peak is a missed beat, not a real pause
        ref_h = np.median(morph[pk])
        refr = int(P['ecg_refractory' if mode == 'ecg' else 'ppg_refractory'] * fs)
        for k in np.flatnonzero(rules['pause']):
            a, b = pk[k - 1] + refr, pk[k] - refr
            if b > a and morph[a:b].max() > P['pause_gap_frac'] * ref_h:
                rules['pause'][k] = False
    conf = confidence_series(q, pk, n, fs, mask)
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
    res = dict(is_peak=is_peak, ibi_ms=ibi_ms, ano=ano, conf=conf, mode=mode, pk=pk, tk=tk)
    if return_debug:
        res.update(q=q, corr=corr, rules=rules, ibi=ibi, morph=morph, intervals=iv)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', '-i', default='ecg.csv')
    ap.add_argument('--output', '-o', default='predictions.csv')
    ap.add_argument('--fs', type=float, default=None, help='sampling rate (Hz) if no time column')
    ap.add_argument('--mode', choices=('auto', 'ecg', 'ppg'), default='auto')
    ap.add_argument('--channel', default=None, help='signal column to analyse')
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    t, x, fs, ch, name = load_signal(a.input, a.fs, a.channel)
    r = run(x, fs, a.mode, name)
    out = pd.DataFrame({'Timestamp': np.round(t, 6), 'Is_peak': r['is_peak'], 'Ibi_ms': np.round(r['ibi_ms'], 2),
                        'ano_flag': r['ano'], 'confidence_score': np.round(r['conf'], 3)})
    out.to_csv(a.output, index=False)
    dt = time.perf_counter() - t0
    print('channel=%s fs=%.1f mode=%s beats=%d mean_HR=%.1f bpm anomalous=%.1f%% mean_conf=%.2f time=%.2fs -> %s'
          % (ch, fs, r['mode'], r['is_peak'].sum(),
             60.0 / np.median(np.diff(r['tk']) / fs) if len(r['tk']) > 1 else 0,
             100 * r['ano'].mean(), r['conf'].mean(), dt, a.output), file=sys.stderr)


if __name__ == '__main__':
    main()
