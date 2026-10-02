#!/usr/bin/env python3
"""Synthetic motion-corrupted wrist PPG with known ground truth.

The provided ecg.csv contains no PPG, so the PPG branch of the pipeline is
validated on signals built here from a documented signal model

    ppg(t) = pulse(t; beats) * (1 + resp AM) + baseline(t) + motion(t) + noise

  beats    sinus rhythm (HR drift + respiratory sinus arrhythmia + jitter) with
           PVCs (premature 60-70 % RR, full compensatory pause, 40 % pulse
           amplitude), dropped beats (2 x RR) and an atrial-fibrillation
           episode (i.i.d. gamma IBIs, CV 0.22)
  pulse    systolic + diastolic Gaussian pair, pulse-transit time 0.25 s after
           the ECG R wave
  baseline 0.05-0.25 Hz sinusoids + low-passed random walk (perspiration drift)
  motion   arm-swing source with slowly varying frequency 0.9-2.2 Hz (inside
           the cardiac band) + 2nd/3rd harmonics, in bursts; enters the PPG
           through a random short FIR; the 3-axis accelerometer sees the same
           source through different gains + sensor noise
  sampling 64 Hz nominal with +-2 % timestamp jitter, NaN runs and a dropped
           block of rows (tests resampling / gap handling)

Output: CSV (Timestamp, ACC_X, PPG, ACC_Y, ACC_Z, ECG) and a WFDB-style text
annotation of the ECG R peaks + rhythm markers, readable by pipeline.py.
This is a TEST FIXTURE, not training data: nothing in the pipeline is fitted to it.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import signal


def make_rhythm(duration, rng, events=True):
    t, beats, types = 1.0, [], []
    hr0 = 68.0
    af = (0.45 * duration, 0.45 * duration + 60.0) if events else (-1, -1)
    pvc_every = 55.0
    next_pvc = 20.0
    drop_times = [0.25 * duration, 0.8 * duration] if events else []
    rhythm = [(0.0, "(N")]
    while t < duration - 1.5:
        hr = hr0 + 6 * np.sin(2 * np.pi * t / 240.0)
        rr = 60.0 / hr * (1 + 0.03 * np.sin(2 * np.pi * 0.25 * t)) \
            * (1 + 0.01 * rng.standard_normal())
        if af[0] <= t < af[1]:
            if not rhythm or rhythm[-1][1] != "(AFIB":
                rhythm.append((t, "(AFIB"))
            rr = rng.gamma(1 / 0.22 ** 2, 0.72 * 0.22 ** 2)
            rr = float(np.clip(rr, 0.35, 1.6))
            beats.append(t); types.append("N"); t += rr
            continue
        if rhythm[-1][1] == "(AFIB":
            rhythm.append((t, "(N"))
        if events and t >= next_pvc:
            beats.append(t); types.append("N")
            short = rr * rng.uniform(0.6, 0.7)
            t += short
            beats.append(t); types.append("V")
            t += 2 * rr - short
            next_pvc += pvc_every * rng.uniform(0.7, 1.3)
            continue
        if drop_times and t >= drop_times[0]:
            drop_times.pop(0)
            beats.append(t); types.append("N")
            t += rr
            types.append("DROP")           # marker: missing beat at t
            beats.append(t)
            t += rr
            continue
        beats.append(t); types.append("N")
        t += rr
    return np.array(beats), np.array(types), rhythm


def synth(duration=600.0, fs=64.0, motion_snr_db=0.0, seed=1, acc=True,
          events=True, motion_frac=0.5, jitter=0.02, gaps=True):
    rng = np.random.default_rng(seed)
    beats, types, rhythm = make_rhythm(duration, rng, events)
    fs_hi = 256.0
    th = np.arange(0, duration, 1 / fs_hi)
    real = types != "DROP"
    bt, bty = beats[real], types[real]
    ptt = 0.25
    pulse = np.zeros_like(th)
    for tb, ty in zip(bt, bty):
        a = 0.4 if ty == "V" else 1.0
        c = tb + ptt
        m = (th > c - 0.6) & (th < c + 0.9)
        tt = th[m] - c
        pulse[m] += a * (np.exp(-0.5 * (tt / 0.07) ** 2)
                         + 0.4 * np.exp(-0.5 * ((tt - 0.28) / 0.11) ** 2))
    pulse *= 1 + 0.1 * np.sin(2 * np.pi * 0.25 * th)
    # ECG reference: narrow QRS-like spikes at R times
    ecg = np.zeros_like(th)
    for tb, ty in zip(bt, bty):
        m = np.abs(th - tb) < 0.1
        w = 0.025 if ty == "V" else 0.012
        ecg[m] += np.exp(-0.5 * ((th[m] - tb) / w) ** 2)
    base = sum(rng.uniform(0.3, 0.8) * np.sin(2 * np.pi * f * th + rng.uniform(0, 6))
               for f in (0.05, 0.11, 0.23))
    rw = np.cumsum(rng.standard_normal(len(th))) / np.sqrt(fs_hi)
    rw = signal.sosfiltfilt(signal.butter(2, 0.05, fs=fs_hi, output="sos"), rw)
    base = base + 0.5 * rw / (rw.std() + 1e-12)
    # motion source
    fm = 1.55 + 0.65 * np.sin(2 * np.pi * th / 170.0 + 1.0)
    phase = 2 * np.pi * np.cumsum(fm) / fs_hi
    src = np.sin(phase) + 0.5 * np.sin(2 * phase + 0.7) + 0.25 * np.sin(3 * phase + 2)
    burst = np.zeros_like(th)
    t_on = 30.0
    while t_on < duration:
        dur = rng.uniform(20, 50)
        burst[(th >= t_on) & (th < t_on + dur)] = 1.0
        t_on += dur / motion_frac
    burst = signal.sosfiltfilt(signal.butter(2, 0.5, fs=fs_hi, output="sos"), burst)
    src = src * np.clip(burst, 0, 1)
    fir = rng.uniform(0.2, 1.0, 5) * np.hanning(7)[1:-1]
    mot = np.convolve(src, fir / fir.sum(), mode="same")
    on = burst > 0.5
    p_sig = np.var(pulse)
    p_mot = np.var(mot[on]) if on.any() else 1.0
    g = np.sqrt(p_sig / (p_mot * 10 ** (motion_snr_db / 10))) if np.isfinite(
        motion_snr_db) else 0.0
    ppg = pulse + base + g * mot + 0.02 * rng.standard_normal(len(th))
    accs = [gain * src + 0.05 * rng.standard_normal(len(th))
            for gain in (1.0, 0.6, -0.4)]
    # non-uniform sampling
    n = int(duration * fs)
    tn = np.arange(n) / fs
    tn = tn + rng.uniform(-jitter, jitter, n) / fs
    tn = np.sort(np.clip(tn, 0, duration - 1 / fs_hi))
    keep = np.ones(n, bool)
    cols = {}
    def s(x):
        return np.interp(tn, th, x)
    cols["Timestamp"] = tn
    if acc:
        cols["ACC_X"] = s(accs[0])
    cols["PPG"] = s(ppg) * 1000 + 20000     # arbitrary ADC-like units
    if acc:
        cols["ACC_Y"] = s(accs[1])
        cols["ACC_Z"] = s(accs[2])
    cols["ECG"] = s(ecg)
    if gaps:
        i0 = int(0.33 * n)
        cols["PPG"][i0:i0 + int(0.4 * fs)] = np.nan          # sensor dropout
        j0 = int(0.66 * n)
        keep[j0:j0 + int(1.5 * fs)] = False                  # lost packets
    cols = {k: v[keep] for k, v in cols.items()}
    truth = {"beats": beats, "types": types, "rhythm": rhythm, "ptt": ptt,
             "t_hi": th, "pulse": pulse, "motion": g * mot, "burst": on}
    return cols, truth


def write_wfdb_ann(path, truth, fs_ann=250.0):
    rows = []
    for t, a in truth["rhythm"]:
        rows.append((t, "+", a))
    for t, ty in zip(truth["beats"], truth["types"]):
        if ty == "DROP":
            rows.append((t, '"', "MISSB"))
        else:
            rows.append((t, ty, ""))
    rows.sort(key=lambda r: r[0])
    with open(path, "w") as fh:
        fh.write("      Time   Sample #  Type  Sub Chan  Num\tAux\n")
        for t, ty, aux in rows:
            m, sec = divmod(t, 60)
            fh.write(f"{int(m):>7d}:{sec:06.3f} {int(round(t * fs_ann)):>9d} "
                     f"{ty:>5s} {0:>4d} {0:>4d} {0:>4d}\t{aux}\n")


def write_csv(path, cols):
    import pandas as pd
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="dev/fixtures")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--fs", type=float, default=64.0)
    ap.add_argument("--snr", type=float, default=0.0, help="motion SNR (dB)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for acc in (True, False):
        cols, truth = synth(a.duration, a.fs, a.snr, a.seed, acc=acc)
        tag = "acc" if acc else "noacc"
        write_csv(out / f"ppg_{tag}.csv", cols)
        write_wfdb_ann(out / f"ppg_{tag}_ann.txt", truth)
        print("wrote", out / f"ppg_{tag}.csv", len(cols["Timestamp"]), "rows")


if __name__ == "__main__":
    main()
