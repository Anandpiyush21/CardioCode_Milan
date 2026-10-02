"""Synthetic PPG driven by (annotated) beat times — generic version of make_data.synth_ppg."""
import numpy as np
from scipy import signal

def synth_ppg(beats, types, n360, fs, motion_amp=0.0, drift_amp=0.0, seed=1):
    r = np.random.default_rng(seed)
    N = int(n360 / 360 * fs); tt = np.arange(N) / fs
    tb = beats / 360.0
    ibi_prev = np.diff(tb, prepend=tb[0] - 1.0)
    med = np.median(ibi_prev)
    ptt = 0.22 + 0.01 * r.standard_normal(len(tb))
    amp = np.clip(ibi_prev / med, 0.3, 1.6) ** 0.6
    amp[~np.isin(types, list('NLRej'))] *= 0.55
    y = np.zeros(N)
    sys_t = tb + ptt
    for k, (t0, a) in enumerate(zip(sys_t, amp)):
        i0, i1 = max(int((t0 - 0.4) * fs), 0), min(int((t0 + 0.9) * fs), N)
        if i1 <= i0: continue
        ts = tt[i0:i1]; w = min(1.0, max(0.5, ibi_prev[k]))
        y[i0:i1] += a * (np.exp(-0.5 * ((ts - t0) / (0.09 * w)) ** 2)
                         + 0.35 * np.exp(-0.5 * ((ts - t0 - 0.28 * w) / (0.12 * w)) ** 2))
    truth = []
    for t0 in sys_t:
        i0, i1 = int((t0 - 0.1) * fs), int((t0 + 0.1) * fs)
        if 0 < i0 and i1 < N: truth.append(i0 + np.argmax(y[i0:i1]))
    y += 0.08 * np.sin(2 * np.pi * 0.25 * tt)
    if drift_amp:
        rw = np.cumsum(r.standard_normal(N)); rw = signal.sosfiltfilt(signal.butter(2, 0.05, fs=fs, output='sos'), rw)
        y += drift_amp * rw / rw.std()
    if motion_amp:
        f_m = 1.2 + 0.4 * np.sin(2 * np.pi * tt / 90)
        ph = 2 * np.pi * np.cumsum(f_m) / fs
        burst = np.convolve((np.sin(2 * np.pi * tt / 60 + 1) > 0).astype(float), np.ones(int(2 * fs)) / (2 * fs), 'same')
        mot = (np.sin(ph) + 0.5 * np.sin(2 * ph + 0.7)) * burst
        imp = np.convolve((r.random(N) < 0.3 / fs) * r.standard_normal(N) * 3, np.hanning(int(0.3 * fs)), 'same')
        y += motion_amp * (mot + imp)
    y += 0.02 * r.standard_normal(N)
    return y, np.array(truth)
