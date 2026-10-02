"""Build stress-test datasets with known ground truth.
ECG: clean record + real NSTDB noise (em/ma/bw) at several SNRs (nst protocol: S = pp_QRS^2 / 8).
PPG: synthetic pulse wave driven by the annotated beat times (+PTT), with drift + in-band periodic motion.
Outputs dev/data/<name>.csv (+ truth in dev/data/<name>.npz)."""
import sys, os
sys.path.insert(0, 'dev')
import numpy as np, pandas as pd
from scipy import signal
from ann import load
FS = 360; rng = np.random.default_rng(0)
os.makedirs('dev/data', exist_ok=True)

def read212(path):
    raw = np.fromfile(path, np.uint8).reshape(-1, 3).astype(np.int32)
    a = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    b = raw[:, 2] | ((raw[:, 1] & 0xF0) << 4)
    a[a > 2047] -= 4096; b[b > 2047] -= 4096
    return np.stack([a, b], 1).reshape(-1, 2).astype(float)

d = pd.read_csv('ecg.csv'); x = d.iloc[:, 1].to_numpy(float); n = len(x)
beats, types, rh, notes = load()
# signal power per nst: median peak-to-peak QRS amplitude
hp = signal.sosfiltfilt(signal.butter(2, 0.5, 'hp', fs=FS, output='sos'), x)
w = int(0.1 * FS); pp = np.median([np.ptp(hp[b - w:b + w]) for b in beats[1:-1]])
S = pp ** 2 / 8
noise = {k: read212('dev/nstdb/%s.dat' % k)[:n, 0] for k in ('em', 'ma', 'bw')}
def save(name, sig, truth_beats, col='MLII', extra=None):
    pd.DataFrame({'sample #': np.arange(len(sig)), col: np.round(sig, 3)}).to_csv('dev/data/%s.csv' % name, index=False)
    np.savez('dev/data/%s.npz' % name, beats=truth_beats, **(extra or {}))
for k in ('em', 'ma', 'bw'):
    nz = noise[k] - noise[k].mean()
    for snr in (12, 6, 0):
        g = np.sqrt(S / (10 ** (snr / 10)) / nz.var())
        save('ecg_%s_%ddB' % (k, snr), x + g * nz, beats)
# worst case: em+ma+bw mixed, each at 6 dB, in 2-min bursts after the first 5 min (nst protocol)
mix = sum(np.sqrt(S / (10 ** 0.6) / noise[k].var()) * (noise[k] - noise[k].mean()) for k in noise)
env = np.zeros(n); t = np.arange(n) / FS
env[(t >= 300) & (((t - 300) // 120) % 2 == 0)] = 1
env = np.convolve(env, np.ones(FS) / FS, 'same')
save('ecg_mix_6dB_bursts', x + env * mix, beats)

# ---------------- synthetic PPG ----------------
def synth_ppg(fs, motion_amp=0.0, drift_amp=0.0, seed=1):
    r = np.random.default_rng(seed)
    N = int(n / FS * fs); tt = np.arange(N) / fs
    tb = beats / FS
    ibi_prev = np.diff(tb, prepend=tb[0] - 1.0)
    med = np.median(ibi_prev)
    ptt = 0.22 + 0.01 * r.standard_normal(len(tb))            # pulse transit time
    amp = np.clip(ibi_prev / med, 0.5, 1.6) ** 0.6            # Frank-Starling: longer filling -> bigger pulse
    amp[np.isin(types, list('VA'))] *= 0.55                   # ectopic beats have weak pulses
    y = np.zeros(N)
    # pulse template: fast systolic rise (gaussian) + dicrotic/diastolic wave
    def pulse(t0, a, width):
        return a * (np.exp(-0.5 * ((tt_seg - t0) / (0.09 * width)) ** 2)
                    + 0.35 * np.exp(-0.5 * ((tt_seg - t0 - 0.28 * width) / (0.12 * width)) ** 2))
    sys_t = tb + ptt
    for k, (t0, a) in enumerate(zip(sys_t, amp)):
        i0, i1 = int((t0 - 0.4) * fs), int((t0 + 0.9) * fs)
        i0 = max(i0, 0); i1 = min(i1, N)
        if i1 <= i0: continue
        tt_seg = tt[i0:i1]
        width = min(1.0, max(0.6, ibi_prev[k]))
        y[i0:i1] += pulse(t0, a, width)
    # systolic peak location of the composite wave (search near t0)
    truth = []
    for t0 in sys_t:
        i0, i1 = int((t0 - 0.1) * fs), int((t0 + 0.1) * fs)
        if 0 < i0 and i1 < N: truth.append(i0 + np.argmax(y[i0:i1]))
    truth = np.array(truth)
    y += 0.08 * np.sin(2 * np.pi * 0.25 * tt)                   # respiration
    if drift_amp:   # perspiration / coupling drift: slow random walk + step changes
        rw = np.cumsum(r.standard_normal(N)) ; rw = signal.sosfiltfilt(signal.butter(2, 0.05, fs=fs, output='sos'), rw)
        y += drift_amp * rw / rw.std()
    if motion_amp:  # rhythmic arm motion in the cardiac band, in bursts, frequency wandering 0.8-2 Hz
        f_m = 1.2 + 0.4 * np.sin(2 * np.pi * tt / 90)
        ph = 2 * np.pi * np.cumsum(f_m) / fs
        burst = (np.sin(2 * np.pi * tt / 60 + 1) > 0.0).astype(float)
        burst = np.convolve(burst, np.ones(int(2 * fs)) / (2 * fs), 'same')
        mot = (np.sin(ph) + 0.5 * np.sin(2 * ph + 0.7)) * burst
        imp = (r.random(N) < 0.3 / fs) * r.standard_normal(N) * 3   # occasional jolts
        imp = np.convolve(imp, np.hanning(int(0.3 * fs)), 'same')
        y += motion_amp * (mot + imp)
    y += 0.02 * r.standard_normal(N)
    return y, truth, sys_t
for fs in (360, 125, 64):
    for tag, m, dft in (('clean', 0, 0), ('drift', 0, 1.0), ('motion05', 0.5, 1.0), ('motion1', 1.0, 1.0)):
        y, truth, st = synth_ppg(fs, m, dft)
        name = 'ppg%d_%s' % (fs, tag)
        pd.DataFrame({'sample #': np.arange(len(y)), 'PPG': np.round(y, 5)}).to_csv('dev/data/%s.csv' % name, index=False)
        np.savez('dev/data/%s.npz' % name, beats=truth, fs=fs)
print('done', sorted(os.listdir('dev/data')))
