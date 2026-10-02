"""Figures for REPORT.md -> plots/fig*.png"""
import sys, json; sys.path[:0] = ['.', 'dev']
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
import pipeline
from ann import load
P = pipeline.P
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
# ---- Fig 1: ECG stages around a normal -> 2:1 block transition (t = 90..110 s)
d = pd.read_csv('ecg.csv'); x = d.iloc[:, 1].to_numpy(float); fs = 360
b, ty, rh, notes = load()
r = pipeline.run(x, fs, 'ecg', '', return_debug=True)
t0, t1 = 88, 112; i0, i1 = t0 * fs, t1 * fs; tt = np.arange(i0, i1) / fs
hp = pipeline.highpass(x, fs, P['ecg_hp']); y = pipeline.bandpass(x, fs, *P['ecg_qrs_band'], order=3)
env = uniform_filter1d(np.gradient(y) ** 2, int(P['ecg_mwi'] * fs))
fig, ax = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
ax[0].plot(tt, x[i0:i1], lw=.6, c='gray'); ax[0].plot(tt, (x - hp)[i0:i1], c='C3', lw=1, label='estimated baseline'); ax[0].set_title('1. raw MLII + baseline wander (removed by 0.5 Hz zero-phase high-pass)'); ax[0].legend(loc='upper right')
ax[1].plot(tt, y[i0:i1], lw=.6); ax[1].set_title('2. QRS band 5-15 Hz')
ax[2].plot(tt, np.sqrt(env[i0:i1]), lw=.8); ax[2].set_title('3. Pan-Tompkins envelope (sqrt of 150 ms MWI of squared derivative)')
pk = r['pk']; s = pk[(pk >= i0) & (pk < i1)]; tr = b[(b >= i0) & (b < i1)]
ax[3].plot(tt, r['morph'][i0:i1], lw=.6); ax[3].plot(s / fs, r['morph'][s], 'o', mfc='none', c='C3', label='detected'); ax[3].plot(tr / fs, r['morph'][tr] + 60, 'v', c='g', ms=4, label='annotated')
for sn, _ in notes:
    if i0 <= sn < i1: ax[3].axvline(sn / fs, c='k', ls=':', lw=.8)
ax[3].set_title('4. R peaks (dotted = annotated dropped beat "MISSB")'); ax[3].legend(loc='upper right', ncol=2)
ibi_t = r['tk'][1:] / fs; m = (ibi_t >= t0) & (ibi_t < t1)
ax[4].step(ibi_t[m], r['ibi'][m] * 1000, where='post', c='C0', label='IBI (ms)'); ax4 = ax[4].twinx()
ax4.fill_between(tt, 0, r['ano'][i0:i1], color='C3', alpha=.2, step='mid', label='ano_flag'); ax4.plot(tt, r['conf'][i0:i1], c='C2', lw=1, label='confidence'); ax4.set_ylim(0, 1.05)
ax[4].set_title('5. IBI, anomaly flag (2:1 AV block onset) and confidence'); ax[4].set_xlabel('time (s)'); ax[4].legend(loc='upper left'); ax4.legend(loc='upper right')
plt.tight_layout(); plt.savefig('plots/fig1_ecg_stages.png', dpi=110); plt.savefig('plots/fig1_ecg_stages.pdf'); plt.close()
# ---- Fig 2: PPG motion suppression
name = 'ppg125_motion05'
t, x, _, _, _ = pipeline.load_signal('dev/data/%s.csv' % name); z = np.load('dev/data/%s.npz' % name); fs = float(z['fs']); tr = z['beats']
yb = pipeline.bandpass(x, fs, *P['ppg_band']); yc, f0, mov, pur, hop = pipeline.motion_suppress(yb, fs)
r = pipeline.run(x, fs, 'ppg', '', return_debug=True)
q = max(1, int(fs // P['ms_fs'])); yd = signal.decimate(yb, q, ftype='fir', zero_phase=True); fsd = fs / q
f, ts, Z = signal.stft(yd, fsd, nperseg=int(8 * fsd), noverlap=int(7 * fsd))
fig, ax = plt.subplots(4, 1, figsize=(11, 9))
sel = (f <= 4); k = ts <= 300
ax[0].pcolormesh(ts[k], f[sel], np.abs(Z[sel][:, k]), shading='auto', cmap='magma'); ax[0].plot(np.arange(len(f0))[:300] * hop, f0[:300], c='cyan', lw=1.2, label='Viterbi HR track')
hr_true = pd.Series(1 / np.diff(tr / fs)).rolling(5, center=True, min_periods=1).median()
ax[0].plot(tr[1:] / fs, hr_true, 'w--', lw=.8, label='true HR'); ax[0].set_xlim(0, 300); ax[0].set_ylabel('Hz'); ax[0].legend(loc='upper right'); ax[0].set_title('1. short-time spectrum (8 s / 1 s) with harmonic-sum Viterbi HR track')
t0, t1 = 55, 75; i0, i1 = int(t0 * fs), int(t1 * fs); tt = np.arange(i0, i1) / fs
ax[1].plot(tt, x[i0:i1], c='gray', lw=.7); ax[1].set_title('2. raw PPG (drift + in-band periodic motion + jolts)')
ax[2].plot(tt, yb[i0:i1], c='gray', lw=.7, label='band-passed 0.5-8 Hz'); ax[2].plot(tt, yc[i0:i1], c='C0', lw=1, label='after spectral motion suppression'); ax[2].legend(loc='upper right'); ax[2].set_title('3. motion suppression')
pk = r['pk']; s = pk[(pk >= i0) & (pk < i1)]; trs = tr[(tr >= i0) & (tr < i1)]
ax[3].plot(tt, r['morph'][i0:i1], lw=.8); ax[3].plot(s / fs, r['morph'][s], 'o', mfc='none', c='C3', label='detected'); ax[3].plot(trs / fs, r['morph'][trs] + .4, 'v', c='g', ms=4, label='true systolic peak')
ax[3].plot(tt, r['conf'][i0:i1] - 3, c='C2', label='confidence (offset -3)'); ax[3].legend(loc='upper right', ncol=3); ax[3].set_title('4. fused Elgendi / HR-guided DP peaks'); ax[3].set_xlabel('time (s)')
plt.tight_layout(); plt.savefig('plots/fig2_ppg_motion.png', dpi=110); plt.savefig('plots/fig2_ppg_motion.pdf'); plt.close()
# ---- Fig 3: benchmark
res = json.load(open('dev/results_v7.json')); v1 = json.load(open('dev/results_v1.json'))
names = [k for k in res if k in v1]
fig, ax = plt.subplots(figsize=(11, 4)); xx = np.arange(len(names))
ax.bar(xx - .2, [v1[k]['beat_f1'] for k in names], .4, label='v1 (Pan-Tompkins / Elgendi)', color='#bbb')
ax.bar(xx + .2, [res[k]['beat_f1'] for k in names], .4, label='final', color='C0')
ax.set_xticks(xx); ax.set_xticklabels(names, rotation=60, ha='right'); ax.set_ylim(.5, 1.01); ax.set_ylabel('beat F1 (±150 ms)'); ax.legend()
ax.set_title('Beat detection F1 on stress sets (NSTDB noise on ECG; synthetic PPG with drift/motion)')
plt.tight_layout(); plt.savefig('plots/fig3_benchmark.png', dpi=110); plt.savefig('plots/fig3_benchmark.pdf'); plt.close()
# ---- Fig 4: AF feature
rows = np.load('dev/af_rows.npy', allow_pickle=True)
isaf = rows[:, 0].astype(bool); masd = rows[:, 1].astype(float)
fig, ax = plt.subplots(figsize=(6, 3.2))
bins = np.linspace(0, .8, 41); ax.hist(masd[isaf], bins, alpha=.6, label='AF windows', density=True); ax.hist(masd[~isaf], bins, alpha=.6, label='non-AF windows', density=True)
ax.axvline(P['af_masd'], c='k', ls='--', label='threshold %.2f' % P['af_masd']); ax.set_xlabel('median |ΔIBI| / median IBI (16 beats)'); ax.legend(); ax.set_title('AF irregularity feature, MIT-BIH 201-222')
plt.tight_layout(); plt.savefig('plots/fig4_af_feature.png', dpi=110); plt.savefig('plots/fig4_af_feature.pdf'); plt.close()
print('ok')
