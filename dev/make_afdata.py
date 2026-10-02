"""Real arrhythmia test sets from MIT-BIH (AF, AFL, VT, bigeminy, ...): ECG (MLII) + synthetic PPG 125 Hz."""
import sys, os; sys.path.insert(0, 'dev')
import numpy as np, pandas as pd
from ann import from_atr, anomaly_truth
from synth import synth_ppg
from make_data_util import read212
os.makedirs('dev/data', exist_ok=True)
for rec in ('201', '202', '203', '210', '219', '221', '222'):
    x = read212('dev/mitdb/%s.dat' % rec)[:, 0] + 1024
    n = len(x)
    b, t, rh, notes = from_atr('dev/mitdb/%s.atr' % rec)
    ano = anomaly_truth(b, t, rh, notes, n)
    pd.DataFrame({'sample #': np.arange(n), 'MLII': x.astype(int)}).to_csv('dev/data/mit%s_ecg.csv' % rec, index=False)
    np.savez('dev/data/mit%s_ecg.npz' % rec, beats=b, fs=360, ano=np.packbits(ano), n=n)
    for tag, m in (('clean', 0), ('motion05', 0.5)):
        fs = 125
        y, tr = synth_ppg(b, t, n, fs, m, 1.0 if m else 0)
        N = len(y); ano_p = ano[np.minimum((np.arange(N) * 360 / fs).astype(int), n - 1)]
        pd.DataFrame({'sample #': np.arange(N), 'PPG': np.round(y, 5)}).to_csv('dev/data/mit%s_ppg_%s.csv' % (rec, tag), index=False)
        np.savez('dev/data/mit%s_ppg_%s.npz' % (rec, tag), beats=tr, fs=fs, ano=np.packbits(ano_p), n=N)
    print(rec, 'beats', len(b), 'anomalous %.2f' % ano.mean(), sorted(set(l for _, l in rh)))
