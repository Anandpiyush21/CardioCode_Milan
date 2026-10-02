"""Run the pipeline over all stress datasets and print a score table.
usage: python dev/stress.py [filter-substring] [--save results.json]"""
import sys, glob, os, time, json
sys.path.insert(0, 'dev'); sys.path.insert(0, '.')
import numpy as np
import pipeline
from evaluate import truth_from_annotations, score
from ann import load
b360, gt360 = truth_from_annotations()
def gt_at(fs, n):
    if fs == 360: return gt360[:n]
    idx = np.minimum((np.arange(n) * 360 / fs).astype(int), len(gt360) - 1)
    return gt360[idx]
argv = sys.argv[1:]
save = argv[argv.index('--save') + 1] if '--save' in argv else None
flt = [a for a in argv if not a.startswith('--') and a != save]
files = ['ecg.csv'] + sorted(glob.glob('dev/data/*.csv'))
if flt: files = [f for f in files if any(s in f for s in flt)]
rows = {}
cols = ['beat_f1', 'beat_se', 'beat_ppv', 'ibi_mae_ms', 'ano_f1', 'ano_se', 'ano_ppv', 'win_f1', 'mean_conf']
print('%-22s %4s %5s' % ('dataset', 'mode', 't(s)') + ''.join('%10s' % c[:10] for c in cols))
for f in files:
    name = os.path.basename(f)[:-4]
    t, x, fs, ch, low = pipeline.load_signal(f)
    if name == 'ecg': truth = b360
    else:
        z = np.load(f[:-4] + '.npz'); truth = z['beats']
        if 'fs' in z: fs = float(z['fs'])
    gt = gt_at(fs, len(x))
    if name != 'ecg' and 'ano' in z:
        gt = np.unpackbits(z['ano'])[:int(z['n'])].astype(bool)
    t0 = time.perf_counter(); r = pipeline.run(x, fs, 'auto', ''); dt = time.perf_counter() - t0
    s = score(r, truth, gt, fs=fs); s['time'] = dt; s['mode'] = r['mode']; rows[name] = s
    print('%-22s %4s %5.2f' % (name, r['mode'], dt) + ''.join('%10.4f' % s[c] for c in cols))
agg = {c: float(np.mean([v[c] for v in rows.values()])) for c in cols}
print('%-22s %4s %5.2f' % ('MEAN', '', np.mean([v['time'] for v in rows.values()])) + ''.join('%10.4f' % agg[c] for c in cols))
if save:
    json.dump(rows, open(save, 'w'), indent=1, default=float)
