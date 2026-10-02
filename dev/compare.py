"""Head-to-head of pipeline variants (final pipeline.py vs design branches A and B) on identical inputs, one scorer.
Both are run as CLI programs on the same CSV; scoring uses predictions.csv columns only:
Is_peak (±150 ms match), Ibi_ms at matched consecutive peaks, ano_flag (per sample + 10-s windows), runtime."""
import sys, os, glob, json, time, subprocess
sys.path[:0] = ['dev', '.']
import numpy as np, pandas as pd
from evaluate import truth_from_annotations, match, prf
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cmp'
b360, gt360 = truth_from_annotations()
PY = '.venv/bin/python'
ALL = {'final': ['pipeline.py'], 'branch_a': ['dev/baselines/branch_a.py'],
       'branch_b': ['dev/baselines/branch_b.py', '--log-level', 'ERROR']}
VARIANTS = {k: ALL[k] for k in os.environ.get('VARIANTS', 'final,branch_a,branch_b').split(',')}

def score(pred, ref, gt, fs):
    det = np.flatnonzero(pred['Is_peak'].to_numpy())
    ok, best = match(ref, det, int(0.15 * fs))
    tp = ok.sum(); se, ppv, f1 = prf(tp, len(det) - tp, len(ref) - tp)
    ibi_col = pred['Ibi_ms'].to_numpy()
    pair = ok[1:] & ok[:-1]
    same = (best[1:] - best[:-1] == 1) & pair
    ibi_ref = np.diff(ref)[same] / fs * 1000
    ibi_det = ibi_col[det[best[1:][same]]]
    mae = float(np.mean(np.abs(ibi_ref - ibi_det))) if same.any() else np.nan
    a = pred['ano_flag'].to_numpy().astype(bool); g = gt[:len(a)]
    _, _, af1 = prf((a & g).sum(), (a & ~g).sum(), (~a & g).sum())
    w = int(10 * fs); m = len(a) // w
    aw = a[:m * w].reshape(m, w).sum(1) >= fs; gw = g[:m * w].reshape(m, w).sum(1) >= fs
    _, _, wf1 = prf((aw & gw).sum(), (aw & ~gw).sum(), (~aw & gw).sum())
    c = pred['confidence_score'].to_numpy()
    return dict(beat_f1=f1, se=se, ppv=ppv, ibi_mae=mae, ano_f1=af1, win_f1=wf1, conf=float(c.mean()))

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    files = ['ecg.csv'] + sorted(glob.glob('dev/data/*.csv'))
    rows = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        if name == 'ecg':
            ref, gt, fs = b360, gt360, 360.0
        else:
            z = np.load(f[:-4] + '.npz'); ref = z['beats']; fs = float(z['fs']) if 'fs' in z else 360.0
            n = sum(1 for _ in open(f)) - 1
            gt = (np.unpackbits(z['ano'])[:int(z['n'])].astype(bool) if 'ano' in z
                  else gt360[np.minimum((np.arange(n) * 360 / fs).astype(int), len(gt360) - 1)])
        rows[name] = {}
        for team, cmd in VARIANTS.items():
            out = os.path.join(OUT, '%s_%s.csv' % (team, name))
            t0 = time.perf_counter()
            p = subprocess.run([PY] + cmd + ['--input', f, '--output', out, '--fs', str(fs)], capture_output=True, text=True)
            dt = time.perf_counter() - t0
            if p.returncode != 0:
                rows[name][team] = dict(error=p.stderr.strip().splitlines()[-1] if p.stderr else 'failed'); continue
            s = score(pd.read_csv(out), ref, gt, fs); s['time'] = dt; rows[name][team] = s
        fmt = lambda d: 'ERR ' + d['error'][:40] if 'error' in d else 'F1 %.3f IBI %5.1f ano %.3f' % (d['beat_f1'], d['ibi_mae'], d['ano_f1'])
        print('%-22s ' % name + '  '.join('%s[%s]' % (k, fmt(v)) for k, v in rows[name].items()), flush=True)
    json.dump(rows, open(os.path.join(OUT, 'compare.json'), 'w'), indent=1, default=float)

