"""Score pipeline variants on the independent synthetic PPG fixtures (dev/make_fixtures.py: ppg_acc / ppg_noacc).
Annotations are ECG R-peak times; PPG peaks lag by the pulse transit time. For each team, the lag is estimated
as the median offset to the nearest reference beat within 0.5 s and removed before matching at ±150 ms
(same procedure for every variant). Anomaly GT: non-(N rhythm segments + spans around non-N beats + MISSB, in time."""
import sys, os, json, subprocess, time
sys.path[:0] = ['dev', '.']
import numpy as np, pandas as pd
from evaluate import prf
def parse(path):
    beats, types, rh = [], [], []
    for L in open(path).readlines()[1:]:
        head, _, aux = L.rstrip('\n').partition('\t'); p = head.split()
        if len(p) < 3: continue
        mm, ss = p[0].split(':'); t = int(mm) * 60 + float(ss); ty = p[2]
        if ty in 'NVAFLRaJSE': beats.append(t); types.append(ty)
        elif ty == '+' and aux.strip().startswith('('): rh.append((t, aux.strip()))
        elif ty == '"' and 'MISSB' in aux: types.append('M'); beats.append(t)
    return np.array(beats), np.array(types), rh
def gt_mask(ts, beats, types, rh):
    g = np.zeros(len(ts), bool); T = ts[-1] + 1
    bounds = [t for t, _ in rh] + [T]
    for (s, lab), e in zip(rh, bounds[1:]):
        if lab != '(N': g |= (ts >= s) & (ts <= e)
    real = types != 'M'; bt = beats[real]
    for t, ty in zip(beats, types):
        if ty in ('V', 'A', 'M'):
            i = np.searchsorted(bt, t); a = bt[max(i - 1, 0)]; b = bt[min(i + 1 if ty != 'M' else i, len(bt) - 1)]
            g |= (ts >= a) & (ts <= b)
    return g, bt
if __name__ == '__main__':
    D = sys.argv[1]; OUT = sys.argv[2]
    rows = {}
    for name in ('ppg_acc', 'ppg_noacc'):
        f = os.path.join(D, name + '.csv')
        beats, types, rh = parse(os.path.join(D, name + '_ann.txt'))
        rows[name] = {}
        ALL = {'final': ['pipeline.py'], 'branch_a': ['dev/baselines/branch_a.py'],
               'branch_b': ['dev/baselines/branch_b.py', '--log-level', 'ERROR']}
        for team, cmd in ((k, ALL[k]) for k in os.environ.get('VARIANTS', 'final,branch_a,branch_b').split(',')):
            out = os.path.join(OUT, '%s_%s.csv' % (team, name)); t0 = time.perf_counter()
            p = subprocess.run(['.venv/bin/python'] + cmd + ['--input', f, '--output', out], capture_output=True, text=True)
            dt = time.perf_counter() - t0
            if p.returncode: rows[name][team] = {'error': p.stderr.strip().splitlines()[-1]}; print(name, team, 'ERROR', p.stderr[-300:]); continue
            pr = pd.read_csv(out); ts = pr['Timestamp'].to_numpy(float)
            g, bt = gt_mask(ts, beats, types, rh)
            dt_det = ts[pr['Is_peak'].to_numpy() == 1]
            j = np.clip(np.searchsorted(bt, dt_det), 1, len(bt) - 1)
            near = np.where(np.abs(dt_det - bt[j - 1]) < np.abs(dt_det - bt[j]), bt[j - 1], bt[j])
            off = dt_det - near; lag = np.median(off[np.abs(off) < 0.5])
            det = dt_det - lag
            k = np.clip(np.searchsorted(det, bt), 1, len(det) - 1)
            dist = np.minimum(np.abs(det[k - 1] - bt), np.abs(det[k] - bt)); tp = int((dist <= 0.15).sum())
            se, ppv, f1 = prf(tp, len(det) - tp, len(bt) - tp)
            # IBI: reference IBI vs detected IBI at matched consecutive beats
            idx = np.where(np.abs(det[k - 1] - bt) < np.abs(det[k] - bt), k - 1, k); ok = dist <= 0.15
            pair = ok[1:] & ok[:-1] & (idx[1:] - idx[:-1] == 1)
            ibi_err = np.abs(np.diff(bt)[pair] - np.diff(det)[idx[1:][pair] - 1]) * 1000 if pair.any() else np.array([np.nan])
            # note: diff(det)[i-1] = det[i]-det[i-1]
            a = pr['ano_flag'].to_numpy().astype(bool)
            _, _, af1 = prf((a & g).sum(), (a & ~g).sum(), (~a & g).sum())
            ase, appv, _ = prf((a & g).sum(), (a & ~g).sum(), (~a & g).sum())
            rows[name][team] = dict(lag_ms=lag * 1000, beat_f1=f1, se=se, ppv=ppv, ibi_mae=float(np.nanmean(ibi_err)),
                                    ano_f1=af1, ano_se=ase, ano_ppv=appv, time=dt)
            print('%-10s %-6s lag %5.0f ms  F1 %.3f (Se %.3f PPV %.3f)  IBI MAE %5.1f ms  ano F1 %.3f (Se %.3f PPV %.3f)  %.1fs'
                  % (name, team, lag * 1000, f1, se, ppv, np.nanmean(ibi_err), af1, ase, appv, dt), flush=True)
    json.dump(rows, open(os.path.join(OUT, 'compare_fixtures.json'), 'w'), indent=1, default=float)

