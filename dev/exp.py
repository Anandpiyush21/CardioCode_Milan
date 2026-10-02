"""In-process experiment runner: python dev/exp.py '{"param": value, ...}' [filter]
Runs pipeline (load_record -> run -> to_rows) on all stress sets + the independent PPG fixtures and prints means."""
import sys, glob, os, json, io, contextlib
sys.path[:0] = ['dev', '.']
import numpy as np, pandas as pd
import pipeline
from evaluate import truth_from_annotations, prf
from compare import score
FIX = os.environ.get('FIXTURES', 'dev/fixtures')      # made by dev/make_fixtures.py
ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
flt = sys.argv[2] if len(sys.argv) > 2 else ''
pipeline.P.update(ov)
b360, gt360 = truth_from_annotations()
res = {}
def run_file(f, fs=None):
    rec = pipeline.load_record(f, fs)
    r = pipeline.run(rec['x'], rec['fs'], 'auto', rec['name'], acc=rec['acc'], valid=rec['valid'])
    return pipeline.to_rows(rec, r)
for f in ['ecg.csv'] + sorted(glob.glob('dev/data/*.csv')):
    name = os.path.basename(f)[:-4]
    if flt and flt not in name: continue
    if name == 'ecg': ref, gt, fs = b360, gt360, 360.0
    else:
        z = np.load(f[:-4] + '.npz'); ref = z['beats']; fs = float(z['fs']) if 'fs' in z else 360.0
        gt = np.unpackbits(z['ano'])[:int(z['n'])].astype(bool) if 'ano' in z else None
    pr = run_file(f, fs)
    if gt is None: gt = gt360[np.minimum((np.arange(len(pr)) * 360 / fs).astype(int), len(gt360) - 1)]
    res[name] = score(pr, ref, gt, fs)
# independent PPG fixtures (lag-compensated)
sys.argv = ['x', FIX, '/tmp']
from compare_fixtures import parse, gt_mask
for name in ('ppg_acc', 'ppg_noacc'):
    if flt and flt not in name: continue
    beats, types, rh = parse(os.path.join(FIX, name + '_ann.txt'))
    pr = run_file(os.path.join(FIX, name + '.csv')); ts = pr['Timestamp'].to_numpy(float)
    g, bt = gt_mask(ts, beats, types, rh)
    d = ts[pr['Is_peak'].to_numpy() == 1]
    j = np.clip(np.searchsorted(bt, d), 1, len(bt) - 1)
    near = np.where(np.abs(d - bt[j - 1]) < np.abs(d - bt[j]), bt[j - 1], bt[j]); off = d - near
    d = d - np.median(off[np.abs(off) < 0.5])
    k = np.clip(np.searchsorted(d, bt), 1, len(d) - 1)
    tp = int((np.minimum(np.abs(d[k - 1] - bt), np.abs(d[k] - bt)) <= 0.15).sum())
    _, _, f1 = prf(tp, len(d) - tp, len(bt) - tp)
    a = pr['ano_flag'].to_numpy().astype(bool)
    _, _, af1 = prf((a & g).sum(), (a & ~g).sum(), (~a & g).sum())
    res['fix_' + name] = dict(beat_f1=f1, ano_f1=af1, win_f1=np.nan, ibi_mae=np.nan)
m = lambda key, sel=lambda k: True: np.nanmean([v[key] for k, v in res.items() if sel(k)])
print('%s | beat %.4f ano %.4f win %.4f | ecg.csv ano %.3f | motion ano %.4f | fixtures ano %s' % (
    json.dumps(ov), m('beat_f1'), m('ano_f1'), m('win_f1'), res.get('ecg', {}).get('ano_f1', np.nan),
    m('ano_f1', lambda k: 'motion' in k), ' '.join('%.3f' % res[k]['ano_f1'] for k in res if k.startswith('fix_'))))
# confidence calibration on noisy / motion sets (per 10-s window): Spearman(conf, window beat F1), AUC(F1>=0.9)
if os.environ.get('CONF'):
    from scipy.stats import spearmanr
    from evaluate import match
    cs, fs_ = [], []
    for f in sorted(glob.glob('dev/data/*motion*.csv')) + sorted(glob.glob('dev/data/ecg_*dB*.csv')):
        z = np.load(f[:-4] + '.npz'); fs = float(z['fs']) if 'fs' in z else 360.0; ref = z['beats']
        pr = run_file(f, fs); det = np.flatnonzero(pr['Is_peak'].to_numpy()); conf = pr['confidence_score'].to_numpy()
        W = int(10 * fs)
        for s0 in range(0, len(pr) - W, W):
            r = ref[(ref >= s0) & (ref < s0 + W)]; d = det[(det >= s0) & (det < s0 + W)]
            if len(r) < 3: continue
            ok = match(r, d, int(.15 * fs))[0] if len(d) else np.zeros(len(r), bool)
            cs.append(conf[s0:s0 + W].mean()); fs_.append(2 * ok.sum() / (len(r) + len(d)))
    cs, fs_ = np.array(cs), np.array(fs_); y = fs_ >= 0.9
    rk = pd.Series(cs).rank().to_numpy(); auc = (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    print('confidence: Spearman %.3f AUC %.3f (windows %d)' % (spearmanr(cs, fs_)[0], auc, len(cs)))
