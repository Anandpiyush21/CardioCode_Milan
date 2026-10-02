"""Score pipeline output against ground truth (beats + anomaly intervals)."""
import sys, time, json
sys.path.insert(0, 'dev'); sys.path.insert(0, '.')
import numpy as np
from ann import load
FS = 360

def truth_from_annotations(path='annotation.txt', n=650000):
    b, t, rh, notes = load(path)
    iv = []
    # rhythm segments that are not normal sinus
    bounds = [s for s, _ in rh] + [n]
    for (s, lab), e in zip(rh, bounds[1:]):
        if lab != '(N':
            iv.append((s, e))
    # ectopic beats: span previous..next beat
    for i, ty in enumerate(t):
        if ty not in 'NR':
            iv.append((b[max(i - 1, 0)], b[min(i + 1, len(b) - 1)]))
    # dropped beats (MISSB): span of the long IBI containing the note
    for s, _ in notes:
        i = np.searchsorted(b, s)
        if 0 < i < len(b):
            iv.append((b[i - 1], b[i]))
    ano = np.zeros(n, bool)
    for s, e in iv:
        ano[s:e + 1] = True
    return b, ano

def match(ref, det, tol):
    """Greedy one-to-one matching of sorted index arrays within tol samples."""
    j = np.searchsorted(det, ref)
    cand = np.stack([np.clip(j - 1, 0, len(det) - 1), np.clip(j, 0, len(det) - 1)], 1)
    d = np.abs(det[cand] - ref[:, None])
    best = cand[np.arange(len(ref)), d.argmin(1)]
    ok = d.min(1) <= tol
    # enforce one-to-one
    used = {}
    for r in np.flatnonzero(ok):
        k = best[r]
        if k in used and abs(det[k] - ref[used[k]]) <= abs(det[k] - ref[r]):
            ok[r] = False
        else:
            if k in used: ok[used[k]] = False
            used[k] = r
    return ok, best

def prf(tp, fp, fn):
    se = tp / max(tp + fn, 1); ppv = tp / max(tp + fp, 1)
    return se, ppv, 2 * se * ppv / max(se + ppv, 1e-12)

def score(res, ref_beats, gt_ano, fs=FS, tol_s=0.15, lag=0):
    det = np.flatnonzero(res['is_peak'])
    ref = ref_beats + lag
    ok, best = match(ref, det, int(tol_s * fs))
    tp = ok.sum(); fn = len(ref) - tp; fp = len(det) - tp
    se, ppv, f1 = prf(tp, fp, fn)
    # IBI error on consecutive matched pairs (use sub-sample times if available)
    tk = res.get('tk', det.astype(float))
    pos = np.searchsorted(det, det)  # identity
    pair = ok[1:] & ok[:-1]
    ibi_ref = np.diff(ref)[pair] / fs * 1000
    ibi_det = (tk[best[1:][pair]] - tk[best[:-1][pair]]) / fs * 1000
    same = best[1:][pair] - best[:-1][pair] == 1
    ibi_mae = np.mean(np.abs(ibi_ref[same] - ibi_det[same])) if same.any() else np.nan
    timing = np.median(np.abs(det[best[ok]] - ref[ok])) / fs * 1000
    # anomaly, sample level
    a = res['ano'].astype(bool); g = gt_ano[:len(a)]
    s_se, s_ppv, s_f1 = prf((a & g).sum(), (a & ~g).sum(), (~a & g).sum())
    # 10 s windows: window positive if >= 1 s anomalous
    w = int(10 * fs); m = len(a) // w
    aw = a[:m * w].reshape(m, w).sum(1) >= fs; gw = g[:m * w].reshape(m, w).sum(1) >= fs
    w_se, w_ppv, w_f1 = prf((aw & gw).sum(), (aw & ~gw).sum(), (~aw & gw).sum())
    return dict(beat_se=se, beat_ppv=ppv, beat_f1=f1, fp=int(fp), fn=int(fn), ibi_mae_ms=ibi_mae,
                timing_med_ms=timing, ano_se=s_se, ano_ppv=s_ppv, ano_f1=s_f1, win_f1=w_f1,
                win_se=w_se, win_ppv=w_ppv, mean_conf=float(np.mean(res['conf'])))

def fmt(d):
    return ' '.join('%s=%.4g' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v) for k, v in d.items())

if __name__ == '__main__':
    import pandas as pd, pipeline
    d = pd.read_csv('ecg.csv'); x = d.iloc[:, 1].to_numpy(float)
    b, gt = truth_from_annotations(n=len(x))
    t0 = time.perf_counter(); r = pipeline.run(x, FS, 'auto', 'mlii'); dt = time.perf_counter() - t0
    print('mode', r['mode'], 'runtime %.2fs' % dt)
    print(fmt(score(r, b, gt)))
    print('GT anomalous fraction %.3f, predicted %.3f' % (gt.mean(), r['ano'].mean()))
