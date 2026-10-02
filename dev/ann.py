"""Parse MIT-BIH style annotation.txt (rdann text output)."""
import numpy as np
BEAT = set('NLRBAaJSVrFejnE/fQ?')
def load(path='annotation.txt'):
    beats, types, rh_changes, notes = [], [], [], []
    for L in open(path).readlines()[1:]:
        if not L.strip(): continue
        head, _, aux = L.rstrip('\n').partition('\t')
        p = head.split(); s = int(p[1]); t = p[2]; aux = aux.strip()
        if t in BEAT: beats.append(s); types.append(t)
        elif t == '+' and aux.startswith('('): rh_changes.append((s, aux))
        elif t == '"': notes.append((s, aux))
    return np.array(beats), np.array(types), rh_changes, notes

CODES = {1: 'N', 2: 'L', 3: 'R', 4: 'a', 5: 'V', 6: 'F', 7: 'J', 8: 'A', 9: 'S', 10: 'E', 11: 'j', 12: '/',
         13: 'Q', 14: '~', 16: '|', 18: 's', 19: 'T', 20: '*', 21: 'D', 22: '"', 23: '=', 24: 'p', 25: 'B',
         26: '^', 27: 't', 28: '+', 29: 'u', 30: '?', 31: '!', 32: '[', 33: ']', 34: 'e', 35: 'n', 36: '@',
         37: 'x', 38: 'f', 39: '(', 40: ')', 41: 'r'}

def read_atr(path):
    """Minimal WFDB MIT-format annotation reader -> (samples, symbols, aux)."""
    w = np.fromfile(path, '<u2')
    i, t, S, Y, X = 0, 0, [], [], []
    while i < len(w):
        a, v = int(w[i]) >> 10, int(w[i]) & 1023
        i += 1
        if a == 0 and v == 0:
            break
        if a == 59:                      # SKIP: 32-bit interval, PDP-11 word order
            t += (int(w[i]) << 16) | int(w[i + 1]); i += 2
        elif a == 63:                    # AUX string for the previous annotation
            b = w[i:i + (v + 1) // 2].tobytes()[:v]; i += (v + 1) // 2
            if X: X[-1] = b.decode('latin1').rstrip('\x00')
        elif a in (60, 61, 62):
            pass
        else:
            t += v; S.append(t); Y.append(CODES.get(a, '?')); X.append('')
    return np.array(S), np.array(Y), X

def from_atr(path):
    """Same outputs as load(): beats, types, rhythm changes, MISSB notes."""
    s, y, x = read_atr(path)
    beat = np.isin(y, list(BEAT))
    rh = [(si, xi) for si, yi, xi in zip(s, y, x) if yi == '+' and xi.startswith('(')]
    notes = [(si, xi) for si, yi, xi in zip(s, y, x) if yi == '"' and 'MISSB' in xi]
    return s[beat], y[beat], rh, notes

NORMAL_CONDUCTED = set('NLRej')

def anomaly_truth(beats, types, rh, notes, n):
    """Per-sample GT: abnormal rhythm segments + spans around ectopic beats + dropped-beat pauses."""
    iv = []
    bounds = [s for s, _ in rh] + [n]
    for (s, lab), e in zip(rh, bounds[1:]):
        if lab not in ('(N', '(SBR'):        # sinus rhythm / sinus bradycardia count as normal
            iv.append((s, e))
    for i, ty in enumerate(types):
        if ty not in NORMAL_CONDUCTED:
            iv.append((beats[max(i - 1, 0)], beats[min(i + 1, len(beats) - 1)]))
    for s, _ in notes:
        i = np.searchsorted(beats, s)
        if 0 < i < len(beats):
            iv.append((beats[i - 1], beats[i]))
    ano = np.zeros(n, bool)
    for s, e in iv:
        ano[s:e + 1] = True
    return ano
