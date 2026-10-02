"""Fig. 5: grouped comparison of design branch A, design branch B and the final pipeline -> plots/fig5_compare.{png,pdf}
usage: python dev/fig_compare.py <compare.json> <compare_fixtures.json>"""
import sys, json
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1])); fx = json.load(open(sys.argv[2]))
G = [('ecg.csv', lambda k: k == 'ecg'), ('ECG + NSTDB\nnoise (10)', lambda k: k.startswith('ecg_')),
     ('real AF/VT\nECG (7)', lambda k: k.startswith('mit') and k.endswith('_ecg')),
     ('real AF/VT\nPPG clean (7)', lambda k: 'ppg_clean' in k), ('real AF/VT\nPPG motion (7)', lambda k: 'ppg_motion' in k),
     ('synthetic PPG\nclean (6)', lambda k: k.startswith('ppg') and ('clean' in k or 'drift' in k)),
     ('synthetic PPG\nmotion (6)', lambda k: k.startswith('ppg') and 'motion' in k)]
teams = [('branch_a', 'design branch A', '#9aa5b1'), ('branch_b', 'design branch B', '#e0a458'), ('final', 'final pipeline', '#1f5fa8')]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, met, lab in zip(axes, ('beat_f1', 'ano_f1'), ('beat F1', 'anomaly F1 (per sample)')):
    rows = [(g, f) for g, f in G] + [('fixture PPG\n+ACC', None), ('fixture PPG\nno ACC', None)]
    x = np.arange(len(rows))
    for i, (t, name, c) in enumerate(teams):
        vals = []
        for g, f in rows:
            if f is None:
                key = 'ppg_acc' if 'ACC' in g and 'no' not in g else 'ppg_noacc'
                vals.append(fx[key][t][met])
            else:
                vals.append(np.mean([r[k][t][met] for k in r if f(k)]))
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=c, label=name)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in rows], fontsize=7, rotation=35, ha='right')
    ax.set_ylim(0.3 if met == 'ano_f1' else 0.6, 1.01); ax.set_ylabel(lab); ax.grid(axis='y', alpha=.3)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(loc='lower left', fontsize=8)
plt.tight_layout(); plt.savefig('plots/fig5_compare.png', dpi=110); plt.savefig('plots/fig5_compare.pdf')
print('ok')
