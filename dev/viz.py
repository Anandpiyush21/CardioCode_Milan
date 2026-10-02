"""viz.py <dataset> <t0> [dur] : plot signal, detections, truth, anomaly & confidence."""
import sys; sys.path.insert(0,'dev'); sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pipeline
name=sys.argv[1]; t0=float(sys.argv[2]); dur=float(sys.argv[3]) if len(sys.argv)>3 else 20
f='ecg.csv' if name=='ecg' else 'dev/data/%s.csv'%name
t,x,fs,ch,low=pipeline.load_signal(f)
if name!='ecg':
    z=np.load(f[:-4]+'.npz'); truth=z['beats']; fs=float(z['fs']) if 'fs' in z else fs
else:
    from ann import load; truth=load()[0]
r=pipeline.run(x,fs,'auto','',return_debug=True)
i0,i1=int(t0*fs),int((t0+dur)*fs); tt=np.arange(i0,i1)/fs
fig,ax=plt.subplots(3,1,figsize=(16,9),sharex=True)
ax[0].plot(tt,x[i0:i1],lw=.6,color='gray'); ax[0].set_title(name+' raw')
m=r['morph']; ax[1].plot(tt,m[i0:i1],lw=.7)
pk=r['pk']; s=pk[(pk>=i0)&(pk<i1)]; ax[1].plot(s/fs,m[s],'ro',ms=5,label='detected')
tr=truth[(truth>=i0)&(truth<i1)]; ax[1].plot(tr/fs,m[tr],'g^',ms=7,mfc='none',label='truth'); ax[1].legend()
ax[2].plot(tt,r['conf'][i0:i1],label='confidence'); ax[2].plot(tt,r['ano'][i0:i1]*0.9,label='ano_flag')
ax[2].legend(); plt.tight_layout(); out='plots/viz_%s_%d.png'%(name,t0); plt.savefig(out,dpi=70); print(out)
