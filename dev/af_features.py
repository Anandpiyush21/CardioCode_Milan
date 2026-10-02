import sys; sys.path.insert(0,'dev')
import numpy as np
from ann import from_atr
W=16
def feats(v):
    d=np.diff(v); med=np.median(v)
    masd=np.median(np.abs(d))/med
    nrmssd=np.sqrt(np.mean(d**2))/np.mean(v)
    m=len(v); tp=np.sum((v[1:-1]-v[:-2])*(v[1:-1]-v[2:])>0); tpz=(tp-(2*m-4)/3)/np.sqrt((16*m-29)/90)
    # Shannon entropy of 8-bin histogram after dropping 2 min/2 max (Dash)
    s=np.sort(v)[2:-2]; h,_=np.histogram(s,bins=8); p=h[h>0]/h.sum(); she=-(p*np.log(p)).sum()/np.log(8)
    # fraction of successive differences > 50 ms relative (pNN50 normalised)
    pnn=np.mean(np.abs(d)>0.05)
    return masd,nrmssd,tpz,she,pnn
rows=[]
for r in ('201','202','203','210','219','221','222'):
    b,t,rh,_=from_atr('dev/mitdb/%s.atr'%r); tb=b/360; ibi=np.diff(tb)
    rs=np.array([s for s,_ in rh])/360; rl=[l for _,l in rh]
    for j in range(0,len(ibi)-W,W//2):
        tc=tb[j+W//2]; lab=rl[max(np.searchsorted(rs,tc,'right')-1,0)]
        rows.append((r,lab)+feats(ibi[j:j+W]))
import collections
names=['masd','nrmssd','tpz','she','pnn50']
for lab in ('(AFIB','(AFL','(N','(B','(T','(NOD','(AB','(SVTA','(VT'):
    X=np.array([x[2:] for x in rows if x[1]==lab])
    if len(X)==0: continue
    print('%-6s n=%4d '%(lab,len(X))+'  '.join('%s p10/50/90=%s'%(nm,np.round(np.percentile(X[:,i],[10,50,90]),2)) for i,nm in enumerate(names)))
np.save('dev/af_rows.npy',np.array([(x[1]=='(AFIB',)+x[2:] for x in rows]),allow_pickle=True)
