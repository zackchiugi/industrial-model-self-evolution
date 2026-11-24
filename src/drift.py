"""Frozen training-reference feature drift, without validation labels.
Engineering alarms, not calibrated significance tests for correlated sensor rows.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from baselines.data import flatten,json_value

def inputs_only(path,paths):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        names,x=flatten(json_value(line)['inputs'])
        if names!=paths: raise ValueError('drift feature schema mismatch')
        rows.append(x)
    a=np.asarray(rows,dtype=float)
    if a.ndim!=2 or len(a)<1 or not np.isfinite(a).all(): raise ValueError('invalid drift input')
    return a

def reference(x,paths):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(paths) or len(x)<64 or not np.isfinite(x).all(): raise ValueError('reference requires >=64 finite rows')
    # Keep a bounded deterministic empirical reference. Fit only on training.
    ix=np.random.default_rng(42).choice(len(x),min(2048,len(x)),replace=False)
    return {'version':1,'paths':paths,'values':x[ix].tolist(),'source_rows':len(x),'seed':42,'sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}

def diagnose(ref,x,paths):
    if ref.get('version')!=1 or ref['paths']!=paths: raise ValueError('reference schema mismatch')
    a=np.asarray(ref['values'],dtype=float); b=np.asarray(x,dtype=float)
    if b.ndim!=2 or b.shape[1]!=a.shape[1] or len(b)<1 or not np.isfinite(b).all(): raise ValueError('invalid observed features')
    n=len(b); ix=np.random.default_rng(43).choice(n,min(2048,n),replace=False); b=b[ix]
    if min(len(a),len(b))<64: return {'status':'insufficient_data','alert':False,'observed_rows':n,'features':[]}
    threshold=max(.10,float(np.sqrt(.5*np.log(2*a.shape[1]/.01)*(1/len(a)+1/len(b)))))
    features=[]
    for j,path in enumerate(paths):
        u=np.sort(a[:,j]);v=np.sort(b[:,j]); grid=np.unique(np.concatenate([u,v]))
        ks=float(np.max(np.abs(np.searchsorted(u,grid,side='right')/len(u)-np.searchsorted(v,grid,side='right')/len(v))))
        cuts=np.unique(np.quantile(u,np.linspace(0,1,11)))
        # Preserve a separate tail bin for constant features as well.
        edges=np.r_[-np.inf,cuts,np.nextafter(cuts[-1],np.inf),np.inf]
        p=(np.histogram(u,edges)[0]+.5)/(len(u)+.5*(len(edges)-1))
        q=(np.histogram(v,edges)[0]+.5)/(len(v)+.5*(len(edges)-1))
        psi=float(np.sum((q-p)*np.log(q/p)))
        shift=float((np.mean(v)-np.mean(u))/max(np.std(u),1e-12))
        features.append({'index':j,'path':path,'ks':ks,'psi':psi,'standardized_mean_shift':shift,'alert':ks>threshold})
    features.sort(key=lambda z:z['ks'],reverse=True)
    return {'status':'measured','alert':any(f['alert'] for f in features),'reference_rows':len(a),'observed_rows':n,'sampled_observed_rows':len(b),'ks_threshold':threshold,'flagged_features':sum(f['alert'] for f in features),'features':features,'interpretation':'Input-distribution alarm; correlated rows, no calibrated p-values; feature ranking is not causal attribution.'}
