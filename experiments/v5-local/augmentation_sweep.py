import os,sys,copy,json,time
os.environ['OPENBLAS_NUM_THREADS']='1';sys.path.insert(0,'final_work/src')
from baselines.model import Model
from baselines.data import read_dataset
from baselines.train import metrics
import numpy as np
p='release/cpu-a';m=Model.load(p+'/checkpoint.npz');a=read_dataset(p+'/public/train.jsonl',m.meta['task'],m.meta['classes']);b=read_dataset(p+'/public/validation.jsonl',m.meta['task'],m.meta['classes']);out=[]
for noise in [.03,.1,.3,.6,1.0]:
 for lr in [.001,.003,.01]:
  n=copy.deepcopy(m);n.meta['learning_rate']=lr;best=0;bestep=0
  for ep in range(1,101):
   rng=np.random.default_rng(np.random.SeedSequence([42,n.meta['epochs'],103]));x=a['x']+rng.normal(size=a['x'].shape)*n.scale*noise
   n.fit(x,a['y'],1)
   if ep%5:continue
   f=metrics(n,b)['macro_f1']
   if f>best:best=f;bestep=ep
  r={'noise':noise,'lr':lr,'best_f1':best,'best_epoch':bestep};out.append(r);print(json.dumps(r),flush=True)
open('final_work/evidence/v5/augmentation_results.json','w').write(json.dumps(out,indent=2))
