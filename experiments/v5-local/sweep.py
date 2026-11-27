import os,sys,time,json,copy
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,'final_work/src')
from baselines.model import Model
from baselines.data import read_dataset
from baselines.train import metrics
results=[]
for k in 'ABC':
 p='release/cpu-'+k.lower();m=Model.load(p+'/checkpoint.npz');a=read_dataset(p+'/public/train.jsonl',m.meta['task'],m.meta['classes']);b=read_dataset(p+'/public/validation.jsonl',m.meta['task'],m.meta['classes']);before=metrics(m,b);metric='macro_f1' if k=='A' else 'rmse'
 for lr in [.0003,.001,.003,.01]:
  n=copy.deepcopy(m);n.meta['learning_rate']=lr;t=time.monotonic();best=before[metric];bestep=0;records=[]
  for ep in range(1,101):
   n.fit(a['x'],a['y'],1)
   if ep%5:continue
   val=metrics(n,b);v=val[metric]
   if (v>best if k=='A' else v<best):best=v;bestep=ep
   records.append({'epoch':ep,**val})
  r={'dataset':k,'lr':lr,'before':before,'best':best,'best_epoch':bestep,'seconds':time.monotonic()-t,'records':records};results.append(r);print(json.dumps({x:y for x,y in r.items() if x!='records'}),flush=True)
  open('final_work/evidence/v5/sweep_results.json','w').write(json.dumps(results,indent=2))
