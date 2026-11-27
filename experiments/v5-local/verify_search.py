import os,sys,tempfile,json
os.environ['OPENBLAS_NUM_THREADS']='1';sys.path.insert(0,'final_work/src')
from baselines.model import Model
from baselines.data import read_dataset
from adaptive_train import improve
results=[]
for key,lr,epochs in [('A',.001,100),('B',.01,100),('C',.003,80)]:
 p='release/cpu-'+key.lower();m=Model.load(p+'/checkpoint.npz');a=read_dataset(p+'/public/train.jsonl',m.meta['task'],m.meta['classes']);b=read_dataset(p+'/public/validation.jsonl',m.meta['task'],m.meta['classes'])
 with tempfile.TemporaryDirectory() as temp:
  r=improve(m,a,b,temp+'/candidate',[lr*.3,lr],epochs)
  n=Model.load(temp+'/candidate/checkpoint.npz');assert n.meta['steps']>m.meta['steps'];assert (n.mean==m.mean).all();assert (n.scale==m.scale).all()
  results.append(r);print(key,r['validation_before'],r['validation_after'],r['elapsed_seconds'],flush=True)
  open('final_work/evidence/v5/local_search_verification.json','w').write(json.dumps(results,indent=2))
