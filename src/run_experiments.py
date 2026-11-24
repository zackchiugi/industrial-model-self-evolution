"""Ten genuine checkpoint-continuation rounds; public validation is model-selection data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,hashlib,shutil,argparse,csv
from pathlib import Path
from collections import Counter
import numpy as np
from baselines.model import Model,checkpoint_digest
from baselines.data import read_dataset
from baselines.train import metrics

def evaluate(model,data):
 out=metrics(model,data)
 pred=model.predict(data['x'])
 if model.meta['task']=='classification':
  p=pred.argmax(1); y=data['y']; cm=np.zeros((6,6),int)
  np.add.at(cm,(y,p),1)
  out['confusion_matrix']=cm.tolist()
  out['macro_f1_fixed6']=float(np.mean([2*cm[i,i]/max(1,cm[i].sum()+cm[:,i].sum()) for i in range(6)]))
 else:
  e=pred-data['y'];out['overprediction_fraction']=float(np.mean(e>0))
  if model.meta['profile']=='C':out['nasa_score']=float(np.expm1(np.where(e<0,-e/13,e/10)).sum())
 return out

def group(row,key):
 m=row['meta']
 return str(m['source_batch']) if key=='A' else m['bearing'] if key=='B' else m['source_subset']+':'+str(m['unit'])

def run(root,out,key):
 d=root/('cpu-'+key.lower()); dest=out/key;dest.mkdir(parents=True,exist_ok=False)
 parent=d/'checkpoint.npz'; model=Model.load(parent)
 tr=read_dataset(d/'public/train.jsonl',model.meta['task'],model.meta['classes'],model.meta['paths'])
 va=read_dataset(d/'public/validation.jsonl',model.meta['task'],model.meta['classes'],model.meta['paths'])
 raws={s:[json.loads(l) for l in (d/f'public/{s}.jsonl').read_text().splitlines() if l.strip()] for s in ('train','validation')}
 groups={s:sorted({group(r,key) for r in rows}) for s,rows in raws.items()}
 assert not set(groups['train'])&set(groups['validation']); assert not set(tr['ids'])&set(va['ids'])
 original=json.loads((d/'training.json').read_text())
 assert tr['sha256']==original['train_sha256'] and va['sha256']==original['validation_sha256']
 assert checkpoint_digest(parent)==original['checkpoint_sha256']
 baseline=evaluate(model,va);champ=baseline;lr=model.meta['learning_rate']; records=[]
 shutil.copy2(parent,dest/'baseline.npz')
 for i in range(1,11):
  # Reading prior history determines whether to shrink the next step.
  history=[dict(r) for r in records]
  if history: lr*=0.8 if history[-1]['accepted'] else 0.5
  candidate=Model.load(parent);candidate.meta['learning_rate']=lr
  candidate.meta['batch_size']=128 if key in 'AB' else 256
  before_steps=candidate.meta['steps'];before_epochs=candidate.meta['epochs']
  started=time.perf_counter(); losses=candidate.fit(tr['x'],tr['y'],5); seconds=time.perf_counter()-started
  measured=evaluate(candidate,va); metric='macro_f1' if key=='A' else 'rmse'
  accepted=measured[metric]>champ[metric]+1e-12 if key=='A' else measured[metric]<champ[metric]-1e-12
  rd=dest/f'round_{i:02}';rd.mkdir();cp=rd/'checkpoint.npz';candidate.save(cp)
  rec={'round':i,'history_ids_read':[x['round'] for x in history],'hypothesis':'Continue the selected parent for five epochs; reduce learning rate after examining prior acceptance.', 'parent_checkpoint_sha256':checkpoint_digest(parent),'candidate_checkpoint_sha256':checkpoint_digest(cp),'learning_rate':lr,'epochs':5,'batch_size':candidate.meta['batch_size'],'initial_steps':before_steps,'final_steps':candidate.meta['steps'],'initial_epochs':before_epochs,'final_epochs':candidate.meta['epochs'],'seed':candidate.meta['seed'],'losses':losses,'training_seconds':seconds,'validation_before':champ,'validation_candidate':measured,'accepted':bool(accepted),'reason':('Accept: strict primary metric improvement.' if accepted else 'Reject: no strict primary metric improvement; retain parent.'),'model_call_ids':[], 'execution':'local deterministic controller; no platform or LLM calls'}
  if accepted:parent=cp;champ=measured
  rec['selected_checkpoint_sha256']=checkpoint_digest(parent); rec['validation_selected']=champ
  records.append(rec);(rd/'record.json').write_text(json.dumps(rec,indent=2))
  print(key,i,metric,measured[metric],accepted,flush=True)
 final=Model.load(parent);shutil.copy2(parent,dest/'selected.npz');pred=final.predict(va['x']);basepred=model.predict(va['x'])
 with (dest/'validation_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','group','actual','baseline','selected'])
  for n,row in enumerate(raws['validation']):w.writerow([row['id'],group(row,key),row['answer'],model.meta['classes'][basepred[n].argmax()] if key=='A' else basepred[n], model.meta['classes'][pred[n].argmax()] if key=='A' else pred[n]])
 subgroup={}
 for g in groups['validation']:
  ix=np.array([group(r,key)==g for r in raws['validation']]);sub={**va,'x':va['x'][ix],'y':va['y'][ix]}
  subgroup[g]={'n':int(ix.sum()),'baseline':evaluate(model,sub),'selected':evaluate(final,sub)}
 summary={'dataset':key,'train_rows':len(tr['y']),'validation_rows':len(va['y']),'groups':groups,'input_features':len(tr['paths']),'parameters':model.meta['parameter_count'],'baseline':baseline,'selected':champ,'accepted_rounds':[r['round'] for r in records if r['accepted']],'training_seconds':sum(r['training_seconds'] for r in records),'train_sha256':tr['sha256'],'validation_sha256':va['sha256'],'baseline_checkpoint_sha256':checkpoint_digest(d/'checkpoint.npz'),'selected_checkpoint_sha256':checkpoint_digest(dest/'selected.npz'),'subgroups':subgroup,'rounds':records,'class_counts':{s:dict(Counter(r['answer'] for r in rows)) for s,rows in raws.items()} if key=='A' else None}
 (dest/'summary.json').write_text(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',default='ABC');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for key in a.tasks:run(a.release,a.output,key)
