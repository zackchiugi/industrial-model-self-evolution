from pathlib import Path
import json,time,argparse
from drift import inputs_only,reference,diagnose
from baselines.model import Model
p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
for key in 'ABC':
 d=args.release/('cpu-'+key.lower());m=Model.load(d/'checkpoint.npz');paths=m.meta['paths'];t=time.perf_counter()
 tr=inputs_only(d/'public/train.jsonl',paths);va=inputs_only(d/'public/validation.jsonl',paths);ref=reference(tr,paths)
 result={'dataset':key,'execution':'local_input_drift_audit','reference_sha256':ref['sha256'],'train':diagnose(ref,tr,paths),'validation':diagnose(ref,va,paths)}
 result['wall_seconds']=time.perf_counter()-t
 (args.output/(key+'.json')).write_text(json.dumps(result,indent=2))
 print(key,result['validation']['flagged_features'],result['validation']['ks_threshold'],result['validation']['features'][0],flush=True)
