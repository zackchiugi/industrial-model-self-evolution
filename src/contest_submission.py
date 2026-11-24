"""Bounded model-guided continuation using contest_harness 1.0.2.
Local experiment results do not establish real-gateway performance.
"""
import os,json,math,tempfile,shutil,time,platform
from pathlib import Path
from functools import lru_cache
from drift import inputs_only,reference,diagnose
from baselines.model import Model,checkpoint_digest
from baselines.train import train_run,metrics
from baselines.data import read_dataset
from baselines.predict import predict as baseline_predict
TASKS=('classification','regression')

def parse_object(text):
    try:
        obj=json.loads(text)
        return obj if isinstance(obj,dict) else {}
    except (ValueError,TypeError):return {}

def numeric_metric(receipt, name):
    # SDK examples expose summary.metrics. Unknown host schemas fail closed.
    value=receipt.get('summary',{}).get('metrics',{}).get(name)
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None

def train(rnd):
    history=rnd.history()
    declared=rnd.parent.descriptor['checkpoint_paths']
    if len(declared)!=1:raise ValueError('Expected exactly one compatible NumPy checkpoint')
    parent_path=rnd.parent.path(declared[0]);parent=Model.load(parent_path)
    key=rnd.dataset_key
    if parent.meta.get('profile')!=key:raise ValueError('Parent profile mismatch')
    if rnd.task.get('task',rnd.task.get('task_type',parent.meta['task']))!=parent.meta['task']:raise ValueError('Task contract mismatch')
    work=Path(tempfile.mkdtemp(prefix='driftguard-',dir=os.environ.get('CONTEST_WORK','/workspace')))
    state=rnd.state_workspace(work/'state')
    previous={}
    if (state/'decision.json').exists():previous=json.loads((state/'decision.json').read_text())
    ref_file=state/'drift_reference.json'
    train_x=inputs_only(rnd.data.train_path,parent.meta['paths'])
    if ref_file.exists(): ref=json.loads(ref_file.read_text())
    else:
        ref=reference(train_x,parent.meta['paths'])
        ref_file.write_text(json.dumps(ref))
    drift_train=diagnose(ref,train_x,parent.meta['paths'])
    drift_validation=diagnose(ref,inputs_only(rnd.data.validation_path,parent.meta['paths']),parent.meta['paths'])
    drift_summary={name:{**{k:v for k,v in obj.items() if k!='features'},'top_features':obj['features'][:5]} for name,obj in [('train_vs_frozen_reference',drift_train),('validation_vs_frozen_reference',drift_validation)]}
    lr=float(previous.get('learning_rate',parent.meta['learning_rate']))
    if previous:lr*=0.8 if previous.get('accepted') else 0.5
    lr=max(1e-7,min(0.01,lr))
    request={'dataset':key,'round':rnd.round_index,'history':history[-10:],'drift_diagnostics':drift_summary,'previous_decision':previous,'proposal':{'epochs':5 if drift_validation.get('alert') or drift_train.get('alert') else 1,'learning_rate':lr},'allowed_epochs':[1,3,5], 'instruction':'Read history and propose a continuation experiment. Reply with JSON only: epochs (1,3,5), learning_rate (1e-7 to 0.01), hypothesis. Keep the parent weights, Adam state and normalization. No code execution or new data.'}
    plan=rnd.model([{'role':'user','content':json.dumps(request,ensure_ascii=False)}],max_tokens=min(512,rnd.max_output_tokens),temperature=0)
    proposal=parse_object(plan.text);epochs=proposal.get('epochs',5)
    if type(epochs)!=int or epochs not in [1,3,5]:epochs=5
    proposed_lr=proposal.get('learning_rate',lr)
    if isinstance(proposed_lr,(int,float)) and not isinstance(proposed_lr,bool) and math.isfinite(proposed_lr) and 1e-7<=proposed_lr<=0.01:lr=float(proposed_lr)
    hyp=proposal.get('hypothesis','Five-epoch continuation with history-based learning-rate control.')
    if not isinstance(hyp,str) or not hyp.strip():hyp='Continue selected parent with bounded optimizer settings.'
    hyp=hyp[:500]
    candidate=work/'candidate'
    fit_started=time.perf_counter()
    report=train_run(key,rnd.data.train_path,rnd.data.validation_path,candidate,resume=parent_path,epochs=epochs,learning_rate=lr,stage_id=f'round-{rnd.round_index}')
    fit_wall_seconds=time.perf_counter()-fit_started
    config={'epochs':epochs,'learning_rate':lr,'batch_size':parent.meta['batch_size'],'controller':'bounded_llm_advice_with_metric_gate','parent_checkpoint_sha256':checkpoint_digest(parent_path)}
    changes='Continue parent weights and Adam state; preserve original input and target scaling.'
    # Include actual inference source as auditable artifact files; /app provides identical code.
    (candidate/'baselines').mkdir()
    for name in ['model.py','data.py','predict.py','__init__.py']:
        shutil.copyfile(Path(__file__).parent/'baselines'/name,candidate/'baselines'/name)
    submission=rnd.submit_candidate(candidate,hypothesis=hyp,change_summary=changes,configuration=config,seed=parent.meta['seed'],checkpoint_paths=['checkpoint.npz'],code_paths=['baselines/'+n for n in ['model.py','data.py','predict.py','__init__.py']],predict_argv=['python3','/app/harness.py','predict'])
    metric='macro_f1' if key=='A' else 'rmse'
    before=numeric_metric(rnd.parent.receipt,metric);after=numeric_metric(submission,metric)
    eligible=before is not None and after is not None and (after>before+1e-12 if key=='A' else after<before-1e-12)
    decision=rnd.model([{'role':'user','content':json.dumps({'instruction':'Read the experiment and host measurements. Reply JSON with select: candidate or parent, and reason. Candidate may only be accepted if eligible is true. Explain the evidence.', 'hypothesis':hyp,'configuration':config,'parent_host_receipt':rnd.parent.receipt.get('summary'),'candidate_host_receipt':submission.get('summary'),'local_training_diagnostics':report,'eligible':eligible},ensure_ascii=False)}],max_tokens=min(512,rnd.max_output_tokens),temperature=0)
    advice=parse_object(decision.text)
    accepted=bool(eligible and advice.get('select')=='candidate')
    selected=submission['artifact_id'] if accepted else rnd.parent.artifact_id
    why='Accepted a strict host-measured primary-metric improvement.' if accepted else 'Retained parent: missing/non-improving host metrics or model did not select candidate.'
    explanation=advice.get('reason','')
    if isinstance(explanation,str):why+=' Model rationale: '+explanation[:700]
    evidence={'round':rnd.round_index,'learning_rate':lr,'accepted':accepted,'selected_artifact_id':selected,'parent_host_metric':before,'candidate_host_metric':after,'reason':why,'model_call_ids':list(rnd.call_ids),'history_ids':[x['id'] for x in history],'plan_response':plan.text,'decision_response':decision.text,'configuration':config,'drift_diagnostics':drift_summary,'fit_and_local_evaluation_wall_seconds':fit_wall_seconds,'model_name':rnd.model_name,'model_usage':[plan.response.get('usage'),decision.response.get('usage')],'python_version':platform.python_version()}
    (state/'decision.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    (state/f'round_{rnd.round_index:04d}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    rnd.save_state(state)
    rnd.finish(selected_artifact_id=selected,hypothesis=hyp,change_summary=changes,reason=why,configuration=config,seed=parent.meta['seed'],history_ids=[x['id'] for x in history],model_call_ids=list(rnd.call_ids))

@lru_cache(maxsize=4)
def load_model(path):return Model.load(path)

def predict(sample,model_dir):
    root=Path(model_dir).resolve();path=Path(os.environ.get('CONTEST_ARTIFACT_PATH',str(root/'checkpoint.npz'))).resolve()
    if not path.is_relative_to(root):raise ValueError('Checkpoint must be inside selected artifact')
    model=load_model(str(path))
    if model.meta['task']!=sample.task:raise ValueError('Prediction task mismatch')
    if sample.dataset_key and model.meta.get('profile')!=sample.dataset_key:raise ValueError('Prediction profile mismatch')
    return baseline_predict(model,sample.inputs)['outputs']
