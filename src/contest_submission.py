"""Model-guided, host-verified industrial checkpoint continuation."""
import os,json,math,tempfile,shutil,platform
from pathlib import Path
from functools import lru_cache
from drift import reference,diagnose
from baselines.model import Model,checkpoint_digest
from baselines.data import read_dataset
from baselines.predict import predict as baseline_predict
from adaptive_train import improve
TASKS=('classification','regression')

def parse_object(text):
    if not isinstance(text,str):return {}
    # Strip completed reasoning and a Markdown fence, never execute model text.
    text=text.rsplit('</think>',1)[-1].strip()
    if text.startswith('```'):
        text=text.split('\n',1)[-1].rsplit('```',1)[0].strip()
    try:
        obj=json.loads(text)
        return obj if isinstance(obj,dict) else {}
    except (ValueError,TypeError):return {}

def numeric_metric(receipt,name):
    summary=receipt.get('summary',{})
    value=summary.get('metrics',{}).get(name) if isinstance(summary,dict) else None
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None

def train(rnd):
    history=rnd.history()
    declared=rnd.parent.descriptor['checkpoint_paths']
    if len(declared)!=1:raise ValueError('Expected one NumPy checkpoint')
    parent_path=rnd.parent.path(declared[0]);parent=Model.load(parent_path);key=rnd.dataset_key
    if parent.meta.get('profile')!=key:raise ValueError('Parent profile mismatch')
    if rnd.task.get('task',rnd.task.get('task_type',parent.meta['task']))!=parent.meta['task']:raise ValueError('Task mismatch')
    work=Path(tempfile.mkdtemp(prefix='industrial-',dir=os.environ.get('CONTEST_WORK','/workspace')))
    state=rnd.state_workspace(work/'state');previous={}
    if (state/'decision.json').exists():previous=json.loads((state/'decision.json').read_text())
    train_data=read_dataset(rnd.data.train_path,parent.meta['task'],parent.meta['classes'],parent.meta['paths'])
    validation=read_dataset(rnd.data.validation_path,parent.meta['task'],parent.meta['classes'],parent.meta['paths'])
    if set(train_data['ids'])&set(validation['ids']):raise ValueError('Train-validation overlap')
    ref_file=state/'drift_reference.json'
    if ref_file.exists():ref=json.loads(ref_file.read_text())
    else:
        ref=reference(train_data['x'],parent.meta['paths']);ref_file.write_text(json.dumps(ref))
    drift={}
    for name,data in [('train',train_data),('validation',validation)]:
        d=diagnose(ref,data['x'],parent.meta['paths'])
        drift[name]={'alert':d['alert'],'flagged_features':d.get('flagged_features',0),'top_features':d['features'][:3]}
    # The model's previous-round advice affects the next bounded experiment.
    defaults={'A':(.001,100),'B':(.01,100),'C':(.003,80)}
    lr,epochs=defaults[key]
    suggested=previous.get('next_learning_rate')
    if isinstance(suggested,(int,float)) and not isinstance(suggested,bool) and .0001<=suggested<=.01:lr=float(suggested)
    if rnd.round_index>1:epochs={'A':20,'B':40,'C':25}[key]
    rates=sorted(set([lr, max(.0001,lr*.3)]))
    candidate=work/'candidate'
    report=improve(parent,train_data,validation,candidate,rates,epochs)
    config={'learning_rates':rates,'maximum_epochs_per_branch':epochs,'check_every':5,
            'selected_training_noise_std':report['selected_configuration']['training_noise_std'],
            'selected_learning_rate':report['selected_configuration']['learning_rate'],
            'selected_additional_epochs':report['selected_configuration']['additional_epochs'],
            'batch_size':parent.meta['batch_size'],'controller':'compact_model_decision_host_gate',
            'parent_checkpoint_sha256':checkpoint_digest(parent_path)}
    hyp='从父检查点及 Adam 状态继续训练；按公开验证指标选择学习率与训练轮次，提高漂移适应性。'
    changes='保留父模型参数、优化器和归一化；梯度仅使用训练集，公开验证集用于选优。'
    (candidate/'baselines').mkdir()
    code=['model.py','data.py','predict.py','__init__.py']
    for name in code:shutil.copyfile(Path(__file__).parent/'baselines'/name,candidate/'baselines'/name)
    submission=rnd.submit_candidate(candidate,hypothesis=hyp,change_summary=changes,configuration=config,
        seed=parent.meta['seed'],checkpoint_paths=['checkpoint.npz'],code_paths=['baselines/'+n for n in code],
        predict_argv=['python3','/app/harness.py','predict'])
    metric='macro_f1' if key=='A' else 'rmse'
    before=numeric_metric(rnd.parent.receipt,metric);after=numeric_metric(submission,metric)
    eligible=before is not None and after is not None and (after>before+1e-9 if key=='A' else after<before-1e-9)
    # Bounded summaries prevent recursive state/history prompt growth.
    past=[{'id':h['id'],'reason':str(h.get('reason',''))[:240]} for h in history[-3:]]
    request={'dataset':key,'round':rnd.round_index,'history':past,
        'previous':{k:previous.get(k) for k in ['accepted','parent_host_metric','candidate_host_metric','reason']},
        'drift':drift,'configuration':config,'metric':metric,'parent_host_metric':before,'candidate_host_metric':after,
        'eligible':eligible,'local_before':report['validation_before'],'local_after':report['validation_after'],
        'instruction':'Act as experiment controller. Review history, drift and this continuation experiment. Select candidate when eligible=true and measured improvement is credible; otherwise parent. Only host metrics authorize acceptance. Reply one JSON object: select (candidate or parent), reason (brief), next_learning_rate (0.0001 to 0.01). No reasoning tags.'}
    responses=[];advice={}
    for budget in [384,1024]:
        response=rnd.model([{'role':'user','content':json.dumps(request,ensure_ascii=False,separators=(',',':'))}],
            max_tokens=min(budget,rnd.max_output_tokens),temperature=0,enable_thinking=False,response_format={'type':'json_object'})
        responses.append({'call_id':response.call_id,'text':response.text,'usage':response.response.get('usage'),
            'finish_reason':response.response.get('choices',[{}])[0].get('finish_reason')})
        advice=parse_object(response.text)
        if advice.get('select') in ('candidate','parent'):break
        request['instruction']='Previous response was invalid. Return ONLY JSON: {"select":"candidate" or "parent","reason":"brief","next_learning_rate":0.001}. Choose candidate only when eligible=true.'
    accepted=bool(eligible and advice.get('select')=='candidate')
    selected=submission['artifact_id'] if accepted else rnd.parent.artifact_id
    why=('采纳：宿主验证指标改善。' if accepted else '保留父模型：候选未获宿主指标与模型决策同时认可。')
    why+=str(advice.get('reason','模型未返回有效选择，安全保留。'))[:500]
    evidence={'round':rnd.round_index,'accepted':accepted,'selected_artifact_id':selected,
        'parent_host_metric':before,'candidate_host_metric':after,'reason':why,
        'next_learning_rate':advice.get('next_learning_rate'), 'model_call_ids':list(rnd.call_ids),
        'history_ids':[h['id'] for h in history],'model_responses':responses,'configuration':config,
        'drift_diagnostics':drift,'training_seconds':report['elapsed_seconds'],'model_name':rnd.model_name,
        'parent_receipt':rnd.parent.receipt.get('summary'),'candidate_receipt':submission.get('summary'),
        'python_version':platform.python_version()}
    (state/'decision.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    (state/f'round_{rnd.round_index:04d}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    rnd.save_state(state)
    rnd.finish(selected_artifact_id=selected,hypothesis=hyp,change_summary=changes,reason=why,
        configuration=config,seed=parent.meta['seed'],history_ids=[h['id'] for h in history],model_call_ids=list(rnd.call_ids))

@lru_cache(maxsize=4)
def load_model(path):return Model.load(path)

@lru_cache(maxsize=4)
def checked_path(model_dir,artifact_path):
    root=Path(model_dir).resolve();path=Path(artifact_path).resolve()
    if not path.is_relative_to(root):raise ValueError('Checkpoint outside artifact')
    return str(path)

def predict(sample,model_dir):
    path=checked_path(str(model_dir),os.environ.get('CONTEST_ARTIFACT_PATH',str(Path(model_dir)/'checkpoint.npz')))
    model=load_model(path)
    if model.meta['task']!=sample.task:raise ValueError('Prediction task mismatch')
    if sample.dataset_key and model.meta.get('profile')!=sample.dataset_key:raise ValueError('Prediction profile mismatch')
    return baseline_predict(model,sample.inputs)['outputs']
