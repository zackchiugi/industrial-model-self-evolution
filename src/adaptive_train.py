"""Bounded continuation search. Gradients use training rows only."""
import copy,time,json
import numpy as np
from pathlib import Path
from baselines.model import checkpoint_digest
from baselines.train import metrics

def improve(parent, train, validation, output, learning_rates, epochs, check_every=5):
    output=Path(output);output.mkdir()
    key=parent.meta['profile'];metric='macro_f1' if key=='A' else 'rmse'
    baseline=metrics(parent,validation)
    best=None; best_value=None; records=[]; started=time.monotonic()
    strategies=[(lr,0.0) for lr in learning_rates]
    if key=='A':strategies.append((.003,.3))
    for lr,noise in strategies:
        model=copy.deepcopy(parent);model.meta['learning_rate']=lr
        for epoch in range(1,epochs+1):
            x=train['x']
            if noise:
                rng=np.random.default_rng(np.random.SeedSequence([parent.meta['seed'],model.meta['epochs'],103]))
                x=x+rng.normal(size=x.shape)*parent.scale*noise
            loss=model.fit(x,train['y'],1)[0]
            if epoch%check_every and epoch!=epochs:continue
            score=metrics(model,validation);value=score[metric]
            records.append({'learning_rate':lr,'training_noise_std':noise,'additional_epochs':epoch,'steps':model.meta['steps'],'loss':loss,'validation':score})
            if best_value is None or (value>best_value if key=='A' else value<best_value):
                best=copy.deepcopy(model);best_value=value;chosen=records[-1]
    best.save(output/'checkpoint.npz')
    report={'profile':key,'parent_initial_steps':parent.meta['steps'],'parent_initial_epochs':parent.meta['epochs'],
            'selected_steps':best.meta['steps'],'selected_epochs':best.meta['epochs'],'selected_configuration':chosen,
            'validation_before':baseline,'validation_after':metrics(best,validation),'training_rows':len(train['y']),
            'validation_rows':len(validation['y']),'train_sha256':train['sha256'],'validation_sha256':validation['sha256'],
            'checkpoint_sha256':checkpoint_digest(output/'checkpoint.npz'),'elapsed_seconds':time.monotonic()-started,
            'records':records,'method':'Resume parent weights and Adam moments; select epoch on public validation; no validation gradients.'}
    (output/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
