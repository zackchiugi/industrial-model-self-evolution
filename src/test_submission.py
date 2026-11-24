import unittest,tempfile,json,os,io,hashlib,base64
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from contest_harness.round import Round
from contest_harness.predict import Sample,normalize_output
from baselines.model import Model,checkpoint_digest
import contest_submission as sub
ROOT=Path(os.environ.get('DRIFTGUARD_RELEASE','release')).resolve()
class Loopback:
    """In-process protocol host. Model replies and scores are explicit test doubles."""
    def __init__(self,mode):self.queue=[];self.frames=[];self.calls=0;self.mode=mode
    def write(self,line):
        if not line.strip():return
        f=json.loads(line);self.frames.append(f);t=f['type'];r={'id':f.get('id')}
        if t=='history_request':r.update(type='history_response',records=[{'id':'prior-1','note':'mock history'}])
        elif t=='model_request':
            self.calls+=1;payload={'epochs':1,'learning_rate':0.001,'hypothesis':'Test continuation.'} if self.calls==1 else {'select':'candidate','reason':'mock comparison'}
            r.update(type='model_response',call_id=f'mock-{self.calls}',response={'choices':[{'message':{'content':json.dumps(payload)}}]})
        elif t=='upload_begin_request':r.update(type='upload_begin_response',upload_id='mock-upload')
        elif t=='upload_chunk_request':r.update(type='upload_chunk_response',offset=f['offset']+len(base64.b64decode(f['data_base64'])))
        elif t=='upload_finish_request':r.update(type='upload_finish_response',upload_id=f['upload_id'])
        elif t=='candidate_request':r.update(type='candidate_response',artifact_id='candidate-test',summary={'metrics':{'macro_f1':0.8 if self.mode=='accept' else 0.4}} if self.mode!='missing' else {})
        elif t=='state_request':r.update(type='state_response',state_sha256='sha256:'+'a'*64)
        elif t=='round_complete':r={'type':'round_ack','accepted':True}
        else:raise AssertionError(t)
        self.queue.append(json.dumps(r)+'\n')
    def flush(self):pass
    def readline(self):return self.queue.pop(0)
class Tests(unittest.TestCase):
    def protocol(self,mode):
        d=ROOT/'cpu-a';p=d/'checkpoint.npz';host=Loopback(mode)
        frame={'resource_limits':{'artifact_upload_protocol':'chunked-v1'},'protocol':'contest.harness.workspace.v2','prompt':'test','prompt_version':'test','dataset_key':'A','round_index':2,'task_contract':{'task':'classification','public_projection':{'train_path':str(d/'public/train.jsonl'),'validation_path':str(d/'public/validation.jsonl')}},'model_name':'test-double','max_output_tokens':1024,'parent_checkpoint_sha256':checkpoint_digest(p),'parent_artifact':{'artifact_id':'parent-test','artifact':{'checkpoint_paths':['checkpoint.npz'],'code_paths':[],'predict_argv':[]},'files':[{'path':'checkpoint.npz','sha256':checkpoint_digest(p)}],'summary':{'metrics':{'macro_f1':0.7}}},'artifacts':[]}
        rnd=Round(frame,host,host);rnd.parent.root=d
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'CONTEST_WORK':temp}):sub.train(rnd)
        rec=host.frames[-1]['record'];self.assertEqual(rec['selected_artifact_id'],'candidate-test' if mode=='accept' else 'parent-test');self.assertEqual(rec['model_call_ids'],['mock-1','mock-2']);self.assertEqual(rec['history_ids'],['prior-1']);self.assertTrue(rnd._finished)
    def test_protocol_accept(self):self.protocol('accept')
    def test_protocol_reject(self):self.protocol('reject')
    def test_unknown_metrics_retain_parent(self):self.protocol('missing')
    def test_bad_model_json(self):self.assertEqual(sub.parse_object('```json\n{}\n```'),{})
    def test_all_validation_predictions(self):
        for key in 'ABC':
            d=ROOT/('cpu-'+key.lower());m=Model.load(d/'checkpoint.npz');context={'dataset_key':key,'artifact_id':'local'}
            for line in (d/'public/validation.jsonl').read_text().splitlines():
                sample=Sample(m.meta['task'],json.loads(line)['inputs'],context)
                normalize_output(sample.task,sub.predict(sample,d))
    def test_parent_digest_tampering(self):
        from contest_harness.round import ParentView
        p=ROOT/'cpu-a/checkpoint.npz';v=ParentView({'parent_checkpoint_sha256':'sha256:'+'0'*64,'parent_artifact':{'artifact_id':'x','artifact':{'checkpoint_paths':['checkpoint.npz'],'code_paths':[]},'files':[{'path':'checkpoint.npz','sha256':'0'*64}]}});v.root=p.parent
        with self.assertRaises(Exception):v.path('checkpoint.npz')
if __name__=='__main__':unittest.main()
