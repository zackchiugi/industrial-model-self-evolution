import unittest
import numpy as np
from drift import reference,diagnose
class DriftTests(unittest.TestCase):
 def setUp(self):
  self.x=np.random.default_rng(21).normal(size=(512,4)); self.paths=['a','b','c','d']; self.ref=reference(self.x,self.paths)
 def test_identity(self):
  d=diagnose(self.ref,self.x,self.paths); self.assertFalse(d['alert']); self.assertEqual(d['features'][0]['ks'],0)
 def test_shift_localization(self):
  x=self.x.copy();x[:,2]+=4;d=diagnose(self.ref,x,self.paths);self.assertTrue(d['alert']);self.assertEqual(d['features'][0]['index'],2)
 def test_schema(self):
  with self.assertRaises(ValueError):diagnose(self.ref,self.x,['wrong']*4)
 def test_nonfinite(self):
  x=self.x.copy();x[0,0]=np.nan
  with self.assertRaises(ValueError):diagnose(self.ref,x,self.paths)
 def test_small_window(self):self.assertEqual(diagnose(self.ref,self.x[:10],self.paths)['status'],'insufficient_data')
 def test_constant_shift(self):
  x=np.zeros((128,1)); d=diagnose(reference(x,['a']),x+1,['a']);self.assertTrue(d['alert']);self.assertGreater(d['features'][0]['psi'],0)
 def test_reference_immutable(self):
  import json
  before=json.dumps(self.ref);diagnose(self.ref,self.x+3,self.paths);self.assertEqual(before,json.dumps(self.ref))
if __name__=='__main__':unittest.main()
