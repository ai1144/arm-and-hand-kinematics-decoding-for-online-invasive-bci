"""Numerical regressions; no patient data required. Run python -m unittest discover -s tests."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from decoder_core.data import Trial
from run_split_pca_model_suite import kalman_filter_trial,temporal_metrics_by_trial,phase_features,SequenceNetFactory
from run_strict_direct_output_suite import qpos_from_target_prediction

class ProtocolTests(unittest.TestCase):
    def test_kalman_matches_observation_space_update(self):
        rng=np.random.default_rng(8);d,c,n=5,13,21
        p={'A':(.7*np.eye(d)).astype('f'),'A_b':rng.normal(size=d).astype('f')*.01,
           'H':rng.normal(size=(c,d)).astype('f'),'H_b':rng.normal(size=c).astype('f')*.01,
           'Q_diag':np.full(d,.04,dtype='f'),'R_diag':np.full(c,.3,dtype='f'),
           'z_mean':np.zeros(d,dtype='f')}
        obs=rng.normal(size=(n,c)).astype('f');z=p['z_mean'].astype(float);cov=np.eye(d);expected=[]
        for x in obs:
            prior=p['A']@z+p['A_b'];cov=p['A']@cov@p['A'].T+np.diag(p['Q_diag'])
            gain=np.linalg.solve(p['H']@cov@p['H'].T+np.diag(p['R_diag']),p['H']@cov).T
            z=prior+gain@(x-(p['H']@prior+p['H_b']));cov=(np.eye(d)-gain@p['H'])@cov;expected.append(z.copy())
        actual=kalman_filter_trial(p,obs)
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)
        np.testing.assert_array_equal(actual,kalman_filter_trial(p,obs))
    def test_trial_boundaries_do_not_enter_jitter(self):
        a=np.arange(20,dtype='f')[:,None]*np.ones((1,32),dtype='f')
        b=a+1000
        result=temporal_metrics_by_trial([a,b],[a,b],100)
        self.assertAlmostEqual(result['jitter'],0.)
        self.assertEqual(result['jitter_valid_trials'],2)
    def test_metrics_do_not_mutate_inputs(self):
        a=np.arange(640,dtype=float).reshape(20,32);before=a.copy()
        temporal_metrics_by_trial([a],[a.copy()],100)
        np.testing.assert_array_equal(a,before)
    def test_velocity_starts_at_initial_posture(self):
        class Stub: qpos=np.full((4,32),3.,dtype='f')
        actual=qpos_from_target_prediction(np.ones((4,32),dtype='f'),Stub(),'qvel',100)
        np.testing.assert_allclose(actual[:,0],[3.,3.01,3.02,3.03],atol=1e-6)
    def test_fixed_clock_is_independent_of_trial_end(self):
        class Stub:
            def __init__(self,n):self.n_time=n
        np.testing.assert_array_equal(phase_features(Stub(30)),phase_features(Stub(300))[:30])
    def test_future_neural_bins_cannot_change_previous_outputs(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(7)
        for family in ['gru','lfads_lite','transformer']:
            model=SequenceNetFactory.make(torch,torch.nn,family,7,32,32,12,False).eval()
            x=torch.randn(1,20,7);changed=x.clone();changed[:,10:]+=10
            mask=torch.ones((1,20),dtype=torch.bool)
            with torch.no_grad():a=model(x,mask)['y'];b=model(changed,mask)['y']
            np.testing.assert_allclose(a[:,:10].numpy(),b[:,:10].numpy(),atol=2e-6,rtol=2e-6)

if __name__=='__main__':unittest.main()
