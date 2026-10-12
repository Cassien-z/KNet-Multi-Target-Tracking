from __future__ import annotations
import unittest
import numpy as np
from scripts.data_generation.prepare_reference_pilot import transform_observations, fit_calibration, build_payload, measured_reference_origin


class ReferencePilotConversionTests(unittest.TestCase):
    def test_pitch_branch_and_measured_xyz_roundtrip(self):
        # -150 degrees encodes positive 30-degree elevation with north-positive bearing.
        r=10000.; own=np.array([[3000000.,4000000.,4000000.]])
        raw=np.array([[r,-150.,20.,0.,0.,0.]])
        identity=np.eye(3)[None]
        result=transform_observations(raw,own,identity,identity,full=True)
        expected=r*np.array([[np.cos(np.pi/6)*np.sin(np.radians(20.)),np.cos(np.pi/6)*np.cos(np.radians(20.)),.5]])
        np.testing.assert_allclose(result['Meas_ECEF']-own,expected,atol=1e-8)
        np.testing.assert_allclose(result['Meas'][0,2],np.pi/6,atol=1e-12)

    def test_missing_range_keeps_bearing_without_fabricating_position(self):
        own=np.array([[3e6,4e6,4e6],[3e6+100,4e6,4e6]])
        raw=np.column_stack((np.zeros(2),[-30.,0.],[60.,0.],own))
        rotation=np.repeat(np.eye(3)[None],2,axis=0)
        result=transform_observations(raw,own,rotation,rotation,full=False)
        np.testing.assert_array_equal(result['valid'],[True,False])
        self.assertGreater(np.linalg.norm(result['Meas'][0]),0.)
        np.testing.assert_array_equal(result['Meas_ECEF'],np.zeros((2,3)))

    def test_zero_measurement_with_nonzero_xyz_remains_invalid(self):
        raw=np.array([[0.,0.,0.,3e6,4e6,4e6]])
        result=transform_observations(raw,np.zeros((1,3)),np.eye(3)[None],np.eye(3)[None],full=True)
        self.assertFalse(result['valid'][0])
        np.testing.assert_array_equal(result['Meas'],np.zeros((1,3)))

    def test_validation_values_cannot_change_calibration(self):
        raw=np.zeros((3,2,6));raw[:,:,0]=1010.;raw[:,:,2]=90.
        own=np.zeros((3,3));target=np.zeros((3,2,3));target[:,:,0]=1000.
        rotation=np.repeat(np.eye(3)[None],3,axis=0);mask=np.zeros((3,2),bool);mask[:,0]=True
        first=fit_calibration(raw,own,target,rotation,mask)
        raw[:,1,:3]=[99999.,80.,-45.];target[:,1]=[9e6,8e6,7e6]
        second=fit_calibration(raw,own,target,rotation,mask)
        self.assertEqual(first,second)
        self.assertEqual(first['range_bias_m'],10.)

    def test_slots_share_ownship_and_no_links_or_esm_are_invented(self):
        n=3;own=np.full((n,3),3e6);rot=np.repeat(np.eye(3)[None],n,axis=0)
        def stream(w):return {'Meas':np.zeros((n,w)),'Meas_ECEF':np.zeros((n,3)),'valid':np.zeros(n,bool)}
        sources={name:stream(3 if name.startswith('全维') else 2) for name in ['全维1','全维2','全维3','全维4','缺维1','缺维2']}
        payload=build_payload(own+100.,np.ones((n,3)),own,rot,sources,{})
        for slot in range(5):np.testing.assert_array_equal(payload['UAV_Pos'][:,slot],own)
        self.assertFalse(any(x['valid'].any() for x in payload['RegisteredTrack'][0]['FastLink']))
        np.testing.assert_array_equal(payload['ReferenceOrigin'],own[0])


    def test_model_inputs_do_not_depend_on_target_truth_origin(self):
        import pickle
        import tempfile
        from pathlib import Path
        from src.reference_pilot_dataset import ReferencePilotDataset
        import torch
        n=4;own=np.full((n,3),3e6);own[:,0]+=np.arange(n)*100.
        rot=np.repeat(np.eye(3)[None],n,axis=0)
        def stream(w):return {'Meas':np.zeros((n,w)),'Meas_ECEF':np.zeros((n,3)),'valid':np.zeros(n,bool)}
        sources={name:stream(3 if name.startswith('全维') else 2) for name in ['全维1','全维2','全维3','全维4','缺维1','缺维2']}
        metadata={'source_target_id':1,'source_start_frame':0}
        payload=build_payload(own+1000.,np.ones((n,3)),own,rot,sources,metadata)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'case.pkl'
            with path.open('wb') as f:pickle.dump(payload,f)
            first=ReferencePilotDataset(temp,n)[0]
            payload['RegisteredTrack'][0]['target_pos']+=50000.
            payload['RegisteredTrack'][0]['target_vel']*=12.
            with path.open('wb') as f:pickle.dump(payload,f)
            second=ReferencePilotDataset(temp,n)[0]
        for key in ['radar','eo','esm','radar_mask','eo_mask','esm_mask','uav_pos','rotmat_b2e','ref_pos','fast_link','slow_link','fast_update','slow_update','fast_age','slow_age','imu','dt']:
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(first['label_abs'],second['label_abs']))



    def test_initialization_waits_for_actual_observation(self):
        own=np.full((4,3),3e6)
        xyz=own+1234.
        valid=np.array([False,False,True,True])
        sources={name:{'valid':valid.copy(),'Meas_ECEF':xyz.copy()} for name in ['全维1','全维2','全维3','全维4']}
        origin,metadata=measured_reference_origin(own,sources)
        self.assertEqual(metadata['frame'],2)
        self.assertEqual(metadata['sensor'],'全维1')
        np.testing.assert_array_equal(origin,xyz[2])
        self.assertFalse(np.array_equal(origin,own[0]))


if __name__=='__main__':unittest.main()
