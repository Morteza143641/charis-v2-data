import numpy as np
from icp_v2.preprocess import PreprocessConfig,detect_abp_beats,build_valid_segments,CausalMAPInput

def synthetic_abp(duration=600.0,fs=50.0,mean=100.0,hr_bpm=60.0):
    t=np.arange(0.0,duration,1/fs); phase=2*np.pi*(hr_bpm/60.0)*t; abp=mean+22*np.sin(phase)+6*np.sin(2*phase-0.3); return t,abp

def test_beat_map_is_integral_not_one_third_formula():
    t,abp=synthetic_abp(); cfg=PreprocessConfig(); beats=detect_abp_beats(t,abp,cfg); valid=[b for b in beats if b['valid']]
    assert len(valid)>500; maps=np.array([b['map_mmHg'] for b in valid[10:-10]]); assert abs(np.median(maps)-100.0)<0.15

def test_segments_and_burnin_possible_on_clean_signal():
    t,abp=synthetic_abp(); cfg=PreprocessConfig(); segs=build_valid_segments(detect_abp_beats(t,abp,cfg),cfg); assert len(segs)==1
    inp,meta=segs[0]; assert meta['duration_s']>500; assert meta['score_start_s']-meta['start_s']==cfg.burn_in_s

def test_causal_map_has_continuous_state_and_consistent_derivative():
    events=np.array([0.,1.,2.,3.]); maps=np.array([100.,110.,90.,100.]); inp=CausalMAPInput(events,maps,tau_s=5.0); eps=1e-6
    for x in [0.5,1.5,2.5]:
        fd=(inp.value(x+eps)-inp.value(x-eps))/(2*eps); assert abs(fd-inp.derivative(x))<1e-5
    for x in [1.0,2.0]: assert abs(inp.value(x-1e-9)-inp.value(x+1e-9))<1e-7

def test_invalid_beat_breaks_segment():
    t,abp=synthetic_abp(duration=500); cfg=PreprocessConfig(); beats=detect_abp_beats(t,abp,cfg); mid=len(beats)//2; beats[mid]['valid']=False; beats[mid]['reason']='synthetic_artifact'
    assert len(build_valid_segments(beats,cfg))==2
