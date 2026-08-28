from __future__ import annotations
import numpy as np
from .model import ModelParams
from .preprocess import PreprocessConfig, detect_abp_beats, build_valid_segments
from .runtime import simulate_rk4_fixed


def predict_from_abp_arrays_runtime(time_s,abp_mmHg,pcfg:PreprocessConfig=PreprocessConfig(),params:ModelParams=ModelParams(),dt:float=0.02):
    t=np.asarray(time_s,float); abp=np.asarray(abp_mmHg,float); beats=detect_abp_beats(t,abp,pcfg); segments=build_valid_segments(beats,pcfg)
    rows=[]; segment_meta=[]; step=1.0/pcfg.output_prediction_hz
    for sid,(inp,meta) in enumerate(segments):
        first=np.ceil(inp.start_s/step)*step; last=np.floor(inp.end_s/step)*step
        if last<=first: continue
        tt=np.arange(first,last+0.5*step,step)
        states,audit=simulate_rk4_fixed(tt,(9.5,params.C_an),inp.value,inp.derivative,p=params,autoregulation=True,dt=dt)
        pred=states[0]; A=np.array([inp.value(float(x)) for x in tt]); Adot=np.array([inp.derivative(float(x)) for x in tt]); valid=(tt>=meta['score_start_s']).astype(int)
        for values in zip(tt,pred,A,Adot,valid): rows.append((float(values[0]),float(values[1]),float(values[2]),float(values[3]),int(sid),int(values[4])))
        segment_meta.append({'segment_id':sid,**meta,**audit})
    dtype=[('time_s','f8'),('predicted_icp_mmHg','f8'),('A_map_mmHg','f8'),('A_dot_mmHg_s','f8'),('segment_id','i4'),('valid_for_score','i1')]
    result=np.array(rows,dtype=dtype)
    meta={'n_beats_total':len(beats),'n_beats_valid':int(sum(bool(b['valid']) for b in beats)),'n_segments':len(segment_meta),'segments':segment_meta,
          'n_prediction_rows':len(result),'n_score_eligible_samples':int(np.sum(result['valid_for_score'])) if len(result) else 0,
          'output_prediction_hz':pcfg.output_prediction_hz,'retained_segment_duration_h':float(sum(m['duration_s'] for m in segment_meta)/3600.0),
          'runtime_solver_version':'V2-runtime-solver-freeze-1','runtime_solver':'classical RK4 dt=0.02 s; validated against frozen DOP853 before ICP unlock'}
    return result,meta,beats
