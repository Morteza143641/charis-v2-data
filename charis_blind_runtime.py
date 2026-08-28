#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
from icp_v2.model import ModelParams
from icp_v2.preprocess import PreprocessConfig
from icp_v2.predictor_runtime import predict_from_abp_arrays_runtime
from icp_v2.wfdb_charis import load_abp_only


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def write_predictions(path,result):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(result.dtype.names)
        for row in result: w.writerow([row[n].item() for n in result.dtype.names])


def main():
    ap=argparse.ArgumentParser(description='Frozen blind CHARIS ABP-only V2 prediction')
    ap.add_argument('header'); ap.add_argument('dat'); ap.add_argument('output_dir')
    args=ap.parse_args(); hea=Path(args.header); dat=Path(args.dat); outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    t,abp,source=load_abp_only(hea,dat)
    st=time.time()
    result,meta,_=predict_from_abp_arrays_runtime(t,abp,PreprocessConfig(),ModelParams(),dt=0.02)
    wall=time.time()-st
    out=outdir/f"{source['record']}_V2_blind.csv"; write_predictions(out,result)
    manifest={'source':source,'header_sha256':sha256_file(hea),'dat_sha256':sha256_file(dat),
              'prediction_sha256':sha256_file(out),'model_version':'V2',
              'preprocessing_version':'V2-preprocess-freeze-1','runtime_solver_version':'V2-runtime-solver-freeze-1',
              'runtime_solver':'classical RK4 dt=0.02 s; validated against frozen DOP853 before ICP unlock',
              'model_params':ModelParams().to_dict(),'preprocess_config':PreprocessConfig().to_dict(),
              **meta,'invasive_icp_values_loaded':False,'scoring_performed':False,'wall_runtime_s':wall}
    mp=outdir/f"{source['record']}_V2_blind.manifest.json"; mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'prediction':str(out),'manifest':str(mp),**{k:meta[k] for k in ['n_beats_total','n_beats_valid','n_segments','n_prediction_rows','n_score_eligible_samples','retained_segment_duration_h']},'wall_runtime_s':wall},indent=2))

if __name__=='__main__': main()
