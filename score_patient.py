#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.lib.stride_tricks import sliding_window_view
from icp_v2.wfdb_charis import parse_header, channel_index

REF_MIN=-50.0; REF_MAX=150.0; MIN_RAW=40

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def load_table(path): return np.genfromtxt(path,delimiter=',',names=True)

def reference_for_predictions(hea_path,dat_path,times):
    h=parse_header(hea_path); idx=channel_index(h,'ICP'); s=h['signals'][idx]; fs=h['fs_hz']
    if abs(fs-50.0)>1e-12: raise ValueError(f'frozen CHARIS fs is 50 Hz, got {fs}')
    raw=np.memmap(dat_path,dtype='<i2',mode='r'); expected=h['n_samples']*h['n_sig']
    if raw.size<expected: raise ValueError('DAT shorter than header')
    times=np.asarray(times,float); ref=np.full(len(times),np.nan); counts=np.zeros(len(times),int)
    for i,t in enumerate(times):
        start=int(round((t-0.5)*fs)); stop=int(round((t+0.5)*fs))
        if start<0 or stop>h['n_samples']: continue
        dig=np.asarray(raw[idx+start*h['n_sig']:idx+stop*h['n_sig']:h['n_sig']],dtype=np.float64)
        vals=(dig-s['baseline'])/s['gain']; good=np.isfinite(vals)&(vals>=REF_MIN)&(vals<=REF_MAX)
        counts[i]=int(np.sum(good))
        if counts[i]>=MIN_RAW: ref[i]=float(np.median(vals[good]))
    meta={'record':h['record'],'fs_hz':fs,'n_samples':h['n_samples'],'selected_channel':'ICP','selected_channel_index':idx,'gain':s['gain'],'baseline':s['baseline'],'unit':s['unit']}
    return ref,counts,meta

def finite_corr(x,y,kind='pearson'):
    if len(x)<2 or np.std(x)==0 or np.std(y)==0: return None
    if kind=='pearson': return float(np.corrcoef(x,y)[0,1])
    r=spearmanr(x,y).statistic; return None if not np.isfinite(r) else float(r)

def absolute_metrics(pred,ref):
    pred=np.asarray(pred,float); ref=np.asarray(ref,float); n=len(pred); d=pred-ref
    if n==0: return {'n':0}
    bias=float(np.mean(d)); sd=float(np.std(d,ddof=1)) if n>1 else float('nan')
    return {'n':int(n),'bias_mmHg':bias,'mae_mmHg':float(np.mean(np.abs(d))),'rmse_mmHg':float(np.sqrt(np.mean(d*d))),
            'median_abs_error_mmHg':float(np.median(np.abs(d))),'pearson_r':finite_corr(pred,ref,'pearson'),'spearman_r':finite_corr(pred,ref,'spearman'),
            'bland_altman_mean_diff_mmHg':bias,'bland_altman_loa_low_mmHg':float(bias-1.96*sd),'bland_altman_loa_high_mmHg':float(bias+1.96*sd)}

def ccc(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2: return None
    mx=x.mean(); my=y.mean(); vx=np.mean((x-mx)**2); vy=np.mean((y-my)**2); cov=np.mean((x-mx)*(y-my)); den=vx+vy+(mx-my)**2
    return None if den==0 else float(2*cov/den)

def trailing_median_strict(values,seg,window=60):
    values=np.asarray(values,float); seg=np.asarray(seg,int); out=np.full(len(values),np.nan)
    for s in np.unique(seg):
        idx=np.where(seg==s)[0]
        if len(idx)<window: continue
        win=sliding_window_view(values[idx],window); good=np.all(np.isfinite(win),axis=1); med=np.full(len(win),np.nan); med[good]=np.median(win[good],axis=1)
        out[idx[window-1:]]=med
    return out

def trend_metrics(pred,ref,time_s,seg,valid,h_seconds):
    pm=trailing_median_strict(pred,seg,60); rm=trailing_median_strict(ref,seg,60); pds=[]; rds=[]
    for s in np.unique(seg):
        idx=np.where(seg==s)[0]
        if len(idx)<=h_seconds: continue
        a=idx[:-h_seconds]; b=idx[h_seconds:]
        ok=(valid[a]&valid[b]&np.isfinite(pm[a])&np.isfinite(pm[b])&np.isfinite(rm[a])&np.isfinite(rm[b])&np.isclose(time_s[b]-time_s[a],h_seconds))
        pds.append(pm[b[ok]]-pm[a[ok]]); rds.append(rm[b[ok]]-rm[a[ok]])
    if not pds: return {'n_pairs':0}
    dp=np.concatenate(pds); dr=np.concatenate(rds); strong=np.abs(dr)>=2.0
    return {'n_pairs':int(len(dp)),'delta_mae_mmHg':float(np.mean(np.abs(dp-dr))),'spearman_r':finite_corr(dp,dr,'spearman'),'ccc':ccc(dp,dr),
            'direction_accuracy':float(np.mean(np.sign(dp)==np.sign(dr))),
            'direction_accuracy_ref_abs_delta_ge_2':(float(np.mean(np.sign(dp[strong])==np.sign(dr[strong]))) if np.any(strong) else None),
            'n_ref_abs_delta_ge_2':int(np.sum(strong))}

def sustained_runs(mask,time_s,seg,min_len=300):
    mask=np.asarray(mask,bool); t=np.asarray(time_s,float); seg=np.asarray(seg,int); runs=[]; start=None; prev=None; prevseg=None
    for i,yes in enumerate(mask):
        contiguous=(prev is not None and seg[i]==prevseg and abs(t[i]-prev-1.0)<1e-9)
        if yes:
            if start is None or not contiguous: start=i
        else:
            if start is not None:
                end=i-1
                if end-start+1>=min_len: runs.append((start,end))
                start=None
        prev=t[i]; prevseg=seg[i]
    if start is not None:
        end=len(mask)-1
        if end-start+1>=min_len: runs.append((start,end))
    return runs

def event_metrics(pred,ref,time_s,seg,valid,threshold):
    eligible=valid&np.isfinite(ref)&np.isfinite(pred); p=pred[eligible]; r=ref[eligible]; tt=time_s[eligible]; ss=seg[eligible]; y=(r>threshold)
    auroc=None; auprc=None
    if len(np.unique(y))==2: auroc=float(roc_auc_score(y,p)); auprc=float(average_precision_score(y,p))
    rr=sustained_runs(y,tt,ss,300); pr=sustained_runs(p>threshold,tt,ss,300); detections=[]; detected=0
    for a,b in rr:
        overlaps=[(c,d) for c,d in pr if not (d<a or c>b)]
        if overlaps: detected+=1; first=min(c for c,d in overlaps); ttd=max(0.0,float(tt[first]-tt[a])); det=True
        else: ttd=None; det=False
        detections.append({'reference_event':{'start_s':float(tt[a]),'end_s':float(tt[b]),'duration_s':int(b-a+1)},'detected':det,'time_to_detection_s':ttd})
    false=sum(1 for c,d in pr if not any(not (d<a or c>b) for a,b in rr)); hours=len(p)/3600.0
    return {'threshold_mmHg':float(threshold),'n_timepoints':int(len(p)),'reference_positive_fraction':float(np.mean(y)) if len(y) else None,
            'timepoint_auroc':auroc,'timepoint_auprc':auprc,'n_reference_sustained_events':int(len(rr)),'n_predicted_sustained_events':int(len(pr)),
            'event_sensitivity':(float(detected/len(rr)) if rr else None),'false_alarm_events':int(false),
            'false_alarm_events_per_hour':(float(false/hours) if hours>0 else None),'detections':detections}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('header');ap.add_argument('dat');ap.add_argument('blind_csv');ap.add_argument('baseline_csv');ap.add_argument('output_dir'); args=ap.parse_args()
    hea=Path(args.header);dat=Path(args.dat);blindp=Path(args.blind_csv);basep=Path(args.baseline_csv);outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    d=load_table(blindp); b=load_table(basep)
    if len(d)!=len(b) or not np.array_equal(d['time_s'],b['time_s']): raise ValueError('blind/baseline alignment mismatch')
    ref,raw_n,src=reference_for_predictions(hea,dat,d['time_s']); valid=d['valid_for_score'].astype(int)==1; paired=valid&np.isfinite(ref)
    paired_path=outdir/f"{src['record']}_V2_paired_1Hz.csv"
    with open(paired_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['time_s','segment_id','valid_for_score','V2_pred_mmHg','B0_mmHg','B1_mmHg','ICP_ref_median_mmHg','reference_raw_valid_n'])
        for i in range(len(d)):
            w.writerow([float(d['time_s'][i]),int(d['segment_id'][i]),int(d['valid_for_score'][i]),float(d['predicted_icp_mmHg'][i]),float(b['B0_constant_9p5_mmHg'][i]),float(b['B1_static_fixedC_mmHg'][i]),float(ref[i]) if np.isfinite(ref[i]) else float('nan'),int(raw_n[i])])
    pred=d['predicted_icp_mmHg']; b0=b['B0_constant_9p5_mmHg']; b1=b['B1_static_fixedC_mmHg']; seg=d['segment_id'].astype(int); tt=d['time_s']
    score={'unlock':{'invasive_icp_loaded':True,'source':src,'reference_gate_mmHg':[REF_MIN,REF_MAX],'min_valid_raw_samples_per_1s_bin':MIN_RAW,
                     'reference_aggregation':'median of centered [t-0.5,t+0.5) 50-Hz samples','lag_optimization':False},
           'coverage':{'prediction_rows':int(len(d)),'blind_score_eligible_rows':int(np.sum(valid)),'valid_reference_rows_among_eligible':int(np.sum(paired)),
                       'paired_rows':int(np.sum(paired)),'reference_qc_excluded_eligible_rows':int(np.sum(valid&~np.isfinite(ref))),'paired_hours':float(np.sum(paired)/3600.0)},
           'absolute':{'V2':absolute_metrics(pred[paired],ref[paired]),'B0_constant_9p5':absolute_metrics(b0[paired],ref[paired]),'B1_static_fixedC':absolute_metrics(b1[paired],ref[paired])},
           'trend':{},'events':{},'frozen_hashes':{'prediction_sha256':sha256_file(blindp),'baseline_sha256':sha256_file(basep),'scoring_freeze_sha256':sha256_file(Path(__file__).with_name('SCORING_FREEZE.md')),
                                                 'dat_sha256':sha256_file(dat),'header_sha256':sha256_file(hea)}}
    for mins in [5,15,30]: score['trend'][f'{mins}min']=trend_metrics(pred,ref,tt,seg,valid,mins*60)
    for thr in [22.0,20.0]: score['events'][f'{int(thr)}mmHg']=event_metrics(pred,ref,tt,seg,valid,thr)
    score_path=outdir/f"{src['record']}_V2_score.json";score_path.write_text(json.dumps(score,indent=2),encoding='utf-8')
    print(json.dumps({'score':str(score_path),'paired':str(paired_path),'coverage':score['coverage'],'absolute':score['absolute'],'trend':score['trend']},indent=2))

if __name__=='__main__':main()
