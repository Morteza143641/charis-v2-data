#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from numba import njit, prange

# Frozen V2 population parameters. Equations are unchanged from icp_v2.model.
R_O_POP = 526.3
R_F = 2.38e3
R_PV = 1.24
C_AN = 0.15
K_E = 0.11
K_R = 4.91e4
Q_N = 12.5
TAU = 20.0
G = 1.5
P_VS = 6.0
P0 = 9.5
GRID = np.logspace(np.log10(0.1), np.log10(20.0), 80)
SUBSTEPS = 2  # 0.5 s RK4; 1-Hz Hermite replay validated against frozen primary outputs.


def load_csv(path):
    with open(path,'r',encoding='utf-8') as f:
        names=f.readline().strip().split(',')
    arr=np.genfromtxt(path,delimiter=',',skip_header=1,dtype=float,filling_values=np.nan,ndmin=2)
    return {name:arr[:,i] for i,name in enumerate(names)}


def corr(x, y, kind='pearson'):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2 or np.std(x)==0 or np.std(y)==0: return None
    if kind=='pearson': return float(np.corrcoef(x,y)[0,1])
    r=spearmanr(x,y).statistic
    return None if not np.isfinite(r) else float(r)


def metrics(pred, ref):
    pred=np.asarray(pred,float); ref=np.asarray(ref,float); d=pred-ref
    return {
        'n': int(len(d)),
        'bias_mmHg': float(np.mean(d)),
        'mae_mmHg': float(np.mean(np.abs(d))),
        'rmse_mmHg': float(np.sqrt(np.mean(d*d))),
        'pearson_r': corr(pred,ref,'pearson'),
        'spearman_r': corr(pred,ref,'spearman'),
    }


@njit(cache=True)
def _hermite(a0,a1,d0,d1,s):
    val=((2*s**3-3*s**2+1)*a0 + (s**3-2*s**2+s)*d0 + (-2*s**3+3*s**2)*a1 + (s**3-s**2)*d1)
    der=((6*s**2-6*s)*a0 + (3*s**2-4*s+1)*d0 + (-6*s**2+6*s)*a1 + (3*s**2-2*s)*d1)
    return val,der

@njit(cache=True)
def _rhs(P,C,A,dA,Ro):
    gap=A-P
    if gap<=1e-4 or P<=0.0 or C<=0.0:
        return np.nan,np.nan
    Ra=K_R*C_AN*C_AN/(C*C*gap*gap)
    q=gap/(Ra+R_PV)
    if q<=0.0:
        return np.nan,np.nan
    x=(q-Q_N)/Q_N
    Delta=0.75 if x<=0.0 else 0.075
    ks=Delta/4.0
    z=G*x/ks
    if z>700.0: z=700.0
    elif z<-700.0: z=-700.0
    ez=np.exp(z)
    Cinf=((C_AN+Delta/2.0)+(C_AN-Delta/2.0)*ez)/(1.0+ez)
    Cdot=(Cinf-C)/TAU
    Pdot=K_E*P/(1.0+K_E*P*C)*(C*dA+Cdot*(A-P)+(R_PV/R_F)*q-(P-P_VS)/Ro)
    return Pdot,Cdot

@njit(cache=True,parallel=True)
def _replay_numba(A,D,seg,multipliers):
    n=A.size; nc=multipliers.size
    out=np.empty((n,nc),dtype=np.float32)
    h=1.0/SUBSTEPS
    for j in prange(nc):
        P=P0; C=C_AN; Ro=R_O_POP*multipliers[j]; out[0,j]=P
        for i in range(n-1):
            if seg[i+1]!=seg[i]:
                P=P0; C=C_AN; out[i+1,j]=P
                continue
            a0=A[i]; a1=A[i+1]; d0=D[i]; d1=D[i+1]
            for sub in range(SUBSTEPS):
                ss=sub*h
                aa,dd=_hermite(a0,a1,d0,d1,ss); k1p,k1c=_rhs(P,C,aa,dd,Ro)
                aa,dd=_hermite(a0,a1,d0,d1,ss+h/2.0); k2p,k2c=_rhs(P+h*k1p/2.0,C+h*k1c/2.0,aa,dd,Ro)
                k3p,k3c=_rhs(P+h*k2p/2.0,C+h*k2c/2.0,aa,dd,Ro)
                aa,dd=_hermite(a0,a1,d0,d1,ss+h); k4p,k4c=_rhs(P+h*k3p,C+h*k3c,aa,dd,Ro)
                P=P+h*(k1p+2.0*k2p+2.0*k3p+k4p)/6.0
                C=C+h*(k1c+2.0*k2c+2.0*k3c+k4c)/6.0
            out[i+1,j]=P
    return out

def replay(blind,multipliers):
    A=np.asarray(blind['A_map_mmHg'],dtype=np.float64)
    D=np.asarray(blind['A_dot_mmHg_s'],dtype=np.float64)
    seg=np.asarray(blind['segment_id'],dtype=np.int64)
    mult=np.asarray(multipliers,dtype=np.float64)
    out=_replay_numba(A,D,seg,mult)
    if not np.all(np.isfinite(out)):
        raise RuntimeError('oracle replay left frozen model domain')
    return out

def align_reference(blind, paired):
    key={(float(t),int(s)):float(r) for t,s,r in zip(paired['time_s'],paired['segment_id'],paired['ICP_ref_median_mmHg'])}
    ref=np.array([key.get((float(t),int(s)),np.nan) for t,s in zip(blind['time_s'],blind['segment_id'])],float)
    eligible=(np.asarray(blind['valid_for_score'],int)==1)&np.isfinite(ref)
    return ref,eligible


def best_ro(preds, ref):
    maes=np.mean(np.abs(preds-ref[:,None]),axis=0); j=int(np.argmin(maes)); return j,maes


def best_ro_offset(preds, ref):
    offsets=np.median(ref[:,None]-preds,axis=0)
    maes=np.mean(np.abs(preds+offsets[None,:]-ref[:,None]),axis=0); j=int(np.argmin(maes)); return j,offsets,maes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('blind_csv'); ap.add_argument('paired_csv'); ap.add_argument('output_json')
    args=ap.parse_args()
    blind=load_csv(args.blind_csv); paired=load_csv(args.paired_csv)
    if len(blind['time_s'])==0 or len(paired['time_s'])==0: raise ValueError('empty input')
    ref,eligible=align_reference(blind,paired)
    idx=np.where(eligible)[0]; refs=ref[idx]; primary=np.asarray(blind['predicted_icp_mmHg'],float)[idx]
    if len(idx)<100: raise ValueError('too few paired rows')

    # Replay the 80 frozen R_o candidates plus exact population R_o in one pass.
    all_pred=replay(blind,np.r_[GRID,1.0])
    pop=all_pred[:,-1].astype(float)
    re=np.abs(pop-np.asarray(blind['predicted_icp_mmHg'],float))
    validation={'mean_abs_error_mmHg':float(np.mean(re)),'p99_abs_error_mmHg':float(np.quantile(re,0.99)),
                'max_abs_error_mmHg':float(np.max(re)),
                'acceptance':{'mean_le_0p02':bool(np.mean(re)<=0.02),'p99_le_0p15':bool(np.quantile(re,0.99)<=0.15),'max_le_0p5':bool(np.max(re)<=0.5)}}
    validation['passed']=bool(all(validation['acceptance'].values()))
    if not validation['passed']: raise RuntimeError(f'replay validation failed: {validation}')

    grid_pred=all_pred[idx,:-1].astype(float)
    del all_pred

    # Whole-record diagnostic oracles (nondeployable upper bounds).
    off=float(np.median(refs-primary)); off_pred=primary+off
    j,maes=best_ro(grid_pred,refs); ro_pred=grid_pred[:,j]
    ro_post_off=float(np.median(refs-ro_pred))
    k,offs,maeso=best_ro_offset(grid_pred,refs); roo_pred=grid_pred[:,k]+offs[k]

    ndev=max(1,int(math.floor(0.25*len(idx)))); dev=np.arange(ndev); test=np.arange(ndev,len(idx))
    if len(test)<1: raise ValueError('empty late test split')
    primary_dev=primary[dev]; primary_test=primary[test]; ref_dev=refs[dev]; ref_test=refs[test]
    gp_dev=grid_pred[dev]; gp_test=grid_pred[test]

    # Offset-only transfer.
    dev_off=float(np.median(ref_dev-primary_dev))
    # R_o-only transfer.
    jr,dev_maes=best_ro(gp_dev,ref_dev)
    # R_o+offset transfer.
    jro,dev_offs,dev_maeso=best_ro_offset(gp_dev,ref_dev)

    out={
      'protocol':'CHARIS V2 oracle stage1 cohort freeze 1',
      'record':str(Path(args.blind_csv).name).split('_V2_blind')[0],
      'population_R_o':R_O_POP,
      'grid':{'n':80,'multiplier_min':0.1,'multiplier_max':20.0,'spacing':'log','multipliers':GRID.tolist()},
      'replay':{'input':'frozen 1-Hz A_map and A_dot; cubic Hermite between samples; RK4 0.5-s substeps','validation':validation},
      'coverage':{'paired_rows':int(len(idx)),'paired_hours':float(len(idx)/3600.0),'dev_rows':int(len(dev)),'test_rows':int(len(test)),'split':'chronological first 25% paired rows development; remaining 75% test'},
      'primary':{'whole':metrics(primary,refs),'dev':metrics(primary_dev,ref_dev),'test':metrics(primary_test,ref_test)},
      'whole_oracle':{
        'offset_only':{'offset_mmHg':off,'metrics':metrics(off_pred,refs)},
        'R_o_only':{'selected_multiplier':float(GRID[j]),'selected_R_o':float(R_O_POP*GRID[j]),'metrics':metrics(ro_pred,refs),
                    'posthoc_offset_at_selected_R_o_mmHg':ro_post_off,'metrics_after_posthoc_offset':metrics(ro_pred+ro_post_off,refs)},
        'R_o_plus_offset':{'selected_multiplier':float(GRID[k]),'selected_R_o':float(R_O_POP*GRID[k]),'offset_mmHg':float(offs[k]),'metrics':metrics(roo_pred,refs),
                           'selected_at_lower_grid_bound':bool(k==0),'selected_at_upper_grid_bound':bool(k==len(GRID)-1)}
      },
      'early25_to_late75':{
        'offset_only':{'dev_offset_mmHg':dev_off,'dev_metrics':metrics(primary_dev+dev_off,ref_dev),'test_metrics':metrics(primary_test+dev_off,ref_test)},
        'R_o_only':{'selected_multiplier':float(GRID[jr]),'selected_R_o':float(R_O_POP*GRID[jr]),'dev_metrics':metrics(gp_dev[:,jr],ref_dev),'test_metrics':metrics(gp_test[:,jr],ref_test),
                    'selected_at_lower_grid_bound':bool(jr==0),'selected_at_upper_grid_bound':bool(jr==len(GRID)-1)},
        'R_o_plus_offset':{'selected_multiplier':float(GRID[jro]),'selected_R_o':float(R_O_POP*GRID[jro]),'dev_offset_mmHg':float(dev_offs[jro]),
                           'dev_metrics':metrics(gp_dev[:,jro]+dev_offs[jro],ref_dev),'test_metrics':metrics(gp_test[:,jro]+dev_offs[jro],ref_test),
                           'selected_at_lower_grid_bound':bool(jro==0),'selected_at_upper_grid_bound':bool(jro==len(GRID)-1)}
      }
    }
    Path(args.output_json).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'record':out['record'],'replay':validation,'whole':out['whole_oracle'],'transfer':out['early25_to_late75']},indent=2))

if __name__=='__main__': main()
