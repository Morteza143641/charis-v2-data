#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.lib.stride_tricks import sliding_window_view

R_O_POP=526.3; R_F=2.38e3; R_PV=1.24; C_AN=0.15; K_E_POP=0.11; K_R=4.91e4; Q_N=12.5; TAU=20.0; G_POP=1.5; P_VS=6.0
SUBSTEPS=2; SEED=20260829
BOUNDS=[(3.0,18.0),(math.log10(0.03),math.log10(20.0)),(math.log10(0.25),math.log10(4.0)),(math.log10(0.10),math.log10(4.0))]

def load_csv(path):
    with open(path,'r',encoding='utf-8') as f: names=f.readline().strip().split(',')
    arr=np.genfromtxt(path,delimiter=',',skip_header=1,dtype=float,filling_values=np.nan,ndmin=2)
    return {name:arr[:,i] for i,name in enumerate(names)}

def corr(x,y,kind='pearson'):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<2 or np.std(x)==0 or np.std(y)==0: return None
    if kind=='pearson': return float(np.corrcoef(x,y)[0,1])
    r=spearmanr(x,y).statistic; return None if not np.isfinite(r) else float(r)

def metrics(pred,ref):
    pred=np.asarray(pred,float); ref=np.asarray(ref,float); ok=np.isfinite(pred)&np.isfinite(ref); pred=pred[ok]; ref=ref[ok]
    if len(pred)==0: return {'n':0}
    d=pred-ref
    return {'n':int(len(d)),'bias_mmHg':float(np.mean(d)),'mae_mmHg':float(np.mean(np.abs(d))),'rmse_mmHg':float(np.sqrt(np.mean(d*d))),'pearson_r':corr(pred,ref,'pearson'),'spearman_r':corr(pred,ref,'spearman')}

@njit(cache=True)
def _hermite(a0,a1,d0,d1,s):
    val=((2*s**3-3*s**2+1)*a0+(s**3-2*s**2+s)*d0+(-2*s**3+3*s**2)*a1+(s**3-s**2)*d1)
    der=((6*s**2-6*s)*a0+(3*s**2-4*s+1)*d0+(-6*s**2+6*s)*a1+(3*s**2-2*s)*d1)
    return val,der

@njit(cache=True)
def _rhs(P,C,A,dA,Ro,kE,G):
    gap=A-P
    if gap<=1e-4 or P<=0.0 or C<=0.0: return np.nan,np.nan
    Ra=K_R*C_AN*C_AN/(C*C*gap*gap); q=gap/(Ra+R_PV)
    if q<=0.0 or not np.isfinite(q): return np.nan,np.nan
    x=(q-Q_N)/Q_N; Delta=0.75 if x<=0.0 else 0.075; ks=Delta/4.0; z=G*x/ks
    if z>700.0: z=700.0
    elif z<-700.0: z=-700.0
    ez=np.exp(z); Cinf=((C_AN+Delta/2.0)+(C_AN-Delta/2.0)*ez)/(1.0+ez); Cdot=(Cinf-C)/TAU
    den=1.0+kE*P*C
    if den<=0.0: return np.nan,np.nan
    Pdot=kE*P/den*(C*dA+Cdot*(A-P)+(R_PV/R_F)*q-(P-P_VS)/Ro)
    if not np.isfinite(Pdot) or not np.isfinite(Cdot): return np.nan,np.nan
    return Pdot,Cdot

@njit(cache=True)
def _step_interval(P,C,a0,a1,d0,d1,Ro,kE,G):
    h=1.0/SUBSTEPS
    for sub in range(SUBSTEPS):
        ss=sub*h; aa,dd=_hermite(a0,a1,d0,d1,ss); k1p,k1c=_rhs(P,C,aa,dd,Ro,kE,G)
        if not np.isfinite(k1p): return np.nan,np.nan
        aa,dd=_hermite(a0,a1,d0,d1,ss+h/2.0); k2p,k2c=_rhs(P+h*k1p/2.0,C+h*k1c/2.0,aa,dd,Ro,kE,G)
        if not np.isfinite(k2p): return np.nan,np.nan
        k3p,k3c=_rhs(P+h*k2p/2.0,C+h*k2c/2.0,aa,dd,Ro,kE,G)
        if not np.isfinite(k3p): return np.nan,np.nan
        aa,dd=_hermite(a0,a1,d0,d1,ss+h); k4p,k4c=_rhs(P+h*k3p,C+h*k3c,aa,dd,Ro,kE,G)
        if not np.isfinite(k4p): return np.nan,np.nan
        P=P+h*(k1p+2*k2p+2*k3p+k4p)/6.0; C=C+h*(k1c+2*k2c+2*k3c+k4c)/6.0
        if not np.isfinite(P) or not np.isfinite(C) or P<=0.0 or C<=0.0: return np.nan,np.nan
    return P,C

@njit(cache=True)
def _objective_mae(A,D,seg,ref,fitmask,last,P0,rm,kem,gm):
    Ro=R_O_POP*rm; kE=K_E_POP*kem; G=G_POP*gm; P=P0; C=C_AN; total=0.0; count=0
    if fitmask[0]: total+=abs(P-ref[0]); count+=1
    for i in range(last):
        if seg[i+1]!=seg[i]: P=P0; C=C_AN
        else:
            P,C=_step_interval(P,C,A[i],A[i+1],D[i],D[i+1],Ro,kE,G)
            if not np.isfinite(P): return 1.0e9
        if fitmask[i+1]: total+=abs(P-ref[i+1]); count+=1
    return total/count if count else 1.0e9

@njit(cache=True)
def _simulate_full(A,D,seg,P0,rm,kem,gm):
    n=A.size; out=np.empty(n,np.float32); Ro=R_O_POP*rm; kE=K_E_POP*kem; G=G_POP*gm; P=P0; C=C_AN; out[0]=P
    for i in range(n-1):
        if seg[i+1]!=seg[i]: P=P0; C=C_AN
        else:
            P,C=_step_interval(P,C,A[i],A[i+1],D[i],D[i+1],Ro,kE,G)
            if not np.isfinite(P): return np.empty(0,np.float32)
        out[i+1]=P
    return out

def unpack(x): return float(x[0]),10.0**float(x[1]),10.0**float(x[2]),10.0**float(x[3])

def align_reference(blind,paired):
    key={(float(t),int(s)):float(r) for t,s,r in zip(paired['time_s'],paired['segment_id'],paired['ICP_ref_median_mmHg'])}
    ref=np.array([key.get((float(t),int(s)),np.nan) for t,s in zip(blind['time_s'],blind['segment_id'])],float)
    eligible=(np.asarray(blind['valid_for_score'],int)==1)&np.isfinite(ref)
    return ref,eligible

def fit_params(A,D,seg,ref,fitmask,last):
    def obj(x):
        P0,rm,kem,gm=unpack(x); return float(_objective_mae(A,D,seg,ref,fitmask,last,P0,rm,kem,gm))
    res=differential_evolution(obj,BOUNDS,seed=SEED,init='sobol',popsize=5,maxiter=10,tol=0.0,atol=0.0,polish=False,updating='immediate',workers=1)
    x=np.asarray(res.x,float); f=float(res.fun); xpop=np.array([9.5,0.0,0.0,0.0]); fpop=obj(xpop)
    source='differential_evolution'
    if fpop<f: x=xpop; f=fpop; source='population_setting'
    return x,f,{'source':source,'nfev':int(res.nfev)+1,'nit':int(res.nit),'optimizer_success':bool(res.success),'message':str(res.message),'fixed_budget':True}

def param_report(x):
    P0,rm,kem,gm=unpack(x); vals=[float(x[i]) for i in range(4)]; loc=[]; low=[]; high=[]
    for v,(a,b) in zip(vals,BOUNDS):
        z=(v-a)/(b-a); loc.append(float(z)); low.append(bool(z<=0.02)); high.append(bool(z>=0.98))
    return {'P0_mmHg':P0,'R_o_multiplier':rm,'R_o':R_O_POP*rm,'k_E_multiplier':kem,'k_E':K_E_POP*kem,'G_multiplier':gm,'G':G_POP*gm,
            'normalized_search_location':dict(zip(['P0','R_o','k_E','G'],loc)),'near_lower_bound':dict(zip(['P0','R_o','k_E','G'],low)),'near_upper_bound':dict(zip(['P0','R_o','k_E','G'],high))}

def trailing_median_strict(values,seg,window=60):
    values=np.asarray(values,float); seg=np.asarray(seg,int); out=np.full(len(values),np.nan)
    for s in np.unique(seg):
        idx=np.where(seg==s)[0]
        if len(idx)<window: continue
        win=sliding_window_view(values[idx],window); good=np.all(np.isfinite(win),axis=1); med=np.full(len(win),np.nan); med[good]=np.median(win[good],axis=1); out[idx[window-1:]]=med
    return out

def ccc(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2:return None
    mx=x.mean(); my=y.mean(); vx=np.mean((x-mx)**2); vy=np.mean((y-my)**2); cov=np.mean((x-mx)*(y-my)); den=vx+vy+(mx-my)**2
    return None if den==0 else float(2*cov/den)

def trend_metrics(pred,ref,time_s,seg,valid,h_seconds):
    p=np.asarray(pred,float).copy(); r=np.asarray(ref,float).copy(); p[~valid]=np.nan; r[~valid]=np.nan
    pm=trailing_median_strict(p,seg,60); rm=trailing_median_strict(r,seg,60); pds=[]; rds=[]
    for s in np.unique(seg):
        idx=np.where(seg==s)[0]
        if len(idx)<=h_seconds: continue
        a=idx[:-h_seconds]; b=idx[h_seconds:]
        ok=(valid[a]&valid[b]&np.isfinite(pm[a])&np.isfinite(pm[b])&np.isfinite(rm[a])&np.isfinite(rm[b])&np.isclose(time_s[b]-time_s[a],h_seconds))
        pds.append(pm[b[ok]]-pm[a[ok]]); rds.append(rm[b[ok]]-rm[a[ok]])
    if not pds:return {'n_pairs':0}
    dp=np.concatenate(pds); dr=np.concatenate(rds); strong=np.abs(dr)>=2.0
    return {'n_pairs':int(len(dp)),'delta_mae_mmHg':float(np.mean(np.abs(dp-dr))),'spearman_r':corr(dp,dr,'spearman'),'ccc':ccc(dp,dr),'direction_accuracy':float(np.mean(np.sign(dp)==np.sign(dr))),
            'direction_accuracy_ref_abs_delta_ge_2':float(np.mean(np.sign(dp[strong])==np.sign(dr[strong]))) if np.any(strong) else None,'n_ref_abs_delta_ge_2':int(np.sum(strong))}

def sustained_runs(mask,time_s,seg,min_len=300):
    mask=np.asarray(mask,bool); t=np.asarray(time_s,float); seg=np.asarray(seg,int); runs=[]; start=None; prev=None; prevseg=None
    for i,yes in enumerate(mask):
        contiguous=(prev is not None and seg[i]==prevseg and abs(t[i]-prev-1.0)<1e-9)
        if yes:
            if start is None or not contiguous:start=i
        elif start is not None:
            end=i-1
            if end-start+1>=min_len:runs.append((start,end))
            start=None
        prev=t[i]; prevseg=seg[i]
    if start is not None:
        end=len(mask)-1
        if end-start+1>=min_len:runs.append((start,end))
    return runs

def event_metrics(pred,ref,time_s,seg,valid,threshold):
    eligible=valid&np.isfinite(ref)&np.isfinite(pred); p=np.asarray(pred)[eligible]; r=np.asarray(ref)[eligible]; tt=np.asarray(time_s)[eligible]; ss=np.asarray(seg)[eligible]; y=r>threshold
    auroc=auprc=None
    if len(np.unique(y))==2: auroc=float(roc_auc_score(y,p)); auprc=float(average_precision_score(y,p))
    rr=sustained_runs(y,tt,ss,300); pr=sustained_runs(p>threshold,tt,ss,300); detected=sum(any(not(d<a or c>b) for c,d in pr) for a,b in rr)
    false=sum(1 for c,d in pr if not any(not(d<a or c>b) for a,b in rr)); hours=len(p)/3600.0
    return {'threshold_mmHg':float(threshold),'n_timepoints':int(len(p)),'reference_positive_fraction':float(np.mean(y)) if len(y) else None,'timepoint_auroc':auroc,'timepoint_auprc':auprc,
            'n_reference_sustained_events':int(len(rr)),'n_predicted_sustained_events':int(len(pr)),'event_sensitivity':float(detected/len(rr)) if rr else None,'false_alarm_events':int(false),'false_alarm_events_per_hour':float(false/hours) if hours>0 else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('blind_csv'); ap.add_argument('paired_csv'); ap.add_argument('output_json'); args=ap.parse_args()
    blind=load_csv(args.blind_csv); paired=load_csv(args.paired_csv); ref,eligible=align_reference(blind,paired)
    A=np.asarray(blind['A_map_mmHg'],np.float64); D=np.asarray(blind['A_dot_mmHg_s'],np.float64); seg=np.asarray(blind['segment_id'],np.int64); tt=np.asarray(blind['time_s'],float); primary=np.asarray(blind['predicted_icp_mmHg'],float); idx=np.where(eligible)[0]
    if len(idx)<100: raise ValueError('too few paired rows')
    pop=_simulate_full(A,D,seg,9.5,1.0,1.0,1.0)
    if len(pop)!=len(primary): raise RuntimeError('population replay left model domain')
    re=np.abs(pop.astype(float)-primary); validation={'mean_abs_error_mmHg':float(np.mean(re)),'p99_abs_error_mmHg':float(np.quantile(re,0.99)),'max_abs_error_mmHg':float(np.max(re)),
      'acceptance':{'mean_le_0p02':bool(np.mean(re)<=0.02),'p99_le_0p15':bool(np.quantile(re,0.99)<=0.15),'max_le_0p5':bool(np.max(re)<=0.5)}}; validation['passed']=bool(all(validation['acceptance'].values()))
    if not validation['passed']: raise RuntimeError(f'replay validation failed: {validation}')
    ndev=max(1,int(math.floor(0.25*len(idx)))); dev_idx=idx[:ndev]; test_idx=idx[ndev:]
    whole_mask=eligible.copy(); dev_mask=np.zeros(len(A),bool); dev_mask[dev_idx]=True; test_mask=np.zeros(len(A),bool); test_mask[test_idx]=True
    xw,fw,optw=fit_params(A,D,seg,ref,whole_mask,int(idx[-1])); pw=_simulate_full(A,D,seg,*unpack(xw))
    if len(pw)!=len(A): raise RuntimeError('whole selected parameters left model domain')
    xd,fd,optd=fit_params(A,D,seg,ref,dev_mask,int(dev_idx[-1])); pt=_simulate_full(A,D,seg,*unpack(xd))
    if len(pt)!=len(A): raise RuntimeError('transfer selected parameters left model domain')
    out={'protocol':'CHARIS V2 oracle stage2 cohort freeze 1','record':str(Path(args.blind_csv).name).split('_V2_blind')[0],
      'search':{'bounds':{'P0_mmHg':[3.0,18.0],'R_o_multiplier':[0.03,20.0],'k_E_multiplier':[0.25,4.0],'G_multiplier':[0.10,4.0]},'seed':SEED,'init':'sobol','popsize':5,'maxiter':10,'polish':False,'objective':'MAE'},
      'replay_validation':validation,'coverage':{'paired_rows':int(len(idx)),'paired_hours':float(len(idx)/3600.0),'dev_rows':int(len(dev_idx)),'test_rows':int(len(test_idx))},
      'primary':{'whole':metrics(primary[idx],ref[idx]),'dev':metrics(primary[dev_idx],ref[dev_idx]),'test':metrics(primary[test_idx],ref[test_idx])},
      'whole_oracle':{'params':param_report(xw),'objective_mae_mmHg':fw,'optimization':optw,'metrics':metrics(pw[idx],ref[idx])},
      'early25_to_late75':{'params':param_report(xd),'dev_objective_mae_mmHg':fd,'optimization':optd,'dev_metrics':metrics(pt[dev_idx],ref[dev_idx]),'test_metrics':metrics(pt[test_idx],ref[test_idx]),'test_trend':{},'test_events':{}}}
    for mins in [5,15,30]: out['early25_to_late75']['test_trend'][f'{mins}min']=trend_metrics(pt,ref,tt,seg,test_mask,mins*60)
    for thr in [22.0,20.0]: out['early25_to_late75']['test_events'][f'{int(thr)}mmHg']=event_metrics(pt,ref,tt,seg,test_mask,thr)
    Path(args.output_json).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'record':out['record'],'replay':validation,'whole':out['whole_oracle'],'transfer':out['early25_to_late75']},indent=2))
if __name__=='__main__': main()
