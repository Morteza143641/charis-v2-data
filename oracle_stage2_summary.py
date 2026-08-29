#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

SEED=20260829; NBOOT=10000

def load_all(root):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(root).rglob('*_oracle_stage2.json'))]

def vals(rows,path):
    z=[]
    for r in rows:
        x=r
        for k in path: x=x.get(k,{}) if isinstance(x,dict) else None
        if x is not None and np.isfinite(x): z.append(float(x))
    return np.asarray(z,float)

def summarize(a,rng):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if len(a)==0:return {'n_patients':0,'mean':None,'mean_bootstrap_95ci':[None,None],'median':None,'q1':None,'q3':None}
    means=np.empty(NBOOT,float)
    for i in range(NBOOT):means[i]=np.mean(a[rng.integers(0,len(a),len(a))])
    return {'n_patients':int(len(a)),'mean':float(np.mean(a)),'mean_bootstrap_95ci':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'median':float(np.median(a)),'q1':float(np.quantile(a,.25)),'q3':float(np.quantile(a,.75))}

def metric_block(rows,prefix,rng): return {k:summarize(vals(rows,prefix+[k]),rng) for k in ['mae_mmHg','rmse_mmHg','bias_mmHg','pearson_r','spearman_r']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input_root');ap.add_argument('output_dir');args=ap.parse_args();rows=load_all(args.input_root)
    if len(rows)!=13:raise RuntimeError(f'expected 13 Stage-2 patient files, found {len(rows)}')
    if not all(r['replay_validation']['passed'] for r in rows):raise RuntimeError('one or more replay validations failed')
    rng=np.random.default_rng(SEED)
    out={'protocol':'CHARIS V2 oracle stage2 cohort freeze 1','n_records':13,'records':[r['record'] for r in rows],
         'coverage':{'total_paired_hours':float(sum(r['coverage']['paired_hours'] for r in rows))},
         'replay_validation':{'n_passed':13,'max_of_patient_max_abs_error_mmHg':float(max(r['replay_validation']['max_abs_error_mmHg'] for r in rows))},
         'whole':{'primary':metric_block(rows,['primary','whole'],rng),'stage2_oracle':metric_block(rows,['whole_oracle','metrics'],rng)},
         'transfer':{'primary_test':metric_block(rows,['primary','test'],rng),'stage2_test':metric_block(rows,['early25_to_late75','test_metrics'],rng)},
         'paired_differences':{},'parameters':{},'trend':{},'events':{}}
    d=np.array([r['early25_to_late75']['test_metrics']['mae_mmHg']-r['primary']['test']['mae_mmHg'] for r in rows],float)
    out['paired_differences']['stage2_minus_primary_test_mae']=summarize(d,rng);out['paired_differences']['n_stage2_lower_test_mae']=int(np.sum(d<0))
    dw=np.array([r['whole_oracle']['metrics']['mae_mmHg']-r['primary']['whole']['mae_mmHg'] for r in rows],float)
    out['paired_differences']['stage2_minus_primary_whole_mae']=summarize(dw,rng);out['paired_differences']['n_stage2_lower_whole_mae']=int(np.sum(dw<0))
    for label,key in [('whole','whole_oracle'),('early25','early25_to_late75')]:
        pp=[r[key]['params'] for r in rows]
        out['parameters'][label]={
          'P0_mmHg':summarize(np.array([p['P0_mmHg'] for p in pp]),rng),'R_o_multiplier':summarize(np.array([p['R_o_multiplier'] for p in pp]),rng),
          'k_E_multiplier':summarize(np.array([p['k_E_multiplier'] for p in pp]),rng),'G_multiplier':summarize(np.array([p['G_multiplier'] for p in pp]),rng),
          'near_lower_bound_counts':{q:int(sum(p['near_lower_bound'][q] for p in pp)) for q in ['P0','R_o','k_E','G']},
          'near_upper_bound_counts':{q:int(sum(p['near_upper_bound'][q] for p in pp)) for q in ['P0','R_o','k_E','G']}}
    for h in ['5min','15min','30min']:
        out['trend'][h]={k:summarize(vals(rows,['early25_to_late75','test_trend',h,k]),rng) for k in ['delta_mae_mmHg','spearman_r','ccc','direction_accuracy','direction_accuracy_ref_abs_delta_ge_2']}
    for thr in ['22mmHg','20mmHg']:
        ev=[r['early25_to_late75']['test_events'][thr] for r in rows]
        out['events'][thr]={'n_patients_with_reference_events':int(sum(e['n_reference_sustained_events']>0 for e in ev)),'total_reference_events':int(sum(e['n_reference_sustained_events'] for e in ev)),
          'total_predicted_events':int(sum(e['n_predicted_sustained_events'] for e in ev)),'patient_auroc':summarize(np.array([e['timepoint_auroc'] for e in ev if e['timepoint_auroc'] is not None]),rng),
          'patient_auprc':summarize(np.array([e['timepoint_auprc'] for e in ev if e['timepoint_auprc'] is not None]),rng),'macro_event_sensitivity':summarize(np.array([e['event_sensitivity'] for e in ev if e['event_sensitivity'] is not None]),rng),
          'false_alarm_events_per_hour':summarize(np.array([e['false_alarm_events_per_hour'] for e in ev if e['false_alarm_events_per_hour'] is not None]),rng)}
    od=Path(args.output_dir);od.mkdir(parents=True,exist_ok=True);(od/'oracle_stage2_cohort_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    fields=['record','paired_hours','primary_test_mae','stage2_test_mae','primary_test_pearson','stage2_test_pearson','primary_test_spearman','stage2_test_spearman','P0','R_o_mult','k_E_mult','G_mult','trend5_spearman','trend15_spearman','trend30_spearman','event22_auroc','event22_sensitivity']
    with open(od/'oracle_stage2_patient_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            e=r['early25_to_late75'];p=e['params'];w.writerow({'record':r['record'],'paired_hours':r['coverage']['paired_hours'],'primary_test_mae':r['primary']['test']['mae_mmHg'],'stage2_test_mae':e['test_metrics']['mae_mmHg'],
              'primary_test_pearson':r['primary']['test']['pearson_r'],'stage2_test_pearson':e['test_metrics']['pearson_r'],'primary_test_spearman':r['primary']['test']['spearman_r'],'stage2_test_spearman':e['test_metrics']['spearman_r'],
              'P0':p['P0_mmHg'],'R_o_mult':p['R_o_multiplier'],'k_E_mult':p['k_E_multiplier'],'G_mult':p['G_multiplier'],'trend5_spearman':e['test_trend']['5min']['spearman_r'],'trend15_spearman':e['test_trend']['15min']['spearman_r'],'trend30_spearman':e['test_trend']['30min']['spearman_r'],
              'event22_auroc':e['test_events']['22mmHg']['timepoint_auroc'],'event22_sensitivity':e['test_events']['22mmHg']['event_sensitivity']})
    t=out['transfer'];pd=out['paired_differences'];md=['# CHARIS V2 Oracle Stage-2 Cohort Summary','', 'Nondeployable post-primary diagnostic. V2 primary is unchanged.','']
    md.append(f"- Records: 13; paired coverage: {out['coverage']['total_paired_hours']:.2f} h; replay gates passed: 13/13.")
    md.append(f"- Whole-record median MAE: primary {out['whole']['primary']['mae_mmHg']['median']:.3f} -> Stage-2 {out['whole']['stage2_oracle']['mae_mmHg']['median']:.3f} mmHg.")
    md.append(f"- Late-75% median MAE: primary {t['primary_test']['mae_mmHg']['median']:.3f} -> Stage-2 {t['stage2_test']['mae_mmHg']['median']:.3f} mmHg; Stage-2 lower in {pd['n_stage2_lower_test_mae']}/13 patients.")
    md.append(f"- Late-75% mean Pearson: {t['stage2_test']['pearson_r']['mean']:.3f} (95% bootstrap CI {t['stage2_test']['pearson_r']['mean_bootstrap_95ci'][0]:.3f} to {t['stage2_test']['pearson_r']['mean_bootstrap_95ci'][1]:.3f}).")
    md.append(f"- Late-75% mean Spearman: {t['stage2_test']['spearman_r']['mean']:.3f} (95% bootstrap CI {t['stage2_test']['spearman_r']['mean_bootstrap_95ci'][0]:.3f} to {t['stage2_test']['spearman_r']['mean_bootstrap_95ci'][1]:.3f}).")
    for h in ['5min','15min','30min']:md.append(f"- {h} trend: mean Spearman {out['trend'][h]['spearman_r']['mean']:.3f}; mean direction accuracy {out['trend'][h]['direction_accuracy']['mean']:.3f}.")
    e=out['events']['22mmHg'];md.append(f"- >22 mmHg: {e['total_reference_events']} late reference sustained events; macro sensitivity mean {e['macro_event_sensitivity']['mean'] if e['macro_event_sensitivity']['mean'] is not None else 'NA'}; mean patient AUROC {e['patient_auroc']['mean'] if e['patient_auroc']['mean'] is not None else 'NA'}.")
    md+=['','Parameter values are oracle search outputs, not physiological estimates. Boundary counts are reported in JSON.']
    (od/'oracle_stage2_cohort_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
