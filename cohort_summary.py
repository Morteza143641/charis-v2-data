#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

SEED=20260829; N_BOOT=10000


def numeric(vals): return np.asarray([v for v in vals if v is not None and np.isfinite(v)],float)

def summarize(vals):
    x=numeric(vals)
    if len(x)==0: return {'n_patients':0,'mean':None,'mean_bootstrap_95ci':[None,None],'median':None,'q1':None,'q3':None}
    rng=np.random.default_rng(SEED)
    boots=np.mean(x[rng.integers(0,len(x),size=(N_BOOT,len(x)))],axis=1) if len(x)>1 else np.repeat(x[0],N_BOOT)
    return {'n_patients':int(len(x)),'mean':float(np.mean(x)),'mean_bootstrap_95ci':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            'median':float(np.median(x)),'q1':float(np.quantile(x,.25)),'q3':float(np.quantile(x,.75))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('scores_dir'); ap.add_argument('output_dir'); args=ap.parse_args()
    paths=sorted(Path(args.scores_dir).rglob('charis*_V2_score.json'))
    if not paths: raise SystemExit('no patient score JSON files found')
    rec=[]
    for p in paths:
        j=json.load(open(p)); name=j['unlock']['source']['record']
        row={'record':name,'paired_hours':j['coverage']['paired_hours'],'paired_rows':j['coverage']['paired_rows'],
             'ref_qc_excluded':j['coverage']['reference_qc_excluded_eligible_rows']}
        for model,key in [('V2','V2'),('B0','B0_constant_9p5'),('B1','B1_static_fixedC')]:
            a=j['absolute'][key]
            for m in ['bias_mmHg','mae_mmHg','rmse_mmHg','pearson_r','spearman_r']: row[f'{model}_{m}']=a.get(m)
        for h in ['5min','15min','30min']:
            tr=j['trend'][h]
            for m in ['n_pairs','delta_mae_mmHg','spearman_r','ccc','direction_accuracy']:
                row[f'trend_{h}_{m}']=tr.get(m)
        for th in ['22mmHg','20mmHg']:
            e=j['events'][th]
            row[f'event_{th}_n_ref']=e.get('n_reference_sustained_events')
            row[f'event_{th}_sensitivity']=e.get('event_sensitivity')
            row[f'event_{th}_auroc']=e.get('timepoint_auroc')
            row[f'event_{th}_auprc']=e.get('timepoint_auprc')
            row[f'event_{th}_n_detected']=sum(1 for d in e.get('detections',[]) if d.get('detected'))
        rec.append(row)
    rec=sorted(rec,key=lambda r:int(r['record'].replace('charis','')))
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    cols=sorted({k for r in rec for k in r.keys()}, key=lambda x:(x!='record',x))
    with open(outdir/'patient_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rec)

    summary={'freeze':'V2 Cohort Analysis Freeze 1','n_records_found':len(rec),'records':[r['record'] for r in rec],
             'coverage':{'total_paired_hours':float(sum(r['paired_hours'] for r in rec)),
                         'total_reference_qc_excluded_rows':int(sum(r['ref_qc_excluded'] for r in rec))},
             'absolute':{},'paired_mae_differences':{},'trend':{},'events':{}}
    for model in ['V2','B0','B1']:
        summary['absolute'][model]={m:summarize([r[f'{model}_{m}'] for r in rec]) for m in ['bias_mmHg','mae_mmHg','rmse_mmHg','pearson_r','spearman_r']}
    for base in ['B0','B1']:
        dif=[r['V2_mae_mmHg']-r[f'{base}_mae_mmHg'] for r in rec if r['V2_mae_mmHg'] is not None and r[f'{base}_mae_mmHg'] is not None]
        summary['paired_mae_differences'][f'V2_minus_{base}']={**summarize(dif),'n_V2_lower_MAE':int(sum(d<0 for d in dif))}
    for h in ['5min','15min','30min']:
        summary['trend'][h]={m:summarize([r[f'trend_{h}_{m}'] for r in rec]) for m in ['delta_mae_mmHg','spearman_r','ccc','direction_accuracy']}
    for th in ['22mmHg','20mmHg']:
        with_events=[r for r in rec if (r[f'event_{th}_n_ref'] or 0)>0]
        summary['events'][th]={
            'n_patients_with_reference_events':len(with_events),
            'total_reference_events':int(sum(r[f'event_{th}_n_ref'] or 0 for r in rec)),
            'total_detected_reference_events':int(sum(r[f'event_{th}_n_detected'] or 0 for r in rec)),
            'macro_event_sensitivity':summarize([r[f'event_{th}_sensitivity'] for r in with_events]),
            'patient_auroc':summarize([r[f'event_{th}_auroc'] for r in rec]),
            'patient_auprc':summarize([r[f'event_{th}_auprc'] for r in rec]),
        }
    (outdir/'cohort_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# CHARIS V2 cohort summary','',f"Records scored: **{len(rec)}**",f"Total paired time: **{summary['coverage']['total_paired_hours']:.2f} h**",'',
        '## Absolute ICP','', '| Model | Median MAE | IQR | Mean MAE (patient bootstrap 95% CI) |','|---|---:|---:|---:|']
    for model in ['V2','B0','B1']:
        s=summary['absolute'][model]['mae_mmHg'];md.append(f"| {model} | {s['median']:.3f} | {s['q1']:.3f}–{s['q3']:.3f} | {s['mean']:.3f} ({s['mean_bootstrap_95ci'][0]:.3f}–{s['mean_bootstrap_95ci'][1]:.3f}) |")
    md += ['','## Paired MAE comparison','']
    for base in ['B0','B1']:
        s=summary['paired_mae_differences'][f'V2_minus_{base}'];md.append(f"- V2 − {base}: median {s['median']:.3f} mmHg; V2 lower MAE in {s['n_V2_lower_MAE']}/{s['n_patients']} evaluable patients.")
    md += ['','## Trend','']
    for h in ['5min','15min','30min']:
        s=summary['trend'][h];md.append(f"- {h}: median Δ-MAE {s['delta_mae_mmHg']['median']:.3f} mmHg; median Spearman {s['spearman_r']['median'] if s['spearman_r']['median'] is not None else 'NA'}; median direction accuracy {s['direction_accuracy']['median'] if s['direction_accuracy']['median'] is not None else 'NA'}.")
    md += ['','## Elevated ICP','']
    for th in ['22mmHg','20mmHg']:
        e=summary['events'][th];md.append(f"- {th}: {e['n_patients_with_reference_events']} patients with sustained reference events; {e['total_detected_reference_events']}/{e['total_reference_events']} reference events detected.")
    (outdir/'cohort_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
