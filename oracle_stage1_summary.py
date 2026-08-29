#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

SEED=20260829
NBOOT=10000

def finite(x):
    return [float(v) for v in x if v is not None and np.isfinite(v)]

def stat(values):
    x=np.asarray(finite(values),float)
    if x.size==0: return {'n_patients':0,'mean':None,'mean_bootstrap_95ci':[None,None],'median':None,'q1':None,'q3':None}
    rng=np.random.default_rng(SEED); n=len(x)
    means=np.empty(NBOOT,float)
    for i in range(NBOOT): means[i]=np.mean(x[rng.integers(0,n,n)])
    return {'n_patients':int(n),'mean':float(np.mean(x)),'mean_bootstrap_95ci':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
            'median':float(np.median(x)),'q1':float(np.quantile(x,.25)),'q3':float(np.quantile(x,.75))}

def getm(j,path,metric='mae_mmHg'):
    cur=j
    for p in path: cur=cur[p]
    return cur[metric]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input_dir');ap.add_argument('output_dir');a=ap.parse_args()
    inp=Path(a.input_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    files=sorted(inp.rglob('charis*_oracle_stage1.json'))
    js=[json.loads(p.read_text()) for p in files]
    js.sort(key=lambda x:int(x['record'].replace('charis','')))
    if len(js)!=13: raise ValueError(f'expected 13 patient oracle JSON files, found {len(js)}')
    if not all(x['replay']['validation']['passed'] for x in js): raise RuntimeError('at least one replay validation failed')

    rows=[]
    for j in js:
        w=j['whole_oracle']; t=j['early25_to_late75']; p=j['primary']
        rows.append({
          'record':j['record'],'paired_hours':j['coverage']['paired_hours'],
          'replay_mean_abs_error_mmHg':j['replay']['validation']['mean_abs_error_mmHg'],'replay_p99_abs_error_mmHg':j['replay']['validation']['p99_abs_error_mmHg'],'replay_max_abs_error_mmHg':j['replay']['validation']['max_abs_error_mmHg'],
          'primary_whole_mae':p['whole']['mae_mmHg'],'offset_whole_mae':w['offset_only']['metrics']['mae_mmHg'],'offset_whole_mmHg':w['offset_only']['offset_mmHg'],
          'ro_whole_mae':w['R_o_only']['metrics']['mae_mmHg'],'ro_whole_multiplier':w['R_o_only']['selected_multiplier'],
          'ro_offset_whole_mae':w['R_o_plus_offset']['metrics']['mae_mmHg'],'ro_offset_whole_multiplier':w['R_o_plus_offset']['selected_multiplier'],'ro_offset_whole_mmHg':w['R_o_plus_offset']['offset_mmHg'],
          'primary_test_mae':p['test']['mae_mmHg'],'offset_test_mae':t['offset_only']['test_metrics']['mae_mmHg'],'offset_dev_mmHg':t['offset_only']['dev_offset_mmHg'],
          'ro_test_mae':t['R_o_only']['test_metrics']['mae_mmHg'],'ro_dev_multiplier':t['R_o_only']['selected_multiplier'],
          'ro_offset_test_mae':t['R_o_plus_offset']['test_metrics']['mae_mmHg'],'ro_offset_dev_multiplier':t['R_o_plus_offset']['selected_multiplier'],'ro_offset_dev_mmHg':t['R_o_plus_offset']['dev_offset_mmHg'],
          'primary_test_pearson':p['test']['pearson_r'],'offset_test_pearson':t['offset_only']['test_metrics']['pearson_r'],'ro_test_pearson':t['R_o_only']['test_metrics']['pearson_r'],'ro_offset_test_pearson':t['R_o_plus_offset']['test_metrics']['pearson_r'],
          'primary_test_spearman':p['test']['spearman_r'],'offset_test_spearman':t['offset_only']['test_metrics']['spearman_r'],'ro_test_spearman':t['R_o_only']['test_metrics']['spearman_r'],'ro_offset_test_spearman':t['R_o_plus_offset']['test_metrics']['spearman_r'],
        })
    csvp=out/'oracle_stage1_patient_metrics.csv'
    with open(csvp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def col(k):return [r[k] for r in rows]
    summary={
      'protocol':'CHARIS V2 oracle stage1 cohort freeze 1','n_patients':13,'total_paired_hours':float(sum(col('paired_hours'))),
      'replay_validation':{'all_passed':True,'mean_abs_error_mmHg':stat(col('replay_mean_abs_error_mmHg')),'p99_abs_error_mmHg':stat(col('replay_p99_abs_error_mmHg')),'max_abs_error_mmHg':stat(col('replay_max_abs_error_mmHg'))},
      'whole_record_non_deployable':{
        'primary_V2_mae_mmHg':stat(col('primary_whole_mae')),'offset_only_mae_mmHg':stat(col('offset_whole_mae')),'R_o_only_mae_mmHg':stat(col('ro_whole_mae')),'R_o_plus_offset_mae_mmHg':stat(col('ro_offset_whole_mae')),
        'offset_only_improves_n':int(sum(r['offset_whole_mae']<r['primary_whole_mae'] for r in rows)),
        'R_o_only_improves_n':int(sum(r['ro_whole_mae']<r['primary_whole_mae'] for r in rows)),
        'R_o_plus_offset_improves_n':int(sum(r['ro_offset_whole_mae']<r['primary_whole_mae'] for r in rows)),
        'R_o_plus_offset_lower_bound_n':int(sum(np.isclose(r['ro_offset_whole_multiplier'],0.1) for r in rows)),
        'R_o_plus_offset_upper_bound_n':int(sum(np.isclose(r['ro_offset_whole_multiplier'],20.0) for r in rows)),
        'selected_R_o_plus_offset_multiplier':stat(col('ro_offset_whole_multiplier')),'selected_offset_mmHg':stat(col('ro_offset_whole_mmHg')),
      },
      'early25_to_late75_transfer':{
        'primary_test_mae_mmHg':stat(col('primary_test_mae')),'offset_only_test_mae_mmHg':stat(col('offset_test_mae')),'R_o_only_test_mae_mmHg':stat(col('ro_test_mae')),'R_o_plus_offset_test_mae_mmHg':stat(col('ro_offset_test_mae')),
        'offset_only_improves_test_n':int(sum(r['offset_test_mae']<r['primary_test_mae'] for r in rows)),
        'R_o_only_improves_test_n':int(sum(r['ro_test_mae']<r['primary_test_mae'] for r in rows)),
        'R_o_plus_offset_improves_test_n':int(sum(r['ro_offset_test_mae']<r['primary_test_mae'] for r in rows)),
        'R_o_plus_offset_dev_lower_bound_n':int(sum(np.isclose(r['ro_offset_dev_multiplier'],0.1) for r in rows)),
        'R_o_plus_offset_dev_upper_bound_n':int(sum(np.isclose(r['ro_offset_dev_multiplier'],20.0) for r in rows)),
        'R_o_plus_offset_test_pearson_r':stat(col('ro_offset_test_pearson')),'R_o_plus_offset_test_spearman_r':stat(col('ro_offset_test_spearman')),
        'offset_only_test_pearson_r':stat(col('offset_test_pearson')),'R_o_only_test_pearson_r':stat(col('ro_test_pearson')),
        'selected_dev_R_o_plus_offset_multiplier':stat(col('ro_offset_dev_multiplier')),'selected_dev_offset_mmHg':stat(col('ro_offset_dev_mmHg')),
      }
    }
    (out/'oracle_stage1_cohort_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=[]
    md.append('# CHARIS V2 Oracle Stage 1 — Cohort Summary')
    md.append('')
    md.append('Diagnostic only. Primary V2 is unchanged. Whole-record oracle values are nondeployable upper bounds; early-25% to late-75% values are the transfer test.')
    md.append('')
    md.append(f"Patients: **13**; paired hours: **{summary['total_paired_hours']:.2f}**; replay validation: **13/13 passed**.")
    md.append('')
    md.append('| Analysis | Macro mean MAE (mmHg) | Median MAE | Improved vs primary |')
    md.append('|---|---:|---:|---:|')
    W=summary['whole_record_non_deployable']
    md.append(f"| Primary whole | {W['primary_V2_mae_mmHg']['mean']:.3f} | {W['primary_V2_mae_mmHg']['median']:.3f} | — |")
    md.append(f"| Whole offset-only oracle | {W['offset_only_mae_mmHg']['mean']:.3f} | {W['offset_only_mae_mmHg']['median']:.3f} | {W['offset_only_improves_n']}/13 |")
    md.append(f"| Whole R_o-only oracle | {W['R_o_only_mae_mmHg']['mean']:.3f} | {W['R_o_only_mae_mmHg']['median']:.3f} | {W['R_o_only_improves_n']}/13 |")
    md.append(f"| Whole R_o+offset oracle | {W['R_o_plus_offset_mae_mmHg']['mean']:.3f} | {W['R_o_plus_offset_mae_mmHg']['median']:.3f} | {W['R_o_plus_offset_improves_n']}/13 |")
    md.append('')
    T=summary['early25_to_late75_transfer']
    md.append('| Late-75% transfer | Macro mean MAE (mmHg) | Median MAE | Improved vs late primary |')
    md.append('|---|---:|---:|---:|')
    md.append(f"| Primary | {T['primary_test_mae_mmHg']['mean']:.3f} | {T['primary_test_mae_mmHg']['median']:.3f} | — |")
    md.append(f"| Early-fit offset-only | {T['offset_only_test_mae_mmHg']['mean']:.3f} | {T['offset_only_test_mae_mmHg']['median']:.3f} | {T['offset_only_improves_test_n']}/13 |")
    md.append(f"| Early-fit R_o-only | {T['R_o_only_test_mae_mmHg']['mean']:.3f} | {T['R_o_only_test_mae_mmHg']['median']:.3f} | {T['R_o_only_improves_test_n']}/13 |")
    md.append(f"| Early-fit R_o+offset | {T['R_o_plus_offset_test_mae_mmHg']['mean']:.3f} | {T['R_o_plus_offset_test_mae_mmHg']['median']:.3f} | {T['R_o_plus_offset_improves_test_n']}/13 |")
    md.append('')
    md.append(f"Whole R_o+offset selected the lower grid boundary in **{W['R_o_plus_offset_lower_bound_n']}/13** patients; early-fit R_o+offset selected it in **{T['R_o_plus_offset_dev_lower_bound_n']}/13**.")
    (out/'oracle_stage1_cohort_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
