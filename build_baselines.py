#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from icp_v2.equilibrium import solve_equilibrium_fixed_c
from icp_v2.model import ModelParams


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('blind_csv'); ap.add_argument('output_csv'); args=ap.parse_args()
    d=np.genfromtxt(args.blind_csv,delimiter=',',names=True)
    p=ModelParams(); b1=np.empty(len(d),float)
    # exact frozen static model; no ICP/reference input is accepted by this program
    for i,A in enumerate(d['A_map_mmHg']):
        b1[i]=solve_equilibrium_fixed_c(float(A),p=p)['P']
    out=Path(args.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['time_s','B0_constant_9p5_mmHg','B1_static_fixedC_mmHg','segment_id','valid_for_score'])
        for i in range(len(d)):
            w.writerow([float(d['time_s'][i]),9.5,float(b1[i]),int(d['segment_id'][i]),int(d['valid_for_score'][i])])
    print(json.dumps({'baseline_file':str(out),'sha256':sha256_file(out),'n_rows':len(d)},indent=2))

if __name__=='__main__': main()
