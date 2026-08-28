import numpy as np
from dataclasses import replace
from icp_v2.model import ModelParams
from icp_v2.equilibrium import solve_equilibrium, solve_equilibrium_fixed_c

FULL_TARGETS={1:9.43,2:12.82,5:22.68,10:37.65,12:42.89}
FIXED_C_TARGETS={1:9.44,2:12.31,5:18.81,10:25.93,12:28.06}

def test_full_equilibrium_regression_table():
    base=ModelParams()
    for mult,target in FULL_TARGETS.items():
        p=replace(base,R_o=base.R_o*mult); out=solve_equilibrium(A=100.0,p=p)
        assert abs(out['P']-target)<0.01; assert abs(out['P_dot'])<1e-10; assert abs(out['C_dot'])<1e-10
        assert out['P']>0 and out['C']>0 and out['q']>0 and out['R_a']>0; assert 100.0-out['P']>p.min_gap

def test_fixed_c_regression_table():
    base=ModelParams()
    for mult,target in FIXED_C_TARGETS.items():
        p=replace(base,R_o=base.R_o*mult); out=solve_equilibrium_fixed_c(A=100.0,p=p)
        assert abs(out['P']-target)<0.01; assert out['P']>0 and out['C']>0 and out['q']>0 and out['R_a']>0
