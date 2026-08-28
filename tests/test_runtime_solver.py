import numpy as np
from icp_v2.model import simulate
from icp_v2.runtime import simulate_rk4_fixed
from icp_v2.synthetic import SinusoidMAP, SmoothStepMAP
from icp_v2.equilibrium import solve_equilibrium

def check(inp):
    e=solve_equilibrium(100.0); y0=(e['P'],e['C']); t=np.arange(0,181,1.0)
    ref=simulate((0,180),y0,inp.value,inp.derivative,t_eval=t,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.1)
    got,audit=simulate_rk4_fixed(t,y0,inp.value,inp.derivative,dt=.02)
    assert np.max(np.abs(ref.y[0]-got[0]))<.005; assert np.max(np.abs(ref.y[1]-got[1]))<5e-5
    assert audit['min_P_mmHg']>0 and audit['min_C']>0 and audit['min_A_minus_P_mmHg']>0

def test_runtime_sinusoid_gate(): check(SinusoidMAP(base=100,amplitude=5,period=120))
def test_runtime_smooth_step_gate(): check(SmoothStepMAP(base=100,amplitude=10,center=60,width=5))
