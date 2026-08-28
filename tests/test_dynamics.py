import numpy as np
from icp_v2.model import ModelParams, simulate
from icp_v2.equilibrium import solve_equilibrium
from icp_v2.synthetic import ConstantMAP,SmoothStepMAP,RampMAP,SinusoidMAP,BandLimitedNoiseMAP

def initial_equilibrium(A=100.0):
    e=solve_equilibrium(A=A); return e['P'],e['C']

def audit(sol,inp):
    P,C=sol.y; A=np.array([inp.value(t) for t in sol.t]); assert np.isfinite(sol.y).all(); assert (P>0).all(); assert (C>0).all(); assert ((A-P)>ModelParams().min_gap).all()

def test_constant_map_preserves_equilibrium():
    inp=ConstantMAP(100.0); y0=initial_equilibrium(100.0); t=np.linspace(0,600,601)
    sol=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,method='DOP853',rtol=1e-10,atol=1e-12)
    audit(sol,inp); assert np.max(np.abs(sol.y[0]-y0[0]))<1e-7; assert np.max(np.abs(sol.y[1]-y0[1]))<1e-7

def test_synthetic_inputs_are_numerically_sane():
    cases=[SmoothStepMAP(),RampMAP(base=100.0,slope=0.01),SinusoidMAP(),BandLimitedNoiseMAP()]; y0=initial_equilibrium(100.0); t=np.linspace(0,600,1201)
    for inp in cases:
        sol=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,max_step=0.5); audit(sol,inp); assert np.max(sol.y[0])<80.0; assert np.max(sol.y[1])<1.0

def test_full_autoregulation_and_fixed_c_are_distinct():
    inp=SmoothStepMAP(base=100,amplitude=10,center=60,width=5); y0=initial_equilibrium(100.0); t=np.linspace(0,600,1201)
    full=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,autoregulation=True,max_step=0.25); fixed=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,autoregulation=False,max_step=0.25)
    audit(full,inp); audit(fixed,inp); assert abs(full.y[0,-1]-fixed.y[0,-1])>0.25; assert abs(full.y[1,-1]-fixed.y[1,-1])>0.005

def test_solver_and_step_convergence():
    inp=SinusoidMAP(base=100,amplitude=5,period=120); y0=initial_equilibrium(100.0); t=np.linspace(0,600,1201)
    ref=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.1)
    alt_solver=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,method='RK45',rtol=1e-9,atol=1e-11,max_step=.1)
    alt_step=simulate((0,600),y0,inp.value,inp.derivative,t_eval=t,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.5)
    assert np.max(np.abs(ref.y[0]-alt_solver.y[0]))<1e-3; assert np.max(np.abs(ref.y[0]-alt_step.y[0]))<1e-3
    assert np.max(np.abs(ref.y[1]-alt_solver.y[1]))<1e-5; assert np.max(np.abs(ref.y[1]-alt_step.y[1]))<1e-5
