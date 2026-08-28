from __future__ import annotations
import numpy as np
from .model import ModelParams, rhs, flow_q


def simulate_rk4_fixed(t_eval,y0,A_fn,dA_fn,p:ModelParams=ModelParams(),autoregulation:bool=True,dt:float=0.02):
    tt=np.asarray(t_eval,dtype=float)
    if tt.ndim!=1 or len(tt)<1 or np.any(np.diff(tt)<=0): raise ValueError('t_eval must be a nonempty strictly increasing vector')
    if dt<=0: raise ValueError('dt must be positive')
    y=np.asarray(y0,dtype=float).copy(); out=np.empty((2,len(tt)),dtype=float); out[:,0]=y
    min_gap=float('inf'); min_P=y[0]; max_P=y[0]; min_C=y[1]; max_C=y[1]
    def f(t,z): return rhs(t,z,A_fn,dA_fn,p=p,autoregulation=autoregulation)
    for i in range(len(tt)-1):
        span=tt[i+1]-tt[i]; n=int(round(span/dt))
        if n<1 or abs(n*dt-span)>1e-9: raise ValueError(f't_eval interval {span} is not an integer multiple of dt={dt}')
        t0=tt[i]
        for j in range(n):
            t=t0+j*dt; k1=f(t,y); k2=f(t+dt/2.0,y+dt*k1/2.0); k3=f(t+dt/2.0,y+dt*k2/2.0); k4=f(t+dt,y+dt*k3)
            y=y+(dt/6.0)*(k1+2*k2+2*k3+k4); A=float(A_fn(t+dt)); flow_q(A,float(y[0]),float(y[1]),p)
            min_gap=min(min_gap,A-float(y[0])); min_P=min(min_P,float(y[0])); max_P=max(max_P,float(y[0])); min_C=min(min_C,float(y[1])); max_C=max(max_C,float(y[1]))
        out[:,i+1]=y
    return out,{'min_A_minus_P_mmHg':float(min_gap),'min_P_mmHg':float(min_P),'max_P_mmHg':float(max_P),'min_C':float(min_C),'max_C':float(max_C)}
