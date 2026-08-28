from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ConstantMAP:
    level: float = 100.0
    def value(self, t): return float(self.level)
    def derivative(self, t): return 0.0

@dataclass(frozen=True)
class SmoothStepMAP:
    base: float = 100.0
    amplitude: float = 10.0
    center: float = 60.0
    width: float = 5.0
    def value(self, t):
        u=(t-self.center)/self.width
        return float(self.base+self.amplitude*0.5*(1.0+np.tanh(u)))
    def derivative(self, t):
        u=(t-self.center)/self.width
        sech2=1.0/np.cosh(u)**2 if abs(u)<350 else 0.0
        return float(self.amplitude*0.5/self.width*sech2)

@dataclass(frozen=True)
class RampMAP:
    base: float = 95.0
    slope: float = 0.02
    def value(self, t): return float(self.base+self.slope*t)
    def derivative(self, t): return float(self.slope)

@dataclass(frozen=True)
class SinusoidMAP:
    base: float = 100.0
    amplitude: float = 5.0
    period: float = 120.0
    def value(self, t):
        w=2*np.pi/self.period
        return float(self.base+self.amplitude*np.sin(w*t))
    def derivative(self, t):
        w=2*np.pi/self.period
        return float(self.amplitude*w*np.cos(w*t))

class BandLimitedNoiseMAP:
    def __init__(self, base=100.0, rms_scale=1.5, seed=20260828, n=8):
        rng=np.random.default_rng(seed); self.base=float(base)
        self.freq=rng.uniform(1/300,1/30,n); self.phase=rng.uniform(0,2*np.pi,n)
        raw=rng.normal(size=n); raw/=np.sqrt(np.mean(raw**2)); self.amp=raw*rms_scale/np.sqrt(n)
    def value(self,t):
        return float(self.base+np.sum(self.amp*np.sin(2*np.pi*self.freq*t+self.phase)))
    def derivative(self,t):
        return float(np.sum(self.amp*(2*np.pi*self.freq)*np.cos(2*np.pi*self.freq*t+self.phase)))
