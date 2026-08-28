from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
from scipy.signal import find_peaks, lfilter


@dataclass(frozen=True)
class PreprocessConfig:
    expected_fs_hz: float = 50.0
    fs_tolerance_hz: float = 0.5
    abp_min_sample_mmHg: float = 0.0
    abp_max_sample_mmHg: float = 300.0
    min_beat_s: float = 0.30
    max_beat_s: float = 2.00
    min_pulse_pressure_mmHg: float = 5.0
    max_pulse_pressure_mmHg: float = 150.0
    min_beat_map_mmHg: float = 20.0
    max_beat_map_mmHg: float = 200.0
    detection_smooth_s: float = 0.08
    peak_prominence_mmHg: float = 5.0
    map_tau_s: float = 5.0
    min_segment_s: float = 150.0
    burn_in_s: float = 120.0
    output_prediction_hz: float = 1.0

    def to_dict(self): return asdict(self)


def _validate_timebase(time_s: np.ndarray, cfg: PreprocessConfig) -> float:
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<10: raise ValueError("time_s must be a 1-D array with >=10 samples.")
    dt=np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt<=0): raise ValueError("time_s must be strictly increasing and finite.")
    fs=1.0/np.median(dt)
    if abs(fs-cfg.expected_fs_hz)>cfg.fs_tolerance_hz:
        raise ValueError(f"Sampling frequency {fs:.3f} Hz is outside frozen {cfg.expected_fs_hz}±{cfg.fs_tolerance_hz} Hz gate.")
    if np.max(np.abs(dt-np.median(dt)))>0.05*np.median(dt): raise ValueError("Irregular sampling exceeds the frozen 5% tolerance.")
    return float(fs)


def detect_abp_beats(time_s,abp_mmHg,cfg:PreprocessConfig=PreprocessConfig())->List[Dict]:
    t=np.asarray(time_s,float); abp=np.asarray(abp_mmHg,float)
    if len(t)!=len(abp): raise ValueError("time_s and abp_mmHg lengths differ.")
    fs=_validate_timebase(t,cfg)
    finite=np.isfinite(abp)
    sample_ok=finite&(abp>=cfg.abp_min_sample_mmHg)&(abp<=cfg.abp_max_sample_mmHg)
    det=abp.copy()
    if not np.all(finite):
        good=np.flatnonzero(finite)
        if len(good)<10: return []
        bad=np.flatnonzero(~finite); det[bad]=np.interp(bad,good,det[good])
    n_smooth=max(1,int(round(cfg.detection_smooth_s*fs))); kernel=np.ones(n_smooth)/n_smooth
    det_smooth=lfilter(kernel,[1.0],det)
    distance=max(1,int(round(cfg.min_beat_s*fs)))
    peaks,_=find_peaks(det_smooth,distance=distance,prominence=cfg.peak_prominence_mmHg)
    if len(peaks)<3: return []
    feet=[]
    for p0,p1 in zip(peaks[:-1],peaks[1:]):
        if p1<=p0+1: continue
        feet.append(p0+int(np.argmin(det_smooth[p0:p1+1])))
    feet=np.array(sorted(set(feet)),dtype=int)
    if len(feet)<2: return []
    beats=[]
    for i0,i1 in zip(feet[:-1],feet[1:]):
        duration=float(t[i1]-t[i0]); seg=abp[i0:i1+1]; seg_t=t[i0:i1+1]; valid=True; reasons=[]
        if duration<cfg.min_beat_s or duration>cfg.max_beat_s: valid=False; reasons.append("duration")
        if not np.all(sample_ok[i0:i1+1]): valid=False; reasons.append("sample_domain")
        if np.all(np.isfinite(seg)):
            sbp=float(np.max(seg)); dbp=float(np.min(seg)); pp=sbp-dbp; map_value=float(np.trapezoid(seg,seg_t)/duration)
        else:
            sbp=dbp=pp=map_value=float("nan"); valid=False; reasons.append("nonfinite")
        if np.isfinite(pp) and not (cfg.min_pulse_pressure_mmHg<=pp<=cfg.max_pulse_pressure_mmHg): valid=False; reasons.append("pulse_pressure")
        if np.isfinite(map_value) and not (cfg.min_beat_map_mmHg<=map_value<=cfg.max_beat_map_mmHg): valid=False; reasons.append("beat_map")
        beats.append({"foot_start_idx":int(i0),"foot_end_idx":int(i1),"start_s":float(t[i0]),"end_s":float(t[i1]),
                      "duration_s":duration,"map_mmHg":map_value,"sbp_mmHg":sbp,"dbp_mmHg":dbp,"pulse_pressure_mmHg":pp,
                      "valid":bool(valid),"reason":"|".join(sorted(set(reasons))) if reasons else ""})
    return beats


class CausalMAPInput:
    def __init__(self,event_times,beat_maps,tau_s):
        event_times=np.asarray(event_times,float); beat_maps=np.asarray(beat_maps,float)
        if len(event_times)<2 or len(event_times)!=len(beat_maps): raise ValueError("A segment needs >=2 completed valid beats.")
        if np.any(np.diff(event_times)<=0): raise ValueError("event_times must increase.")
        if tau_s<=0: raise ValueError("tau_s must be positive.")
        self.event_times=event_times; self.targets=beat_maps; self.tau=float(tau_s)
        self.a_at_event=np.empty_like(event_times); self.a_at_event[0]=beat_maps[0]
        for k in range(len(event_times)-1):
            dt=event_times[k+1]-event_times[k]; target=beat_maps[k]
            self.a_at_event[k+1]=target+(self.a_at_event[k]-target)*np.exp(-dt/self.tau)
    @property
    def start_s(self): return float(self.event_times[0])
    @property
    def end_s(self): return float(self.event_times[-1])
    def _index(self,t):
        if t<self.start_s-1e-12 or t>self.end_s+1e-12: raise ValueError(f"t={t} outside MAP segment [{self.start_s},{self.end_s}]")
        k=int(np.searchsorted(self.event_times,t,side="right")-1); return min(max(k,0),len(self.event_times)-1)
    def value(self,t):
        k=self._index(float(t)); dt=float(t)-self.event_times[k]
        return float(self.targets[k]+(self.a_at_event[k]-self.targets[k])*np.exp(-dt/self.tau))
    def derivative(self,t):
        a=self.value(float(t)); k=self._index(float(t)); return float((self.targets[k]-a)/self.tau)


def build_valid_segments(beats:List[Dict],cfg:PreprocessConfig=PreprocessConfig())->List[Tuple[CausalMAPInput,Dict]]:
    runs=[]; current=[]
    for b in beats:
        if b["valid"]: current.append(b)
        else:
            if current: runs.append(current); current=[]
    if current: runs.append(current)
    out=[]
    for run in runs:
        if len(run)<2: continue
        event_times=np.array([b["end_s"] for b in run],float); maps=np.array([b["map_mmHg"] for b in run],float)
        span=event_times[-1]-event_times[0]
        if span<cfg.min_segment_s: continue
        inp=CausalMAPInput(event_times,maps,cfg.map_tau_s)
        meta={"start_s":inp.start_s,"end_s":inp.end_s,"score_start_s":inp.start_s+cfg.burn_in_s,"n_beats":len(run),"duration_s":float(span)}
        out.append((inp,meta))
    return out
