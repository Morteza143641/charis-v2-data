from __future__ import annotations
from pathlib import Path
import re
import numpy as np

GAIN_RE=re.compile(r"(?P<gain>[+-]?\d+(?:\.\d+)?)(?:\((?P<baseline>[+-]?\d+)\))?/(?P<unit>\S+)")

def parse_header(hea_path:str|Path):
    hea_path=Path(hea_path); lines=hea_path.read_text(encoding='utf-8').splitlines()
    if not lines: raise ValueError('empty header')
    first=lines[0].split()
    if len(first)<4: raise ValueError(f'unsupported record header: {lines[0]}')
    record=first[0]; n_sig=int(first[1]); fs=float(first[2]); n_samples=int(first[3]); sig=[]
    for line in lines[1:1+n_sig]:
        parts=line.split()
        if len(parts)<3: raise ValueError(f'unsupported signal line: {line}')
        fmt=parts[1].split('x')[0]
        if fmt!='16': raise ValueError(f'CHARIS V2 loader supports WFDB format 16 only, got {parts[1]}')
        m=GAIN_RE.fullmatch(parts[2])
        if not m: raise ValueError(f'cannot parse gain/baseline/unit token: {parts[2]}')
        baseline=int(m.group('baseline') or 0)
        sig.append({'file':parts[0],'format':fmt,'gain':float(m.group('gain')),'baseline':baseline,'unit':m.group('unit'),'name':parts[-1]})
    if len(sig)!=n_sig: raise ValueError(f'header declared {n_sig} signals, parsed {len(sig)}')
    return {'record':record,'n_sig':n_sig,'fs_hz':fs,'n_samples':n_samples,'signals':sig}

def channel_index(header:dict,name:str)->int:
    names=[s['name'] for s in header['signals']]
    if names.count(name)!=1: raise ValueError(f'expected exactly one {name} channel; got {names}')
    return names.index(name)

def load_channel(hea_path,dat_path,name):
    h=parse_header(hea_path); idx=channel_index(h,name); raw=np.memmap(dat_path,dtype='<i2',mode='r'); expected=h['n_samples']*h['n_sig']
    if raw.size<expected: raise ValueError(f'DAT shorter than header: {raw.size} < {expected}')
    dig=np.asarray(raw[idx:expected:h['n_sig']],dtype=np.float64); s=h['signals'][idx]; physical=(dig-s['baseline'])/s['gain']
    return physical,{'record':h['record'],'fs_hz':h['fs_hz'],'n_samples':h['n_samples'],'selected_channel':name,'selected_channel_index':idx,'gain':s['gain'],'baseline':s['baseline'],'unit':s['unit']}

def load_abp_only(hea_path,dat_path):
    h=parse_header(hea_path)
    if abs(h['fs_hz']-50.0)>1e-12: raise ValueError(f'frozen CHARIS sampling frequency is 50 Hz, got {h["fs_hz"]}')
    abp,meta=load_channel(hea_path,dat_path,'ABP'); t=np.arange(h['n_samples'],dtype=np.float64)/h['fs_hz']; meta['reference_channel_values_loaded']=False
    return t,abp,meta
