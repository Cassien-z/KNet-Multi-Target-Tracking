from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_association.pipeline_core import _align_stream, time_align_interp

RUN_DIR = ROOT / 'artifacts' / 'reports' / 'software_test' / 'IT-01-03' / '20260923_run01'
FIG_DIR = RUN_DIR / 'figures'
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

checks: list[dict] = []

def check(check_id: str, description: str, condition: bool, actual, expected):
    row = {
        'check_id': check_id,
        'description': description,
        'passed': bool(condition),
        'actual': actual,
        'expected': expected,
    }
    checks.append(row)
    if not condition:
        raise AssertionError(f'{check_id}: {description}; actual={actual!r}, expected={expected!r}')

# 1) >=4 points: PCHIP branch, shape preservation and independent recomputation.
t_src = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_src = np.array([0.0, 1.0, 0.2, 1.4, 0.6])[:, None]
t_dense = np.arange(0.0, 2.0001, 0.05)
pchip_out, pchip_valid = time_align_interp(t_src, y_src, t_dense, np.ones(len(t_dense), bool), 0.45)
pchip_ref = PchipInterpolator(t_src, y_src[:, 0], extrapolate=False)(t_dense)
pchip_diff = float(np.nanmax(np.abs(pchip_out[:, 0] - pchip_ref)))
check('PCHIP-REFERENCE', '不少于4点时与独立PCHIP复算一致', pchip_diff < 1e-12, pchip_diff, '<1e-12')
check('PCHIP-VALID', 'PCHIP样例覆盖区间内全部有效', bool(np.all(pchip_valid)), int(pchip_valid.sum()), len(t_dense))
interval_ok = True
for i in range(len(t_src)-1):
    sel=(t_dense>=t_src[i])&(t_dense<=t_src[i+1])
    lo=min(y_src[i,0],y_src[i+1,0])-1e-12; hi=max(y_src[i,0],y_src[i+1,0])+1e-12
    interval_ok &= bool(np.all((pchip_out[sel,0]>=lo)&(pchip_out[sel,0]<=hi)))
check('PCHIP-NO-OVERSHOOT', '转折序列逐区间无过冲', interval_ok, bool(interval_ok), True)

# 2) 2 and 3 points: linear branch.
linear_cases=[]
for n,times,values,query in [
    (2,np.array([0.,1.]),np.array([2.,6.]),np.array([0.,0.25,0.5,0.75,1.])),
    (3,np.array([0.,0.5,1.]),np.array([0.,2.,1.]),np.array([0.,0.25,0.5,0.75,1.])),
]:
    out,valid=time_align_interp(times,values[:,None],query,np.ones(len(query),bool),0.45)
    ref=np.interp(query,times,values)
    diff=float(np.nanmax(np.abs(out[valid,0]-ref[valid])))
    check(f'LINEAR-{n}PT',f'{n}点时与线性插值独立复算一致',diff<1e-12,diff,'<1e-12')
    linear_cases.append((n,times,values,query,out[:,0],valid))

# 3) single point and zero point behavior through outer stream wrapper.
t_grid=np.arange(5,dtype=float)
raw=np.full((5,1),np.nan); raw[2,0]=42.0
single,single_valid=_align_stream(raw,np.array([False,False,True,False,False]),t_grid)
check('SINGLE-POINT', '单点流只保留原始有效点', np.array_equal(single_valid,[False,False,True,False,False]) and single[2,0]==42.0, single_valid.astype(int).tolist(), [0,0,1,0,0])
zero,zero_valid=_align_stream(np.full((5,1),np.nan),np.zeros(5,bool),t_grid)
check('ZERO-POINT', '零点流全部保持无效', not np.any(zero_valid) and np.all(np.isnan(zero)), int(zero_valid.sum()), 0)

# 4) no extrapolation.
q=np.array([-0.1,0.,0.5,1.,1.1])
out,noext_valid=time_align_interp(np.array([0.,1.]),np.array([[0.],[10.]]),q,np.ones(len(q),bool),0.45)
check('NO-EXTRAP', '覆盖区间外禁止外推', np.array_equal(noext_valid,[False,True,False,True,False]) and np.isnan(out[[0,4],0]).all(), noext_valid.astype(int).tolist(), [0,1,0,1,0])

# 5) exact 0.45-second boundary.
bq=np.array([0.44,0.45,0.46])
bout,bvalid=time_align_interp(np.array([0.,1.]),np.array([[0.],[1.]]),bq,np.ones(3,bool),0.45)
check('GAP-BOUNDARY', '0.44和0.45秒有效，0.46秒无效', np.array_equal(bvalid,[True,True,False]), bvalid.astype(int).tolist(), [1,1,0])

# 6-8) independent sensor streams, target isolation, validity counts, auditable source timestamps.
grid=np.round(np.arange(0.,4.0001,0.1),10)
sensor_defs={
    'Radar': (np.arange(0,len(grid),5), 10.0),
    'EO': (np.arange(1,len(grid),7), 100.0),
    'ESM': (np.arange(2,len(grid),10), 1000.0),
}
sensor_summary=[]
sensor_outputs={}
for sensor,(ids,offset) in sensor_defs.items():
    raw=np.full((len(grid),1),np.nan); valid=np.zeros(len(grid),bool); source=np.full(len(grid),np.nan)
    valid[ids]=True; source[ids]=grid[ids]+0.02
    raw[ids,0]=offset+np.arange(len(ids),dtype=float)
    aligned,av=_align_stream(raw,valid,grid,source)
    finite=aligned[av,0]
    low=float(np.min(finite)) if len(finite) else None; high=float(np.max(finite)) if len(finite) else None
    isolated = bool(low>=offset-1e-9 and high<offset+100.0)
    check(f'SENSOR-{sensor.upper()}',f'{sensor}独立对齐且数值范围不混插',isolated,[low,high],f'[{offset},{offset+100})')
    sensor_summary.append({'sensor':sensor,'source_valid':int(valid.sum()),'aligned_valid':int(av.sum()),'source_rate':float(valid.mean()),'aligned_rate':float(av.mean()),'value_min':low,'value_max':high,'source_timestamps':';'.join(f'{x:.2f}' for x in source[valid])})
    sensor_outputs[sensor]=(aligned,av,valid)

# Target isolation with different offsets.
base_raw=np.full((len(grid),1),np.nan); base_valid=np.zeros(len(grid),bool); ids=np.arange(0,len(grid),5); base_valid[ids]=True; base_raw[ids,0]=np.arange(len(ids))
t0,v0=_align_stream(base_raw,base_valid,grid)
t1,v1=_align_stream(base_raw+10000.0,base_valid,grid)
common=v0&v1
check('TARGET-ISOLATION','不同目标偏置在对齐后保持分离', bool(np.allclose(t1[common,0]-t0[common,0],10000.0)), float(np.min(t1[common,0]-t0[common,0])), 10000.0)

# 9) nonmonotonic and duplicate timestamps: independently reproduce sort+first-unique rule.
dup_t=np.array([2.,0.,1.,1.,3.]); dup_y=np.array([20.,0.,10.,999.,30.])[:,None]; dup_q=np.array([0.,1.,1.5,2.,3.])
dup_out,dup_valid=time_align_interp(dup_t,dup_y,dup_q,np.ones(len(dup_q),bool),0.45)
order=np.argsort(dup_t); sorted_t=dup_t[order]; sorted_y=dup_y[order,0]; unique_t,unique_idx=np.unique(sorted_t,return_index=True); kept=sorted_y[unique_idx]
expected_at_one=float(kept[np.where(unique_t==1.0)[0][0]])
actual_at_one=float(dup_out[np.where(dup_q==1.0)[0][0],0])
check('SORT-DEDUP','非单调时标排序并按冻结代码去重', dup_valid[np.where(dup_q==1.0)[0][0]] and actual_at_one==expected_at_one, actual_at_one, expected_at_one)

# 10) grid-coincident values are preserved.
coincident=np.isin(t_dense,t_src)
check('COINCIDENT-PRESERVE','源时标与公共网格重合点数值保持', bool(np.allclose(pchip_out[coincident,0],y_src[:,0])), pchip_out[coincident,0].tolist(), y_src[:,0].tolist())

# Plot 1: interpolation branches.
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
axes[0].plot(t_dense,pchip_out[:,0],label='PCHIP对齐值',lw=2,color='#2f6690'); axes[0].scatter(t_src,y_src[:,0],label='原始异步点',color='#d95f02',zorder=3)
axes[0].set_title('不少于4点：PCHIP保持局部形状'); axes[0].set_xlabel('时间/s'); axes[0].set_ylabel('样例值'); axes[0].grid(alpha=.25); axes[0].legend()
n,times,values,query,lout,lvalid=linear_cases[0]
axes[1].plot(query,lout,label='线性对齐值',marker='o',color='#5f9e6e'); axes[1].scatter(times,values,label='2个源点',color='#a23b72',s=60,zorder=3)
axes[1].set_title('2至3点：线性插值'); axes[1].set_xlabel('时间/s'); axes[1].grid(alpha=.25); axes[1].legend()
fig.suptitle('IT-01-03 插值分支独立复算',fontsize=16); fig.tight_layout(); fig.savefig(FIG_DIR/'figure_1_interpolation_branches.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Plot 2: boundary.
fig,ax=plt.subplots(figsize=(8,4.8)); colors=['#2e8b57' if x else '#c44536' for x in bvalid]
ax.bar([str(x) for x in bq],bvalid.astype(int),color=colors,width=.6); ax.axvline(1.5,color='#555',ls='--',lw=1)
for i,v in enumerate(bvalid): ax.text(i,0.52,'有效' if v else '无效',ha='center',va='center',color='white',fontweight='bold')
ax.set_ylim(0,1.15); ax.set_yticks([0,1],['无效','有效']); ax.set_xlabel('距最近真实观测时间/s'); ax.set_title('0.45秒最大插值间隔边界（≤0.45秒有效）'); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(FIG_DIR/'figure_2_gap_boundary.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Plot 3: mask behaviors.
mask_matrix=np.vstack([noext_valid,single_valid,zero_valid]).astype(int)
fig,ax=plt.subplots(figsize=(10,3.8)); im=ax.imshow(mask_matrix,aspect='auto',cmap=plt.get_cmap('RdYlGn',2),vmin=0,vmax=1)
ax.set_yticks(range(3),['禁止外推样例','单点流','零点流']); ax.set_xticks(range(5),['区间前','首点','区间内','末点','区间后'])
for i in range(3):
 for j in range(5): ax.text(j,i,'有效' if mask_matrix[i,j] else '无效',ha='center',va='center',fontsize=9)
ax.set_title('缺失保持与禁止外推的输出mask'); fig.colorbar(im,ax=ax,ticks=[0,1],label='mask'); fig.tight_layout(); fig.savefig(FIG_DIR/'figure_3_missing_masks.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Plot 4: per-sensor source/aligned validity.
fig,ax=plt.subplots(figsize=(9,5.2)); names=[x['sensor'] for x in sensor_summary]; x=np.arange(len(names)); w=.34
ax.bar(x-w/2,[r['source_valid'] for r in sensor_summary],w,label='原始有效点',color='#3d5a80')
ax.bar(x+w/2,[r['aligned_valid'] for r in sensor_summary],w,label='对齐后有效点',color='#98c1d9')
for i,r in enumerate(sensor_summary): ax.text(i+w/2,r['aligned_valid']+.5,str(r['aligned_valid']),ha='center',fontsize=10)
ax.set_xticks(x,names); ax.set_ylabel('有效点数'); ax.set_title('雷达、光电、电子侦察独立时间对齐有效性'); ax.legend(); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(FIG_DIR/'figure_4_multisensor_validity.png',dpi=180,bbox_inches='tight'); plt.close(fig)

with (RUN_DIR/'scenario_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['check_id','description','passed','actual','expected']); writer.writeheader(); writer.writerows(checks)
with (RUN_DIR/'boundary_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['distance_s','valid','aligned_value']); writer.writeheader()
    for t,v,y in zip(bq,bvalid,bout[:,0]): writer.writerow({'distance_s':t,'valid':bool(v),'aligned_value':None if not np.isfinite(y) else float(y)})
with (RUN_DIR/'multisensor_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(sensor_summary[0].keys())); writer.writeheader(); writer.writerows(sensor_summary)

result={
    'test_case':'IT-01-03',
    'generated_at':datetime.now(timezone.utc).isoformat(),
    'implementation':'src/data_association/pipeline_core.py::time_align_interp,_align_stream',
    'python':sys.version,
    'platform':platform.platform(),
    'passed':all(r['passed'] for r in checks),
    'check_count':len(checks),
    'passed_count':sum(r['passed'] for r in checks),
    'method_selection':{'pchip_min_points':4,'linear_points':[2,3],'single_point':'outer wrapper preserves original point','zero_point':'all invalid'},
    'max_interp_gap_s':0.45,
    'boundary_valid':{'0.44':bool(bvalid[0]),'0.45':bool(bvalid[1]),'0.46':bool(bvalid[2])},
    'pchip_max_abs_diff_vs_reference':pchip_diff,
    'pchip_no_overshoot':bool(interval_ok),
    'sensor_summary':sensor_summary,
    'sort_dedup':{'input_times':dup_t.tolist(),'sorted_unique_times':unique_t.tolist(),'kept_values':kept.tolist(),'returned_value_at_1s':actual_at_one},
    'figures':[p.name for p in sorted(FIG_DIR.glob('*.png'))],
}
(RUN_DIR/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':result['passed'],'checks':result['check_count'],'passed_count':result['passed_count'],'boundary_valid':result['boundary_valid'],'sensor_summary':sensor_summary},ensure_ascii=False,indent=2))
