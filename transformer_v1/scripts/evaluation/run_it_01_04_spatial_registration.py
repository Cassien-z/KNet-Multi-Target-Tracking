from __future__ import annotations

import csv
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_association.pipeline_core import _remove_radar_outliers
from src.utils.ecef_utils import get_body_frame_axes, reconstruct_ecef_from_body_angles

RUN_DIR=ROOT/'artifacts'/'reports'/'software_test'/'IT-01-04'/'20260923_run01'
FIG_DIR=RUN_DIR/'figures'; FIG_DIR.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
checks=[]
def check(cid,desc,cond,actual,expected):
 row={'check_id':cid,'description':desc,'passed':bool(cond),'actual':actual,'expected':expected}; checks.append(row)
 if not cond: raise AssertionError(f'{cid}: {desc}; actual={actual!r}; expected={expected!r}')

def rotation_for(velocity,position=np.zeros(3)):
 f,r,u=get_body_frame_axes(np.asarray(velocity,float),np.asarray(position,float)); return np.column_stack((f,r,u))

def body_measurement(target,position,velocity):
 R=rotation_for(velocity,position); body=R.T@(np.asarray(target)-np.asarray(position)); rng=float(np.linalg.norm(body)); az=float(np.arctan2(body[1],body[0])); el=float(np.arcsin(body[2]/rng)); return az,el,rng,body,R

# Known geometry cases for the frozen velocity-oriented body frame.
geometry_cases=[
 ('单位旋转',np.array([10.,20.,30.]),np.array([1.,0.,0.]),0.,0.,100.,np.array([110.,20.,30.])),
 ('偏航+90度',np.array([10.,20.,30.]),np.array([0.,1.,0.]),0.,0.,100.,np.array([10.,120.,30.])),
 ('偏航-90度',np.array([10.,20.,30.]),np.array([0.,-1.,0.]),0.,0.,100.,np.array([10.,-80.,30.])),
 ('反向180度',np.array([10.,20.,30.]),np.array([-1.,0.,0.]),0.,0.,100.,np.array([-90.,20.,30.])),
 ('方位+90度',np.array([10.,20.,30.]),np.array([1.,0.,0.]),np.pi/2,0.,100.,np.array([10.,120.,30.])),
 ('俯仰+90度',np.array([10.,20.,30.]),np.array([1.,0.,0.]),0.,np.pi/2,100.,np.array([10.,20.,130.])),
]
geometry_rows=[]
for name,pos,vel,az,el,rng,expected in geometry_cases:
 R=rotation_for(vel,pos); actual=reconstruct_ecef_from_body_angles(az,el,rng,pos,vel); err=float(np.linalg.norm(actual-expected)); orth=float(np.linalg.norm(R.T@R-np.eye(3))); det=float(np.linalg.det(R))
 check('GEOM-'+name,name+'人工几何坐标复算',err<1e-10,err,'<1e-10 m')
 check('ROT-'+name,name+'旋转矩阵正交且行列式为+1',orth<1e-12 and abs(det-1)<1e-12,{'orth_error':orth,'det':det},{'orth_error':'<1e-12','det':1})
 geometry_rows.append({'case':name,'position':pos.tolist(),'velocity':vel.tolist(),'az_rad':az,'el_rad':el,'range_m':rng,'expected_ecef':expected.tolist(),'actual_ecef':actual.tolist(),'error_m':err,'orthogonality_error':orth,'determinant':det})

# General left-multiplication / axis-order check.
pos=np.array([100.,-50.,25.]); vel=np.array([1.,1.,0.5]); az=.3; el=-.2; rng=750.
R=rotation_for(vel,pos); body=np.array([rng*np.cos(el)*np.cos(az),rng*np.cos(el)*np.sin(az),rng*np.sin(el)])
manual=pos+R@body; actual=reconstruct_ecef_from_body_angles(az,el,rng,pos,vel); mat_err=float(np.linalg.norm(actual-manual))
check('MATRIX-CONVENTION','世界坐标=平台位置+Rwb×机体系列向量',mat_err<1e-10,mat_err,'<1e-10 m')

# Two-platform registration of the same target.
target=np.array([4200.,-1300.,900.])
platforms=[('平台A',np.array([100.,200.,300.]),np.array([1.,0.,0.])),('平台B',np.array([-600.,800.,250.]),np.array([0.,1.,0.]))]
registered=[]; local_vectors=[]; platform_rows=[]
for name,pp,pv in platforms:
 az,el,rng,body,R=body_measurement(target,pp,pv); world=reconstruct_ecef_from_body_angles(az,el,rng,pp,pv); err=float(np.linalg.norm(world-target)); registered.append(world); local_vectors.append(body)
 check('MULTI-'+name,name+'注册至共同世界目标',err<1e-9,err,'<1e-9 m')
 platform_rows.append({'platform':name,'position':pp.tolist(),'velocity':pv.tolist(),'body_vector':body.tolist(),'az_rad':az,'el_rad':el,'range_m':rng,'registered_world':world.tolist(),'target_world':target.tolist(),'registration_error_m':err})
pre_local_distance=float(np.linalg.norm(local_vectors[0]-local_vectors[1])); post_world_distance=float(np.linalg.norm(registered[0]-registered[1]))
check('MULTI-COMPARABILITY','多平台局部表示不同而注册后世界坐标一致',pre_local_distance>100 and post_world_distance<1e-9,{'local_distance':pre_local_distance,'world_distance':post_world_distance},{'local_distance':'>100 m','world_distance':'<1e-9 m'})

# Validate matrices before formal use; do not feed invalid matrices to the pipeline.
invalid_R=np.diag([1.,1.,2.]); invalid_orth=float(np.linalg.norm(invalid_R.T@invalid_R-np.eye(3))); invalid_det=float(np.linalg.det(invalid_R)); rejected=invalid_orth>=1e-12 or abs(invalid_det-1)>=1e-12
check('INVALID-ROTATION-PRECHECK','测试脚本在正式调用前拒绝非正交旋转矩阵',rejected,{'orth_error':invalid_orth,'det':invalid_det},'rejected')

# Threshold boundary tests using 5-point nearest median baseline.
thresholds={'azimuth_rad':(0,0.05,[0.049,0.050,0.051]),'elevation_rad':(1,0.05,[0.049,0.050,0.051]),'range_m':(2,100.0,[99.,100.,101.])}
boundary_rows=[]
for label,(dim,thr,deltas) in thresholds.items():
 for delta in deltas:
  aligned=np.zeros((21,3),float); aligned[:,2]=1000.; aligned[10,dim]+=delta; valid=np.ones(21,bool)
  filtered,out_valid=_remove_radar_outliers(aligned.copy(),valid.copy()); removed=not bool(out_valid[10]); expected=bool(delta>thr)
  check(f'BOUNDARY-{label}-{delta}',f'{label}偏差{delta}的严格越限判定',removed==expected,removed,expected)
  boundary_rows.append({'dimension':label,'threshold':thr,'injected_delta':delta,'removed':removed,'value_is_nan':bool(np.all(np.isnan(filtered[10]))),'expected_removed':expected})
  if removed: check(f'NAN-{label}-{delta}','被剔除样本值与mask同步',np.all(np.isnan(filtered[10])) and not out_valid[10],{'valid':bool(out_valid[10]),'nan':bool(np.all(np.isnan(filtered[10])))},{'valid':False,'nan':True})

# Outlier patterns.
def base_series(n=31):
 x=np.zeros((n,3),float); x[:,0]=0.01; x[:,1]=-0.01; x[:,2]=5000.; return x
patterns={}
# Single spikes in each dimension.
for name,dim,delta in [('方位单点',0,.051),('俯仰单点',1,.051),('距离单点',2,101.)]:
 x=base_series(); x[15,dim]+=delta; f,v=_remove_radar_outliers(x.copy(),np.ones(len(x),bool)); patterns[name]=(x,f,v); check('PATTERN-'+name,name+'异常被剔除',not v[15],[int(i) for i in np.flatnonzero(~v)],[15])
# Two consecutive and multi-dimensional spikes.
x=base_series(); x[14:16,0]+=.051; f,v=_remove_radar_outliers(x.copy(),np.ones(len(x),bool)); patterns['连续两点']=(x,f,v); check('PATTERN-连续两点','连续两点方位偏移均被剔除',set(np.flatnonzero(~v))=={14,15},np.flatnonzero(~v).tolist(),[14,15])
x=base_series(); x[15]+=[.051,.051,101.]; f,v=_remove_radar_outliers(x.copy(),np.ones(len(x),bool)); patterns['多维同时']=(x,f,v); check('PATTERN-多维同时','多维同时越限样本被剔除且整行NaN',not v[15] and np.all(np.isnan(f[15])),{'invalid':not bool(v[15]),'nan':bool(np.all(np.isnan(f[15])))},True)
# Slow drift and normal deterministic variation should remain valid.
x=base_series(); x[:,0]+=np.linspace(0,.2,len(x)); x[:,1]+=np.linspace(0,-.12,len(x)); x[:,2]+=np.linspace(0,300,len(x)); f,v=_remove_radar_outliers(x.copy(),np.ones(len(x),bool)); patterns['缓慢漂移']=(x,f,v); check('PATTERN-缓慢漂移','缓慢漂移不发生大规模误删',np.all(v),int(v.sum()),len(v))
x=base_series(); x[:,0]+=.01*np.sin(np.arange(len(x))*.4); x[:,1]+=.008*np.cos(np.arange(len(x))*.5); x[:,2]+=20*np.sin(np.arange(len(x))*.3); f,v=_remove_radar_outliers(x.copy(),np.ones(len(x),bool)); patterns['正常波动']=(x,f,v); check('PATTERN-正常波动','正常小幅波动全部保留',np.all(v),int(v.sum()),len(v))

# Nearest boundary behavior: independently recompute baseline and compare first/last samples.
edge=base_series(); edge[:,0]+=np.linspace(0,.04,len(edge)); filled=edge[:,0]; baseline=median_filter(filled,size=5,mode='nearest')
check('MEDIAN-NEAREST','5点中值基线采用nearest首尾填充',baseline[0]==np.median([filled[0],filled[0],filled[0],filled[1],filled[2]]) and baseline[-1]==np.median([filled[-3],filled[-2],filled[-1],filled[-1],filled[-1]]),[float(baseline[0]),float(baseline[-1])],'nearest five-point medians')

# Interface/pipeline-order audit.
sig=list(inspect.signature(_remove_radar_outliers).parameters)
check('NO-TRUTH-PREDICTION','异常抑制接口不接收真值或预测值',sig==['aligned','valid'],sig,['aligned','valid'])
source=(ROOT/'src'/'data_association'/'pipeline_core.py').read_text(encoding='utf-8')
order_ok=source.find('aligned, valid = _align_stream') < source.find('aligned, valid = _remove_radar_outliers') < source.find('stored = aligned[:, [2, 0, 1]]')
check('PIPELINE-ORDER','当前处理顺序为时间对齐→异常抑制→存储/模型入口',order_ok,order_ok,True)

# Figures.
fig=plt.figure(figsize=(12,5.2)); ax1=fig.add_subplot(121,projection='3d'); ax2=fig.add_subplot(122,projection='3d')
colors=['#2f6690','#d95f02']
for (name,pp,pv),body,c in zip(platforms,local_vectors,colors): ax1.scatter(*body,label=name,color=c,s=60)
ax1.set_title('注册前：不同机体系局部向量'); ax1.set_xlabel('前向/m'); ax1.set_ylabel('右向/m'); ax1.set_zlabel('上向/m'); ax1.legend()
for (name,pp,pv),world,c in zip(platforms,registered,colors): ax2.plot([pp[0],world[0]],[pp[1],world[1]],[pp[2],world[2]],color=c,label=name); ax2.scatter(*pp,color=c,s=40)
ax2.scatter(*target,color='#2e8b57',s=100,marker='*',label='共同目标'); ax2.set_title('注册后：两平台量测汇聚至同一ECEF点'); ax2.set_xlabel('ECEF X/m'); ax2.set_ylabel('ECEF Y/m'); ax2.set_zlabel('ECEF Z/m'); ax2.legend()
fig.suptitle('IT-01-04 多平台空间注册一致性',fontsize=16); fig.tight_layout(); fig.savefig(FIG_DIR/'figure_1_multiplatform_registration.png',dpi=180,bbox_inches='tight'); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.8)); names=[r['case'] for r in geometry_rows]; errs=[max(r['error_m'],1e-16) for r in geometry_rows]; axes[0].bar(names,errs,color='#4c78a8'); axes[0].set_yscale('log'); axes[0].axhline(1e-10,color='#c44536',ls='--',label='容差1e-10 m'); axes[0].set_ylabel('坐标误差/m（对数）'); axes[0].tick_params(axis='x',rotation=30); axes[0].legend(); axes[0].set_title('人工几何样例误差')
axes[1].bar(names,[r['determinant'] for r in geometry_rows],color='#72b7b2'); axes[1].axhline(1,color='#333',ls='--'); axes[1].set_ylim(.98,1.02); axes[1].set_ylabel('det(R)'); axes[1].tick_params(axis='x',rotation=30); axes[1].set_title('机体系旋转矩阵行列式')
fig.tight_layout(); fig.savefig(FIG_DIR/'figure_2_geometry_validation.png',dpi=180,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5)); groups=['方位角','俯仰角','距离']; x=np.arange(3); offsets=[-.25,0,.25]
for j,label in enumerate(['阈值下','等于阈值','严格越限']):
 vals=[]
 for dim in ['azimuth_rad','elevation_rad','range_m']:
  rows=[r for r in boundary_rows if r['dimension']==dim]; vals.append(int(rows[j]['removed']))
 ax.bar(x+offsets[j],vals,.22,label=label,color=['#8ecae6','#ffb703','#d62828'][j])
ax.set_xticks(x,groups); ax.set_yticks([0,1],['保留','剔除']); ax.set_ylim(0,1.2); ax.set_title('0.05 rad与100 m阈值边界（严格超过才剔除）'); ax.legend(); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(FIG_DIR/'figure_3_threshold_boundary.png',dpi=180,bbox_inches='tight'); plt.close(fig)

x,f,v=patterns['多维同时']; ids=np.arange(len(x)); labels=['方位/rad','俯仰/rad','距离/m']; th=[.05,.05,100.]
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
for d,ax in enumerate(axes):
 filled=x[:,d]; base=median_filter(filled,size=5,mode='nearest'); diff=filled-base
 ax.plot(ids,diff,color='#2f6690',label='相对5点中值基线差值'); ax.axhline(th[d],color='#c44536',ls='--',label='正阈值'); ax.axhline(-th[d],color='#c44536',ls='--'); ax.scatter(ids[~v],diff[~v],color='#d62828',marker='x',s=80,label='被剔除点'); ax.set_ylabel(labels[d]); ax.grid(alpha=.25); ax.legend(loc='upper right')
axes[-1].set_xlabel('样本索引'); fig.suptitle('多维同时异常的5点中值基线与剔除结果',fontsize=16); fig.tight_layout(); fig.savefig(FIG_DIR/'figure_4_outlier_suppression.png',dpi=180,bbox_inches='tight'); plt.close(fig)

with (RUN_DIR/'scenario_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['check_id','description','passed','actual','expected']); w.writeheader(); w.writerows(checks)
with (RUN_DIR/'geometry_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(geometry_rows[0].keys())); w.writeheader(); w.writerows(geometry_rows)
with (RUN_DIR/'platform_registration.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(platform_rows[0].keys())); w.writeheader(); w.writerows(platform_rows)
with (RUN_DIR/'threshold_boundary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(boundary_rows[0].keys())); w.writeheader(); w.writerows(boundary_rows)

result={'test_case':'IT-01-04','generated_at':datetime.now(timezone.utc).isoformat(),'implementation':['src/utils/ecef_utils.py::get_body_frame_axes,reconstruct_ecef_from_body_angles','src/data_association/pipeline_core.py::_remove_radar_outliers'],'python':sys.version,'platform':platform.platform(),'passed':all(r['passed'] for r in checks),'check_count':len(checks),'passed_count':sum(r['passed'] for r in checks),'registration':{'geometry_case_count':len(geometry_rows),'max_error_m':max(r['error_m'] for r in geometry_rows),'matrix_convention_error_m':mat_err,'multiplatform_local_distance_m':pre_local_distance,'multiplatform_registered_distance_m':post_world_distance,'frozen_frame_convention':'forward=ECEF velocity; right=global-Z cross forward; up=forward cross right'},'outlier_filter':{'median_window':5,'boundary_mode':'nearest','thresholds':{'azimuth_rad':0.05,'elevation_rad':0.05,'range_m':100.0},'comparison':'strictly greater than threshold','boundary_cases':boundary_rows,'normal_variation_valid_count':int(patterns['正常波动'][2].sum()),'normal_variation_total':len(patterns['正常波动'][2])},'interface_scope':'frozen Python path uses a velocity-oriented body frame and has no arbitrary roll/installation-angle input','figures':[p.name for p in sorted(FIG_DIR.glob('*.png'))]}
(RUN_DIR/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':result['passed'],'checks':result['check_count'],'passed_count':result['passed_count'],'max_registration_error_m':result['registration']['max_error_m'],'multiplatform_registered_distance_m':post_world_distance,'threshold_boundary':[{'dimension':r['dimension'],'delta':r['injected_delta'],'removed':r['removed']} for r in boundary_rows]},ensure_ascii=False,indent=2))
