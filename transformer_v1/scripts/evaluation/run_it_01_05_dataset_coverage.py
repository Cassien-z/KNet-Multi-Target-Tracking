from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.data_generation.build_v8_10hz_compat import select_resample_indices
from src.data_association.pipeline_core import process_file
from tracking_system.backend.scenario_catalog import DATASET_CATALOG,V7_QUALITY_PROFILE,V8_QUALITY_PROFILE,scenario_class

RUN=ROOT/'artifacts'/'reports'/'software_test'/'IT-01-05'/'20260923_run01'; FIG=RUN/'figures'; FIG.mkdir(parents=True,exist_ok=True)
WORK=ROOT/'.codex_work'/'it_01_05_audit'; WORK.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
checks=[]; anomalies=[]
def check(cid,desc,cond,actual,expected):
 row={'check_id':cid,'description':desc,'passed':bool(cond),'actual':actual,'expected':expected}; checks.append(row)
 if not cond: anomalies.append({'check_id':cid,'description':desc,'actual':actual,'expected':expected})

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def critical_equal(a,b):
 def arr(x): return np.asarray(x)
 if not np.array_equal(arr(a['t_grid']),arr(b['t_grid'])): return False
 if not np.array_equal(arr(a['UAV_Pos']),arr(b['UAV_Pos'])): return False
 ta=a.get('RegisteredTrack',[]); tb=b.get('RegisteredTrack',[])
 if len(ta)!=len(tb): return False
 for xa,xb in zip(ta,tb):
  for k in ('target_pos','target_vel'):
   if not np.array_equal(arr(xa[k]),arr(xb[k])): return False
  for sensor in ('Radar','EO','ESM','FastLink'):
   if len(xa.get(sensor,[]))!=len(xb.get(sensor,[])): return False
   for ra,rb in zip(xa.get(sensor,[]),xb.get(sensor,[])):
    for k in ('Meas','Meas_ECEF','valid'):
     if k in ra or k in rb:
      if k not in ra or k not in rb or not np.array_equal(arr(ra[k]),arr(rb[k]),equal_nan=True): return False
 return True

paths={
 'v7_raw':ROOT/'data'/'val_data_v7', 'v7_registered':ROOT/'data'/'val_data_v7_registered',
 'train_v7_registered':ROOT/'data'/'train_data_v7_registered',
 'v8_raw':ROOT/'data'/'val_data_v8_interference', 'v8_registered':ROOT/'data'/'val_data_v8_interference_registered',
 'v8_10hz':ROOT/'data'/'val_data_v8_interference_registered_10hz',
}
v7_raw=sorted(paths['v7_raw'].glob('*.mat')); v7_reg=sorted(paths['v7_registered'].glob('*.pkl')); train_reg=sorted(paths['train_v7_registered'].glob('*.pkl'))
v8_raw=sorted(paths['v8_raw'].glob('*.mat')); v8_reg=sorted(paths['v8_registered'].glob('*.pkl')); v8_10=sorted(paths['v8_10hz'].glob('*.pkl'))
class_raw=Counter(scenario_class(p) for p in v7_raw); class_reg=Counter(scenario_class(p) for p in v7_reg); class_train=Counter(scenario_class(p) for p in train_reg)
check('V7-RAW-COUNT','V7原始验证文件数',len(v7_raw)==2000,len(v7_raw),2000); check('V7-REG-COUNT','V7注册验证文件数',len(v7_reg)==2000,len(v7_reg),2000)
check('V7-CLASS-COVERAGE','V7四类运动各500场景',class_raw==Counter({'CA':500,'CV':500,'CT':500,'CSSW':500}),dict(class_raw),{'CA':500,'CV':500,'CT':500,'CSSW':500})
check('V7-PAIRING','V7原始与注册文件名一一对应',{p.stem for p in v7_raw}=={p.stem for p in v7_reg},len({p.stem for p in v7_raw}^{p.stem for p in v7_reg}),0)
check('V8-INVENTORY','V8包含1个原始、1个40Hz注册和1个10Hz兼容场景',(len(v8_raw),len(v8_reg),len(v8_10))==(1,1,1),(len(v8_raw),len(v8_reg),len(v8_10)),(1,1,1))

# Full validation-file readability and integrity hashes.
manifest=[]; raw_errors=[]; reg_errors=[]; sensor_presence=Counter(); target_counts=Counter(); frame_counts=Counter(); platform_counts=Counter()
for i,p in enumerate(v7_raw,1):
 try:
  variables={name for name,shape,dtype in sio.whosmat(p)}; ok={'Data_Input','Data_Label','truth'}.issubset(variables)
  if not ok: raw_errors.append(f'{p.name}:missing={sorted({"Data_Input","Data_Label","truth"}-variables)}')
 except Exception as e: ok=False; raw_errors.append(f'{p.name}:{type(e).__name__}:{e}')
 manifest.append({'dataset':'V7 raw','file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p),'readable':ok})
 if i%250==0: print(f'V7 raw audit {i}/{len(v7_raw)}',flush=True)
for i,p in enumerate(v7_reg,1):
 try:
  with p.open('rb') as f: d=pickle.load(f)
  tracks=d.get('RegisteredTrack',[]); tg=np.asarray(d.get('t_grid',[])); up=np.asarray(d.get('UAV_Pos',[]))
  ok=bool(tracks) and tg.size>0 and up.ndim==3
  target_counts[len(tracks)]+=1; frame_counts[len(tg)]+=1; platform_counts[up.shape[1] if up.ndim==3 else -1]+=1
  for s in ('Radar','EO','ESM','FastLink'):
   if all(s in t and len(t[s])>0 for t in tracks): sensor_presence[s]+=1
  if not ok: reg_errors.append(f'{p.name}:structure')
 except Exception as e: ok=False; reg_errors.append(f'{p.name}:{type(e).__name__}:{e}')
 manifest.append({'dataset':'V7 registered','file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p),'readable':ok})
 if i%250==0: print(f'V7 registered audit {i}/{len(v7_reg)}',flush=True)
check('V7-RAW-READABLE','全部V7原始MAT头部可读取且关键字段完整',not raw_errors,len(raw_errors),0)
check('V7-REG-READABLE','全部V7注册PKL可反序列化且结构完整',not reg_errors,len(reg_errors),0)
check('V7-SHAPE','V7注册样本均为3目标、100帧、5平台',target_counts==Counter({3:2000}) and frame_counts==Counter({100:2000}) and platform_counts==Counter({5:2000}),{'targets':dict(target_counts),'frames':dict(frame_counts),'platforms':dict(platform_counts)},{'targets':{3:2000},'frames':{100:2000},'platforms':{5:2000}})
check('V7-MODALITIES','全部V7注册场景包含Radar/EO/ESM/FastLink',all(sensor_presence[s]==2000 for s in ('Radar','EO','ESM','FastLink')),dict(sensor_presence),{'Radar':2000,'EO':2000,'ESM':2000,'FastLink':2000})

# Full registered train/validation exact-file hash isolation.
train_hashes={};
for i,p in enumerate(train_reg,1):
 train_hashes[sha256(p)]=p.name
 if i%1000==0: print(f'train hash {i}/{len(train_reg)}',flush=True)
val_hashes={row['sha256']:row['file'] for row in manifest if row['dataset']=='V7 registered'}
overlap=set(train_hashes)&set(val_hashes)
check('SPLIT-HASH-ISOLATION','V7训练与验证注册文件无完全相同SHA-256',not overlap,len(overlap),0)
# Critical tensor comparisons for first/last file of every class.
sampled_split=[]; critical_duplicates=0
for cls in ('CV','CA','CT','CSSW'):
 tp=[p for p in train_reg if scenario_class(p)==cls]; vp=[p for p in v7_reg if scenario_class(p)==cls]
 for t,v in [(tp[0],vp[0]),(tp[-1],vp[-1])]:
  with t.open('rb') as f: td=pickle.load(f)
  with v.open('rb') as f: vd=pickle.load(f)
  same=critical_equal(td,vd); critical_duplicates+=int(same); sampled_split.append({'class':cls,'train_file':t.name,'validation_file':v.name,'critical_tensors_equal':same,'train_sha256':sha256(t),'validation_sha256':sha256(v)})
check('SPLIT-TENSOR-ISOLATION','各运动类别首尾样本关键张量均不相同',critical_duplicates==0,critical_duplicates,0)

# Two acceptance scenarios and catalog semantics.
coop=DATASET_CATALOG['scenario_cooperative']; jam=DATASET_CATALOG['scenario_maneuver_jamming']
check('SCENARIO-COUNT','scenario_catalog定义两类验收想定',all(k in DATASET_CATALOG for k in ('scenario_cooperative','scenario_maneuver_jamming')),2,2)
check('COOP-SCENARIO','协同编队想定覆盖CV/CA和四类协同输入',tuple(coop['scenario_classes'])==('CV','CA') and set(coop['sensor_modalities'])=={'雷达','光电','电子侦察','协同数据链'},{'classes':coop['scenario_classes'],'modalities':coop['sensor_modalities']},'CV/CA + Radar/EO/ESM/datalink')
check('JAM-SCENARIO','高机动强干扰想定含三类干扰和四类协同输入',set(jam['interference_modes'])=={'调幅闪烁','软压制','距离抖动'} and set(jam['sensor_modalities'])=={'雷达','光电','电子侦察','协同数据链'},{'interference':jam['interference_modes'],'modalities':jam['sensor_modalities']},'3 modes + 4 inputs')
check('MOTION-NOT-SCENARIO','CV/CA/CT/CSSW作为运动类型统计而非想定数量',set(class_raw)=={'CV','CA','CT','CSSW'},sorted(class_raw),'four maneuver classes')

# V8 full structure and 40Hz -> 10Hz compatibility.
raw8=sio.loadmat(v8_raw[0],simplify_cells=True); frames8=list(raw8['Data_Input']); labels8=np.asarray(raw8['Data_Label'],float)
with v8_reg[0].open('rb') as f: reg40=pickle.load(f)
with v8_10[0].open('rb') as f: reg10=pickle.load(f)
check('V8-RAW-SHAPE','V8原始样本为400帧、5目标、5平台',len(frames8)==400 and labels8.shape[:2]==(400,5) and len(frames8[0]['UAV'])==5,{'frames':len(frames8),'labels':labels8.shape,'platforms':len(frames8[0]['UAV'])},{'frames':400,'targets':5,'platforms':5})
check('V8-REG-SHAPE','V8注册源为400帧，兼容视图为100帧',len(reg40['t_grid'])==400 and len(reg10['t_grid'])==100,(len(reg40['t_grid']),len(reg10['t_grid'])),(400,100))
indices=select_resample_indices(np.asarray(reg40['t_grid']),0.1)
compat_ok=np.array_equal(np.asarray(reg10['t_grid']),np.asarray(reg40['t_grid'])[indices]) and np.array_equal(np.asarray(reg10['UAV_Pos']),np.asarray(reg40['UAV_Pos'])[indices])
for a,b in zip(reg40['RegisteredTrack'],reg10['RegisteredTrack']):
 compat_ok &= np.array_equal(np.asarray(b['target_pos']),np.asarray(a['target_pos'])[indices])
 for s in ('Radar','EO','ESM','FastLink'):
  for sa,sb in zip(a[s],b[s]):
   for k in ('Meas','Meas_ECEF','valid'):
    if k in sa: compat_ok &= np.array_equal(np.asarray(sb[k]),np.asarray(sa[k])[indices],equal_nan=True)
check('V8-COMPATIBILITY','V8所有时间对齐字段使用同一每4帧抽样索引',bool(compat_ok),{'indices':indices[:5].tolist(),'last':int(indices[-1]),'count':len(indices)},{'first':[0,4,8,12,16],'last':396,'count':100})

# Actual interference manifestations.
t=np.array([float(f['GlobalTime']) for f in frames8]); radar_mask=np.array([np.asarray(f['UAV'][0]['Radar']['Mask']).reshape(-1) for f in frames8],bool); radar_meas=np.array([np.asarray(f['UAV'][0]['Radar']['Meas'],float) for f in frames8])
all_zero=np.all(~radar_mask,axis=1); groups=[]
for idx in np.flatnonzero(all_zero):
 if not groups or idx>groups[-1][-1]+1: groups.append([int(idx)])
 else: groups[-1].append(int(idx))
blink_groups=[g for g in groups if len(g)>=8]
blink_intervals=[{'start_s':float(t[g[0]]),'end_s':float(t[g[-1]]),'frames':len(g)} for g in blink_groups]
check('INTERFERENCE-BLINK','实测出现4段约0.2秒雷达全盲区',len(blink_groups)==4 and all(len(g)==8 for g in blink_groups),blink_intervals,'4 intervals x 8 frames at 40Hz')
positions=labels8[:,:,[0,2,4]]; master_pos=np.array([f['UAV'][0]['Pose']['PositionWorld'] for f in frames8],float); true_range=np.linalg.norm(positions-master_pos[:,None,:],axis=2)
blink_frame=np.zeros(len(t),bool)
for g in blink_groups: blink_frame[g]=True
bin_defs=[('≤12 km',true_range<=12000),('12–18 km',(true_range>12000)&(true_range<=18000)),('>18 km',true_range>18000)]
range_bins=[]
for name,sel in bin_defs:
 eligible=sel&(~blink_frame[:,None]); total=int(eligible.sum()); missing=int((~radar_mask&eligible).sum()); rate=missing/total if total else None; range_bins.append({'range_bin':name,'slots':total,'missing':missing,'missing_rate':rate})
check('INTERFERENCE-SUPPRESSION','排除闪烁后雷达缺失率随距离区间上升',range_bins[0]['missing_rate']<range_bins[1]['missing_rate']<range_bins[2]['missing_rate'],[r['missing_rate'] for r in range_bins],'strictly increasing')
valid=radar_mask&(~blink_frame[:,None]); residual=radar_meas[:,:,2][valid]-true_range[valid]
smear={'samples':int(residual.size),'mean_m':float(np.mean(residual)),'std_m':float(np.std(residual)),'p95_abs_m':float(np.percentile(np.abs(residual),95)),'fraction_abs_gt_240m':float(np.mean(np.abs(residual)>240)),'max_abs_m':float(np.max(np.abs(residual)))}
check('INTERFERENCE-SMEARING','实测距离残差存在显著大于基础80米噪声的长尾',smear['p95_abs_m']>240 and smear['max_abs_m']>800,smear,{'p95_abs_m':'>240','max_abs_m':'>800'})

# Re-run one frozen raw sample twice and compare output bytes.
repro_a=WORK/'repro_a.pkl'; repro_b=WORK/'repro_b.pkl'; sample_raw=paths['v7_raw']/'track_cv_0000.mat'
process_file(sample_raw,repro_a,verbose=False); process_file(sample_raw,repro_b,verbose=False); ha=sha256(repro_a); hb=sha256(repro_b)
check('REPRODUCIBLE-REGISTRATION','同一冻结V7原始样本两次注册输出SHA-256一致',ha==hb,{'first':ha,'second':hb},'equal')

# Write evidence tables.
for p in v8_raw+v8_reg+v8_10:
 manifest.append({'dataset':p.parent.name,'file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p),'readable':True})
with (RUN/'dataset_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['dataset','file','bytes','sha256','readable']); w.writeheader(); w.writerows(manifest)
with (RUN/'scenario_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['check_id','description','passed','actual','expected']); w.writeheader(); w.writerows(checks)
with (RUN/'split_isolation_sample.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(sampled_split[0].keys())); w.writeheader(); w.writerows(sampled_split)
with (RUN/'interference_observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
 rows=[{'evidence':'blinking','item':f"{x['start_s']:.3f}-{x['end_s']:.3f}s",'value':x['frames']} for x in blink_intervals]+[{'evidence':'soft_suppression','item':x['range_bin'],'value':x['missing_rate']} for x in range_bins]+[{'evidence':'range_smearing','item':k,'value':v} for k,v in smear.items()]
 w=csv.DictWriter(f,fieldnames=['evidence','item','value']); w.writeheader(); w.writerows(rows)
(RUN/'anomalies.json').write_text(json.dumps(anomalies,ensure_ascii=False,indent=2),encoding='utf-8')

# Figures.
fig,ax=plt.subplots(figsize=(10,5.5)); classes=['CV','CA','CT','CSSW','V8复合干扰']; raw_counts=[class_raw.get(c,0) for c in classes[:-1]]+[len(v8_raw)]; reg_counts=[class_reg.get(c,0) for c in classes[:-1]]+[len(v8_10)]; x=np.arange(len(classes)); w=.36
ax.bar(x-w/2,raw_counts,w,label='原始MAT',color='#3d5a80'); ax.bar(x+w/2,reg_counts,w,label='注册/兼容PKL',color='#98c1d9')
for i,v in enumerate(raw_counts): ax.text(i-w/2,v+15 if v>10 else v+.05,str(v),ha='center');
for i,v in enumerate(reg_counts): ax.text(i+w/2,v+15 if v>10 else v+.05,str(v),ha='center')
ax.set_xticks(x,classes); ax.set_ylabel('场景文件数'); ax.set_title('V7四类运动与V8复合干扰样例清单'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/'figure_1_dataset_inventory.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Scenario coverage matrix.
cols=['雷达','光电','电子侦察','协同数据链','CV','CA','CT','CSSW','复合高机动','调幅闪烁','软压制','距离抖动']; matrix=np.array([[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,1,1,1,1]])
fig,ax=plt.subplots(figsize=(12,3.8)); im=ax.imshow(matrix,cmap=plt.get_cmap('Blues',2),vmin=0,vmax=1,aspect='auto'); ax.set_yticks([0,1],['想定一：协同编队目标跟踪','想定二：高机动强干扰跟踪']); ax.set_xticks(range(len(cols)),cols,rotation=30,ha='right')
for i in range(2):
 for j in range(len(cols)): ax.text(j,i,'覆盖' if matrix[i,j] else '—',ha='center',va='center',color='white' if matrix[i,j] else '#555')
ax.set_title('两类验收想定的模态、运动与干扰覆盖矩阵'); fig.tight_layout(); fig.savefig(FIG/'figure_2_scenario_coverage.png',dpi=180,bbox_inches='tight'); plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(11,7.5)); axes[0].plot(t,radar_mask.sum(axis=1),color='#2f6690',lw=1.3,label='主雷达有效目标数')
for g in blink_groups: axes[0].axvspan(t[g[0]],t[g[-1]]+.025,color='#d62828',alpha=.2)
axes[0].set_ylabel('有效目标数/5'); axes[0].set_title('V8调幅闪烁：四段约0.2秒雷达盲区'); axes[0].grid(alpha=.25); axes[0].legend()
axes[1].bar([r['range_bin'] for r in range_bins],[r['missing_rate']*100 for r in range_bins],color=['#90be6d','#f9c74f','#f94144']);
for i,r in enumerate(range_bins): axes[1].text(i,r['missing_rate']*100+1,f"{r['missing_rate']*100:.1f}%",ha='center')
axes[1].set_ylabel('排除闪烁后的缺失率/%'); axes[1].set_title('V8软压制及视场效应：缺失率随距离增加'); axes[1].grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/'figure_3_interference_manifestation.png',dpi=180,bbox_inches='tight'); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.8)); axes[0].hist(residual,bins=50,color='#4c78a8',alpha=.85); axes[0].axvline(240,color='#d62828',ls='--',label='±240 m'); axes[0].axvline(-240,color='#d62828',ls='--'); axes[0].set_xlabel('雷达距离残差/m'); axes[0].set_ylabel('样本数'); axes[0].set_title(f"距离抖动长尾：P95绝对残差{smear['p95_abs_m']:.1f} m"); axes[0].legend(); axes[0].grid(alpha=.2)
axes[1].plot(np.arange(len(indices)),indices,marker='.',lw=1,color='#2e8b57'); axes[1].plot([0,len(indices)-1],[0,396],color='#555',ls='--',label='每4帧抽取'); axes[1].set_xlabel('10 Hz输出索引'); axes[1].set_ylabel('40 Hz源帧索引'); axes[1].set_title('V8兼容视图：400帧同步抽取为100帧'); axes[1].legend(); axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/'figure_4_smearing_and_compatibility.png',dpi=180,bbox_inches='tight'); plt.close(fig)

result={'test_case':'IT-01-05','generated_at':datetime.now(timezone.utc).isoformat(),'passed':all(x['passed'] for x in checks),'check_count':len(checks),'passed_count':sum(x['passed'] for x in checks),'anomaly_count':len(anomalies),'datasets':{'v7_raw_files':len(v7_raw),'v7_registered_files':len(v7_reg),'v7_class_counts':dict(class_raw),'v7_registered_shapes':{'targets':dict(target_counts),'frames':dict(frame_counts),'platforms':dict(platform_counts)},'v8_raw_files':len(v8_raw),'v8_registered_40hz_files':len(v8_reg),'v8_registered_10hz_files':len(v8_10)},'acceptance_scenarios':[{'id':'scenario_cooperative','name':coop['name'],'classes':coop['scenario_classes'],'modalities':coop['sensor_modalities'],'interference':coop['interference_modes']},{'id':'scenario_maneuver_jamming','name':jam['name'],'classes':jam['scenario_classes'],'modalities':jam['sensor_modalities'],'interference':jam['interference_modes']}],'split_isolation':{'train_registered_files':len(train_reg),'validation_registered_files':len(v7_reg),'exact_sha256_overlap':len(overlap),'sampled_critical_tensor_duplicates':critical_duplicates},'v8_compatibility':{'source_frames':400,'output_frames':100,'indices_first':indices[:5].tolist(),'last_index':int(indices[-1]),'all_time_aligned_fields_match':bool(compat_ok)},'interference_evidence':{'blinking_intervals':blink_intervals,'range_bin_missing':range_bins,'range_smearing':smear},'reproducibility':{'sample':sample_raw.name,'output_sha256':ha,'repeat_sha256':hb,'equal':ha==hb},'scope_note':'V8 is one composite-interference scenario and supports coverage/robustness evidence only; it is not a large-sample performance conclusion.','figures':[p.name for p in sorted(FIG.glob('*.png'))]}
(RUN/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':result['passed'],'checks':result['check_count'],'passed_count':result['passed_count'],'anomalies':result['anomaly_count'],'v7_counts':result['datasets']['v7_class_counts'],'split_overlap':result['split_isolation'],'v8_compatibility':result['v8_compatibility'],'interference':result['interference_evidence']},ensure_ascii=False,indent=2))

