"""Run a reproducible fine-tuning/validation pilot on prepared reference data.

Uses the production CoopTrackerModel and train.forward_batch, preserving
native 1 Hz timing. Calibration was fitted solely by the preparation command.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.model import CoopTrackerModel, radar_body_to_world_xyz
from src.reference_pilot_dataset import ReferencePilotDataset
from train import forward_batch
from scripts.data_generation.prepare_reference_pilot import sha256_file


def error_stats(errors):
    errors=np.asarray(errors,dtype=float)
    if not errors.size:return None
    if not np.isfinite(errors).all():raise FloatingPointError('Nonfinite evaluation errors')
    return {'frames':int(errors.size),'rmse_m':float(np.sqrt(np.mean(errors**2))),
            'mean_m':float(np.mean(errors)),'p95_m':float(np.quantile(errors,.95)),
            'max_m':float(np.max(errors))}


@torch.inference_mode()
def evaluate(model,loader,device,keep_arrays=False):
    model.eval();errors=[];velocity_errors=[];per_window=[];examples=[];initialization_frames=[]
    for batch in loader:
        prediction,labels,_,_=forward_batch(model,batch,device)
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite model output')
        distance=torch.linalg.vector_norm(prediction[:,:,[0,2,4]]-labels[:,:,[0,2,4]],dim=-1).cpu().numpy()
        velocity_distance=torch.linalg.vector_norm(prediction[:,:,[1,3,5]]-labels[:,:,[1,3,5]],dim=-1).cpu().numpy()
        errors.append(distance);velocity_errors.append(velocity_distance);initialization_frames.extend(batch['initialization_frame'].tolist())
        for i in range(len(distance)):
            per_window.append({'source_target_id':int(batch['source_target_id'][i]),
                'source_start_frame':int(batch['source_start_frame'][i]),'initialization_frame':int(batch['initialization_frame'][i]),**error_stats(distance[i])})
        if keep_arrays and not examples:
            examples.append({'pred':prediction[0].cpu().numpy(),'label':labels[0].cpu().numpy(),
                'target_id':int(batch['source_target_id'][0]),'start':int(batch['source_start_frame'][0])})
    distances=np.concatenate(errors);vel=np.concatenate(velocity_errors);per_target={}
    for target_id in sorted({x['source_target_id'] for x in per_window}):
        select=np.array([x['source_target_id']==target_id for x in per_window])
        per_target[str(target_id)]=error_stats(distances[select])
    initialized_mask=np.arange(distances.shape[1])[None,:]>=np.asarray(initialization_frames)[:,None]
    result={'after_initialization':error_stats(distances[initialized_mask]),'initialization_wait_frames':initialization_frames,'all_frames':error_stats(distances),'after_5s_warmup':error_stats(distances[initialized_mask & (np.arange(distances.shape[1])[None,:]>=5)]),
        'velocity_all_frames':error_stats(vel),'macro_target_rmse_m':float(np.mean([x['rmse_m'] for x in per_target.values()])),
        'per_target':per_target,'per_window':per_window}
    return result, {'position_errors':distances,'velocity_errors':vel,'initialized_mask':initialized_mask,'example':examples[0] if examples else None}


def observation_baselines(dataset):
    errors={name:[] for name in ['全维1','全维2','全维3','全维4']}
    for sample in dataset:
        xyz=radar_body_to_world_xyz(sample['radar'],sample['uav_pos'],sample['rotmat_b2e'])
        reference=sample['label_pos_rel']
        for i,name in enumerate(errors):
            valid=sample['radar_mask'][:,i].bool()
            errors[name].extend(torch.linalg.vector_norm(xyz[valid,i]-reference[valid],dim=-1).numpy().tolist())
    return {name:error_stats(value) for name,value in errors.items()}


def plots(output,history,baseline,initial,final,initial_arrays,final_arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    ax=axes[0,0];epochs=[x['epoch'] for x in history]
    ax.plot(epochs,[x['train_rmse_m'] for x in history],label='训练集',color='#2879b9');ax.plot(epochs,[x['val_rmse_m'] for x in history],label='验证集',color='#e68135')
    ax.axhline(initial['after_initialization']['rmse_m'],ls='--',color='#777777',label='微调前（已校准）');ax.set(xlabel='训练轮数',ylabel='三维位置 RMSE（米）',title='训练与验证曲线');ax.legend();ax.grid(alpha=.2)
    ax=axes[0,1];labels=['原始观测\nV7直接推理','训练集校准\nV7直接推理','校准后\nV7微调'];vals=[baseline['after_initialization']['rmse_m'],initial['after_initialization']['rmse_m'],final['after_initialization']['rmse_m']]
    bars=ax.bar(labels,vals,color=['#999999','#2879b9','#e68135'])
    for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.1f} 米',ha='center',va='bottom')
    ax.set(ylabel='验证集三维位置 RMSE（米）',title='拆分校准与微调带来的变化');ax.set_ylim(0,max(vals)*1.18);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    for values,name,color in [(initial_arrays['position_errors'][initial_arrays['initialized_mask']],'微调前（已校准）','#2879b9'),(final_arrays['position_errors'][final_arrays['initialized_mask']],'微调后','#e68135')]:
        x=np.sort(values.ravel());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=name,color=color)
    ax.set(xlabel='三维位置误差（米）',ylabel='累计比例',title='独立验证轨迹的误差分布');ax.legend();ax.grid(alpha=.2)
    ax=axes[1,1];ex=final_arrays['example'];origin=ex['label'][0,[0,2,4]];gt=ex['label'][:,[0,2,4]]-origin;pred=ex['pred'][:,[0,2,4]]-origin
    ax.plot(gt[:,0],gt[:,1],color='#333333',label='真值');ax.plot(pred[:,0],pred[:,1],color='#e68135',label='微调后')
    ax.set(xlabel='相对 X（米）',ylabel='相对 Y（米）',title=f'验证轨迹 {ex["target_id"]}，起始 {ex["start"]} 秒');ax.legend();ax.grid(alpha=.2);ax.axis('equal')
    fig.suptitle('参考数据 40 条轨迹小规模微调实验',fontsize=18,y=.985)
    fig.text(.04,.017,'30 条训练、10 条验证；1 Hz 原始采样；按目标轨迹隔离，偏差仅由训练集拟合；初始化后评分；属于同场景的小样本验证。',fontsize=10,color='#555555')
    fig.tight_layout(rect=(.02,.05,1,.96),h_pad=2);fig.savefig(output/'pilot_results.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(ROOT/'data/reference_pilot_40_20261011_measured_anchor'))
    parser.add_argument('--checkpoint',default=str(ROOT/'results/runs_40off(v7)/run_20260728_103728/best_model.pt'))
    parser.add_argument('--output',default='')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=20261011)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=Path(args.data).resolve();manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    tag=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=Path(args.output).resolve() if args.output else ROOT/'results'/f'reference_pilot_{tag}'
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    train=ReferencePilotDataset(data/'train',manifest['seq_len']);val=ReferencePilotDataset(data/'val',manifest['seq_len']);raw_val=ReferencePilotDataset(data/'val_uncalibrated',manifest['seq_len'])
    train_ids={m['source_target_id'] for m in train.metadata};val_ids={m['source_target_id'] for m in val.metadata}
    assert train_ids.isdisjoint(val_ids) and train_ids==set(manifest['calibration_training_target_ids'])
    assert all(float(x['dt'])==1. for ds in [train,val,raw_val] for x in ds)
    sample=train[0]
    assert sample['radar'].shape==(manifest['seq_len'],5,3) and sample['eo'].shape==(manifest['seq_len'],5,2)
    assert all(int(s['initialization_frame']) < manifest['seq_len'] for ds in [train,val,raw_val] for s in ds)
    assert all(torch.all(x['target_valid_mask']==1) for ds in [train,val,raw_val] for x in ds)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator,num_workers=0)
    eval_train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0)
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    raw_loader=DataLoader(raw_val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=CoopTrackerModel(d_model=96).to(device)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'model_state_dict' in state:state=state['model_state_dict']
    model.load_state_dict(state,strict=True)
    config={'args':vars(args),'device':str(device),'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'initial_checkpoint':str(Path(args.checkpoint).resolve()),'initial_checkpoint_sha256':sha256_file(Path(args.checkpoint)),
        'data_manifest':str(data/'manifest.json'),'manifest_sha256':sha256_file(data/'manifest.json'),
        'train_tracks':len(train_ids),'val_tracks':len(val_ids),'train_windows':len(train),'val_windows':len(val),
        'seq_len':manifest['seq_len'],'dt_s':1.,'parameter_count':sum(p.numel() for p in model.parameters()),
        'reference_origin':'first actual full-dimensional position observation, rounded to float32 ECEF; tracking scored after its arrival; no target truth in model inputs',
        'loss':'initialized frames only; training-only isotropic standardized SmoothL1(position)+0.1*SmoothL1(position differences)+ramp(0.01)*SmoothL1(velocity)',
        'selection':'best validation RMSE after measurement initialization, including epoch 0',
        'validation_scope':'held-out target columns in same flight scene; ideal column associations; offline bidirectional model',
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[run]',output,flush=True);print('[device]',config['gpu_name'],flush=True)
    baseline,_=evaluate(model,raw_loader,device)
    initial,initial_arrays=evaluate(model,val_loader,device,keep_arrays=True)
    print('[baseline] raw',baseline['all_frames'],'calibrated',initial['all_frames'],flush=True)
    (output/'initial_metrics.json').write_text(json.dumps({'raw':baseline,'calibrated':initial},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    y=torch.stack([s['label_pos_rel'] for s in train]);v=torch.stack([s['label_abs'][:,[1,3,5]] for s in train])
    pos_mean=y.mean((0,1)).view(1,1,3).to(device);pos_std=y.std((0,1),unbiased=False).mean().clamp_min(1.).reshape(1,1,1).to(device)
    vel_mean=v.mean((0,1)).view(1,1,3).to(device);vel_std=v.std((0,1),unbiased=False).mean().clamp_min(5.).reshape(1,1,1).to(device)
    (output/'normalization.json').write_text(json.dumps({'position_mean':pos_mean.cpu().flatten().tolist(),'position_isotropic_std':float(pos_std),'velocity_mean':vel_mean.cpu().flatten().tolist(),'velocity_isotropic_std':float(vel_std)},indent=2),encoding='utf-8')
    criterion=nn.SmoothL1Loss();optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best_rmse=initial['after_initialization']['rmse_m'];best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};history=[];started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];train_errors=[];epstart=time.perf_counter();vel_weight=.01*min(1.,epoch/10.)
        for batch in train_loader:
            pred,label,rel_label,rel_pred=forward_batch(model,batch,device)
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite training prediction')
            initialized=torch.arange(rel_pred.shape[1],device=device)[None,:]>=batch['initialization_frame'].to(device)[:,None]
            loss=criterion(((rel_pred-pos_mean)/pos_std)[initialized],((rel_label-pos_mean)/pos_std)[initialized])
            pairs=initialized[:,1:] & initialized[:,:-1]
            loss=loss+.1*criterion(((rel_pred[:,1:]-rel_pred[:,:-1])/pos_std)[pairs],((rel_label[:,1:]-rel_label[:,:-1])/pos_std)[pairs])
            loss=loss+vel_weight*criterion(((pred[:,:,[1,3,5]]-vel_mean)/vel_std)[initialized],((label[:,:,[1,3,5]]-vel_mean)/vel_std)[initialized])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()));train_errors.append(torch.linalg.vector_norm(pred[:,:,[0,2,4]].detach()-label[:,:,[0,2,4]],dim=-1)[initialized].cpu().numpy())
        metric,_=evaluate(model,val_loader,device)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_rmse_m':error_stats(np.concatenate(train_errors))['rmse_m'],
             'val_rmse_m':metric['after_initialization']['rmse_m'],'val_p95_m':metric['after_initialization']['p95_m'],
             'vel_loss_weight':vel_weight,'elapsed_s':time.perf_counter()-epstart}
        history.append(row)
        if row['val_rmse_m']<best_rmse:
            best_rmse=row['val_rmse_m'];best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,output/'best_model.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False),encoding='utf-8')
        print(f'[epoch {epoch:02d}/{args.epochs}] loss={row["train_loss"]:.6f} train_RMSE={row["train_rmse_m"]:.2f} val_RMSE={row["val_rmse_m"]:.2f} P95={row["val_p95_m"]:.2f} time={row["elapsed_s"]:.1f}s best={best_epoch}',flush=True)
    torch.save(model.state_dict(),output/'last_model.pt');model.load_state_dict(best_state);torch.save(best_state,output/'best_model.pt')
    final,final_arrays=evaluate(model,val_loader,device,keep_arrays=True);train_final,_=evaluate(model,eval_train_loader,device)
    results={'config':config,'training_seconds':time.perf_counter()-started,'best_epoch':best_epoch,'initial_raw_validation':baseline,
             'initial_calibrated_validation':initial,'best_validation':final,'best_training':train_final,
             'raw_observation_validation':observation_baselines(raw_val),'calibrated_observation_validation':observation_baselines(val),
             'fine_tune_rmse_reduction_percent':100.*(initial['after_initialization']['rmse_m']-final['after_initialization']['rmse_m'])/initial['after_initialization']['rmse_m'],
             'total_rmse_reduction_percent':100.*(baseline['after_initialization']['rmse_m']-final['after_initialization']['rmse_m'])/baseline['after_initialization']['rmse_m']}
    (output/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'history.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    np.savez_compressed(output/'validation_predictions.npz',position_errors=final_arrays['position_errors'],velocity_errors=final_arrays['velocity_errors'],example_prediction=final_arrays['example']['pred'],example_truth=final_arrays['example']['label'],initialized_mask=final_arrays['initialized_mask'])
    plots(output,history,baseline,initial,final,initial_arrays,final_arrays)
    print('[complete]',json.dumps({'output':str(output),'best_epoch':best_epoch,'raw_baseline':baseline['all_frames'],'calibrated_initial':initial['all_frames'],'final':final['all_frames']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
