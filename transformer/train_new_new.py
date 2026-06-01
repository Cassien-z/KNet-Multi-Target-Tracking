import argparse
import json
import os
import random
import time
from datetime import datetime
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import SingerDataset
from model import CoopTrackerModel


matplotlib.rcParams["axes.unicode_minus"] = False


def setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    for name in candidates:
        try:
            matplotlib.rcParams["font.family"] = name
            fig = plt.figure()
            plt.text(0.1, 0.1, "中文")
            plt.close(fig)
            print(f"[Info] 使用中文字体: {name}")
            return
        except Exception:
            continue
    print("[Warn] 未找到可用中文字体，图中文字可能显示异常。")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_basic_metrics(pos_err):
    pos_err = np.asarray(pos_err, dtype=np.float64).reshape(-1)
    if pos_err.size == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "rmse": float(np.sqrt(np.mean(pos_err ** 2))),
        "mae": float(np.mean(np.abs(pos_err))),
        "p95": float(np.percentile(pos_err, 95)),
        "max": float(np.max(pos_err)),
    }


def compute_basic_metrics_ignore_first(pos_err_by_seq, ignore_first=0):
    arr = np.asarray(pos_err_by_seq, dtype=np.float64)
    if arr.ndim != 2:
        return compute_basic_metrics(arr.reshape(-1))
    if ignore_first <= 0 or ignore_first >= arr.shape[1]:
        return compute_basic_metrics(arr.reshape(-1))
    return compute_basic_metrics(arr[:, ignore_first:].reshape(-1))


class MetricMeter:
    def __init__(self):
        self.errors = []

    def reset(self):
        self.errors = []

    def update(self, pred, gt):
        dist = torch.norm(pred[:, :, [0, 2, 4]] - gt[:, :, [0, 2, 4]], dim=-1)
        self.errors.append(dist.detach().cpu().numpy())

    def summary(self):
        if len(self.errors) == 0:
            return {"rmse": np.nan, "p95": np.nan}
        err = np.concatenate(self.errors, axis=None)
        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "p95": float(np.percentile(err, 95)),
        }


def compute_pos_rel_stats(loader, device):
    sums = None
    sq_sums = None
    count = 0
    for batch in loader:
        y = batch["label_pos_rel"].to(device)
        if sums is None:
            sums = y.sum(dim=(0, 1))
            sq_sums = (y ** 2).sum(dim=(0, 1))
        else:
            sums += y.sum(dim=(0, 1))
            sq_sums += (y ** 2).sum(dim=(0, 1))
        count += y.shape[0] * y.shape[1]

    mean = sums / count
    var = (sq_sums / count) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-6))
    floor = torch.tensor([1.0, 1.0, 1.0], device=device)
    std = torch.maximum(std, floor)
    return mean, std


def compute_vel_stats(loader, device):
    sums = None
    sq_sums = None
    count = 0
    for batch in loader:
        v = batch["label_abs"].to(device)[:, :, [1, 3, 5]]
        if sums is None:
            sums = v.sum(dim=(0, 1))
            sq_sums = (v ** 2).sum(dim=(0, 1))
        else:
            sums += v.sum(dim=(0, 1))
            sq_sums += (v ** 2).sum(dim=(0, 1))
        count += v.shape[0] * v.shape[1]

    mean = sums / count
    var = (sq_sums / count) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-6))
    floor = torch.tensor([1.0, 1.0, 1.0], device=device)
    std = torch.maximum(std, floor)
    return mean, std


def forward_batch(model, batch, device, mode="full"):
    radar = batch["radar"].to(device)
    eo = batch["eo"].to(device)
    esm = batch["esm"].to(device)
    uav_pos = batch["uav_pos"].to(device)
    r_mask = batch["radar_mask"].to(device)
    e_mask = batch["eo_mask"].to(device)
    esm_mask = batch["esm_mask"].to(device)
    label_abs = batch["label_abs"].to(device)
    label_pos_rel = batch["label_pos_rel"].to(device)
    ref_pos = batch["ref_pos"].to(device)
    dt = batch["dt"].to(device).view(-1)
    rotmat_b2e = batch["rotmat_b2e"].to(device)
    imu = batch["imu"].to(device)
    fast_link = batch["fast_link"].to(device)
    slow_link = batch["slow_link"].to(device)
    fast_age = batch["fast_age"].to(device)
    slow_age = batch["slow_age"].to(device)
    fast_update = batch["fast_update"].to(device)
    slow_update = batch["slow_update"].to(device)
    esm_mode = batch["esm_mode"].to(device)

    if mode == "radar_only":
        eo = torch.zeros_like(eo)
        esm = torch.zeros_like(esm)
        e_mask = torch.zeros_like(e_mask)
        esm_mask = torch.zeros_like(esm_mask)
        use_gat = True
    elif mode == "dual_no_gat":
        use_gat = False
    else:
        use_gat = True

    pred = model(
        radar=radar,
        eo=eo,
        esm=esm,
        radar_mask=r_mask,
        eo_mask=e_mask,
        esm_mask=esm_mask,
        ref_pos=ref_pos,
        rotmat_b2e=rotmat_b2e,
        uav_pos=uav_pos,
        fast_link=fast_link,
        slow_link=slow_link,
        fast_update=fast_update,
        slow_update=slow_update,
        imu=imu,
        fast_age=fast_age,
        slow_age=slow_age,
        esm_mode=esm_mode,
        dt=dt,
        use_gat=use_gat,
    )
    pred_pos_rel = pred[:, :, [0, 2, 4]] - ref_pos.unsqueeze(1)
    return pred, label_abs, label_pos_rel, pred_pos_rel


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    pos_mean,
    pos_std,
    vel_mean,
    vel_std,
    vel_loss_weight,
    device,
):
    model.train()
    meter = MetricMeter()
    meter.reset()
    loss_sum = 0.0

    for batch in loader:
        pred, label_abs, label_pos_rel, pred_pos_rel = forward_batch(model, batch, device, mode="full")

        pred_n = (pred_pos_rel - pos_mean) / pos_std
        label_n = (label_pos_rel - pos_mean) / pos_std

        regression_loss = criterion(pred_n, label_n)
        # 建议将 smooth_loss 也进行标准化（例如除以 vel_std）
        pred_diff = (pred_pos_rel[:, 1:] - pred_pos_rel[:, :-1]) / vel_std[:, :, :3]
        label_diff = (label_pos_rel[:, 1:] - label_pos_rel[:, :-1]) / vel_std[:, :, :3]
        smooth_loss = criterion(pred_diff, label_diff)

        # smooth_loss = criterion(
        #     pred_pos_rel[:, 1:] - pred_pos_rel[:, :-1],
        #     label_pos_rel[:, 1:] - label_pos_rel[:, :-1],
        # )

        pred_vel_n = (pred[:, :, [1, 3, 5]] - vel_mean) / vel_std
        label_vel_n = (label_abs[:, :, [1, 3, 5]] - vel_mean) / vel_std
        vel_loss = criterion(pred_vel_n, label_vel_n)

        loss = regression_loss + 0.1 * smooth_loss + vel_loss_weight * vel_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += float(loss.item())
        meter.update(pred.detach(), label_abs.detach())

    return loss_sum / max(len(loader), 1), meter.summary()


@torch.no_grad()
def evaluate(model, loader, criterion, pos_mean, pos_std, device):
    model.eval()
    meter = MetricMeter()
    meter.reset()
    loss_sum = 0.0

    for batch in loader:
        pred, label_abs, label_pos_rel, pred_pos_rel = forward_batch(model, batch, device, mode="full")
        pred_n = (pred_pos_rel - pos_mean) / pos_std
        label_n = (label_pos_rel - pos_mean) / pos_std
        regression_loss = criterion(pred_n, label_n)
        loss_sum += float(regression_loss.item())
        meter.update(pred, label_abs)

    return loss_sum / max(len(loader), 1), meter.summary()


@torch.no_grad()
def collect_mode_errors(model, loader, device, mode="full", max_batches=0):
    model.eval()
    pos_err_list = []
    vel_err_list = []

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        pred, label_abs, _, _ = forward_batch(model, batch, device, mode=mode)
        pos_err = torch.norm(pred[:, :, [0, 2, 4]] - label_abs[:, :, [0, 2, 4]], dim=-1)
        vel_err = torch.norm(pred[:, :, [1, 3, 5]] - label_abs[:, :, [1, 3, 5]], dim=-1)
        pos_err_list.append(pos_err.cpu().numpy())
        vel_err_list.append(vel_err.cpu().numpy())

    pos_all = np.concatenate(pos_err_list, axis=0)
    vel_all = np.concatenate(vel_err_list, axis=0)
    return pos_all, vel_all


@torch.no_grad()
def predict_single_mode(model, sample, device, mode="full"):
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v
    pred, label_abs, _, _ = forward_batch(model, batch, device, mode=mode)
    return pred.squeeze(0).cpu().numpy(), label_abs.squeeze(0).cpu().numpy()


@torch.no_grad()
def benchmark_latency_distribution_mode(model, subset, device, mode="full", max_samples=80):
    model.eval()
    n = min(len(subset), max_samples)
    lat = []
    for i in range(n):
        sample = subset[i]
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_batch(model, batch, device, mode=mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)
    return np.array(lat, dtype=np.float64)


def resolve_data_dir(path):
    if os.path.isdir(path):
        return path
    for p in ["./train_data", "./Dataset_Singer_v2", "./Dataset_Singer", "./Dataset_Clean"]:
        if os.path.isdir(p):
            return p
    return path


def load_best_phase2_params(results_dir="results"):
    json_files = sorted(glob(os.path.join(results_dir, "phase2_tuning_*.json")))
    if not json_files:
        return None
    latest_json = json_files[-1]
    try:
        with open(latest_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        print(f"[Warn] 读取调参结果失败: {latest_json}, error={exc}")
        return None

    ranked = obj.get("ranked_ok", [])
    if not ranked:
        return None

    best = ranked[0]
    return {
        "source_json": os.path.abspath(latest_json),
        "lr": float(best["lr"]),
        "phase2_vel_weight": float(best["vel_weight"]),
        "phase2_start_epoch": int(best["phase2_start"]),
        "reference_run_dir": best.get("run_dir", ""),
    }


def build_summary(
    run_dir,
    data_dir,
    best_model_path,
    args,
    best_phase2,
    metrics,
    metrics_wo_warmup,
    vel_err_vec,
    mode_specs,
    mode_curves,
    mode_metrics,
    mode_latency,
    hard_t0,
    t_start,
    t_end,
):
    return {
        "run_dir": os.path.abspath(run_dir),
        "data_dir": data_dir,
        "best_model_path": os.path.abspath(best_model_path),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "d_model": int(args.d_model),
        "warmup_ignore": int(args.warmup_ignore),
        "phase2_vel_weight": float(args.phase2_vel_weight),
        "phase2_start_epoch": int(args.phase2_start_epoch),
        "auto_best_phase2": bool(args.auto_best_phase2),
        "best_phase2": best_phase2,
        "metrics_all": {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "p95": float(metrics["p95"]),
            "max": float(metrics["max"]),
        },
        "metrics_wo_warmup": {
            "rmse": float(metrics_wo_warmup["rmse"]),
            "mae": float(metrics_wo_warmup["mae"]),
            "p95": float(metrics_wo_warmup["p95"]),
            "max": float(metrics_wo_warmup["max"]),
        },
        "speed_error_mean": float(np.mean(vel_err_vec)),
        "mode_metrics": {
            mode: {
                "label": mode_curves[mode]["label"],
                "rmse": float(mode_metrics[mode]["rmse"]),
                "mae": float(mode_metrics[mode]["mae"]),
                "p95": float(mode_metrics[mode]["p95"]),
                "max": float(mode_metrics[mode]["max"]),
                "vel_rmse": float(mode_metrics[mode]["vel_rmse"]),
            }
            for mode, _, _ in mode_specs
        },
        "mode_latency_ms": {
            mode: {
                "mean": float(np.mean(mode_latency[mode])) if mode_latency[mode].size > 0 else None,
                "p95": float(np.percentile(mode_latency[mode], 95)) if mode_latency[mode].size > 0 else None,
            }
            for mode, _, _ in mode_specs
        },
        "hard_case_window": {
            "center_frame": int(hard_t0),
            "start_frame": int(t_start),
            "end_frame": int(t_end - 1),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./train_data")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--num_uavs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency_samples", type=int, default=80)
    parser.add_argument("--warmup_ignore", type=int, default=5)
    parser.add_argument("--phase2_vel_weight", type=float, default=0.08)
    parser.add_argument("--phase2_start_epoch", type=int, default=5)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    parser.add_argument("--auto_best_phase2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard_case_start", type=int, default=-1)
    parser.add_argument("--hard_case_len", type=int, default=20)
    args = parser.parse_args()

    setup_chinese_font()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 设备: {device}")

    data_dir = resolve_data_dir(args.data_dir)
    print(f"[Info] 数据目录: {data_dir}")

    best_phase2 = None
    if args.auto_best_phase2:
        best_phase2 = load_best_phase2_params("results")
        if best_phase2 is not None:
            args.lr = best_phase2["lr"]
            args.phase2_vel_weight = best_phase2["phase2_vel_weight"]
            args.phase2_start_epoch = best_phase2["phase2_start_epoch"]
            print(
                "[Info] 已自动加载最优二阶段参数: "
                f"lr={args.lr}, vel_weight={args.phase2_vel_weight}, "
                f"phase2_start={args.phase2_start_epoch}"
            )
            print(f"[Info] 参数来源: {best_phase2['source_json']}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Info] 本次结果目录: {os.path.abspath(run_dir)}")

    dataset = SingerDataset(
        data_dir=data_dir,
        seq_len=args.seq_len,
        num_uavs=args.num_uavs,
        return_extra=True,
    )
    if len(dataset) == 0:
        print("[Error] 未找到数据。")
        return

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_set = Subset(dataset, indices[:split])
    val_set = Subset(dataset, indices[split:])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CoopTrackerModel(d_model=args.d_model).to(device)
    if args.resume_path and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print(f"[Info] 已加载预训练模型: {args.resume_path}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    pos_mean, pos_std = compute_pos_rel_stats(train_loader, device)
    vel_mean, vel_std = compute_vel_stats(train_loader, device)
    pos_mean = pos_mean.view(1, 1, -1)
    pos_std = pos_std.view(1, 1, -1)
    vel_mean = vel_mean.view(1, 1, -1)
    vel_std = vel_std.view(1, 1, -1)

    train_losses = []
    val_losses = []
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        vel_loss_weight = args.phase2_vel_weight if epoch >= args.phase2_start_epoch else 0.0
        tr_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            pos_mean=pos_mean,
            pos_std=pos_std,
            vel_mean=vel_mean,
            vel_std=vel_std,
            vel_loss_weight=vel_loss_weight,
            device=device,
        )
        va_loss, va_m = evaluate(model, val_loader, criterion, pos_mean, pos_std, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_m["rmse"] < best_rmse:
            best_rmse = va_m["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} "
            f"val_loss={va_loss:.6f} val_RMSE={va_m['rmse']:.4f}m "
            f"val_P95={va_m['p95']:.4f}m vel_w={vel_loss_weight:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    best_model_path = os.path.join(run_dir, "best_model.pt")
    torch.save(model.state_dict(), best_model_path)
    print(f"[Info] 最优模型已保存: {best_model_path}")

    pos_err_all, vel_err_all = collect_mode_errors(
        model, val_loader, device, mode="full", max_batches=args.eval_max_batches
    )
    pos_err_vec = pos_err_all.reshape(-1)
    vel_err_vec = vel_err_all.reshape(-1)
    metrics = compute_basic_metrics(pos_err_vec)
    metrics_wo_warmup = compute_basic_metrics_ignore_first(pos_err_all, args.warmup_ignore)

    mode_specs = [
        ("full", "完整模型", "#e15759"),
        ("dual_no_gat", "无图协同", "#4e79a7"),
        ("radar_only", "仅雷达主流", "#59a96a"),
    ]
    mode_curves = {}
    mode_metrics = {}
    mode_latency = {}

    sample = val_set[0]
    gt_one = None
    for mode, label, color in mode_specs:
        pred_one, gt_tmp = predict_single_mode(model, sample, device, mode=mode)
        mode_curves[mode] = {
            "label": label,
            "color": color,
            "pred": pred_one,
            "pos_err": np.linalg.norm(pred_one[:, [0, 2, 4]] - gt_tmp[:, [0, 2, 4]], axis=1),
            "vel_err": np.linalg.norm(pred_one[:, [1, 3, 5]] - gt_tmp[:, [1, 3, 5]], axis=1),
        }
        if gt_one is None:
            gt_one = gt_tmp

        pos_mode, vel_mode = collect_mode_errors(
            model, val_loader, device, mode=mode, max_batches=args.eval_max_batches
        )
        pos_mode_vec = pos_mode.reshape(-1)
        vel_mode_vec = vel_mode.reshape(-1)
        basic = compute_basic_metrics(pos_mode_vec)
        mode_metrics[mode] = {
            "rmse": float(basic["rmse"]),
            "mae": float(basic["mae"]),
            "p95": float(basic["p95"]),
            "max": float(basic["max"]),
            "vel_rmse": float(np.sqrt(np.mean(vel_mode_vec ** 2))),
            "pos_errors": pos_mode_vec,
        }

        if args.skip_latency:
            mode_latency[mode] = np.array([], dtype=np.float64)
        else:
            mode_latency[mode] = benchmark_latency_distribution_mode(
                model, val_set, device, mode=mode, max_samples=args.latency_samples
            )

    hard_case_len = max(8, int(args.hard_case_len))
    if args.hard_case_start >= 0:
        t_start = min(int(args.hard_case_start), max(0, len(gt_one) - 2))
        t_end = min(len(gt_one), t_start + hard_case_len)
        hard_t0 = t_start + (t_end - t_start) // 2
    else:
        gain_curve = mode_curves["dual_no_gat"]["pos_err"] - mode_curves["full"]["pos_err"]
        if hard_case_len >= len(gain_curve):
            t_start = 0
            t_end = len(gain_curve)
            hard_t0 = len(gain_curve) // 2
        else:
            window_scores = np.convolve(gain_curve, np.ones(hard_case_len, dtype=np.float64), mode="valid")
            t_start = int(np.argmax(window_scores))
            t_end = min(len(gain_curve), t_start + hard_case_len)
            hard_t0 = t_start + (t_end - t_start) // 2

    summary = build_summary(
        run_dir,
        data_dir,
        best_model_path,
        args,
        best_phase2,
        metrics,
        metrics_wo_warmup,
        vel_err_vec,
        mode_specs,
        mode_curves,
        mode_metrics,
        mode_latency,
        hard_t0,
        t_start,
        t_end,
    )

    summary_path = os.path.join(run_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.skip_plots:
        print(f"[Info] summary saved: {os.path.abspath(summary_path)}")
        return

    # 图1：轨迹对比图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(gt_one[:, 0], gt_one[:, 2], "k-", linewidth=2.4, label="真实轨迹")
    axes[1].plot(gt_one[:, 0], gt_one[:, 4], "k-", linewidth=2.4, label="真实轨迹")
    for mode, _, _ in mode_specs:
        pred = mode_curves[mode]["pred"]
        color = mode_curves[mode]["color"]
        label = mode_curves[mode]["label"]
        axes[0].plot(pred[:, 0], pred[:, 2], linewidth=2.0, color=color, label=label)
        axes[1].plot(pred[:, 0], pred[:, 4], linewidth=2.0, color=color, label=label)
    axes[0].set_xlabel("X位置（米）")
    axes[0].set_ylabel("Y位置（米）")
    axes[0].set_title("XY平面轨迹对比")
    axes[0].grid(alpha=0.3)
    axes[0].axis("equal")
    axes[1].set_xlabel("X位置（米）")
    axes[1].set_ylabel("Z高度（米）")
    axes[1].set_title("XZ平面轨迹对比")
    axes[1].grid(alpha=0.3)
    axes[1].axis("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("图1 目标轨迹对比结果", fontsize=14)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "图1_轨迹对比_XY_XZ.png"), dpi=300)
    plt.close(fig)

    # 图2：误差CDF
    plt.figure(figsize=(8.5, 5.5))
    for mode, _, _ in mode_specs:
        sorted_err = np.sort(mode_metrics[mode]["pos_errors"])
        cdf = np.arange(1, sorted_err.size + 1) / sorted_err.size
        plt.plot(sorted_err, cdf, linewidth=2.2, color=mode_curves[mode]["color"], label=mode_curves[mode]["label"])
    plt.xlabel("三维位置误差（米）")
    plt.ylabel("累计概率")
    plt.title("图2 不同方法三维位置误差CDF对比")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "图2_误差CDF对比.png"), dpi=300)
    plt.close()

    # 图3：相对改进图
    baseline_modes = ["dual_no_gat", "radar_only"]
    metric_names = ["RMSE降幅", "MAE降幅", "P95降幅"]
    improve_mat = []
    for metric_key in ["rmse", "mae", "p95"]:
        row = []
        full_val = mode_metrics["full"][metric_key]
        for b_mode in baseline_modes:
            base_val = mode_metrics[b_mode][metric_key]
            row.append((base_val - full_val) / max(base_val, 1e-6) * 100.0)
        improve_mat.append(row)
    improve_mat = np.asarray(improve_mat)

    x = np.arange(len(metric_names))
    width = 0.32
    plt.figure(figsize=(8.5, 5.5))
    for i, b_mode in enumerate(baseline_modes):
        vals = improve_mat[:, i]
        bars = plt.bar(x + (i - 0.5) * width, vals, width=width, label=f"相对{mode_curves[b_mode]['label']}")
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, metric_names)
    plt.ylabel("相对下降幅度（%）")
    plt.title("图3 完整模型相对基线方法的误差下降幅度")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "图3_相对误差下降幅度.png"), dpi=300)
    plt.close()

    # 图4：模块消融图
    ablation_labels = [mode_curves[m]["label"] for m in ["full", "dual_no_gat", "radar_only"]]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, metric_key, title, color in zip(
        axes,
        ["rmse", "mae", "p95"],
        ["RMSE", "MAE", "P95"],
        ["#e15759", "#4e79a7", "#59a96a"],
    ):
        vals = [mode_metrics[m][metric_key] for m in ["full", "dual_no_gat", "radar_only"]]
        bars = ax.bar(np.arange(len(ablation_labels)), vals, color=color)
        ax.set_xticks(np.arange(len(ablation_labels)))
        ax.set_xticklabels(ablation_labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel("误差（米）")
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("图4 模块消融实验结果对比", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "图4_模块消融实验.png"), dpi=300)
    plt.close(fig)

    # 图5：困难片段放大图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(gt_one[t_start:t_end, 0], gt_one[t_start:t_end, 2], "k-", linewidth=2.4, label="真实轨迹")
    for mode, _, _ in mode_specs:
        pred = mode_curves[mode]["pred"]
        color = mode_curves[mode]["color"]
        label = mode_curves[mode]["label"]
        axes[0].plot(pred[t_start:t_end, 0], pred[t_start:t_end, 2], linewidth=2.0, color=color, label=label)
        axes[1].plot(
            np.arange(t_start, t_end),
            mode_curves[mode]["pos_err"][t_start:t_end],
            linewidth=2.0,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("X位置（米）")
    axes[0].set_ylabel("Y位置（米）")
    axes[0].set_title(f"困难机动片段轨迹放大（帧 {t_start} - {t_end - 1}）")
    axes[0].grid(alpha=0.3)
    axes[0].axis("equal")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("三维位置误差（米）")
    axes[1].set_title("对应片段逐帧误差对比")
    axes[1].grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("图5 困难机动片段局部放大结果", fontsize=14)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "图5_困难片段局部放大.png"), dpi=300)
    plt.close(fig)

    # 图6：推理时延
    latency_labels = [mode_curves[m]["label"] for m in ["full", "dual_no_gat", "radar_only"]]
    latency_means = []
    latency_p95s = []
    for mode in ["full", "dual_no_gat", "radar_only"]:
        arr = mode_latency[mode]
        if arr.size == 0:
            latency_means.append(float("nan"))
            latency_p95s.append(float("nan"))
        else:
            latency_means.append(float(np.mean(arr)))
            latency_p95s.append(float(np.percentile(arr, 95)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    if mode_latency["full"].size > 0:
        axes[0].hist(mode_latency["full"], bins=18, color="#f28e2b", alpha=0.80)
        axes[0].set_xlabel("单样本推理时延（毫秒）")
        axes[0].set_ylabel("频次")
        axes[0].set_title("完整模型时延分布")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "已跳过时延评估", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    x = np.arange(len(latency_labels))
    width = 0.32
    bars1 = axes[1].bar(x - width / 2, latency_means, width=width, color="#4e79a7", label="平均时延")
    bars2 = axes[1].bar(x + width / 2, latency_p95s, width=width, color="#e15759", label="P95时延")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(latency_labels)
    axes[1].set_ylabel("毫秒")
    axes[1].set_title("不同方法推理时延对比")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("图6 推理时延统计结果", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "图6_推理时延统计.png"), dpi=300)
    plt.close(fig)

    print(f"[结果][全帧] RMSE: {metrics['rmse']:.6f} 米")
    print(f"[结果][全帧] MAE: {metrics['mae']:.6f} 米")
    print(f"[结果][全帧] P95: {metrics['p95']:.6f} 米")
    print(
        f"[结果][去前{args.warmup_ignore}帧] RMSE/MAE/P95: "
        f"{metrics_wo_warmup['rmse']:.6f}/{metrics_wo_warmup['mae']:.6f}/{metrics_wo_warmup['p95']:.6f}"
    )
    print(f"[结果] 速度误差均值: {float(np.mean(vel_err_vec)):.6f} 米/秒")
    print(f"[结果] 图片与模型输出目录: {os.path.abspath(run_dir)}")
    print(f"[Info] summary saved: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
