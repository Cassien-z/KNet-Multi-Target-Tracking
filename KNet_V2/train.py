# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MultiTrackingDataset, collate_fn, NORM_CFG
from knet_model import KNet_Tracker
import os
import math

# ================= 配置区域 =================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
BATCH_SIZE = 64  # 利用精简架构后的显存红利
EPOCHS = 400
LEARNING_RATE = 1e-4  # 地形平滑后，适当放大学习率
SEQ_LEN = 150  # 合理的机动窗口，防 BPTT 梯度消失
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🟢 配置：微调与保存
DO_FINE_TUNE = False
MODEL_PATH = "knet_weights.pth"
SAVE_PATH = "knet_weights.pth"

# 🟢 配置：学习率衰减机制
LR_PATIENCE = 10  # 连续 10 个 Epoch 验证集不降则衰减
LR_FACTOR = 0.5  # 更加平滑的衰减因子


# ===========================================

def train():
    print(f"🔥 正在使用设备：{DEVICE}")

    # 分别加载训练集和验证集
    train_dataset = MultiTrackingDataset(data_dir=TRAIN_DIR, seq_length=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_dataset = MultiTrackingDataset(data_dir=VAL_DIR, seq_length=SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = KNet_Tracker().to(DEVICE)

    if DO_FINE_TUNE and os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"✅ 已加载预训练模型：{MODEL_PATH} (开始微调...)")
        except Exception as e:
            print(f"⚠️ 加载模型失败：{e}，将从随机初始化开始训练。")
    else:
        print("ℹ️ 从随机初始化开始训练。")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # ================== 1. 训练阶段 (Train) ==================
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            target_states = batch['states'].to(DEVICE)
            baselines = batch['baselines'].to(DEVICE)  # 模型内部已不再使用，但保持接口兼容
            meas = batch['meas'].to(DEVICE)
            mask_input = batch['mask_input'].to(DEVICE)
            mask_loss = batch['mask_loss'].to(DEVICE)
            dt = batch['dt'].to(DEVICE)

            pred_states, _, _ = model(meas, baselines, mask_input, dt)

            mask_3d = mask_loss.unsqueeze(-1).expand(-1, -1, 3)
            valid_count = mask_3d.sum() + 1e-8

            # 🌟 核心：在归一化空间计算 SmoothL1Loss，彻底告别量纲爆炸
            pred_pos_norm = pred_states[:, :, [0, 3, 6]]
            target_pos_norm = target_states[:, :, [0, 3, 6]]
            pred_vel_norm = pred_states[:, :, [1, 4, 7]]
            target_vel_norm = target_states[:, :, [1, 4, 7]]
            pred_acc_norm = pred_states[:, :, [2, 5, 8]]
            target_acc_norm = target_states[:, :, [2, 5, 8]]

            # ========== 修改这里 ==========
            # 删掉 criterion = nn.SmoothL1Loss(reduction='none')
            # 换成直接暴力的 MSE (L2) 损失
            criterion = nn.MSELoss(reduction='none')

            loss_pos_raw = criterion(pred_pos_norm, target_pos_norm)
            loss_vel_raw = criterion(pred_vel_norm, target_vel_norm)
            loss_acc_raw = criterion(pred_acc_norm, target_acc_norm)
            # ==============================

            # 🌟 预热机制：给 GRU 建立速度记忆的时间
            seq_length = pred_states.shape[1]
            temporal_weights = torch.ones(seq_length, device=DEVICE)
            warmup_len = min(50, seq_length)
            temporal_weights[:warmup_len] = torch.linspace(0.5, 1.0, steps=warmup_len, device=DEVICE)
            temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(-1)

            loss_pos_val = (loss_pos_raw * mask_3d * temporal_weights).sum() / valid_count
            loss_vel_val = (loss_vel_raw * mask_3d * temporal_weights).sum() / valid_count
            loss_acc_val = (loss_acc_raw * mask_3d * temporal_weights).sum() / valid_count

            # 🌟 之前的代码：
            # weight_pos, weight_vel, weight_acc = 2.0, 3.0, 2.0

            # 🌟 换成全新的统治级权重：
            weight_pos = 100.0  # 强迫网络把 99% 的精力放在算准位置上！
            weight_vel = 5.0  # 速度给一点约束，保证轨迹平滑
            weight_acc = 0.5  # 加速度几乎不管，让 GRU 自由发挥

            total_loss_val = weight_pos * loss_pos_val + weight_vel * loss_vel_val + weight_acc * loss_acc_val

            if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
                print(f"⚠️ 警告：当前 Batch 产生 NaN，紧急跳过！")
                optimizer.zero_grad()
                continue

            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss_val.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0

        # ================== 2. 验证阶段 (Validation) ==================
        model.eval()
        val_loss = 0.0
        val_rmse_m = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                target_states = batch['states'].to(DEVICE)
                baselines = batch['baselines'].to(DEVICE)
                meas = batch['meas'].to(DEVICE)
                mask_input = batch['mask_input'].to(DEVICE)
                mask_loss = batch['mask_loss'].to(DEVICE)
                dt = batch['dt'].to(DEVICE)

                pred_states, _, _ = model(meas, baselines, mask_input, dt)

                mask_3d = mask_loss.unsqueeze(-1).expand(-1, -1, 3)
                valid_count = mask_3d.sum() + 1e-8

                pred_pos_norm = pred_states[:, :, [0, 3, 6]]
                target_pos_norm = target_states[:, :, [0, 3, 6]]
                pred_vel_norm = pred_states[:, :, [1, 4, 7]]
                target_vel_norm = target_states[:, :, [1, 4, 7]]
                pred_acc_norm = pred_states[:, :, [2, 5, 8]]
                target_acc_norm = target_states[:, :, [2, 5, 8]]

                # A. 归一化空间算误差 (直面全局误差，不加预热权重)
                criterion = nn.SmoothL1Loss(reduction='none')
                loss_pos_val_batch = (criterion(pred_pos_norm, target_pos_norm) * mask_3d).sum() / valid_count
                loss_vel_val_batch = (criterion(pred_vel_norm, target_vel_norm) * mask_3d).sum() / valid_count
                loss_acc_val_batch = (criterion(pred_acc_norm, target_acc_norm) * mask_3d).sum() / valid_count

                total_loss_val_batch = weight_pos * loss_pos_val_batch + weight_vel * loss_vel_val_batch + weight_acc * loss_acc_val_batch
                val_loss += total_loss_val_batch.item()

                # B. 计算给人类看的绝对物理误差 (RMSE，单位：米)
                pred_pos_m = pred_pos_norm * NORM_CFG["pos"]
                target_pos_m = target_pos_norm * NORM_CFG["pos"]
                physical_mse_raw = torch.nn.functional.mse_loss(pred_pos_m, target_pos_m, reduction='none')
                physical_mse_mean = (physical_mse_raw * mask_3d).sum() / valid_count

                val_rmse_m += math.sqrt(physical_mse_mean.item())
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_rmse = val_rmse_m / val_batches if val_batches > 0 else 0

        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch + 1:>3}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val RMSE: {avg_val_rmse:.1f} m")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🏆 模型进化！保存最佳验证权重 (Loss: {best_val_loss:.6f})")

    print(f"\n🎉 训练结束！最终最佳 Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()