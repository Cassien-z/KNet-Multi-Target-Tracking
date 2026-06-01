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
DATA_DIR = "knet_train_data"
BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 1e-4
SEQ_LEN = 600
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🟢 配置：微调与保存
DO_FINE_TUNE = False
MODEL_PATH = "knet_weights_shorter.pth"
SAVE_PATH = "knet_weights_shorter.pth"

# 🟢 配置：学习率衰减机制
LR_PATIENCE = 10      # 如果连续 10 个 Epoch Loss 都没有下降，就降低学习率
LR_FACTOR = 0.1       # 每次触发降低时，学习率乘以 0.5
# ===========================================

# 🌟 将原来的 nn.MSELoss() 替换为：
criterion_pos = nn.SmoothL1Loss(reduction='none')
criterion_vel = nn.SmoothL1Loss(reduction='none')
criterion_acc = nn.SmoothL1Loss(reduction='none')  # 🌟 必须加上这行！专门用于计算加速度误差
criterion_az = nn.SmoothL1Loss(reduction='none')

def train():
    print(f"正在使用设备：{DEVICE}")

    should_fine_tune = DO_FINE_TUNE
    model_path = MODEL_PATH
    save_path = SAVE_PATH

    dataset = MultiTrackingDataset(data_dir=DATA_DIR, seq_length=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = KNet_Tracker().to(DEVICE)

    if should_fine_tune and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ 已加载预训练模型：{model_path} (开始微调...)")
        except Exception as e:
            print(f"⚠️ 加载模型失败：{e}，将从随机初始化开始训练。")
            should_fine_tune = False
    else:
        if should_fine_tune:
            print(f"ℹ️ 未找到预训练模型 {model_path}，将从随机初始化开始训练。")
        else:
            print("ℹ️ 微调模式已关闭，从随机初始化开始训练。")

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        avg_pos_rmse_m = 0.0
        count_batches = 0

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            # 1. 转移到 GPU/CPU
            target_states = batch['states'].to(DEVICE)
            baselines = batch['baselines'].to(DEVICE)
            meas = batch['meas'].to(DEVICE)
            mask_input = batch['mask_input'].to(DEVICE)
            mask_loss = batch['mask_loss'].to(DEVICE)
            dt = batch['dt'].to(DEVICE)

            # 2. 前向传播 (根据你的 KNet_Tracker 实际需要的输入传参)
            # 这里默认传 meas, baselines, mask_input, dt
            pred_states, _, _ = model(meas, baselines, mask_input, dt)

            # 3. 生成 3D 掩码，去除 DataLoader 中由于 Padding 补零的无效数据
            # mask_loss 原始形状 [batch_size, seq_len]，扩展为 [batch_size, seq_len, 3]
            mask_3d = mask_loss.unsqueeze(-1).expand(-1, -1, 3)
            valid_count = mask_3d.sum() + 1e-8  # 防止除以 0

            # 🌟 杀手锏 2：换回 MSELoss，精准打击 RMSE 指标
            criterion_mse = nn.MSELoss(reduction='none')

            # ================= 核心修改区域：物理单位对齐 =================
            # A. 提取位置、速度、加速度，并乘回真实的物理单位
            pred_pos_m = pred_states[:, :, [0, 3, 6]] * NORM_CFG["pos"]
            target_pos_m = target_states[:, :, [0, 3, 6]] * NORM_CFG["pos"]

            pred_vel_ms = pred_states[:, :, [1, 4, 7]] * NORM_CFG["vel"]
            target_vel_ms = target_states[:, :, [1, 4, 7]] * NORM_CFG["vel"]

            pred_acc = pred_states[:, :, [2, 5, 8]] * NORM_CFG["acc"]
            target_acc = target_states[:, :, [2, 5, 8]] * NORM_CFG["acc"]

            # B. 使用 MSE 求解误差平方
            loss_pos_raw = criterion_mse(pred_pos_m, target_pos_m)
            loss_vel_raw = criterion_mse(pred_vel_ms, target_vel_ms)
            loss_acc_raw = criterion_mse(pred_acc, target_acc)

            # 🌟 杀手锏 1：创建时间序列渐进权重 (Burn-in 预热机制)
            seq_length = pred_states.shape[1]
            temporal_weights = torch.ones(seq_length, device=DEVICE)
            warmup_len = min(50, seq_length)  # 前 50 帧作为预热期
            # 前 50 帧的权重从 0.1 线性增长到 1.0
            temporal_weights[:warmup_len] = torch.linspace(0.5, 1.0, steps=warmup_len, device=DEVICE)
            # 调整形状以匹配 loss_pos_raw: [1, seq_len, 1]
            temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(-1)

            # C. 乘以掩码、乘以时间权重并求均值 (将开局的误差降权)
            loss_pos_val = (loss_pos_raw * mask_3d * temporal_weights).sum() / valid_count
            loss_vel_val = (loss_vel_raw * mask_3d * temporal_weights).sum() / valid_count
            loss_acc_val = (loss_acc_raw * mask_3d * temporal_weights).sum() / valid_count

            weight_pos, weight_vel, weight_acc = 1.0, 5.0, 10.0

            total_loss_val = weight_pos * loss_pos_val + weight_vel * loss_vel_val + weight_acc * loss_acc_val
            # =============================================================

            # 4. NaN 崩溃保护
            if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
                print(f"⚠️ 警告：当前 Batch 产生 NaN 损失，紧急跳过！")
                optimizer.zero_grad()
                continue

            # 5. 反向传播与梯度裁剪
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 6. 统计数据
            total_loss += total_loss_val.item()
            # 因为 loss_pos_val 是 MSE 平方误差，为了方便观察，开根号转回近似 RMSE (米)
            avg_pos_rmse_m += math.sqrt(loss_pos_val.item())
            count_batches += 1

        # ====== 打印当前 Epoch 结果 ======
        avg_loss = total_loss / count_batches if count_batches > 0 else 0
        avg_rmse = avg_pos_rmse_m / count_batches if count_batches > 0 else 0

        # 学习率调度器更新
        scheduler.step(avg_loss)

        # 每一轮打印一次，直观看到你的网络追踪精度（单位：米）
        print(f"Epoch [{epoch + 1:>3}/{EPOCHS}] | Total Loss: {avg_loss:.2f} | 平均位置误差评估: {avg_rmse:.2f} 米")

        # 默认每 10 轮保存一次模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🏆 发现最佳模型！Loss 降至 {best_loss:.2f}，已保存至 {SAVE_PATH}")

    print(f"\n🎉 训练结束！最佳 Loss: {best_loss:.6f}")
    print(f"📂 请使用模型文件：{save_path}")

if __name__ == "__main__":
    train()