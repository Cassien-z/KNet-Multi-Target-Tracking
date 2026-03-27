# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MultiTrackingDataset, collate_fn
from knet_model import KNet_Tracker
import os

# ================= 配置区域 =================
DATA_DIR = "knet_train_data"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5
SEQ_LEN = 600
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🟢 配置：是否微调
DO_FINE_TUNE = True
MODEL_PATH = "knet_weights.pth"
SAVE_PATH = "knet_weights_finetuned.pth" if DO_FINE_TUNE else "knet_weights.pth"


# ===========================================

def train():
    print(f"正在使用设备：{DEVICE}")

    # 🟢 在这里明确读取配置变量的值到局部变量，避免作用域问题
    should_fine_tune = DO_FINE_TUNE
    model_path = MODEL_PATH
    save_path = SAVE_PATH

    dataset = MultiTrackingDataset(data_dir=DATA_DIR, seq_length=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = KNet_Tracker().to(DEVICE)

    # 🟢 安全的加载逻辑
    if should_fine_tune and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ 已加载预训练模型：{model_path} (开始微调...)")
        except Exception as e:
            print(f"⚠️ 加载模型失败：{e}，将从随机初始化开始训练。")
            should_fine_tune = False  # 仅在局部变量修改，不影响外部
    else:
        if should_fine_tune:
            print(f"ℹ️ 未找到预训练模型 {model_path}，将从随机初始化开始训练。")
        else:
            print("ℹ️ 微调模式已关闭，从随机初始化开始训练。")

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        count_batches = 0

        for batch in loader:
            states = batch['states'].to(DEVICE)
            meas = batch['meas'].to(DEVICE)
            dts = batch['dt'].to(DEVICE)
            obs_mask = batch['mask'].to(DEVICE)
            lengths = batch['lengths']

            B, T, _ = states.shape

            optimizer.zero_grad()

            hx = None
            # ✅ 现在改成这样（模拟 Inference 时的真实起步环境）：
            x_prev = torch.zeros_like(states[:, 0, :])
            # 将位置替换为第一帧的带噪观测（反归一化再转换），但为了简单可以直接加点噪声
            x_prev[:, [0, 2, 4]] = states[:, 0, [0, 2, 4]] + torch.randn_like(states[:, 0, [0, 2, 4]]) * 0.01
            # 速度强制保持为 0，逼迫 GRU 学习如何收敛速度！
            x_prev[:, [1, 3, 5]] = 0.0

            pred_states = []

            for t in range(T):
                dt_t = dts[:, t]
                meas_t = meas[:, t, :]
                mask_t = obs_mask[:, t, :]

                x_new, hx = model(meas_t, mask_t, dt_t, x_prev, hx)
                pred_states.append(x_new)
                x_prev = x_new

            pred_states = torch.stack(pred_states, dim=1)

            is_valid = (obs_mask.sum(dim=2, keepdim=True) > 0).float()
            state_mask = is_valid.expand(-1, -1, 6)

            # 🌟 新增：构建时间加权张量 (Time-Weighted Loss)
            # 权重随时间步呈指数增长，使得模型更加关注序列后期的稳态误差
            gamma = 0.99  # 衰减因子
            # 生成形如 [gamma^(T-1), gamma^(T-2), ..., 1] 的序列
            time_weights = (gamma ** torch.arange(T - 1, -1, -1, device=DEVICE)).view(1, T, 1)
            # 归一化权重，保证平均值为 1，不影响总体的 Learning Rate 尺度
            time_weights = time_weights / time_weights.mean()

            # 🌟 修改：在计算基础 MSE 时，乘上时间权重
            loss_pos = criterion(pred_states[:, :, [0, 2, 4]], states[:, :, [0, 2, 4]]) * time_weights
            loss_vel = criterion(pred_states[:, :, [1, 3, 5]], states[:, :, [1, 3, 5]]) * time_weights

            weight_pos = 3.0
            weight_vel = 5.0

            mask_pos = state_mask[:, :, [0, 2, 4]]
            mask_vel = state_mask[:, :, [1, 3, 5]]

            valid_count_pos = mask_pos.sum() + 1e-6
            valid_count_vel = mask_vel.sum() + 1e-6

            loss_pos_val = (loss_pos * mask_pos).sum() / valid_count_pos
            loss_vel_val = (loss_vel * mask_vel).sum() / valid_count_vel

            total_loss_val = weight_pos * loss_pos_val + weight_vel * loss_vel_val

            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += total_loss_val.item()
            count_batches += 1

        avg_loss = total_loss / count_batches
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"💾 保存最佳模型 ({save_path}): Loss {best_loss:.6f}")

    print(f"\n🎉 训练结束！最佳 Loss: {best_loss:.6f}")
    print(f"📂 请使用模型文件：{save_path}")


if __name__ == "__main__":
    train()