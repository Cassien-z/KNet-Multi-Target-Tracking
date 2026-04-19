# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MultiTrackingDataset, collate_fn
from knet_model import KNet_Tracker
import os
import math

# ================= 配置区域 =================
DATA_DIR = "knet_train_data"
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-5
SEQ_LEN = 100  # 🌟 终极修复 2：从 600 改为 100（甚至是 50），极大降低梯度爆炸概率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🟢 配置：是否微调
DO_FINE_TUNE = True
MODEL_PATH = "knet_weights_finetuned.pth"
SAVE_PATH = "knet_weights_finetuned.pth" if DO_FINE_TUNE else "knet_weights.pth"

# ===========================================

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_loss = float('inf')

    # 🌟 修复 1：移除多余的重复 epoch 循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        count_batches = 0
        avg_pos_rmse_m = 0.0

        for batch_data in loader:
            states = batch_data['states'].to(DEVICE)
            meas = batch_data['meas'].to(DEVICE)

            if torch.isnan(states).any() or torch.isnan(meas).any():
                print("⚠️ 警告：DataLoader 加载到了 NaN 数据，请检查 JSON 文件！")
                continue

            mask_input = batch_data['mask_input'].to(DEVICE)
            mask_loss = batch_data['mask_loss'].to(DEVICE)
            dt = batch_data['dt'].to(DEVICE)
            baselines = batch_data['baselines'].to(DEVICE)

            Batch, Time, _ = states.shape

            # ==============================================================
            # 🌟 FIX 1: Realistic "Cold Start" Initialization
            # Instead of True State + Noise, use the raw Master radar measurement
            # to initialize x_prev, exactly like inference.py does.
            # ==============================================================
            x_prev = torch.zeros(Batch, 9, device=DEVICE)

            # Extract Master's Radar measurement from the first frame (meas[:, 0, 0:3])
            # meas is normalized: [az_norm, el_norm, r_norm]
            m_meas_0 = meas[:, 0, 0:3]
            m_mask_0 = mask_input[:, 0, 0:3]

            # Un-normalize to calculate Cartesian coordinates
            az = m_meas_0[:, 0] * math.pi
            el = m_meas_0[:, 1] * math.pi
            r = m_meas_0[:, 2] * 150000.0

            # Convert to Cartesian (x, y, z)
            x_init = r * torch.cos(el) * torch.cos(az)
            y_init = r * torch.cos(el) * torch.sin(az)
            z_init = r * torch.sin(el)

            # ... 前面 x_init 等计算保持不变 ...
            x_prev[:, 0] = x_init / 150000.0
            x_prev[:, 3] = y_init / 150000.0
            x_prev[:, 6] = z_init / 150000.0

            # 🌟 严禁作弊！速度严格设置为 0，逼迫网络学会“冷启动提速”
            x_prev[:, 1] = 0.0
            x_prev[:, 4] = 0.0
            x_prev[:, 7] = 0.0

            hx = None
            optimizer.zero_grad()
            pred_states = []

            pred_states.append(x_prev)

            for t in range(1, Time):
                dt_t = dt[:, t]
                meas_t = meas[:, t, :]
                mask_t = mask_input[:, t, :]
                base_t = baselines[:, t, :]

                # 🌟 核心修复：绝对不能在这里清空 hx！让 GRU 保持完整的序列记忆！
                x_pred, hx, _ = model(meas_t, mask_t, dt_t, base_t, x_prev, hx, step=t)

                pred_states.append(x_pred)
                x_prev = x_pred

            pred_states = torch.stack(pred_states, dim=1)
            state_mask = mask_loss.unsqueeze(2).expand(-1, -1, 9)  # 🌟 修复：扩展为 9 维

            time_weights = torch.ones(1, Time, 1, device=DEVICE)

            # 🌟 修复：使用 0,3,6 和 1,4,7 取位置和速度
            loss_pos = criterion(pred_states[:, :, [0, 3, 6]], states[:, :, [0, 3, 6]]) * time_weights
            loss_vel = criterion(pred_states[:, :, [1, 4, 7]], states[:, :, [1, 4, 7]]) * time_weights

            # 计算极坐标余弦相似度损失
            pred_x = pred_states[:, :, 0]
            pred_y = pred_states[:, :, 3]  # 🌟 y轴索引改为 3
            gt_x = states[:, :, 0]
            gt_y = states[:, :, 3]  # 🌟 y轴索引改为 3

            pred_r_xy = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-8)
            gt_r_xy = torch.sqrt(gt_x ** 2 + gt_y ** 2 + 1e-8)
            cos_sim = (pred_x * gt_x + pred_y * gt_y) / (pred_r_xy * gt_r_xy)
            loss_az = (1.0 - cos_sim) * time_weights.squeeze(-1)

            # 调整权重：因为 Cosine Loss 是稳定且有上限的，我们将权重调整为合理的比例
            weight_pos = 3.0
            weight_vel = 5.0
            weight_az = 5.0  # 🌟 适当降低惩罚，让网络有喘息空间

            mask_pos = state_mask[:, :, [0, 3, 6]]  # 🌟 修复索引
            mask_vel = state_mask[:, :, [1, 4, 7]]  # 🌟 修复索引
            mask_az = mask_loss

            valid_count_pos = mask_pos.sum() + 1e-6
            valid_count_vel = mask_vel.sum() + 1e-6
            valid_count_az = mask_az.sum() + 1e-6

            loss_pos_val = (loss_pos * mask_pos).sum() / valid_count_pos
            loss_vel_val = (loss_vel * mask_vel).sum() / valid_count_vel
            loss_az_val = (loss_az * mask_az).sum() / valid_count_az

            total_loss_val = weight_pos * loss_pos_val + weight_vel * loss_vel_val + weight_az * loss_az_val

            # 🌟 终极修复 3：如果算出来的 Loss 已经是 NaN 了，直接抛弃这个 Batch，绝不污染模型！
            if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
                print(f"⚠️ 警告：当前 Batch 产生 NaN 损失，紧急跳过！")
                optimizer.zero_grad()
                continue

            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += total_loss_val.item()
            # 🌟 提取物理误差用于监控
            avg_pos_rmse_m += math.sqrt(loss_pos_val.item()) * 150000.0
            count_batches += 1

        avg_loss = total_loss / count_batches
        avg_rmse = avg_pos_rmse_m / count_batches
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            # 🌟 现在你能看到真实的物理误差了，几万米就是没收敛，几百米才是成功！
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, 均方误差: {avg_rmse:.1f} 米")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"💾 保存最佳模型 ({save_path}): Loss {best_loss:.6f}")

    print(f"\n🎉 训练结束！最佳 Loss: {best_loss:.6f}")
    print(f"📂 请使用模型文件：{save_path}")

if __name__ == "__main__":
    train()