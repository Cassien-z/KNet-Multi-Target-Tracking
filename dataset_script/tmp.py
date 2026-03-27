import matplotlib.pyplot as plt
import numpy as np

# 数据提取自您的日志
# 格式: (Epoch, Loss)
# 注意：日志中每10个epoch打印一次平均Loss，中间穿插着"保存最佳模型"的实时Loss。
# 为了绘图清晰，我们主要绘制每10个epoch的平均Loss（代表该阶段的整体水平），
# 并标注出最佳模型的保存点。

epochs_avg = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
loss_avg = [
    0.327559, 0.339475, 0.256240, 0.295517, 0.275269,
    0.276496, 0.277508, 0.319112, 0.297652, 0.293361,
    0.268510, 0.259328, 0.304489, 0.307689, 0.308695
]

# 最佳模型保存记录 (Epoch是估算的，基于打印顺序)
# 前几个epoch下降极快，从70000降到0.3
best_loss_points = [
    (1, 70410.8), (2, 3031.8), (3, 4.76), (4, 0.59),
    (5, 0.43), (6, 0.40), (7, 0.34), (8, 0.327), # 对应Epoch 10前的快速下降
    (9, 0.303), (10, 0.295), (11, 0.286), (12, 0.271), # 10-20之间
    (15, 0.256), # 30之前
    (18, 0.253), # 50之前
    (22, 0.245)  # 70之前，这是全局最低点
]

# 分离坐标
best_epochs = [p[0] for p in best_loss_points]
best_losses = [p[1] for p in best_loss_points]

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

# 由于初始Loss极大(70000)，直接画会压缩后面的细节。
# 策略：使用对数坐标(Y轴) 或者 分段显示。这里推荐使用对数坐标以展示全貌。
ax.semilogy(epochs_avg, loss_avg, 'o-', label='Average Loss (per 10 epochs)', color='blue', linewidth=2, markersize=8)
ax.scatter(best_epochs, best_losses, color='red', s=40, label='Best Model Saved', zorder=5, alpha=0.7)

# 标注最低点
min_loss = min(best_losses)
min_epoch = best_epochs[best_losses.index(min_loss)]
ax.annotate(f'Global Min\nLoss: {min_loss:.4f}',
            xy=(min_epoch, min_loss),
            xytext=(min_epoch + 10, min_loss * 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            fontsize=10, color='darkred', fontweight='bold')

# 标题和标签
plt.title('KNet Training Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Log Scale)', fontsize=12)
plt.legend(loc='upper right')

# 网格优化
ax.grid(True, which="both", ls="-", alpha=0.6)

# 添加文本说明
text_str = (f"Observations:\n"
            f"1. Rapid convergence in first 10 epochs (70k -> 0.3).\n"
            f"2. Global minimum at Epoch ~{min_epoch} (Loss={min_loss:.4f}).\n"
            f"3. Slight oscillation/stagnation after Epoch 30 (0.25 - 0.31).")
plt.text(0.02, 0.02, text_str, transform=ax.transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()