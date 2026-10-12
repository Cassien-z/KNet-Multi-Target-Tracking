import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

matplotlib.rcParams["axes.unicode_minus"] = False
# Setup English font
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

labels = ["Full Model", "w/o Cross-Modal", "w/o GAT", "w/o Temporal"]
rmse = [11.41, 11.62, 13.16, 42.93]

x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(x, rmse, width, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"], edgecolor="black", linewidth=1.2)

ax.set_ylabel("RMSE (m)", fontsize=13, fontweight='bold')
ax.set_title("Ablation Study of Core Network Modules", fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, 50)

# Add text annotations on top of the bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    if i == 0:
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.2f}m',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    else:
        # Calculate percentage increase
        increase = ((height - rmse[0]) / rmse[0]) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.2f}m\n(+{increase:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()

# Save to local workspace figure folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
fig_folder = str(PROJECT_ROOT / "artifacts" / "figures")
os.makedirs(fig_folder, exist_ok=True)
workspace_path = os.path.join(fig_folder, "ablation_bar_chart.png")
plt.savefig(workspace_path, dpi=300, bbox_inches='tight')
print(f"Saved bar chart to {workspace_path}")
