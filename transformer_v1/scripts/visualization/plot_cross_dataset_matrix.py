import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def plot_matrix():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman']

    out_dir = str(PROJECT_ROOT / "artifacts" / "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if results exist
    results_path = PROJECT_ROOT / "artifacts" / "metrics" / "cross_dataset_rmse.json"
    if not os.path.exists(results_path):
        print("Results file not found. Wait for evaluation to finish.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = list(data.keys())
    # Separate datasets into 3x3 and V8
    datasets_3x3 = [d for d in data[models[0]].keys() if 'v8' not in d.lower()]
    dataset_v8 = [d for d in data[models[0]].keys() if 'v8' in d.lower()][0]

    # --- Plot 1: 3x3 Heatmap ---
    matrix_3x3 = np.zeros((len(models), len(datasets_3x3)))
    for i, m in enumerate(models):
        for j, d in enumerate(datasets_3x3):
            matrix_3x3[i, j] = data[m][d]

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matrix_3x3, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('3D Position RMSE (m)', fontsize=12)

    ax.set_xticks(np.arange(len(datasets_3x3)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(datasets_3x3, fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(datasets_3x3)):
            val = matrix_3x3[i, j]
            color = "white" if val > (matrix_3x3.max() + matrix_3x3.min()) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    ax.set_title("Cross-Dataset Generalization (SNR Adaptation)", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Validation Scenarios", fontsize=13, labelpad=10)
    ax.set_ylabel("Trained Models (Varying Quality Constraints)", fontsize=13, labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cross_dataset_heatmap_3x3.png'), dpi=300)
    plt.close(fig)

    # --- Plot 2: V8 Interference Bar Chart ---
    v8_errors = [data[m][dataset_v8] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, v8_errors, color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.8, edgecolor='black', width=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('3D Position RMSE (m)', fontsize=13)
    ax.set_title("Robustness Analysis under Extreme Interference (V8)", fontsize=14, pad=15, fontweight='bold')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.2f}",
                ha='center', va='bottom', fontweight='bold', fontsize=11)
                
    # Emphasize the "Sweet Spot" (V6 is the middle bar)
    ax.annotate('Sweet Spot', xy=(1, v8_errors[1]), xytext=(1, v8_errors[1] + 5),
                ha='center', va='bottom',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, fontweight='bold', color='red')

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'interference_robustness_bar.png'), dpi=300)
    plt.close(fig)

    print("Heatmap and Bar Chart generated successfully.")

if __name__ == "__main__":
    plot_matrix()
