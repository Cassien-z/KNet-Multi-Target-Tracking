import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Simulation parameters
T_start = 0.0
T_end = 1.0 # Plot 1 second to show detail
dt = 0.025 # 25ms base step
time_stamps = np.arange(T_start, T_end, dt)

radar_times = []
eo_times = []
esm_times = []

for t in time_stamps:
    if np.random.rand() < 0.95:
        radar_times.append(t)
    
    if np.isclose(t % 0.05, 0) and np.random.rand() < 0.90:
        eo_times.append(t)
        
    if np.isclose(t % 0.20, 0) and np.random.rand() < 0.85:
        esm_times.append(t)

fig, ax = plt.subplots(figsize=(10, 3.5))

ax.plot(radar_times, [3]*len(radar_times), 'o', color='magenta', markersize=8, alpha=0.7, label='Radar (25ms, 40Hz)')
ax.plot(eo_times, [2]*len(eo_times), 's', color='cyan', markersize=8, alpha=0.9, label='EO (50ms, 20Hz)')
ax.plot(esm_times, [1]*len(esm_times), 'D', color='orange', markersize=9, alpha=0.9, label='ESM (200ms, 5Hz)')

# Formatting
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['ESM\n(Passive)', 'EO\n(Electro-Optical)', 'Radar\n(Active)'], fontsize=12)
ax.set_ylim(0.5, 3.5)
ax.set_xlim(0, 1.0)
ax.set_xlabel('Simulation Time (s)', fontsize=12)
ax.set_title('Asynchronous Multi-frequency Downsampling Mechanism (1s Window)', fontsize=14, pad=10)

ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.set_xticks(np.arange(0, 1.01, 0.025), minor=True)
ax.grid(which='minor', axis='x', linestyle=':', alpha=0.4)
ax.grid(which='major', axis='x', linestyle='-', alpha=0.6)
ax.grid(which='major', axis='y', linestyle='--', alpha=0.5)

ax.legend(loc='lower right', fontsize=10)

fig.tight_layout()
fig.savefig(FIGURES_DIR / 'async_sampling_plot.png', dpi=300)
plt.close(fig)

print("Asynchronous sampling plot generated (English).")
