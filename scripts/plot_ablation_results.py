#!/usr/bin/env python3
"""
Plot ablation study results: radar chart and heatmap
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Set style for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Load data
with open('/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026/data/outputs/ablation/ablation_results.json', 'r') as f:
    data = json.load(f)

# Models and metrics
models = list(data.keys())
metrics = ['top1_acc', 'top3_acc', 'mae', 'attn_acc', 'regret']
metric_labels = ['Top-1 Acc', 'Top-3 Acc', 'MAE', 'Attn Acc', 'Regret']

# For radar chart, we need to normalize all metrics to [0, 1] where higher is better
# For MAE and Regret, we need to invert (1 - normalized_value)

def normalize_for_radar(values, metrics):
    """Normalize metrics for radar chart (all should be higher-is-better)

    Approach:
    - For accuracy metrics (higher is better): scale to [0.5, 1.0] based on value/max
    - For error metrics (lower is better): scale to [0.5, 1.0] using min/value ratio
    """
    normalized = []
    for i, m in enumerate(metrics):
        if m in ['mae', 'regret']:
            # For error metrics: lower is better
            # Use ratio: min_value / current_value, scaled to reasonable range
            all_vals = [data[key][m] for key in data.keys()]
            min_val = min(all_vals)
            max_val = max(all_vals)
            # Map to [0.4, 1.0] range: min_val -> 1.0, max_val -> 0.4
            if values[i] > 0:
                # Linear interpolation
                ratio = (max_val - values[i]) / (max_val - min_val) if max_val != min_val else 1.0
                norm = 0.4 + 0.6 * ratio  # Scale to [0.4, 1.0]
            else:
                norm = 1.0
        else:
            # For accuracy metrics: higher is better, directly use the value
            # Map [min_val, max_val] to [0.5, 1.0]
            all_vals = [data[key][m] for key in data.keys()]
            min_val = min(all_vals)
            max_val = max(all_vals)
            if max_val != min_val:
                norm = 0.5 + 0.5 * (values[i] - min_val) / (max_val - min_val)
            else:
                norm = 1.0
        normalized.append(norm)
    return normalized

# Prepare data for radar chart
radar_data = {}
for model in models:
    values = [data[model][m] for m in metrics]
    radar_data[model] = normalize_for_radar(values, metrics)

# Colors for each model
colors = {
    'Full': '#2E86AB',
    'No-Memory': '#A23B72',
    'No-Topology': '#F18F01',
    'No-Extractor': '#C73E1D',
    'No-Multi-step': '#6A994E'
}

# Create figure with two subplots
fig = plt.figure(figsize=(16, 7))

# ============================================================================
# Radar Chart
# ============================================================================
ax1 = fig.add_subplot(1, 2, 1, projection='polar')

# Number of variables
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each model
for model in models:
    values = radar_data[model]
    values += values[:1]  # Complete the circle
    ax1.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors.get(model, '#333333'))
    ax1.fill(angles, values, alpha=0.15, color=colors.get(model, '#333333'))

# Set labels
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metric_labels, size=13)
ax1.set_ylim(0, 1.1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Add legend
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, shadow=False)

ax1.set_title('Ablation Study - Radar Chart', size=15, pad=20, fontweight='bold')

# ============================================================================
# Heatmap
# ============================================================================
ax2 = fig.add_subplot(1, 2, 2)

# For heatmap, use normalized values relative to Full model
# Create a matrix where Full = 1.0 (or 100%), others are relative
heatmap_data = []
for model in models:
    row = []
    for m in metrics:
        if m in ['mae', 'regret']:
            # For error metrics: Full is baseline (lower is better, so we invert)
            full_val = data['Full'][m]
            if full_val > 0:
                # Lower is better, so ratio should show degradation
                rel_val = full_val / data[model][m]  # < 1 means worse
                # Scale so that Full = 1.0
                rel_val = (data[model][m] / full_val)
            else:
                rel_val = 1.0
        else:
            # For accuracy metrics: higher is better
            full_val = data['Full'][m]
            if full_val > 0:
                rel_val = data[model][m] / full_val
            else:
                rel_val = 1.0
        row.append(rel_val)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

# Plot heatmap
im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.05)

# Set ticks and labels
ax2.set_xticks(np.arange(len(metrics)))
ax2.set_yticks(np.arange(len(models)))
ax2.set_xticklabels(metric_labels, size=12)
ax2.set_yticklabels(models, size=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Relative Performance (Full=1.0)', size=11)

# Add text annotations
for i in range(len(models)):
    for j in range(len(metrics)):
        text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="black", size=10, fontweight='bold')

ax2.set_title('Ablation Study - Heatmap (Relative to Full)', size=15, pad=15, fontweight='bold')

# Add grid
ax2.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
ax2.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
ax2.grid(which="minor", color="gray", linestyle='-', linewidth=1)
ax2.tick_params(which="minor", size=0)

plt.tight_layout()

# Save the figure
output_dir = '/home/supa_2/Projects/IROS_Gaze/IROS_Gaze_2026/data/outputs/ablation'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/ablation_plots.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/ablation_plots.pdf', bbox_inches='tight')
print(f"Saved to {output_dir}/ablation_plots.png")

plt.show()
