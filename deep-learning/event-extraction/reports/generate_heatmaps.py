import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "reports"
JSON_FILE = BASE / "reports" / "lora_weight_metrics.json"

if not JSON_FILE.exists():
    raise FileNotFoundError(f"Missing {JSON_FILE}. Run calculate_metrics.py first.")

with open(JSON_FILE, "r") as f:
    metrics = json.load(f)

# 1. Module-level Norms
modules = metrics["mean_norms_per_module"]
mod_names = list(modules.keys())
mod_vals = list(modules.values())

# 2. Layer-wise total norms
layers_data = metrics["layer_total_norms"]
# Sort layers correctly 0 to N
max_layer = max([int(k) for k in layers_data.keys()])
layers = list(range(max_layer + 1))
layer_vals = [layers_data.get(str(l), 0) for l in layers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Module norms (Bar plot formatted like a heatmap)
sns.barplot(x=mod_names, y=mod_vals, ax=ax1, palette="viridis")
ax1.set_title("Mean Adapter Magnitude by Projection Module", fontsize=14)
ax1.set_ylabel("Frobenius Norm of $\\Delta W$", fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(mod_vals):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

# Plot 2: Layer-wise heatmap
layer_matrix = np.array(layer_vals).reshape(-1, 1) # Treat as 1D heatmap
sns.heatmap(layer_matrix, ax=ax2, cmap="YlOrRd", annot=True, fmt=".2f",
            yticklabels=layers, xticklabels=False, cbar=True)
ax2.set_title("Adapter Magnitude by Transformer Layer", fontsize=14)
ax2.set_ylabel("Transformer Layer (0 = earliest, 27 = deepest)", fontsize=12)

plt.tight_layout()
out_path = REPORTS_DIR / "lora_heatmaps.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved heatmaps to {out_path}")
