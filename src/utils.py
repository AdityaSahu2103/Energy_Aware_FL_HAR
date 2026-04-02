"""
Utility functions for logging, metrics, and reproducibility.
"""
import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from src.config import SEED, RESULTS_DIR


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _convert_keys(obj):
    """Recursively convert dict keys to strings for JSON."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys(i) for i in obj]
    return obj


def save_metrics(metrics, filename):
    """Save metrics dict to JSON file."""
    ensure_dir(RESULTS_DIR)
    filepath = os.path.join(RESULTS_DIR, filename)
    clean = _convert_keys(metrics)
    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2, cls=NumpyEncoder)
    print(f"[Saved] Metrics → {filepath}")


def print_round_summary(round_num, total_rounds, metrics):
    """Print a formatted summary for one FL round."""
    acc = metrics.get("accuracy", 0) * 100
    loss = metrics.get("loss", 0)
    energy = metrics.get("total_energy", 0)
    active = metrics.get("active_clients", 0)
    avg_battery = metrics.get("avg_battery", 0)

    bar_len = 20
    filled = int(bar_len * (round_num + 1) / total_rounds)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(
        f"  Round [{bar}] {round_num + 1:3d}/{total_rounds} │ "
        f"Acc: {acc:5.1f}% │ Loss: {loss:.4f} │ "
        f"Energy: {energy:6.1f}% │ "
        f"Clients: {active:2d} │ "
        f"Avg Battery: {avg_battery:5.1f}%"
    )


def print_experiment_header(name, energy_aware):
    """Print experiment header."""
    mode = "ENERGY-AWARE" if energy_aware else "STANDARD"
    print("\n" + "=" * 80)
    print(f"  🔬 EXPERIMENT: {name}")
    print(f"  ⚡ Mode: {mode} Federated Learning")
    print(f"  🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)


def print_final_comparison(standard_metrics, energy_metrics):
    """Print a final comparison table between standard and energy-aware FL."""
    print("\n" + "=" * 80)
    print("  📊 FINAL COMPARISON: Standard FL vs Energy-Aware FL")
    print("=" * 80)

    std_acc = standard_metrics["final_accuracy"] * 100
    ea_acc = energy_metrics["final_accuracy"] * 100
    std_energy = standard_metrics["total_energy"]
    ea_energy = energy_metrics["total_energy"]
    energy_saving = (1 - ea_energy / max(std_energy, 1e-8)) * 100

    print(f"\n  {'Metric':<30} {'Standard FL':>15} {'Energy-Aware FL':>15} {'Δ':>10}")
    print(f"  {'─' * 70}")
    print(f"  {'Final Accuracy':<30} {std_acc:>14.1f}% {ea_acc:>14.1f}% {ea_acc - std_acc:>+9.1f}%")
    print(f"  {'Total Energy Consumed':<30} {std_energy:>14.1f}% {ea_energy:>14.1f}% {energy_saving:>+9.1f}%")
    print(f"  {'Avg Clients/Round':<30} {standard_metrics['avg_clients_per_round']:>15.1f} {energy_metrics['avg_clients_per_round']:>15.1f}")
    print(f"  {'Best Accuracy':<30} {standard_metrics['best_accuracy']*100:>14.1f}% {energy_metrics['best_accuracy']*100:>14.1f}%")
    print(f"  {'Best Round':<30} {standard_metrics['best_round']:>15d} {energy_metrics['best_round']:>15d}")

    print(f"\n  ⚡ Energy Savings: {energy_saving:.1f}%")
    if abs(ea_acc - std_acc) < 3.0:
        print(f"  ✅ Accuracy maintained within 3% while saving {energy_saving:.1f}% energy!")
    print("=" * 80 + "\n")
