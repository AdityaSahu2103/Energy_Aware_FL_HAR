"""
Main entry point for Energy-Aware Federated Learning for HAR.
Runs the full pipeline: data loading → FL training → visualization.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.federated_train import run_full_comparison
from visualize.dashboard import generate_all_plots


def main():
    print("\n🚀 Energy-Aware Federated Learning for Human Activity Recognition")
    print("   Using UCI HAR Smartphones Dataset\n")

    # Run both experiments: Standard FL vs Energy-Aware FL
    standard_history, energy_history = run_full_comparison(device="cpu")

    # Generate visualization dashboard
    generate_all_plots(standard_history, energy_history)

    print("\n🎉 All done! Check the 'results/' folder for plots and metrics.")
    print("   Key files:")
    print("   ├── results/fl_dashboard.png          — Main 6-panel dashboard")
    print("   ├── results/confusion_matrices.png     — Confusion matrices")
    print("   ├── results/energy_savings_summary.png — Energy savings infographic")
    print("   └── results/experiment_metrics.json    — Raw metrics data")


if __name__ == "__main__":
    main()
