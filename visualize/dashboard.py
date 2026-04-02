"""
Visualization Dashboard for Energy-Aware Federated Learning.
Generates a rich multi-panel figure comparing Standard FL vs Energy-Aware FL.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import ACTIVITY_LABELS, NUM_ROUNDS, NUM_CLASSES, RESULTS_DIR
from src.utils import ensure_dir


# ─── Style ───
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 10,
})

# Color palette
COLORS = {
    "standard": "#f85149",     # Red
    "energy_aware": "#3fb950", # Green
    "accent1": "#58a6ff",      # Blue
    "accent2": "#d2a8ff",      # Purple
    "accent3": "#79c0ff",      # Light blue
    "accent4": "#ffa657",      # Orange
    "bg_dark": "#0d1117",
    "bg_card": "#161b22",
    "border": "#30363d",
    "text": "#c9d1d9",
    "text_dim": "#8b949e",
}


def plot_main_dashboard(standard_history, energy_history):
    """Generate the main 6-panel comparison dashboard."""
    ensure_dir(RESULTS_DIR)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Energy-Aware Federated Learning for Human Activity Recognition",
        fontsize=18, fontweight="bold", color="#58a6ff", y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "UCI HAR Dataset  •  30 Clients  •  FedAvg Aggregation  •  50 Rounds",
        ha="center", fontsize=11, color=COLORS["text_dim"],
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3, top=0.92, bottom=0.06, left=0.07, right=0.95)
    rounds = list(range(1, NUM_ROUNDS + 1))

    # ── Panel 1: Accuracy vs Rounds ──
    ax1 = fig.add_subplot(gs[0, 0])
    std_acc = [a * 100 for a in standard_history["accuracy"]]
    ea_acc = [a * 100 for a in energy_history["accuracy"]]
    ax1.plot(rounds, std_acc, color=COLORS["standard"], linewidth=2, label="Standard FL", alpha=0.9)
    ax1.plot(rounds, ea_acc, color=COLORS["energy_aware"], linewidth=2, label="Energy-Aware FL", alpha=0.9)
    ax1.fill_between(rounds, std_acc, alpha=0.1, color=COLORS["standard"])
    ax1.fill_between(rounds, ea_acc, alpha=0.1, color=COLORS["energy_aware"])
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("📈 Global Model Accuracy", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # ── Panel 2: Energy per Round ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        [r - 0.2 for r in rounds],
        standard_history["total_energy_per_round"],
        width=0.35, color=COLORS["standard"], alpha=0.8, label="Standard FL",
    )
    ax2.bar(
        [r + 0.2 for r in rounds],
        energy_history["total_energy_per_round"],
        width=0.35, color=COLORS["energy_aware"], alpha=0.8, label="Energy-Aware FL",
    )
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Energy Consumed (battery %)")
    ax2.set_title("⚡ Energy Consumption per Round", fontweight="bold", pad=10)
    ax2.legend(loc="upper right", framealpha=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Cumulative Energy ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        rounds, standard_history["cumulative_energy"],
        color=COLORS["standard"], linewidth=2.5, label="Standard FL", alpha=0.9,
    )
    ax3.plot(
        rounds, energy_history["cumulative_energy"],
        color=COLORS["energy_aware"], linewidth=2.5, label="Energy-Aware FL", alpha=0.9,
    )
    ax3.fill_between(
        rounds, standard_history["cumulative_energy"],
        energy_history["cumulative_energy"],
        alpha=0.15, color=COLORS["energy_aware"], label="Energy Saved",
    )
    saving = (
        standard_history["cumulative_energy"][-1]
        - energy_history["cumulative_energy"][-1]
    )
    saving_pct = saving / standard_history["cumulative_energy"][-1] * 100
    ax3.annotate(
        f"💡 {saving_pct:.0f}% Energy\n    Saved!",
        xy=(NUM_ROUNDS * 0.7, (standard_history["cumulative_energy"][-1] + energy_history["cumulative_energy"][-1]) / 2),
        fontsize=12, fontweight="bold", color=COLORS["energy_aware"],
        ha="center",
    )
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("Cumulative Energy (battery %)")
    ax3.set_title("🔋 Cumulative Energy Over Training", fontweight="bold", pad=10)
    ax3.legend(loc="upper left", framealpha=0.8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Client Participation Heatmap ──
    ax4 = fig.add_subplot(gs[1, 1])
    all_clients = sorted(set(
        cid for part_list in energy_history["client_participation"]
        for cid in part_list
    ))
    if not all_clients:
        all_clients = list(range(1, 31))

    participation_matrix = np.zeros((len(all_clients), NUM_ROUNDS))
    for r, part_list in enumerate(energy_history["client_participation"]):
        for cid in part_list:
            if cid in all_clients:
                idx = all_clients.index(cid)
                participation_matrix[idx, r] = 1

    sns.heatmap(
        participation_matrix,
        ax=ax4, cmap=["#161b22", "#3fb950"],
        cbar=False,
        xticklabels=5, yticklabels=all_clients,
        linewidths=0.1, linecolor="#21262d",
    )
    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("Client ID")
    ax4.set_title("👥 Client Participation (Energy-Aware)", fontweight="bold", pad=10)

    # ── Panel 5: Battery Distribution Over Time ──
    ax5 = fig.add_subplot(gs[2, 0])
    sample_rounds = list(range(0, NUM_ROUNDS, max(1, NUM_ROUNDS // 10)))
    if (NUM_ROUNDS - 1) not in sample_rounds:
        sample_rounds.append(NUM_ROUNDS - 1)

    battery_data_ea = []
    battery_positions_ea = []
    for r in sample_rounds:
        batteries = list(energy_history["battery_states"][r].values())
        battery_data_ea.append(batteries)
        battery_positions_ea.append(r + 1)

    bp = ax5.boxplot(
        battery_data_ea, positions=battery_positions_ea, widths=2,
        patch_artist=True, showfliers=False,
    )
    for box in bp["boxes"]:
        box.set_facecolor(COLORS["energy_aware"])
        box.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")

    # Also show standard FL battery
    battery_data_std = []
    for r in sample_rounds:
        batteries = list(standard_history["battery_states"][r].values())
        battery_data_std.append(batteries)

    bp2 = ax5.boxplot(
        battery_data_std, positions=[p + 2.5 for p in battery_positions_ea], widths=2,
        patch_artist=True, showfliers=False,
    )
    for box in bp2["boxes"]:
        box.set_facecolor(COLORS["standard"])
        box.set_alpha(0.7)
    for median in bp2["medians"]:
        median.set_color("white")

    ax5.set_xlabel("Communication Round")
    ax5.set_ylabel("Battery Level (%)")
    ax5.set_title("🔋 Battery Distribution Over Time", fontweight="bold", pad=10)
    ax5.legend(
        [bp["boxes"][0], bp2["boxes"][0]],
        ["Energy-Aware FL", "Standard FL"],
        loc="lower left", framealpha=0.8,
    )
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_ylim([0, 110])

    # ── Panel 6: Per-Activity Accuracy ──
    ax6 = fig.add_subplot(gs[2, 1])
    activity_names = [ACTIVITY_LABELS[i + 1].replace("_", "\n") for i in range(NUM_CLASSES)]
    std_per_class = standard_history["final_per_class_accuracy"]
    ea_per_class = energy_history["final_per_class_accuracy"]

    x = np.arange(NUM_CLASSES)
    width = 0.35
    bars1 = ax6.bar(x - width / 2, [std_per_class[i] * 100 for i in range(NUM_CLASSES)],
                    width, color=COLORS["standard"], alpha=0.85, label="Standard FL")
    bars2 = ax6.bar(x + width / 2, [ea_per_class[i] * 100 for i in range(NUM_CLASSES)],
                    width, color=COLORS["energy_aware"], alpha=0.85, label="Energy-Aware FL")

    # Value labels
    for bar in bars1:
        ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{bar.get_height():.0f}', ha="center", va="bottom", fontsize=7, color=COLORS["text_dim"])
    for bar in bars2:
        ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{bar.get_height():.0f}', ha="center", va="bottom", fontsize=7, color=COLORS["text_dim"])

    ax6.set_xticks(x)
    ax6.set_xticklabels(activity_names, fontsize=8)
    ax6.set_ylabel("Accuracy (%)")
    ax6.set_title("🏃 Per-Activity Classification Accuracy", fontweight="bold", pad=10)
    ax6.legend(loc="lower right", framealpha=0.8)
    ax6.grid(True, alpha=0.3, axis="y")
    ax6.set_ylim([0, 110])

    plt.savefig(
        os.path.join(RESULTS_DIR, "fl_dashboard.png"),
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"[Saved] Dashboard → {os.path.join(RESULTS_DIR, 'fl_dashboard.png')}")


def plot_confusion_matrices(standard_history, energy_history):
    """Plot confusion matrices for both experiments."""
    ensure_dir(RESULTS_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Confusion Matrices — Final Model",
        fontsize=16, fontweight="bold", color="#58a6ff", y=1.02,
    )

    activity_names_short = [ACTIVITY_LABELS[i + 1] for i in range(NUM_CLASSES)]

    for idx, (history, title, cmap) in enumerate([
        (standard_history, "Standard FL", "Reds"),
        (energy_history, "Energy-Aware FL", "Greens"),
    ]):
        labels = history["final_labels"]
        preds = history["final_predictions"]
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
        cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

        ax = axes[idx]
        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap=cmap,
            xticklabels=activity_names_short,
            yticklabels=activity_names_short,
            ax=ax, cbar_kws={"label": "% Correct"},
            linewidths=0.5, linecolor="#30363d",
        )
        ax.set_xlabel("Predicted", color=COLORS["text"])
        ax.set_ylabel("True", color=COLORS["text"])
        ax.set_title(title, fontweight="bold", pad=10, color=COLORS["text"])
        ax.tick_params(axis="both", colors=COLORS["text_dim"])

    plt.savefig(
        os.path.join(RESULTS_DIR, "confusion_matrices.png"),
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"[Saved] Confusion Matrices → {os.path.join(RESULTS_DIR, 'confusion_matrices.png')}")


def plot_energy_savings_summary(standard_history, energy_history):
    """Generate an energy savings infographic."""
    ensure_dir(RESULTS_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "⚡ Energy Savings Summary",
        fontsize=16, fontweight="bold", color=COLORS["energy_aware"], y=1.02,
    )

    # Panel 1: Total energy comparison (donut chart)
    ax = axes[0]
    std_total = standard_history["total_energy"]
    ea_total = energy_history["total_energy"]
    savings = std_total - ea_total
    savings_pct = savings / std_total * 100

    sizes = [ea_total, savings]
    colors_donut = [COLORS["energy_aware"], "#21262d"]
    wedges, texts = ax.pie(
        sizes, colors=colors_donut, startangle=90,
        wedgeprops=dict(width=0.35),
    )
    ax.text(0, 0, f"{savings_pct:.0f}%\nSaved", ha="center", va="center",
            fontsize=16, fontweight="bold", color=COLORS["energy_aware"])
    ax.set_title("Total Energy Reduction", fontweight="bold", color=COLORS["text"])

    # Panel 2: Accuracy comparison
    ax = axes[1]
    categories = ["Standard FL", "Energy-Aware FL"]
    accuracies = [
        standard_history["final_accuracy"] * 100,
        energy_history["final_accuracy"] * 100,
    ]
    bars = ax.bar(categories, accuracies,
                  color=[COLORS["standard"], COLORS["energy_aware"]], alpha=0.85)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 105])
    ax.set_title("Final Accuracy", fontweight="bold", color=COLORS["text"])
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Avg clients/round
    ax = axes[2]
    std_clients = standard_history["avg_clients_per_round"]
    ea_clients = energy_history["avg_clients_per_round"]
    bars = ax.bar(
        categories, [std_clients, ea_clients],
        color=[COLORS["standard"], COLORS["energy_aware"]], alpha=0.85,
    )
    for bar, val in zip(bars, [std_clients, ea_clients]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold",
                fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Avg Clients per Round")
    ax.set_title("Client Utilization", fontweight="bold", color=COLORS["text"])
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(
        os.path.join(RESULTS_DIR, "energy_savings_summary.png"),
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"[Saved] Energy Summary → {os.path.join(RESULTS_DIR, 'energy_savings_summary.png')}")


def generate_all_plots(standard_history, energy_history):
    """Generate all visualization outputs."""
    print("\n📊 Generating visualizations...")
    plot_main_dashboard(standard_history, energy_history)
    plot_confusion_matrices(standard_history, energy_history)
    plot_energy_savings_summary(standard_history, energy_history)
    print(f"\n✅ All plots saved to: {RESULTS_DIR}")
