"""
Federated Training Orchestrator.
Runs the complete FL training pipeline for both Standard and Energy-Aware modes.
"""
import numpy as np
import copy
import time

from src.client import FLClient
from src.server import FLServer
from src.data_loader import load_and_partition_data
from src.utils import (
    set_seed, print_round_summary, print_experiment_header,
    print_final_comparison, save_metrics,
)
from src.config import (
    NUM_ROUNDS, SEED, COMPRESSION_ENABLED,
)


def create_clients(client_data, device="cpu"):
    """Create FLClient instances from partitioned data."""
    clients = {}
    for subject_id, data in client_data.items():
        clients[subject_id] = FLClient(subject_id, data, device)
    return clients


def run_fl_experiment(
    client_data, test_dataset, energy_aware=False,
    experiment_name="Experiment", device="cpu"
):
    """
    Run one complete Federated Learning experiment.

    Args:
        client_data: dict of {subject_id: (X, y)}
        test_dataset: global test dataset
        energy_aware: whether to use energy-aware strategies
        experiment_name: name for logging
        device: torch device

    Returns:
        metrics_history: dict with all round-by-round metrics
    """
    set_seed(SEED)
    rng = np.random.RandomState(SEED)

    print_experiment_header(experiment_name, energy_aware)

    # Create fresh clients and server
    clients = create_clients(client_data, device)
    server = FLServer(test_dataset, device)

    # Metrics tracking
    history = {
        "accuracy": [],
        "loss": [],
        "total_energy_per_round": [],
        "cumulative_energy": [],
        "active_clients_per_round": [],
        "client_participation": [],      # List of lists: which clients participated
        "battery_states": [],            # Battery levels per round
        "per_class_accuracy": [],
        "round_times": [],
        "compression_stats": [],
    }

    cumulative_energy = 0.0

    for round_num in range(NUM_ROUNDS):
        round_start = time.time()

        # ── Client Selection ──
        if energy_aware:
            selected_ids = server.select_clients_energy_aware(clients, rng)
        else:
            selected_ids = server.select_clients_standard(clients, rng)

        # ── Local Training ──
        global_params = server.get_global_params()
        client_updates = []
        round_energy = 0.0
        round_compression_stats = []

        for cid in selected_ids:
            client = clients[cid]
            model_update, metrics = client.train(
                global_params,
                energy_aware=energy_aware,
                compress=energy_aware and COMPRESSION_ENABLED,
            )
            client_updates.append((model_update, client.data_size))
            round_energy += metrics["energy"]["total_energy"]

            if metrics["compression"]:
                round_compression_stats.append(metrics["compression"])

        # Mark idle clients
        for cid, client in clients.items():
            if cid not in selected_ids:
                client.idle_round()

        # ── Aggregation ──
        if client_updates:
            server.aggregate_fedavg(client_updates)

        # ── Evaluation ──
        accuracy, loss, per_class_acc, preds, labels = server.evaluate()

        # ── Track Energy ──
        cumulative_energy += round_energy

        # Battery snapshot
        battery_snapshot = {}
        for cid, client in clients.items():
            status = client.get_energy_status()
            battery_snapshot[cid] = status["battery"]

        avg_battery = np.mean(list(battery_snapshot.values()))

        # ── Record Metrics ──
        history["accuracy"].append(accuracy)
        history["loss"].append(loss)
        history["total_energy_per_round"].append(round_energy)
        history["cumulative_energy"].append(cumulative_energy)
        history["active_clients_per_round"].append(len(selected_ids))
        history["client_participation"].append(sorted(selected_ids))
        history["battery_states"].append(battery_snapshot)
        history["per_class_accuracy"].append(per_class_acc)
        history["round_times"].append(time.time() - round_start)
        history["compression_stats"].append(round_compression_stats)

        # ── Print Progress ──
        print_round_summary(round_num, NUM_ROUNDS, {
            "accuracy": accuracy,
            "loss": loss,
            "total_energy": round_energy,
            "active_clients": len(selected_ids),
            "avg_battery": avg_battery,
        })

    # ── Final Evaluation ──
    final_acc, final_loss, final_per_class, final_preds, final_labels = server.evaluate()

    # Summary metrics
    best_round = int(np.argmax(history["accuracy"]))
    history["final_accuracy"] = final_acc
    history["final_loss"] = final_loss
    history["total_energy"] = cumulative_energy
    history["best_accuracy"] = max(history["accuracy"])
    history["best_round"] = best_round
    history["avg_clients_per_round"] = np.mean(history["active_clients_per_round"])
    history["final_predictions"] = final_preds.tolist()
    history["final_labels"] = final_labels.tolist()
    history["final_per_class_accuracy"] = {int(k): float(v) for k, v in final_per_class.items()}

    total_time = sum(history["round_times"])
    print(f"\n  ✅ {experiment_name} Complete!")
    print(f"     Final Accuracy: {final_acc * 100:.1f}%")
    print(f"     Best Accuracy:  {max(history['accuracy']) * 100:.1f}% (Round {best_round + 1})")
    print(f"     Total Energy:   {cumulative_energy:.1f}%")
    print(f"     Total Time:     {total_time:.1f}s")

    return history


def run_full_comparison(device="cpu"):
    """
    Run both Standard FL and Energy-Aware FL, then compare.

    Returns:
        standard_history, energy_history
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " ENERGY-AWARE FEDERATED LEARNING FOR HUMAN ACTIVITY RECOGNITION ".center(78) + "║")
    print("║" + " UCI HAR Dataset — 30 Clients — FedAvg ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Load data once
    print("\n📂 Loading and partitioning UCI HAR dataset...")
    client_data, test_dataset, scaler = load_and_partition_data()

    # Experiment 1: Standard FL
    standard_history = run_fl_experiment(
        client_data, test_dataset,
        energy_aware=False,
        experiment_name="Standard Federated Learning",
        device=device,
    )

    # Experiment 2: Energy-Aware FL
    energy_history = run_fl_experiment(
        client_data, test_dataset,
        energy_aware=True,
        experiment_name="Energy-Aware Federated Learning",
        device=device,
    )

    # Comparison
    print_final_comparison(standard_history, energy_history)

    # Save metrics
    save_metrics(
        {
            "standard": {k: v for k, v in standard_history.items()
                         if k not in ("final_predictions", "final_labels")},
            "energy_aware": {k: v for k, v in energy_history.items()
                            if k not in ("final_predictions", "final_labels")},
        },
        "experiment_metrics.json",
    )

    return standard_history, energy_history
