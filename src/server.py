"""
Federated Learning Server.
Handles client selection, model aggregation (FedAvg), and global evaluation.
"""
import torch
import numpy as np
import copy
from collections import OrderedDict

from src.model import HARClassifier
from src.data_loader import get_test_dataloader
from src.config import (
    NUM_FEATURES, NUM_CLASSES, CLIENT_FRACTION,
    BATTERY_THRESHOLD, COMPRESSION_ENABLED,
)


class FLServer:
    """
    Federated Learning Server.

    Manages:
    - Global model
    - Client selection (standard random vs energy-aware)
    - FedAvg aggregation
    - Global model evaluation
    """

    def __init__(self, test_dataset, device="cpu"):
        self.device = device
        self.global_model = HARClassifier(NUM_FEATURES, NUM_CLASSES).to(device)
        self.test_loader = get_test_dataloader(test_dataset)

        # Track history
        self.round_history = []

    def get_global_params(self):
        """Get current global model parameters."""
        return self.global_model.get_params()

    def select_clients_standard(self, clients, rng):
        """
        Standard FL: randomly select a fraction of clients.
        """
        num_selected = max(1, int(len(clients) * CLIENT_FRACTION))
        selected_ids = rng.choice(list(clients.keys()), num_selected, replace=False)
        return list(selected_ids)

    def select_clients_energy_aware(self, clients, rng):
        """
        Energy-Aware client selection:
        1. Filter out clients below battery threshold
        2. Prioritize clients that are charging
        3. Among remaining, prefer higher battery clients
        """
        candidates = []

        for cid, client in clients.items():
            status = client.get_energy_status()
            battery = status["battery"]
            is_charging = status["is_charging"]

            if battery >= BATTERY_THRESHOLD:
                # Score: higher is better
                # Charging clients get a bonus
                score = battery + (30.0 if is_charging else 0.0)
                candidates.append((cid, score))

        if not candidates:
            # Fallback: select top clients by battery even if below threshold
            for cid, client in clients.items():
                status = client.get_energy_status()
                candidates.append((cid, status["battery"]))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top fraction (with some randomness to avoid always picking same)
        num_desired = max(1, int(len(clients) * CLIENT_FRACTION))
        # Take top 60% deterministically, rest randomly from remaining
        num_top = max(1, int(num_desired * 0.6))
        num_random = num_desired - num_top

        selected = [c[0] for c in candidates[:num_top]]

        remaining = [c[0] for c in candidates[num_top:]]
        if remaining and num_random > 0:
            num_random = min(num_random, len(remaining))
            random_picks = rng.choice(remaining, num_random, replace=False)
            selected.extend(random_picks.tolist())

        return selected

    def aggregate_fedavg(self, client_updates):
        """
        FedAvg: weighted average of model parameters by data size.

        Args:
            client_updates: list of (state_dict, data_size) tuples
        """
        total_data = sum(ds for _, ds in client_updates)

        # Weighted average
        avg_state = OrderedDict()
        for key in client_updates[0][0].keys():
            avg_state[key] = torch.zeros_like(client_updates[0][0][key], dtype=torch.float32)
            for state_dict, data_size in client_updates:
                weight = data_size / total_data
                avg_state[key] += state_dict[key].float() * weight

        self.global_model.set_params(avg_state)

    def evaluate(self):
        """
        Evaluate global model on the test set.

        Returns:
            accuracy, loss, per_class_accuracy, predictions, true_labels
        """
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.global_model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / total

        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class_acc = {}
        for c in range(NUM_CLASSES):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class_acc[c] = (all_preds[mask] == all_labels[mask]).mean()
            else:
                per_class_acc[c] = 0.0

        return accuracy, avg_loss, per_class_acc, all_preds, all_labels
