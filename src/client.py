"""
Federated Learning Client.
Handles local training, energy tracking, and model update compression.
"""
import torch
import torch.nn as nn
import numpy as np
import copy

from src.model import HARClassifier
from src.energy_model import DeviceEnergyModel
from src.compression import compress_model_update
from src.data_loader import get_client_dataloader
from src.config import (
    LOCAL_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    COMPRESSION_ENABLED, ADAPTIVE_EPOCHS, LOW_BATTERY_EPOCHS,
    NUM_FEATURES, NUM_CLASSES,
)


class FLClient:
    """
    Federated Learning client representing one smartphone user.

    Each client:
    - Holds local HAR data (from one subject)
    - Maintains a local model copy
    - Simulates battery/energy state
    - Trains locally and returns compressed updates
    """

    def __init__(self, client_id, data, device="cpu"):
        self.client_id = client_id
        self.device = device

        # Data
        self.X, self.y = data
        self.data_size = len(self.y)
        self.dataloader = get_client_dataloader(self.X, self.y, BATCH_SIZE)

        # Model (initialized each round from global model)
        self.model = HARClassifier(NUM_FEATURES, NUM_CLASSES).to(self.device)

        # Energy
        self.energy_model = DeviceEnergyModel(client_id, self.data_size)

    def get_energy_status(self):
        """Get current battery and charging status."""
        return self.energy_model.get_status()

    def estimate_training_cost(self, num_epochs):
        """Estimate energy cost before committing to training."""
        return self.energy_model.estimate_training_cost(num_epochs)

    def get_adaptive_epochs(self, energy_aware=False):
        """
        Determine number of local epochs based on battery state.
        Energy-aware mode uses fewer epochs for low-battery clients.
        """
        if energy_aware and ADAPTIVE_EPOCHS and self.energy_model.battery < 50:
            return LOW_BATTERY_EPOCHS
        return LOCAL_EPOCHS

    def train(self, global_params, energy_aware=False, compress=False):
        """
        Perform one round of local training.

        Args:
            global_params: Global model state dict to start from
            energy_aware: Whether to use adaptive epochs
            compress: Whether to compress model update

        Returns:
            model_update: state dict of trained model
            metrics: dict with training loss, energy, etc.
        """
        # Load global model
        self.model.set_params(copy.deepcopy(global_params))
        self.model.train()

        # Determine epochs
        num_epochs = self.get_adaptive_epochs(energy_aware)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=LEARNING_RATE, momentum=0.9
        )

        # Local training loop
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for X_batch, y_batch in self.dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Get model update
        model_update = self.model.get_params()

        # Compress if enabled
        compression_stats = {}
        if compress and COMPRESSION_ENABLED:
            model_update, compression_stats = compress_model_update(model_update)

        # Consume energy
        energy_info = self.energy_model.consume_energy(num_epochs)

        # Apply charging for next round
        self.energy_model.apply_charging()

        metrics = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "epochs_trained": num_epochs,
            "data_size": self.data_size,
            "energy": energy_info,
            "compression": compression_stats,
            "battery_after": self.energy_model.battery,
        }

        return model_update, metrics

    def idle_round(self):
        """Record that this client was idle (not selected) for a round."""
        self.energy_model.idle_round()
        self.energy_model.apply_charging()
