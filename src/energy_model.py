"""
Energy Model for simulating battery and energy consumption on edge devices.
Based on realistic energy profiling from recent FL papers (EnFed 2024, etc.)
"""
import numpy as np
from src.config import (
    INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX,
    CHARGING_PROBABILITY, CHARGE_RATE_PER_ROUND,
    ENERGY_PER_EPOCH, ENERGY_PER_COMM, ENERGY_COMPUTE_FACTOR,
)


class DeviceEnergyModel:
    """
    Simulates a mobile device's energy state for FL training.

    Tracks:
        - Battery level (0-100%)
        - Charging state
        - Cumulative energy consumed
        - Per-round energy breakdown (compute + communication)
    """

    def __init__(self, client_id, data_size, rng=None):
        self.client_id = client_id
        self.data_size = data_size
        self.rng = rng or np.random.RandomState(client_id)

        # Initialize battery state
        self.battery = self.rng.uniform(INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX)
        self.is_charging = self.rng.random() < CHARGING_PROBABILITY
        self.total_energy_consumed = 0.0
        self.round_energy_history = []

    def estimate_training_cost(self, num_epochs):
        """
        Estimate energy cost for a training round (before deciding to participate).

        Args:
            num_epochs: Number of local epochs

        Returns:
            Estimated energy cost in battery %
        """
        compute_energy = (
            num_epochs * ENERGY_PER_EPOCH
            + self.data_size * ENERGY_COMPUTE_FACTOR
        )
        comm_energy = ENERGY_PER_COMM
        return compute_energy + comm_energy

    def consume_energy(self, num_epochs):
        """
        Simulate energy consumption for one FL round.

        Args:
            num_epochs: Actual local epochs trained

        Returns:
            dict with energy breakdown
        """
        compute_energy = (
            num_epochs * ENERGY_PER_EPOCH
            + self.data_size * ENERGY_COMPUTE_FACTOR
        )
        comm_energy = ENERGY_PER_COMM

        total_cost = compute_energy + comm_energy
        self.battery = max(0.0, self.battery - total_cost)
        self.total_energy_consumed += total_cost

        energy_info = {
            "compute_energy": compute_energy,
            "comm_energy": comm_energy,
            "total_energy": total_cost,
            "battery_after": self.battery,
        }
        self.round_energy_history.append(energy_info)
        return energy_info

    def apply_charging(self):
        """Simulate battery charging between rounds."""
        # Randomly toggle charging state with some probability
        if self.rng.random() < 0.1:
            self.is_charging = not self.is_charging

        if self.is_charging:
            self.battery = min(100.0, self.battery + CHARGE_RATE_PER_ROUND)

    def idle_round(self):
        """Record an idle round (client not selected / too low battery)."""
        self.round_energy_history.append({
            "compute_energy": 0.0,
            "comm_energy": 0.0,
            "total_energy": 0.0,
            "battery_after": self.battery,
        })

    def get_status(self):
        """Get current device status."""
        return {
            "client_id": self.client_id,
            "battery": self.battery,
            "is_charging": self.is_charging,
            "total_energy_consumed": self.total_energy_consumed,
        }
