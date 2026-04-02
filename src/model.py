"""
HAR Classifier Model for Federated Learning.
A 3-layer MLP with BatchNorm and Dropout — small but effective for UCI HAR.
"""
import torch
import torch.nn as nn
import copy


class HARClassifier(nn.Module):
    """
    Multi-Layer Perceptron for Human Activity Recognition.
    Architecture: 561 → 256 → 128 → 64 → 6
    """

    def __init__(self, input_dim=561, num_classes=6):
        super(HARClassifier, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if x.size(0) == 1 and self.training:
            # BatchNorm1d can't handle batch_size=1 during training
            self.eval()
            out = self.network(x)
            self.train()
            return out
        return self.network(x)

    def get_params(self):
        """Return a deep copy of model parameters."""
        return copy.deepcopy(self.state_dict())

    def set_params(self, state_dict):
        """Load parameters into the model."""
        self.load_state_dict(state_dict)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_bytes(self):
        """Estimate model size in bytes (FP32)."""
        return sum(p.numel() * 4 for p in self.parameters())  # 4 bytes per FP32
