"""
Configuration for Energy-Aware Federated Learning for HAR.
All hyperparameters and settings in one place.
"""
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "human+activity+recognition+using+smartphones",
    "UCI HAR Dataset",
    "UCI HAR Dataset",
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
NUM_CLASSES = 6
NUM_FEATURES = 561
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# ─────────────────────────────────────────────
# Federated Learning
# ─────────────────────────────────────────────
NUM_CLIENTS = 30          # One per subject
NUM_ROUNDS = 50           # Communication rounds
LOCAL_EPOCHS = 5          # Local training epochs per round
BATCH_SIZE = 64
LEARNING_RATE = 0.01
CLIENT_FRACTION = 0.5     # Fraction of clients selected per round
SEED = 42

# ─────────────────────────────────────────────
# Energy Model
# ─────────────────────────────────────────────
INITIAL_BATTERY_MIN = 20    # Minimum initial battery (%)
INITIAL_BATTERY_MAX = 100   # Maximum initial battery (%)
CHARGING_PROBABILITY = 0.25 # Probability a client is charging
CHARGE_RATE_PER_ROUND = 3.0 # Battery % gained per round if charging

# Energy costs (% battery per unit)
ENERGY_PER_EPOCH = 1.5      # Battery drain per local epoch
ENERGY_PER_COMM = 1.0       # Battery drain per communication round
ENERGY_COMPUTE_FACTOR = 0.003  # Scales with data size

# ─────────────────────────────────────────────
# Energy-Aware Settings
# ─────────────────────────────────────────────
BATTERY_THRESHOLD = 40.0    # Min battery % to participate
ADAPTIVE_EPOCHS = True      # Reduce epochs for low-battery clients
LOW_BATTERY_EPOCHS = 1      # Epochs for clients with battery < 50%

# ─────────────────────────────────────────────
# Model Compression
# ─────────────────────────────────────────────
COMPRESSION_ENABLED = True
PRUNING_RATE = 0.3          # 30% magnitude pruning
QUANTIZATION_BITS = 8       # INT8 quantization simulation
