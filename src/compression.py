"""
Model Compression utilities for reducing communication and compute costs.
Implements magnitude pruning and quantization simulation.
"""
import torch
import copy
from src.config import PRUNING_RATE, QUANTIZATION_BITS


def magnitude_pruning(state_dict, pruning_rate=PRUNING_RATE):
    """
    Apply magnitude-based pruning to model parameters.
    Zeroes out the smallest weights by absolute value.

    Args:
        state_dict: Model state dict
        pruning_rate: Fraction of weights to prune (0.0 to 1.0)

    Returns:
        Pruned state dict and compression stats
    """
    pruned_dict = copy.deepcopy(state_dict)
    total_params = 0
    pruned_params = 0

    for key, tensor in pruned_dict.items():
        if "weight" in key and tensor.dim() >= 2:
            flat = tensor.abs().flatten()
            total_params += flat.numel()

            # Find threshold for pruning
            k = int(flat.numel() * pruning_rate)
            if k > 0:
                threshold = torch.topk(flat, k, largest=False).values.max()
                mask = tensor.abs() > threshold
                pruned_dict[key] = tensor * mask.float()
                pruned_params += k

    sparsity = pruned_params / max(total_params, 1)
    stats = {
        "total_params": total_params,
        "pruned_params": pruned_params,
        "sparsity": sparsity,
        "compression_ratio": 1.0 / (1.0 - sparsity + 1e-8),
    }
    return pruned_dict, stats


def simulate_quantization(state_dict, bits=QUANTIZATION_BITS):
    """
    Simulate quantization of model parameters (FP32 → INT8).
    This doesn't actually quantize but estimates the size reduction.

    Args:
        state_dict: Model state dict
        bits: Target bit width

    Returns:
        Size stats (original vs quantized)
    """
    original_size = 0
    quantized_size = 0

    for key, tensor in state_dict.items():
        param_bytes = tensor.numel() * 4  # FP32 = 4 bytes
        quant_bytes = tensor.numel() * (bits / 8)  # Target bits

        original_size += param_bytes
        quantized_size += quant_bytes

    stats = {
        "original_size_kb": original_size / 1024,
        "quantized_size_kb": quantized_size / 1024,
        "size_reduction": 1.0 - (quantized_size / max(original_size, 1)),
        "compression_ratio": original_size / max(quantized_size, 1),
    }
    return stats


def compress_model_update(state_dict, enable_pruning=True, enable_quantization=True):
    """
    Apply full compression pipeline to model update before sending.

    Returns:
        compressed_dict, compression_stats
    """
    stats = {}

    if enable_pruning:
        state_dict, prune_stats = magnitude_pruning(state_dict)
        stats["pruning"] = prune_stats

    if enable_quantization:
        quant_stats = simulate_quantization(state_dict)
        stats["quantization"] = quant_stats

    return state_dict, stats
