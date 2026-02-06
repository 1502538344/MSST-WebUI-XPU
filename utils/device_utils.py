"""
XPU Device Utilities for MSST

Provides device detection and selection with priority: XPU > CPU
"""

import torch

# Safe import of Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    if not IPEX_AVAILABLE:
        return False
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def get_device(device_id: int = 0) -> torch.device:
    """
    Get the best available device with priority: XPU > CPU

    Args:
        device_id: Device index for XPU (default: 0)

    Returns:
        torch.device: The selected device
    """
    if is_xpu_available():
        return torch.device(f"xpu:{device_id}")
    return torch.device("cpu")


def get_device_name() -> str:
    """Get a human-readable description of the current device."""
    if is_xpu_available():
        try:
            device_name = torch.xpu.get_device_name(0)
            return f"Intel XPU: {device_name}"
        except Exception:
            return "Intel XPU"
    return "CPU"


def empty_cache():
    """Clear device memory cache."""
    if is_xpu_available():
        torch.xpu.empty_cache()
