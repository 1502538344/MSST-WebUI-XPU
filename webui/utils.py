"""
WebUI Shared Utilities
"""

import os
import json
from typing import List, Optional


def list_files(directory: str, extensions: List[str]) -> List[str]:
    """List files with specific extensions in a directory."""
    if not os.path.exists(directory):
        return []

    files = []
    for f in os.listdir(directory):
        if any(f.endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return sorted(files)


def list_configs(configs_dir: str) -> List[str]:
    """List all YAML config files."""
    return list_files(configs_dir, [".yaml", ".yml"])


def list_checkpoints(models_dir: str) -> List[str]:
    """List all checkpoint files, including those in subfolders."""
    if not os.path.exists(models_dir):
        return []

    extensions = [".ckpt", ".pth", ".pt", ".bin", ".th"]
    files = []

    # Scan root directory
    for f in os.listdir(models_dir):
        filepath = os.path.join(models_dir, f)
        if os.path.isfile(filepath) and any(f.endswith(ext) for ext in extensions):
            files.append(filepath)

    # Scan subfolders (category folders)
    for subfolder in os.listdir(models_dir):
        subfolder_path = os.path.join(models_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                filepath = os.path.join(subfolder_path, f)
                if os.path.isfile(filepath) and any(
                    f.endswith(ext) for ext in extensions
                ):
                    files.append(filepath)

    return sorted(files)


def load_json_settings(filepath: str) -> dict:
    """Load settings from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json_settings(filepath: str, settings: dict) -> None:
    """Save settings to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_preferred_device() -> str:
    """Get preferred device from settings. Returns 'auto', 'xpu:0', 'xpu:0,xpu:1', or 'cpu'."""
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".webui_config.json"
    )
    settings = load_json_settings(config_file)
    return settings.get("preferred_device", "auto")


def get_device_ids() -> list:
    """Get list of device IDs for multi-GPU training. Returns list like [0, 1] or [0]."""
    pref = get_preferred_device()

    if pref == "cpu" or pref == "auto":
        # For auto, return all available XPUs
        try:
            import torch
            import intel_extension_for_pytorch as ipex

            if torch.xpu.is_available():
                return list(range(torch.xpu.device_count()))
        except (ImportError, AttributeError):
            pass
        return []

    if "," in pref:
        # Multiple XPUs: "xpu:0,xpu:1" -> [0, 1]
        ids = []
        for p in pref.split(","):
            p = p.strip()
            if p.startswith("xpu:"):
                try:
                    ids.append(int(p.split(":")[1]))
                except (ValueError, IndexError):
                    pass
        return ids

    if pref.startswith("xpu:"):
        try:
            return [int(pref.split(":")[1])]
        except (ValueError, IndexError):
            pass

    return [0]


def get_actual_device():
    """Get the actual torch device based on preference and availability."""
    import torch

    pref = get_preferred_device()

    if pref == "cpu":
        return torch.device("cpu")

    try:
        import intel_extension_for_pytorch as ipex

        if torch.xpu.is_available():
            if pref.startswith("xpu:"):
                # Specific XPU device, e.g., "xpu:0", "xpu:1"
                try:
                    device_idx = int(pref.split(":")[1])
                    if device_idx < torch.xpu.device_count():
                        return torch.device(f"xpu:{device_idx}")
                except (ValueError, IndexError):
                    pass
            # Default to first XPU
            return torch.device("xpu:0")
    except (ImportError, AttributeError):
        pass

    return torch.device("cpu")


def get_settings_paths() -> dict:
    """Get configured paths from settings."""
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".webui_config.json"
    )
    settings = load_json_settings(config_file)
    return {
        "output_dir": settings.get("output_dir", "output"),
        "models_dir": settings.get("models_dir", "checkpoints"),
        "configs_dir": settings.get("configs_dir", "configs"),
    }
