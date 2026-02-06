"""
WebUI Configuration
"""

import os

# Default paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODELS_DIR = "checkpoints"
DEFAULT_CONFIGS_DIR = "configs"

# Server settings
DEFAULT_SERVER_PORT = 7860
DEFAULT_SERVER_NAME = "0.0.0.0"

# UI settings
APP_TITLE = "MSST - Music Source Separation Training & Inference"
APP_THEME = "default"

# Get paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_output_dir():
    return os.path.join(PROJECT_ROOT, DEFAULT_OUTPUT_DIR)


def get_models_dir():
    return os.path.join(PROJECT_ROOT, DEFAULT_MODELS_DIR)


def get_configs_dir():
    return os.path.join(PROJECT_ROOT, DEFAULT_CONFIGS_DIR)
