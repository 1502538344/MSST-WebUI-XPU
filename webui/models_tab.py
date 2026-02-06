"""
Models Tab - Model Management Interface
"""

import os
import shutil
import gradio as gr
from webui.i18n import t, get_lang
from webui.utils import get_settings_paths


# Supported model types for MSST
MSST_MODEL_TYPES = [
    "mdx23c",
    "htdemucs",
    "segm_models",
    "mel_band_roformer",
    "bs_roformer",
    "swin_upernet",
    "bandit",
    "bandit_v2",
    "scnet",
    "torchseg",
    "apollo",
    "bs_mamba2",
    "conformer",
    "bs_conformer",
    "scnet_tran",
    "scnet_masked",
]

# Model categories for organizing in subfolders
MODEL_CATEGORY_FOLDERS = {
    "multi_stem_models": "Multi-Stem Models (多音轨分离)",
    "single_stem_models": "Single-Stem Models (单音轨分离)",
    "vocal_models": "Vocal Models (人声分离)",
    "UVR_VR_Models": "UVR VR Models",
}


def detect_model_type(filename):
    """Detect model type from filename."""
    name_lower = filename.lower()

    if name_lower.endswith(".onnx"):
        return "uvr_mdx"
    elif "mel_band_roformer" in name_lower or "melbandroformer" in name_lower:
        return "mel_band_roformer"
    elif "bs_roformer" in name_lower or "bsroformer" in name_lower:
        return "bs_roformer"
    elif "roformer" in name_lower:
        return "bs_roformer"
    elif "htdemucs" in name_lower or "demucs" in name_lower:
        return "htdemucs"
    elif "mdx23c" in name_lower:
        return "mdx23c"
    elif "scnet" in name_lower:
        return "scnet"
    elif "bandit" in name_lower:
        return "bandit"
    elif "swin_upernet" in name_lower or "swin-upernet" in name_lower:
        return "swin_upernet"
    elif "segm_models" in name_lower:
        return "segm_models"
    elif "apollo" in name_lower:
        return "apollo"
    elif "bs_mamba" in name_lower:
        return "bs_mamba2"
    elif "conformer" in name_lower:
        return "bs_conformer"
    else:
        return "unknown"


def detect_type_from_config(config_path):
    """Detect model type from config file content."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check for model_type in config
        if "model_type" in config:
            return config["model_type"]

        # Check audio config for hints
        if "audio" in config:
            audio = config["audio"]
            # Different models have different audio configs
            if "hop_length" in audio and audio.get("hop_length") == 512:
                if "dim" in config.get("model", {}):
                    return "bs_roformer"

        # Check model config
        if "model" in config:
            model = config["model"]
            if "dim" in model and "depth" in model:
                if "freq_transformer_depth" in model:
                    return "bs_roformer"
                elif "mel_scale" in model or "mel_band" in model.get("type", ""):
                    return "mel_band_roformer"
            if "encoder" in model:
                return "segm_models"
            if "channels" in model and "growth" in model:
                return "htdemucs"
    except Exception:
        pass

    return None


def ensure_category_folders():
    """Create category subfolders in checkpoints directory."""
    paths = get_settings_paths()
    models_dir = paths.get("models_dir", "checkpoints")

    for folder in MODEL_CATEGORY_FOLDERS.keys():
        folder_path = os.path.join(models_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

    return models_dir


def detect_model_category(filename, model_type=None):
    """Detect which category folder a model belongs to."""
    name_lower = filename.lower()

    # UVR models
    if name_lower.endswith(".onnx"):
        return "UVR_VR_Models"

    # Check for multi-stem indicators
    multi_stem_keywords = [
        "4stem",
        "6stem",
        "drums",
        "bass",
        "piano",
        "guitar",
        "musdb",
        "drumsep",
    ]
    for kw in multi_stem_keywords:
        if kw in name_lower:
            return "multi_stem_models"

    # Check for vocal-specific models
    vocal_keywords = ["vocal", "voc_", "_voc", "karaoke", "kara"]
    for kw in vocal_keywords:
        if kw in name_lower:
            return "vocal_models"

    # Single stem models (denoise, dereverb, crowd, etc.)
    single_stem_keywords = [
        "denoise",
        "dereverb",
        "reverb",
        "echo",
        "deecho",
        "crowd",
        "noise",
        "other",
        "inst_",
        "instrumental",
    ]
    for kw in single_stem_keywords:
        if kw in name_lower:
            return "single_stem_models"

    # Default based on model type
    if model_type:
        # htdemucs often used for multi-stem
        if model_type == "htdemucs":
            return "multi_stem_models"
        # roformers often for vocals
        if model_type in ["mel_band_roformer", "bs_roformer"]:
            return "vocal_models"

    # Default to single_stem
    return "single_stem_models"


def organize_existing_models():
    """Move existing models from checkpoints root to category subfolders."""
    paths = get_settings_paths()
    models_dir = paths.get("models_dir", "checkpoints")
    configs_dir = paths.get("configs_dir", "configs")

    ensure_category_folders()

    moved_count = 0
    errors = []

    # Scan root of checkpoints for model files
    if not os.path.exists(models_dir):
        return 0, []

    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)

        # Skip directories and non-model files
        if os.path.isdir(filepath):
            continue
        if not filename.endswith((".ckpt", ".pth", ".pt", ".bin", ".th", ".onnx")):
            continue

        try:
            # Detect model type and category
            model_type = detect_model_type(filename)
            category = detect_model_category(filename, model_type)

            # Move model file
            dest_dir = os.path.join(models_dir, category)
            dest_path = os.path.join(dest_dir, filename)

            if not os.path.exists(dest_path):
                shutil.move(filepath, dest_path)
                moved_count += 1

                # Note: Config files stay in configs/ directory, don't move them

        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    return moved_count, errors


# Model categories
MODEL_CATEGORIES = {
    "multi_stem_models": "Multi-Stem (多音轨分离)",
    "single_stem_models": "Single-Stem (单音轨分离)",
    "vocal_models": "Vocal Models (人声分离)",
    "UVR_VR_Models": "UVR VR Models",
}

# Predefined models available for download
AVAILABLE_MODELS = [
    # ========== Vocal Models ==========
    {
        "name": "MDX23C Vocals",
        "category": "vocal_models",
        "type": "mdx23c",
        "description": "Vocals/Other, SDR 10.17",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mdx23c.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
    },
    {
        "name": "BS Roformer Vocals (ViperX)",
        "category": "vocal_models",
        "type": "bs_roformer",
        "description": "Vocals/Other, SDR 10.87",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    {
        "name": "MelBand Roformer Vocals (ViperX)",
        "category": "vocal_models",
        "type": "mel_band_roformer",
        "description": "Vocals/Other, SDR 9.67",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    },
    {
        "name": "MelBand Roformer Vocals (KimberleyJensen)",
        "category": "vocal_models",
        "type": "mel_band_roformer",
        "description": "Vocals/Other, SDR 10.98",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "checkpoint_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    },
    {
        "name": "HTDemucs4 Vocals",
        "category": "vocal_models",
        "type": "htdemucs",
        "description": "Vocals/Other, SDR 8.78",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_htdemucs.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt",
    },
    {
        "name": "Segm Models VitLarge23 Vocals",
        "category": "vocal_models",
        "type": "segm_models",
        "description": "Vocals/Other, SDR 9.77",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt",
    },
    {
        "name": "Swin Upernet Vocals",
        "category": "vocal_models",
        "type": "swin_upernet",
        "description": "Vocals/Other, SDR 10.67",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/config_vocals_swin_upernet.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/model_swin_upernet_ep_56_sdr_10.6703.ckpt",
    },
    {
        "name": "BS Mamba2 Vocals",
        "category": "vocal_models",
        "type": "bs_mamba2",
        "description": "Vocals/Other, SDR 8.82",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.19/config_bs_mamba2_vocals.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.19/bs_mamba2_vocals.ckpt",
    },
    # ========== Multi-Stem Models ==========
    {
        "name": "HTDemucs4 (4 stems)",
        "category": "multi_stem_models",
        "type": "htdemucs",
        "description": "Bass/Drums/Vocals/Other, SDR 9.16",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    },
    {
        "name": "HTDemucs4 (6 stems)",
        "category": "multi_stem_models",
        "type": "htdemucs",
        "description": "Bass/Drums/Vocals/Other/Piano/Guitar",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    },
    {
        "name": "Demucs3 MMI (4 stems)",
        "category": "multi_stem_models",
        "type": "htdemucs",
        "description": "Bass/Drums/Vocals/Other, SDR 8.88",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_demucs3_mmi.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
    },
    {
        "name": "BandIt Plus (Speech/Music/Effects)",
        "category": "multi_stem_models",
        "type": "bandit",
        "description": "DnR SDR 11.50",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt",
    },
    {
        "name": "SCNet Large (4 stems)",
        "category": "multi_stem_models",
        "type": "scnet",
        "description": "MUSDB SDR 9.32",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/config_musdb18_scnet_large.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/model_scnet_sdr_9.3244.ckpt",
    },
    {
        "name": "SCNet XL (4 stems)",
        "category": "multi_stem_models",
        "type": "scnet",
        "description": "MUSDB SDR 9.80",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt",
    },
    {
        "name": "SCNet XL IHF (4 stems)",
        "category": "multi_stem_models",
        "type": "scnet",
        "description": "MUSDB SDR 10.08",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
    },
    {
        "name": "BS Roformer (4 stems)",
        "category": "multi_stem_models",
        "type": "bs_roformer",
        "description": "MUSDB SDR 9.65",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
    },
    {
        "name": "BS Conformer (4 stems)",
        "category": "multi_stem_models",
        "type": "bs_conformer",
        "description": "MUSDB SDR 9.18",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.18/config_musdb18_bs_conformer_infer.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.18/fused_model_bs_conformer_sdr_9.18.ckpt",
    },
    {
        "name": "DrumSep HTDemucs (kick/snare/cymbals/toms)",
        "category": "multi_stem_models",
        "type": "htdemucs",
        "description": "Drum separation",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_drumsep.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th",
    },
    {
        "name": "DrumSep MDX23C (5 stems)",
        "category": "multi_stem_models",
        "type": "mdx23c",
        "description": "kick/snare/toms/hh/cymbals",
        "config_url": "https://github.com/jarredou/models/releases/download/DrumSep/config_mdx23c.yaml",
        "checkpoint_url": "https://github.com/jarredou/models/releases/download/DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt",
    },
    # ========== Single-Stem Models ==========
    {
        "name": "HTDemucs4 Drums",
        "category": "single_stem_models",
        "type": "htdemucs",
        "description": "Drums extraction, SDR 11.13",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
    },
    {
        "name": "HTDemucs4 Bass",
        "category": "single_stem_models",
        "type": "htdemucs",
        "description": "Bass extraction, SDR 11.96",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
    },
    {
        "name": "HTDemucs4 Other",
        "category": "single_stem_models",
        "type": "htdemucs",
        "description": "Other extraction, SDR 5.85",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
    },
    {
        "name": "BS Roformer Other (ViperX)",
        "category": "single_stem_models",
        "type": "bs_roformer",
        "description": "Other stem, SDR 6.85",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
    },
    {
        "name": "MelBand Roformer Crowd",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "Crowd separation, SDR 5.99",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
    },
    {
        "name": "MelBand Roformer Denoise",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "Denoise, SDR 27.99",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/model_mel_band_roformer_denoise.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    {
        "name": "MelBand Roformer Denoise Aggressive",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "Aggressive denoise",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/model_mel_band_roformer_denoise.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    },
    {
        "name": "MelBand Roformer DeReverb (anvuew)",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "DeReverb, SDR 19.17",
        "config_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml",
        "checkpoint_url": "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    },
    {
        "name": "MelBand Roformer DeReverb/DeEcho",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "DeReverb+DeEcho, SDR 10.01",
        "config_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml",
        "checkpoint_url": "https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    },
    {
        "name": "MelBand Roformer Aspiration",
        "category": "single_stem_models",
        "type": "mel_band_roformer",
        "description": "Aspiration removal, SDR 9.85",
        "config_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml",
        "checkpoint_url": "https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_sdr_18.9845.ckpt",
    },
    {
        "name": "Apollo LQ MP3 Restoration",
        "category": "single_stem_models",
        "type": "apollo",
        "description": "Low quality audio restoration",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_apollo.yaml",
        "checkpoint_url": "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin",
    },
    {
        "name": "MDX23C Phantom Centre",
        "category": "single_stem_models",
        "type": "mdx23c",
        "description": "Centre extraction",
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt",
    },
    # ========== UVR VR Models ==========
    # Download from: https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models
    {
        "name": "Kim Vocal 1",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Kim vocal model v1",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx",
    },
    {
        "name": "Kim Vocal 2",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Kim vocal model v2",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
    },
    {
        "name": "Kim Inst",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Kim instrumental model",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Inst.onnx",
    },
    {
        "name": "UVR-MDX-NET Inst HQ 3",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "High quality instrumental separation",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
    },
    {
        "name": "UVR-MDX-NET Inst HQ 4",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "High quality instrumental v4",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_4.onnx",
    },
    {
        "name": "UVR-MDX-NET Voc FT",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Fine-tuned vocal separation",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
    },
    {
        "name": "UVR MDXNET Main",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Main vocal separation",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_Main.onnx",
    },
    {
        "name": "UVR MDXNET KARA 2",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Karaoke track extraction",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
    },
    {
        "name": "UVR-MDX-NET Crowd HQ 1",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Crowd/noise separation",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET_Crowd_HQ_1.onnx",
    },
    {
        "name": "Reverb HQ By FoxJoy",
        "category": "UVR_VR_Models",
        "type": "uvr_vr",
        "description": "Reverb removal",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx",
    },
    {
        "name": "Kuielab A Vocals",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Kuielab vocal model A",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/kuielab_a_vocals.onnx",
    },
    {
        "name": "Kuielab B Vocals",
        "category": "UVR_VR_Models",
        "type": "uvr_mdx",
        "description": "Kuielab vocal model B",
        "config_url": "",
        "checkpoint_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/kuielab_b_vocals.onnx",
    },
]


def get_file_size_str(size_bytes):
    """Convert bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def scan_local_models():
    """Scan for local model files in category subfolders."""
    paths = get_settings_paths()
    models_dir = paths.get("models_dir", "checkpoints")

    models = []

    if not os.path.exists(models_dir):
        return models

    # Scan category subfolders and root
    scan_dirs = [models_dir]  # Include root for legacy models
    for folder in MODEL_CATEGORY_FOLDERS.keys():
        folder_path = os.path.join(models_dir, folder)
        if os.path.exists(folder_path):
            scan_dirs.append(folder_path)

    for scan_dir in scan_dirs:
        # Determine category from folder name
        folder_name = os.path.basename(scan_dir)
        if folder_name in MODEL_CATEGORY_FOLDERS:
            category = MODEL_CATEGORY_FOLDERS[folder_name]
        else:
            category = "Root (未分类)"

        for f in os.listdir(scan_dir):
            filepath = os.path.join(scan_dir, f)

            # Skip directories
            if os.path.isdir(filepath):
                continue

            if f.endswith((".ckpt", ".pth", ".pt", ".bin", ".th", ".onnx")):
                size = os.path.getsize(filepath)
                model_type = detect_model_type(f)

                # Use relative path for unique identification
                rel_path = os.path.relpath(filepath, models_dir)

                models.append(
                    {
                        "name": f,
                        "rel_path": rel_path,  # For dropdown unique identification
                        "category": category,
                        "type": model_type,
                        "size": get_file_size_str(size),
                        "path": filepath,
                    }
                )

    return models


def download_file(url, dest_path, progress_callback=None):
    """Download a file with progress tracking."""
    import urllib.request
    import ssl

    # Create SSL context that doesn't verify certificates (for some URLs)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        # Get file size
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
    except Exception:
        total_size = 0

    # Download file
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")

    downloaded = 0
    block_size = 8192

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
        with open(dest_path, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded / total_size)

    return True


def create_models_tab():
    """Create the models tab for model management."""
    with gr.Tab(t("tab_models")) as tab:
        gr.Markdown(f"## {t('models_title')}")
        gr.Markdown(t("models_desc"))

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {t('local_models')}")

                local_models = gr.Dataframe(
                    headers=["Name", "Category", "Type", "Size", "Path"],
                    label=t("installed_models"),
                    interactive=False,
                    value=[],
                )

                with gr.Row():
                    refresh_btn = gr.Button(t("refresh_list"))
                    organize_btn = gr.Button(
                        f"🗂️ {t('organize_models')}", variant="secondary"
                    )

                organize_status = gr.Textbox(
                    label=t("status"), interactive=False, visible=False
                )

                with gr.Row():
                    delete_model = gr.Dropdown(label=t("select_delete"), choices=[])
                    delete_btn = gr.Button(t("delete"), variant="stop")

            with gr.Column():
                gr.Markdown(f"### {t('download_models')}")

                # Category filter
                category_filter = gr.Dropdown(
                    label="Category / 分类",
                    choices=["All / 全部"] + list(MODEL_CATEGORIES.values()),
                    value="All / 全部",
                )

                # Show available models with category
                available_data = [
                    [
                        m["name"],
                        MODEL_CATEGORIES.get(m["category"], m["category"]),
                        m["type"],
                        m["description"],
                    ]
                    for m in AVAILABLE_MODELS
                ]
                available_models = gr.Dataframe(
                    headers=["Name", "Category", "Type", "Description"],
                    label=t("available_download"),
                    interactive=False,
                    value=available_data,
                )

                with gr.Row():
                    download_model = gr.Dropdown(
                        label=t("select_download"),
                        choices=[m["name"] for m in AVAILABLE_MODELS],
                    )
                    download_btn = gr.Button(t("download"), variant="primary")

                download_status = gr.Textbox(
                    label=t("download_status"), interactive=False
                )

        with gr.Accordion(t("model_details"), open=False):
            model_info = gr.JSON(label=t("model_config"))

        # ========== Import Custom Model Section ==========
        with gr.Accordion(t("import_custom_model"), open=False):
            with gr.Row():
                # MSST Model Import
                with gr.Column():
                    gr.Markdown(f"### {t('import_msst_model')}")
                    gr.Markdown(
                        "MSST models require a checkpoint file (.ckpt/.pth/.th) and a config file (.yaml)"
                    )

                    msst_model_file = gr.File(
                        label=t("select_model_file"),
                        file_types=[".ckpt", ".pth", ".pt", ".bin", ".th"],
                    )
                    msst_config_file = gr.File(
                        label=t("select_config_file"),
                        file_types=[".yaml", ".yml"],
                    )
                    msst_model_name = gr.Textbox(
                        label=t("model_name"),
                        placeholder=t("model_name_placeholder"),
                    )
                    msst_category = gr.Dropdown(
                        label="Category / 分类",
                        choices=list(MODEL_CATEGORY_FOLDERS.values())[
                            :-1
                        ],  # Exclude UVR
                        value=MODEL_CATEGORY_FOLDERS["vocal_models"],
                    )
                    msst_model_type = gr.Dropdown(
                        label=t("detected_type"),
                        choices=MSST_MODEL_TYPES,
                        value=None,
                        allow_custom_value=True,
                    )
                    msst_import_btn = gr.Button(t("import_btn"), variant="primary")
                    msst_import_status = gr.Textbox(
                        label=t("status"), interactive=False
                    )

                # UVR Model Import
                with gr.Column():
                    gr.Markdown(f"### {t('import_uvr_model')}")
                    gr.Markdown("UVR models only require an ONNX file (.onnx)")

                    uvr_model_file = gr.File(
                        label=t("select_model_file"),
                        file_types=[".onnx"],
                    )
                    uvr_model_name = gr.Textbox(
                        label=t("model_name"),
                        placeholder=t("model_name_placeholder"),
                    )
                    uvr_import_btn = gr.Button(t("import_btn"), variant="primary")
                    uvr_import_status = gr.Textbox(label=t("status"), interactive=False)

        # Import event handlers
        def detect_msst_type(config_file):
            """Auto-detect model type from config file."""
            if config_file is None:
                return None

            try:
                import yaml

                with open(config_file.name, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Try to detect model type from config
                if "model_type" in config:
                    return config["model_type"]

                # Check filename patterns
                filename = os.path.basename(config_file.name).lower()
                for model_type in MSST_MODEL_TYPES:
                    if model_type.replace("_", "") in filename.replace("_", "").replace(
                        "-", ""
                    ):
                        return model_type

                return None
            except Exception:
                return None

        def import_msst_model(
            model_file, config_file, model_name, category, model_type
        ):
            """Import MSST model to checkpoints category subfolder."""
            if model_file is None:
                return t("no_file_selected")
            if config_file is None:
                return t("select_config_file") + " - " + t("no_file_selected")

            paths = get_settings_paths()
            models_dir = paths.get("models_dir", "checkpoints")
            configs_dir = paths.get("configs_dir", "configs")

            # Find category folder key from display value
            category_folder = "vocal_models"  # default
            for k, v in MODEL_CATEGORY_FOLDERS.items():
                if v == category:
                    category_folder = k
                    break

            # Create category folder for model
            dest_dir = os.path.join(models_dir, category_folder)
            os.makedirs(dest_dir, exist_ok=True)
            os.makedirs(configs_dir, exist_ok=True)

            try:
                # Determine filenames
                if model_name and model_name.strip():
                    base_name = model_name.strip()
                    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
                        base_name = base_name.replace(char, "_")
                else:
                    base_name = os.path.splitext(os.path.basename(model_file.name))[0]

                # Get extensions
                model_ext = os.path.splitext(model_file.name)[1]
                config_ext = os.path.splitext(config_file.name)[1]

                # Copy model file to category folder
                dest_model = os.path.join(dest_dir, base_name + model_ext)
                shutil.copy2(model_file.name, dest_model)

                # Copy config file to configs/ directory (not category folder)
                dest_config = os.path.join(configs_dir, base_name + config_ext)
                shutil.copy2(config_file.name, dest_config)

                return f"✓ {t('import_success')}\n{t('file_copied')}: {dest_model}\n{t('file_copied')}: {dest_config}"

            except Exception as e:
                return f"✗ {t('import_error')}: {str(e)}"

        def import_uvr_model(model_file, model_name):
            """Import UVR ONNX model to UVR_VR_Models subfolder."""
            if model_file is None:
                return t("no_file_selected")

            paths = get_settings_paths()
            models_dir = paths.get("models_dir", "checkpoints")

            # UVR models go to UVR_VR_Models folder
            dest_dir = os.path.join(models_dir, "UVR_VR_Models")
            os.makedirs(dest_dir, exist_ok=True)

            try:
                # Determine filename
                if model_name and model_name.strip():
                    base_name = model_name.strip()
                    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
                        base_name = base_name.replace(char, "_")
                    dest_name = base_name + ".onnx"
                else:
                    dest_name = os.path.basename(model_file.name)

                # Copy model file
                dest_path = os.path.join(dest_dir, dest_name)
                shutil.copy2(model_file.name, dest_path)

                return f"✓ {t('import_success')}\n{t('file_copied')}: {dest_path}"

            except Exception as e:
                return f"✗ {t('import_error')}: {str(e)}"

        # Wire up import events
        msst_config_file.change(
            fn=detect_msst_type,
            inputs=[msst_config_file],
            outputs=[msst_model_type],
        )

        msst_import_btn.click(
            fn=import_msst_model,
            inputs=[
                msst_model_file,
                msst_config_file,
                msst_model_name,
                msst_category,
                msst_model_type,
            ],
            outputs=[msst_import_status],
        )

        uvr_import_btn.click(
            fn=import_uvr_model,
            inputs=[uvr_model_file, uvr_model_name],
            outputs=[uvr_import_status],
        )

        # Event handlers
        def filter_by_category(category):
            if category == "All / 全部":
                filtered = AVAILABLE_MODELS
            else:
                # Find category key from display value
                cat_key = None
                for k, v in MODEL_CATEGORIES.items():
                    if v == category:
                        cat_key = k
                        break
                filtered = [m for m in AVAILABLE_MODELS if m.get("category") == cat_key]

            data = [
                [
                    m["name"],
                    MODEL_CATEGORIES.get(m["category"], m["category"]),
                    m["type"],
                    m["description"],
                ]
                for m in filtered
            ]
            choices = [m["name"] for m in filtered]
            return data, gr.update(choices=choices)

        category_filter.change(
            fn=filter_by_category,
            inputs=[category_filter],
            outputs=[available_models, download_model],
        )

        def refresh_models():
            models = scan_local_models()
            data = [
                [m["name"], m["category"], m["type"], m["size"], m["path"]]
                for m in models
            ]
            # Use rel_path for unique identification in dropdown
            choices = [m["rel_path"] for m in models]
            return data, gr.update(choices=choices)

        def do_organize_models():
            """Organize existing models into category folders."""
            moved_count, errors = organize_existing_models()
            if errors:
                error_msg = "\n".join(errors[:5])  # Show first 5 errors
                return gr.update(
                    visible=True,
                    value=f"{t('organize_errors').format(count=moved_count)}\n{error_msg}",
                )
            elif moved_count > 0:
                return gr.update(
                    visible=True,
                    value=f"✓ {t('organize_success').format(count=moved_count)}",
                )
            else:
                return gr.update(visible=True, value=t("no_models_to_organize"))

        organize_btn.click(
            fn=do_organize_models,
            inputs=[],
            outputs=[organize_status],
        )

        refresh_btn.click(
            fn=refresh_models,
            inputs=[],
            outputs=[local_models, delete_model],
        )

        def do_download(model_name):
            if not model_name:
                return t("please_select_model")

            # Find model info
            model_info = None
            for m in AVAILABLE_MODELS:
                if m["name"] == model_name:
                    model_info = m
                    break

            if not model_info:
                return t("model_not_found")

            paths = get_settings_paths()
            models_dir = paths.get("models_dir", "checkpoints")
            configs_dir = paths.get("configs_dir", "configs")

            # Get category folder from model info
            category = model_info.get("category", "single_stem_models")
            category_dir = os.path.join(models_dir, category)

            os.makedirs(category_dir, exist_ok=True)
            os.makedirs(configs_dir, exist_ok=True)

            try:
                # Get original filename from URL
                url_filename = model_info["checkpoint_url"].split("/")[-1]
                if "?" in url_filename:
                    url_filename = url_filename.split("?")[0]

                # Initialize safe_name for MSST models
                safe_name = None

                # For UVR models (onnx/pth without config), use original URL filename
                # For MSST models (with config), use sanitized display name
                if model_info.get("config_url"):
                    # MSST model: use display name
                    safe_name = model_info["name"]
                    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
                        safe_name = safe_name.replace(char, "_")
                    safe_name = safe_name.replace(" ", "_")

                    # Determine extension
                    if url_filename.endswith(".ckpt"):
                        ckpt_ext = ".ckpt"
                    elif url_filename.endswith(".pth"):
                        ckpt_ext = ".pth"
                    elif url_filename.endswith(".pt"):
                        ckpt_ext = ".pt"
                    elif url_filename.endswith(".bin"):
                        ckpt_ext = ".bin"
                    elif url_filename.endswith(".th"):
                        ckpt_ext = ".th"
                    elif url_filename.endswith(".onnx"):
                        ckpt_ext = ".onnx"
                    else:
                        ckpt_ext = ".ckpt"

                    ckpt_name = f"{safe_name}{ckpt_ext}"
                else:
                    # UVR model: use original URL filename
                    ckpt_name = url_filename

                # Save to category subfolder
                ckpt_path = os.path.join(category_dir, ckpt_name)

                yield t("downloading").format(name=ckpt_name)

                if not os.path.exists(ckpt_path):
                    download_file(model_info["checkpoint_url"], ckpt_path)

                # Download config (if available)
                if model_info.get("config_url"):
                    # Get config extension from URL
                    config_url_filename = model_info["config_url"].split("/")[-1]
                    if "?" in config_url_filename:
                        config_url_filename = config_url_filename.split("?")[0]

                    if config_url_filename.endswith(".yml"):
                        config_ext = ".yml"
                    else:
                        config_ext = ".yaml"

                    config_name = f"{safe_name}{config_ext}"
                    config_path = os.path.join(configs_dir, config_name)

                    yield t("downloading").format(name=config_name)

                    if not os.path.exists(config_path):
                        download_file(model_info["config_url"], config_path)

                yield f"✓ {model_name} {t('download_success')}"

            except Exception as e:
                yield f"Error: {str(e)}"

        download_btn.click(
            fn=do_download,
            inputs=[download_model],
            outputs=[download_status],
        )

        def do_delete(model_name):
            if not model_name:
                return t("please_select_model")

            paths = get_settings_paths()
            configs_dir = paths.get("configs_dir", "configs")

            # Find the model's full path from scan_local_models
            # model_name is now rel_path (e.g., "vocal_models/model.ckpt")
            models = scan_local_models()
            model_path = None
            actual_filename = None
            for m in models:
                if m["rel_path"] == model_name:
                    model_path = m["path"]
                    actual_filename = m["name"]
                    break

            if not model_path or not os.path.exists(model_path) or not actual_filename:
                return t("file_not_found")

            deleted_files = []

            try:
                os.remove(model_path)
                deleted_files.append(actual_filename)

                # Try to delete corresponding config file
                # Get base name without extension
                base_name = os.path.splitext(actual_filename)[0]

                # Check for matching config files in configs/ directory
                for config_ext in [".yaml", ".yml"]:
                    config_path = os.path.join(configs_dir, base_name + config_ext)
                    if os.path.exists(config_path):
                        try:
                            os.remove(config_path)
                            deleted_files.append(base_name + config_ext)
                        except Exception:
                            pass

                    # Also check in same folder as model
                    model_dir = os.path.dirname(model_path)
                    config_path_same_dir = os.path.join(
                        model_dir, base_name + config_ext
                    )
                    if os.path.exists(config_path_same_dir):
                        try:
                            os.remove(config_path_same_dir)
                            deleted_files.append(base_name + config_ext)
                        except Exception:
                            pass

                if len(deleted_files) > 1:
                    return (
                        f"✓ {t('deleted_files').format(files=', '.join(deleted_files))}"
                    )
                else:
                    return f"✓ {model_name} {t('deleted')}"
            except Exception as e:
                return f"Error: {str(e)}"

        delete_btn.click(
            fn=do_delete,
            inputs=[delete_model],
            outputs=[download_status],
        )

        def show_model_info(model_name):
            for m in AVAILABLE_MODELS:
                if m["name"] == model_name:
                    return m
            return {}

        download_model.change(
            fn=show_model_info,
            inputs=[download_model],
            outputs=[model_info],
        )

        # Initial load
        tab.select(fn=refresh_models, inputs=[], outputs=[local_models, delete_model])

    return tab
