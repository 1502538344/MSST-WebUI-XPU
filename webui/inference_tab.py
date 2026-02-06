"""
Inference Tab - Audio Separation Interface
Split into MSST and UVR separation modules
"""

import gradio as gr
import os
import tempfile
import shutil
from webui.i18n import t
from webui.utils import (
    get_settings_paths,
    get_device_ids,
)
from webui.models_tab import AVAILABLE_MODELS, MODEL_CATEGORIES, MODEL_CATEGORY_FOLDERS


def get_installed_msst_models():
    """Get list of installed MSST models (with config files)."""
    paths = get_settings_paths()
    models_dir = paths["models_dir"]
    configs_dir = paths["configs_dir"]

    installed = []

    if not os.path.exists(models_dir):
        return installed

    # Scan root and category subfolders
    scan_dirs = [(models_dir, "")]
    for folder in MODEL_CATEGORY_FOLDERS.keys():
        folder_path = os.path.join(models_dir, folder)
        if os.path.exists(folder_path):
            scan_dirs.append((folder_path, folder))

    for scan_dir, category_folder in scan_dirs:
        for f in os.listdir(scan_dir):
            filepath = os.path.join(scan_dir, f)

            # Skip directories
            if os.path.isdir(filepath):
                continue

            if f.endswith((".ckpt", ".pth", ".pt", ".bin", ".th")):
                base_name = os.path.splitext(f)[0]
                ckpt_path = filepath

                # Find matching config - check same folder first, then configs_dir
                config_path = None
                for ext in [".yaml", ".yml"]:
                    # Check same folder as model
                    cfg = os.path.join(scan_dir, base_name + ext)
                    if os.path.exists(cfg):
                        config_path = cfg
                        break
                    # Check configs_dir
                    cfg = os.path.join(configs_dir, base_name + ext)
                    if os.path.exists(cfg):
                        config_path = cfg
                        break

                # Skip if no config (not an MSST model)
                if not config_path:
                    continue

                # Try to find model info from AVAILABLE_MODELS
                model_info = None
                for m in AVAILABLE_MODELS:
                    if m.get("category") == "UVR_VR_Models":
                        continue
                    safe_name = m["name"]
                    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
                        safe_name = safe_name.replace(char, "_")
                    safe_name = safe_name.replace(" ", "_")

                    if base_name == safe_name:
                        model_info = m
                        break

                if model_info:
                    installed.append(
                        {
                            "name": model_info["name"],
                            "display": f"{model_info['name']} ({model_info['description']})",
                            "type": model_info["type"],
                            "category": category_folder
                            or model_info.get("category", ""),
                            "config_path": config_path,
                            "checkpoint_path": ckpt_path,
                        }
                    )
                else:
                    model_type = detect_model_type(f)
                    installed.append(
                        {
                            "name": base_name,
                            "display": base_name,
                            "type": model_type,
                            "category": category_folder,
                            "config_path": config_path,
                            "checkpoint_path": ckpt_path,
                        }
                    )

    return installed


def get_installed_uvr_models():
    """Get list of installed UVR models (ONNX files)."""
    paths = get_settings_paths()
    models_dir = paths["models_dir"]

    installed = []

    if not os.path.exists(models_dir):
        return installed

    # Scan root and UVR_VR_Models subfolder
    scan_dirs = [models_dir]
    uvr_folder = os.path.join(models_dir, "UVR_VR_Models")
    if os.path.exists(uvr_folder):
        scan_dirs.append(uvr_folder)

    for scan_dir in scan_dirs:
        for f in os.listdir(scan_dir):
            filepath = os.path.join(scan_dir, f)

            # Skip directories
            if os.path.isdir(filepath):
                continue

            if f.endswith(".onnx"):
                ckpt_path = filepath
                base_name = os.path.splitext(f)[0]

                # Try to find model info from AVAILABLE_MODELS
                model_info = None
                for m in AVAILABLE_MODELS:
                    if m.get("category") != "UVR_VR_Models":
                        continue
                    # Match by URL filename
                    url_filename = m["checkpoint_url"].split("/")[-1]
                    if f == url_filename:
                        model_info = m
                        break

                if model_info:
                    installed.append(
                        {
                            "name": model_info["name"],
                            "display": f"{model_info['name']} ({model_info['description']})",
                            "type": model_info["type"],
                            "checkpoint_path": ckpt_path,
                        }
                    )
                else:
                    installed.append(
                        {
                            "name": base_name,
                            "display": base_name,
                            "type": "uvr_mdx",
                            "checkpoint_path": ckpt_path,
                        }
                    )

    return installed


def detect_model_type(filename):
    """Detect model type from filename."""
    name_lower = filename.lower()
    if "mel" in name_lower and "roformer" in name_lower:
        return "mel_band_roformer"
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
    elif "apollo" in name_lower:
        return "apollo"
    elif filename.endswith(".onnx"):
        return "uvr_mdx"
    else:
        return "mel_band_roformer"


def get_msst_model_choices():
    """Get MSST model choices for dropdown."""
    models = get_installed_msst_models()
    return [m["display"] for m in models]


def get_uvr_model_choices():
    """Get UVR model choices for dropdown."""
    models = get_installed_uvr_models()
    return [m["display"] for m in models]


def refresh_msst_models():
    return gr.update(choices=get_msst_model_choices())


def refresh_uvr_models():
    return gr.update(choices=get_uvr_model_choices())


def get_msst_model_info_by_display(display_name):
    models = get_installed_msst_models()
    for m in models:
        if m["display"] == display_name:
            return m
    return None


def get_uvr_model_info_by_display(display_name):
    models = get_installed_uvr_models()
    for m in models:
        if m["display"] == display_name:
            return m
    return None


def collect_audio_files(audio_files, input_folder):
    """Collect all audio files from uploads and folder."""
    all_audio_files = []

    if audio_files:
        for audio in audio_files:
            if hasattr(audio, "name"):
                all_audio_files.append(audio.name)
            else:
                all_audio_files.append(audio)

    # Strip quotes from input folder path (only matching pairs at start/end)
    if input_folder:
        input_folder = input_folder.strip()
        # Remove matching quote pairs at start and end
        if (input_folder.startswith('"') and input_folder.endswith('"')) or (
            input_folder.startswith("'") and input_folder.endswith("'")
        ):
            input_folder = input_folder[1:-1]

    if input_folder and os.path.isdir(input_folder):
        audio_extensions = (
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
            ".wma",
            ".aiff",
            ".opus",
        )
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    all_audio_files.append(os.path.join(root, file))

    return all_audio_files


def convert_to_wav(src_path, dest_dir, ffmpeg_path):
    """Convert audio file to WAV format."""
    src_name = os.path.basename(src_path)
    base_name, ext = os.path.splitext(src_name)
    ext_lower = ext.lower()

    if ext_lower == ".wav":
        dest_path = os.path.join(dest_dir, src_name)
        shutil.copy(src_path, dest_path)
        return dest_path

    wav_name = base_name + ".wav"
    wav_path = os.path.join(dest_dir, wav_name)

    try:
        import subprocess

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            src_path,
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            wav_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                src_path,
                "-ar",
                "44100",
                "-ac",
                "2",
                wav_path,
            ]
            subprocess.run(
                cmd, capture_output=True, timeout=300, encoding="utf-8", errors="ignore"
            )

        if os.path.exists(wav_path):
            return wav_path
        else:
            shutil.copy(src_path, dest_dir)
            return os.path.join(dest_dir, src_name)
    except Exception as e:
        print(f"FFmpeg conversion failed for {src_name}: {e}")
        shutil.copy(src_path, dest_dir)
        return os.path.join(dest_dir, src_name)


def run_msst_separation(
    audio_files,
    input_folder,
    model_display_name,
    output_dir,
    extract_instrumental,
    use_tta,
    normalize,
    batch_size,
    num_overlap,
    chunk_size,
    output_format,
):
    """Run MSST audio separation."""
    all_audio_files = collect_audio_files(audio_files, input_folder)

    if not all_audio_files:
        return t("error_no_audio"), None, None

    if not model_display_name:
        return t("error_no_model"), None, None

    model_info = get_msst_model_info_by_display(model_display_name)
    if not model_info:
        return t("error_model_not_found"), None, None

    model_type = model_info["type"]
    config_path = model_info["config_path"]
    checkpoint_path = model_info["checkpoint_path"]

    if not config_path or not os.path.exists(config_path):
        return t("error_no_config") + f" ({model_info['name']})", None, None

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return t("error_no_checkpoint") + f" ({model_info['name']})", None, None

    try:
        from inference import proc_folder

        # Clear XPU memory before starting
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
        except Exception:
            pass

        paths = get_settings_paths()
        temp_input = tempfile.mkdtemp()

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ffmpeg_path = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")

        for src_path in all_audio_files:
            convert_to_wav(src_path, temp_input, ffmpeg_path)

        if not output_dir:
            output_dir = paths["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Record start time to filter new files
        import time

        start_time = time.time()

        device_ids = get_device_ids()
        if not device_ids:
            device_ids = [0]

        args = {
            "model_type": model_type,
            "config_path": config_path,
            "start_check_point": checkpoint_path,
            "input_folder": temp_input,
            "store_dir": output_dir,
            "extract_instrumental": extract_instrumental,
            "use_tta": use_tta,
            "device_ids": device_ids,
            "force_cpu": False,
            "disable_detailed_pbar": False,
            "flac_file": output_format == "flac",
            "pcm_type": "FLOAT",
            "draw_spectro": 0,
            "lora_checkpoint_peft": "",
            "lora_checkpoint_loralib": "",
            "filename_template": "{file_name}/{instr}",
            "batch_size": int(batch_size),
            "num_overlap": int(num_overlap),
            "chunk_size": int(chunk_size),
            "normalize": normalize,
            "output_format": output_format,
        }

        proc_folder(args)

        # Clear XPU memory after inference
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
        except Exception:
            pass

        # Collect only NEW output files (created after start_time)
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith((".wav", ".flac", ".mp3")):
                    full_path = os.path.join(root, file)
                    # Check if file was created/modified after start_time
                    if os.path.getmtime(full_path) >= start_time:
                        output_files.append(full_path)

        shutil.rmtree(temp_input, ignore_errors=True)

        if output_files:
            output_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return (
                t("separation_complete").format(count=len(output_files)),
                output_files[:10],
                output_files[0],
            )
        else:
            return t("separation_no_output"), None, None

    except Exception as e:
        import traceback

        error_msg = f"{t('error')}: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None, None


def run_uvr_separation(
    audio_files,
    input_folder,
    model_display_name,
    output_dir,
    output_format,
    window_size,
    aggression,
    batch_size,
    post_process_threshold,
    invert_spectrogram,
    enable_tta,
    high_end_process,
    enable_post_process,
):
    """Run UVR audio separation using ONNX."""
    all_audio_files = collect_audio_files(audio_files, input_folder)

    if not all_audio_files:
        return t("error_no_audio"), None, None

    if not model_display_name:
        return t("error_no_model"), None, None

    model_info = get_uvr_model_info_by_display(model_display_name)
    if not model_info:
        return t("error_model_not_found"), None, None

    checkpoint_path = model_info["checkpoint_path"]

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return t("error_no_checkpoint") + f" ({model_info['name']})", None, None

    try:
        from webui.uvr_inference import MDXNetONNX

        paths = get_settings_paths()
        temp_input = tempfile.mkdtemp()

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ffmpeg_path = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")

        # Convert audio files to WAV
        for src_path in all_audio_files:
            convert_to_wav(src_path, temp_input, ffmpeg_path)

        if not output_dir:
            output_dir = paths["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Record start time
        import time

        start_time = time.time()

        # Create UVR model
        model = MDXNetONNX(
            checkpoint_path,
            batch_size=int(batch_size),
            window_size=int(window_size),
            aggression=int(aggression),
            enable_tta=enable_tta,
            enable_post_process=enable_post_process,
            post_process_threshold=float(post_process_threshold),
            high_end_process=high_end_process,
            invert_spectrogram=invert_spectrogram,
        )

        # Process each file
        for wav_file in os.listdir(temp_input):
            if wav_file.endswith(".wav"):
                file_path = os.path.join(temp_input, wav_file)
                try:
                    model.separate(file_path, output_dir, output_format)
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")

        # Cleanup
        shutil.rmtree(temp_input, ignore_errors=True)

        # Collect new output files
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith((".wav", ".flac", ".mp3")):
                    full_path = os.path.join(root, file)
                    if os.path.getmtime(full_path) >= start_time:
                        output_files.append(full_path)

        if output_files:
            output_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return (
                t("separation_complete").format(count=len(output_files)),
                output_files[:10],
                output_files[0],
            )
        else:
            return t("separation_no_output"), None, None

    except ImportError as e:
        return t("uvr_install_onnxruntime") + f"\n{e}", None, None
    except Exception as e:
        import traceback

        error_msg = f"{t('error')}: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None, None


def on_msst_model_select(model_display_name):
    model_info = get_msst_model_info_by_display(model_display_name)
    if model_info:
        category = MODEL_CATEGORIES.get(model_info.get("category", ""), "")
        info_text = f"**{t('model_type')}:** {model_info['type']}\n"
        if category:
            info_text += f"**{t('category_filter')}:** {category}\n"
        if model_info.get("config_path"):
            info_text += f"**{t('config_file')}:** {os.path.basename(model_info['config_path'])}\n"
        info_text += f"**{t('checkpoint_file')}:** {os.path.basename(model_info['checkpoint_path'])}"
        return info_text
    return ""


def on_uvr_model_select(model_display_name):
    model_info = get_uvr_model_info_by_display(model_display_name)
    if model_info:
        info_text = f"**{t('model_type')}:** {model_info['type']}\n"
        info_text += f"**{t('checkpoint_file')}:** {os.path.basename(model_info['checkpoint_path'])}"
        return info_text
    return ""


def create_inference_tab():
    """Create the inference tab with MSST and UVR subtabs."""
    with gr.Tab(t("tab_inference")) as tab:
        gr.Markdown(f"## {t('inference_title')}")
        gr.Markdown(t("inference_desc"))

        with gr.Tabs():
            # ========== MSST Separation Tab ==========
            with gr.Tab(t("msst_separation")) as msst_tab:
                with gr.Row():
                    with gr.Column():
                        msst_audio_input = gr.File(
                            label=t("upload_audio"),
                            file_count="multiple",
                            file_types=["audio"],
                        )
                        msst_input_folder = gr.Textbox(
                            label=t("input_folder"),
                            placeholder=t("input_folder_placeholder"),
                            value="",
                        )

                        with gr.Row():
                            msst_model_dropdown = gr.Dropdown(
                                label=t("select_model"),
                                choices=get_msst_model_choices(),
                                allow_custom_value=False,
                            )
                            msst_refresh_btn = gr.Button("🔄", scale=0, min_width=40)

                        msst_model_info = gr.Markdown(value="")

                        paths = get_settings_paths()
                        msst_output_dir = gr.Textbox(
                            label=t("output_dir"),
                            value=paths["output_dir"],
                        )

                        with gr.Row():
                            msst_extract_instrumental = gr.Checkbox(
                                label=t("extract_instrumental"), value=False
                            )
                            msst_use_tta = gr.Checkbox(label=t("use_tta"), value=False)
                            msst_normalize = gr.Checkbox(
                                label=t("normalize"), value=False
                            )

                        with gr.Row():
                            msst_batch_size = gr.Slider(
                                label=t("batch_size"),
                                minimum=1,
                                maximum=16,
                                step=1,
                                value=1,
                                info=t("batch_size_info"),
                            )
                            msst_num_overlap = gr.Slider(
                                label=t("num_overlap"),
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=2,
                                info=t("num_overlap_info"),
                            )
                            msst_chunk_size = gr.Slider(
                                label=t("chunk_size"),
                                minimum=100000,
                                maximum=1000000,
                                step=50000,
                                value=352800,
                                info=t("chunk_size_info"),
                            )

                        with gr.Row():
                            msst_output_format = gr.Dropdown(
                                label=t("output_format"),
                                choices=["wav", "flac", "mp3"],
                                value="wav",
                            )

                        msst_separate_btn = gr.Button(
                            t("separate_btn"), variant="primary"
                        )

                    with gr.Column():
                        msst_output_status = gr.Textbox(
                            label=t("status"), lines=5, interactive=False
                        )
                        msst_output_files = gr.File(
                            label=t("separated_files"),
                            file_count="multiple",
                            interactive=False,
                        )
                        msst_audio_preview = gr.Audio(
                            label=t("preview"), interactive=False
                        )

                # MSST Event handlers
                msst_refresh_btn.click(
                    fn=refresh_msst_models, inputs=[], outputs=[msst_model_dropdown]
                )
                msst_model_dropdown.change(
                    fn=on_msst_model_select,
                    inputs=[msst_model_dropdown],
                    outputs=[msst_model_info],
                )
                msst_separate_btn.click(
                    fn=run_msst_separation,
                    inputs=[
                        msst_audio_input,
                        msst_input_folder,
                        msst_model_dropdown,
                        msst_output_dir,
                        msst_extract_instrumental,
                        msst_use_tta,
                        msst_normalize,
                        msst_batch_size,
                        msst_num_overlap,
                        msst_chunk_size,
                        msst_output_format,
                    ],
                    outputs=[msst_output_status, msst_output_files, msst_audio_preview],
                )

            # ========== UVR Separation Tab ==========
            with gr.Tab(t("uvr_separation")) as uvr_tab:
                with gr.Row():
                    with gr.Column():
                        uvr_audio_input = gr.File(
                            label=t("upload_audio"),
                            file_count="multiple",
                            file_types=["audio"],
                        )
                        uvr_input_folder = gr.Textbox(
                            label=t("input_folder"),
                            placeholder=t("input_folder_placeholder"),
                            value="",
                        )

                        with gr.Row():
                            uvr_model_dropdown = gr.Dropdown(
                                label=t("select_model"),
                                choices=get_uvr_model_choices(),
                                allow_custom_value=False,
                            )
                            uvr_refresh_btn = gr.Button("🔄", scale=0, min_width=40)

                        uvr_model_info = gr.Markdown(value="")

                        uvr_output_dir = gr.Textbox(
                            label=t("output_dir"),
                            value=paths["output_dir"],
                        )

                        with gr.Row():
                            uvr_output_format = gr.Dropdown(
                                label=t("output_format"),
                                choices=["wav", "flac", "mp3"],
                                value="wav",
                            )

                        with gr.Row():
                            uvr_window_size = gr.Slider(
                                label=t("window_size"),
                                minimum=256,
                                maximum=1024,
                                step=256,
                                value=512,
                                info=t("window_size_info"),
                            )
                            uvr_aggression = gr.Slider(
                                label=t("aggression"),
                                minimum=-100,
                                maximum=100,
                                step=1,
                                value=5,
                                info=t("aggression_info"),
                            )
                            uvr_batch_size = gr.Slider(
                                label=t("batch_size"),
                                minimum=1,
                                maximum=16,
                                step=1,
                                value=1,
                                info=t("batch_size_info"),
                            )

                        with gr.Row():
                            uvr_post_process_threshold = gr.Slider(
                                label=t("post_process_threshold"),
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.2,
                            )

                        with gr.Row():
                            uvr_invert_spectrogram = gr.Checkbox(
                                label=t("invert_spectrogram"), value=False
                            )
                            uvr_enable_tta = gr.Checkbox(
                                label=t("enable_tta"), value=False
                            )
                            uvr_high_end_process = gr.Checkbox(
                                label=t("high_end_process"), value=False
                            )
                            uvr_enable_post_process = gr.Checkbox(
                                label=t("enable_post_process"), value=False
                            )

                        uvr_separate_btn = gr.Button(
                            t("separate_btn"), variant="primary"
                        )

                    with gr.Column():
                        uvr_output_status = gr.Textbox(
                            label=t("status"), lines=5, interactive=False
                        )
                        uvr_output_files = gr.File(
                            label=t("separated_files"),
                            file_count="multiple",
                            interactive=False,
                        )
                        uvr_audio_preview = gr.Audio(
                            label=t("preview"), interactive=False
                        )

                # UVR Event handlers
                uvr_refresh_btn.click(
                    fn=refresh_uvr_models, inputs=[], outputs=[uvr_model_dropdown]
                )
                uvr_model_dropdown.change(
                    fn=on_uvr_model_select,
                    inputs=[uvr_model_dropdown],
                    outputs=[uvr_model_info],
                )
                uvr_separate_btn.click(
                    fn=run_uvr_separation,
                    inputs=[
                        uvr_audio_input,
                        uvr_input_folder,
                        uvr_model_dropdown,
                        uvr_output_dir,
                        uvr_output_format,
                        uvr_window_size,
                        uvr_aggression,
                        uvr_batch_size,
                        uvr_post_process_threshold,
                        uvr_invert_spectrogram,
                        uvr_enable_tta,
                        uvr_high_end_process,
                        uvr_enable_post_process,
                    ],
                    outputs=[uvr_output_status, uvr_output_files, uvr_audio_preview],
                )

    return tab
