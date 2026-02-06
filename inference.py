# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.device_utils import get_device, is_xpu_available

import warnings

warnings.filterwarnings("ignore")


def run_folder(
    model: "torch.nn.Module",
    args: "argparse.Namespace",
    config: dict,
    device: "torch.device",
    verbose: bool = False,
) -> None:
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    # Recursively collect all files from input directory
    mixture_paths = sorted(
        glob.glob(os.path.join(args.input_folder, "**/*.*"), recursive=True)
    )
    mixture_paths = [p for p in mixture_paths if os.path.isfile(p)]

    sample_rate: int = getattr(config.audio, "sample_rate", 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments: list[str] = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    # Determine whether to use detailed progress bar
    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        # Get relative path from input folder
        relative_path: str = os.path.relpath(path, args.input_folder)
        # Extract directory and file name
        dir_name: str = os.path.dirname(relative_path)
        file_name: str = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f"Cannot read track: {format(path)}")
            print(f"Error message: {str(e)}")
            continue

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if "num_channels" in config.audio:
                if config.audio["num_channels"] == 2:
                    print("Convert mono track to stereo...")
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Normalize input audio if enabled
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Perform source separation
        waveforms_orig = demix(
            config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar
        )

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config, model, mix, waveforms_orig, device, args.model_type
            )

        # Extract instrumental track if requested
        if args.extract_instrumental:
            instr = "vocals" if "vocals" in instruments else instruments[0]
            waveforms_orig["instrumental"] = mix_orig - waveforms_orig[instr]
            if "instrumental" not in instruments:
                instruments.append("instrumental")

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            peak: float = float(np.abs(estimates).max())

            # Determine output format
            output_format = getattr(args, "output_format", "wav")
            if output_format == "flac" or args.flac_file:
                codec = "flac"
                subtype = args.pcm_type
            elif output_format == "mp3":
                codec = "mp3"
                subtype = None
            else:
                # wav
                if peak <= 1.0 and args.pcm_type != "FLOAT":
                    codec = "flac"
                else:
                    codec = "wav"
                subtype = args.pcm_type

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=dir_name,
                model_type=args.model_type,
                model=os.path.splitext(os.path.basename(args.start_check_point))[0],
            )

            # Create output directory
            output_dir: str = os.path.join(args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path: str = os.path.join(output_dir, f"{fname}.{codec}")

            if codec == "mp3":
                # Write to wav first, then convert to mp3 using ffmpeg
                temp_wav_path = os.path.join(output_dir, f"{fname}_temp.wav")
                sf.write(temp_wav_path, estimates.T, sr, subtype="FLOAT")

                # Convert to mp3 using ffmpeg
                import subprocess

                ffmpeg_path = os.path.join(
                    os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe"
                )
                if not os.path.exists(ffmpeg_path):
                    ffmpeg_path = "ffmpeg"

                try:
                    subprocess.run(
                        [
                            ffmpeg_path,
                            "-y",
                            "-i",
                            temp_wav_path,
                            "-b:a",
                            "320k",
                            output_path,
                        ],
                        capture_output=True,
                        timeout=300,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    os.remove(temp_wav_path)
                except Exception as e:
                    print(f"MP3 conversion failed: {e}, keeping WAV file")
                    os.rename(temp_wav_path, output_path.replace(".mp3", ".wav"))
            else:
                sf.write(output_path, estimates.T, sr, subtype=subtype)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def format_filename(template, **kwargs):
    """
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    """
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)

    # Device detection: XPU > CPU
    if args.force_cpu:
        device = "cpu"
    elif is_xpu_available():
        device_id = (
            args.device_ids[0] if isinstance(args.device_ids, list) else args.device_ids
        )
        device = f"xpu:{device_id}"
        print("Intel XPU is available.")
    else:
        device = "cpu"

    print("Using device: ", device)

    model_load_start_time = time.time()

    model, config = get_model_from_config(args.model_type, args.config_path)
    if "model_type" in config.training:
        args.model_type = config.training.model_type

    # Override config with user-provided inference parameters
    if dict_args is not None:
        if "batch_size" in dict_args:
            config.inference.batch_size = dict_args["batch_size"]
        if "num_overlap" in dict_args:
            config.inference.num_overlap = dict_args["num_overlap"]
        if "chunk_size" in dict_args:
            if hasattr(config, "inference") and hasattr(config.inference, "chunk_size"):
                config.inference.chunk_size = dict_args["chunk_size"]
            elif hasattr(config, "audio"):
                config.audio.chunk_size = dict_args["chunk_size"]
        if "normalize" in dict_args:
            if not hasattr(config, "inference"):
                config.inference = {}
            config.inference.normalize = dict_args["normalize"]

    if args.start_check_point:
        checkpoint = torch.load(
            args.start_check_point, weights_only=False, map_location="cpu"
        )
        load_start_checkpoint(args, model, checkpoint, type_="inference")

    print("Instruments: {}".format(config.training.instruments))

    # Multi-GPU support (XPU DataParallel when multiple device_ids provided)
    if (
        isinstance(args.device_ids, list)
        and len(args.device_ids) > 1
        and not args.force_cpu
        and is_xpu_available()
    ):
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    try:
        run_folder(model, args, config, device, verbose=True)
    finally:
        # Clean up after inference
        del model
        import gc

        gc.collect()
        if is_xpu_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()


if __name__ == "__main__":
    proc_folder(None)
