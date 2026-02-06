"""
UVR Inference - MDX-NET Model Inference with PyTorch + XPU
Based on UVR (Ultimate Vocal Remover) MDX-NET architecture
Adapted from GPT-SoVITS/RVC implementation for Intel XPU support
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch

# Fix for onnx2torch compatibility with newer NumPy versions
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "complex"):
    np.complex = np.complex128


def get_device():
    """Get the best available device (XPU > CUDA > CPU)."""
    try:
        import intel_extension_for_pytorch as ipex

        if torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConvTDFNet:
    """Helper class for STFT/ISTFT operations."""

    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = n_fft // 2 + 1
        self.chunk_size = hop * (dim_t - 1)
        self.window = torch.hann_window(window_length=n_fft, periodic=True).to(device)
        self.device = device
        self.dim_c = 4
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - dim_f, dim_t]).to(
            device
        )

    def stft(self, x):
        """Compute STFT and format for model input."""
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        """Compute inverse STFT from model output."""
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, 2, self.chunk_size])


class MDXNetONNX:
    """MDX-NET Model for audio source separation using PyTorch + XPU."""

    def __init__(
        self,
        model_path,
        hop_length=1024,
        batch_size=1,
        window_size=512,
        aggression=5,
        enable_tta=False,
        enable_post_process=False,
        post_process_threshold=0.2,
        high_end_process=False,
        invert_spectrogram=False,
    ):
        self.model_path = model_path
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.window_size = window_size
        self.aggression = aggression
        self.enable_tta = enable_tta
        self.enable_post_process = enable_post_process
        self.post_process_threshold = post_process_threshold
        self.high_end_process = high_end_process
        self.invert_spectrogram = invert_spectrogram

        # Model dimensions - will be set from model
        self.n_fft = None
        self.dim_c = None
        self.dim_f = None
        self.dim_t = None

        self.model = None
        self.device = None
        self.stft_helper = None

    def load_model(self):
        """Load ONNX model and convert to PyTorch."""
        try:
            import onnx
            from onnx2torch import convert

            print(f"Loading ONNX model: {self.model_path}")

            # Load ONNX model
            onnx_model = onnx.load(self.model_path)

            # Get input shape from ONNX model
            input_info = onnx_model.graph.input[0]
            input_shape = [
                dim.dim_value for dim in input_info.type.tensor_type.shape.dim
            ]
            print(f"Model input shape: {input_shape}")

            if len(input_shape) == 4:
                self.dim_c = input_shape[1] if input_shape[1] > 0 else 4
                self.dim_f = input_shape[2] if input_shape[2] > 0 else 2048
                self.dim_t = input_shape[3] if input_shape[3] > 0 else 256
            else:
                self.dim_c = 4
                self.dim_f = 2048
                self.dim_t = 256

            # Calculate n_fft from dim_f
            self.n_fft = (self.dim_f - 1) * 2

            print(
                f"Dimensions: channels={self.dim_c}, freq={self.dim_f}, time={self.dim_t}, n_fft={self.n_fft}"
            )

            # Convert ONNX to PyTorch
            print("Converting ONNX to PyTorch...")
            self.model = convert(onnx_model)

            # Get device and move model
            self.device = get_device()
            print(f"Using device: {self.device}")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Create STFT helper
            self.stft_helper = ConvTDFNet(
                device=self.device,
                dim_f=self.dim_f,
                dim_t=self.dim_t,
                n_fft=self.n_fft,
                hop=self.hop_length,
            )

            # Clear XPU cache if using XPU
            if self.device.type == "xpu":
                torch.xpu.empty_cache()

            print(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def separate(self, audio_path, output_dir, output_format="wav", sr=44100):
        """Separate audio into stems."""
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model")

        # Load audio
        print(f"Loading audio: {audio_path}")
        mix, sr = librosa.load(audio_path, sr=sr, mono=False)

        # Ensure stereo
        if mix.ndim == 1:
            mix = np.asarray([mix, mix])

        # Transpose to (samples, channels)
        mix = mix.T

        # Run demix
        print(f"Running inference on {self.device}...")
        sources = self.demix(mix.T)

        # sources is the "other" stem (instrumental for vocal model)
        opt = sources.T

        # Primary = mix - other
        primary = mix - opt

        # Invert if needed
        if self.invert_spectrogram:
            primary, opt = opt, primary

        # Determine output names based on model name
        model_name = os.path.basename(self.model_path).lower()
        if "vocal" in model_name or "voc" in model_name:
            stem1_name = "Vocals"
            stem2_name = "Instrumental"
            stem1_data = primary
            stem2_data = opt
        elif "inst" in model_name:
            stem1_name = "Instrumental"
            stem2_name = "Vocals"
            stem1_data = opt
            stem2_data = primary
        elif "reverb" in model_name:
            stem1_name = "No_Reverb"
            stem2_name = "Reverb"
            stem1_data = primary
            stem2_data = opt
        elif "echo" in model_name:
            stem1_name = "No_Echo"
            stem2_name = "Echo"
            stem1_data = primary
            stem2_data = opt
        elif "noise" in model_name or "denoise" in model_name:
            stem1_name = "No_Noise"
            stem2_name = "Noise"
            stem1_data = primary
            stem2_data = opt
        else:
            stem1_name = "Primary"
            stem2_name = "Secondary"
            stem1_data = primary
            stem2_data = opt

        # Save output files
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_files = []

        ext = output_format if output_format != "mp3" else "wav"

        path1 = os.path.join(output_dir, f"{basename}_{stem1_name}.{ext}")
        sf.write(path1, stem1_data, sr)
        output_files.append(path1)
        print(f"Saved: {path1}")

        path2 = os.path.join(output_dir, f"{basename}_{stem2_name}.{ext}")
        sf.write(path2, stem2_data, sr)
        output_files.append(path2)
        print(f"Saved: {path2}")

        # Convert to mp3 if needed
        if output_format == "mp3":
            output_files = self._convert_to_mp3(output_files)

        # Clear XPU cache
        if self.device and self.device.type == "xpu":
            torch.xpu.empty_cache()

        return output_files

    def demix(self, mix):
        """Demix audio using MDX-NET model."""
        samples = mix.shape[-1]
        margin = 44100
        chunk_size = 10 * 44100  # 10 seconds chunks

        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        """Core demix implementation."""
        chunked_sources = []

        for mix_key in mixes:
            cmix = mixes[mix_key]
            n_sample = cmix.shape[1]

            trim = self.n_fft // 2
            gen_size = self.stft_helper.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size

            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))),
                axis=1,
            )

            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + self.stft_helper.chunk_size])
                mix_waves.append(waves)
                i += gen_size

            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                # Compute STFT
                spek = self.stft_helper.stft(mix_waves)

                # Run model
                if self.enable_tta:
                    # Denoise mode: average of normal and inverted
                    spec_pred = -self.model(-spek) * 0.5 + self.model(spek) * 0.5
                else:
                    spec_pred = self.model(spek)

                # Compute inverse STFT
                tar_waves = self.stft_helper.istft(spec_pred)
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()[:, :-pad]
                )

                start = 0 if mix_key == 0 else margin_size
                end = None if mix_key == list(mixes.keys())[-1] else -margin_size
                if margin_size == 0:
                    end = None

                chunked_sources.append(tar_signal[:, start:end])

        sources = np.concatenate(chunked_sources, axis=-1)
        return sources

    def _convert_to_mp3(self, wav_files):
        """Convert WAV files to MP3."""
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ffmpeg_path = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")

        mp3_files = []
        for wav_path in wav_files:
            mp3_path = wav_path.replace(".wav", ".mp3")
            try:
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", wav_path, "-b:a", "320k", mp3_path],
                    capture_output=True,
                    timeout=300,
                    encoding="utf-8",
                    errors="ignore",
                )
                os.remove(wav_path)
                mp3_files.append(mp3_path)
            except Exception as e:
                print(f"MP3 conversion failed: {e}")
                mp3_files.append(wav_path)

        return mp3_files
