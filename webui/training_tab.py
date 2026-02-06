"""
Training Tab - Model Training, Validation, and Dataset Guide
"""

import gradio as gr
import os
import subprocess
import threading
from datetime import datetime
from webui.i18n import t
from webui.utils import (
    list_configs,
    list_checkpoints,
    get_settings_paths,
    get_device_ids,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_training_process = None
_training_lock = threading.Lock()
_training_log = []
_training_log_file = None

_validation_process = None
_validation_lock = threading.Lock()
_validation_log = []
_validation_log_file = None

MODEL_TYPES = [
    "mel_band_roformer",
    "bs_roformer",
    "htdemucs",
    "mdx23c",
    "scnet",
    "scnet_unofficial",
    "segm_models",
    "swin_upernet",
    "bandit",
    "bandit_v2",
    "apollo",
    "torchseg",
]

TRAIN_METRICS_LIST = [
    "sdr",
    "k_sdr",
    "l1_freq",
    "si_sdr",
    "log_wmse",
    "aura_stft",
    "aura_mrstft",
    "bleedless",
    "fullness",
]

VALID_METRICS_LIST = [
    "sdr",
    "k_sdr",
    "l1_freq",
    "si_sdr",
    "neg_log_wmse",
    "aura_stft",
    "aura_mrstft",
    "bleedless",
    "fullness",
]


def get_config_choices():
    paths = get_settings_paths()
    configs_dir = paths["configs_dir"]
    configs = list_configs(configs_dir)
    return [os.path.basename(c) for c in configs]


def get_checkpoint_choices():
    paths = get_settings_paths()
    models_dir = paths["models_dir"]
    checkpoints = list_checkpoints(models_dir)
    choices = [""]
    for c in checkpoints:
        rel_path = os.path.relpath(c, models_dir)
        choices.append(rel_path)
    return choices


def refresh_configs():
    return gr.update(choices=get_config_choices())


def refresh_checkpoints():
    return gr.update(choices=get_checkpoint_choices())


def read_process_output(process, log_list, log_file=None):
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                stripped = line.strip()
                log_list.append(stripped)
                if len(log_list) > 500:
                    log_list[:] = log_list[-500:]
                if log_file:
                    try:
                        log_file.write(line)
                        log_file.flush()
                    except Exception:
                        pass
    except Exception:
        pass


def start_training(
    model_type,
    config_name,
    checkpoint_name,
    data_paths,
    valid_paths,
    results_path,
    dataset_type,
    num_workers,
    batch_size,
    device_ids_str,
    seed,
    pin_memory,
    use_multistft_loss,
    use_mse_loss,
    use_l1_loss,
    pre_valid,
    metrics,
    metric_for_scheduler,
):
    global _training_process, _training_log

    with _training_lock:
        if _training_process is not None and _training_process.poll() is None:
            return t("training_running"), get_training_log()

    if not config_name:
        return t("error_no_config"), ""
    if not data_paths:
        return "Please provide data paths", ""
    if not results_path:
        return "Please provide results path", ""

    try:
        paths = get_settings_paths()
        configs_dir = paths["configs_dir"]
        models_dir = paths["models_dir"]

        config_path = os.path.join(configs_dir, config_name)
        if not os.path.exists(config_path):
            return f"Config not found: {config_path}", ""

        try:
            device_ids = [
                int(x.strip()) for x in device_ids_str.split(",") if x.strip()
            ]
        except ValueError:
            device_ids = [0]

        cmd = [
            "python",
            "train.py",
            "--model_type",
            model_type,
            "--config_path",
            config_path,
            "--results_path",
            results_path,
            "--dataset_type",
            str(dataset_type),
            "--num_workers",
            str(int(num_workers)),
            "--batch_size",
            str(int(batch_size)),
            "--seed",
            str(int(seed)),
            "--device_ids",
        ] + [str(d) for d in device_ids]

        data_path_list = [p.strip() for p in data_paths.split(";") if p.strip()]
        if data_path_list:
            cmd.extend(["--data_path"] + data_path_list)

        if valid_paths:
            valid_path_list = [p.strip() for p in valid_paths.split(";") if p.strip()]
            if valid_path_list:
                cmd.extend(["--valid_path"] + valid_path_list)

        if checkpoint_name:
            checkpoint_path = os.path.join(models_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                cmd.extend(["--start_check_point", checkpoint_path])

        if pin_memory:
            cmd.append("--pin_memory")

        if use_multistft_loss:
            cmd.append("--use_multistft_loss")
        if use_mse_loss:
            cmd.append("--use_mse_loss")
        if use_l1_loss:
            cmd.append("--use_l1_loss")
        if pre_valid:
            cmd.append("--pre_valid")

        if metrics:
            cmd.extend(["--metrics"] + list(metrics))

        if metric_for_scheduler:
            cmd.extend(["--metric_for_scheduler", metric_for_scheduler])

        os.makedirs(results_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"train_{timestamp}.log")
        _training_log_file = open(log_filename, "w", encoding="utf-8")

        _training_log = [
            f"Starting training with command:",
            " ".join(cmd),
            f"Log file: {log_filename}",
            "",
        ]
        _training_log_file.write(
            f"Starting training with command:\n{' '.join(cmd)}\n\n"
        )
        _training_log_file.flush()

        _training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        output_thread = threading.Thread(
            target=read_process_output,
            args=(_training_process, _training_log, _training_log_file),
            daemon=True,
        )
        output_thread.start()

        return t("training_started"), get_training_log()

    except Exception as e:
        import traceback

        error_msg = f"{t('training_error')}: {str(e)}\n{traceback.format_exc()}"
        return error_msg, ""


def stop_training():
    global _training_process, _training_log

    with _training_lock:
        if _training_process is not None and _training_process.poll() is None:
            try:
                import platform

                if platform.system() == "Windows":
                    import subprocess as sp

                    sp.run(
                        ["taskkill", "/F", "/T", "/PID", str(_training_process.pid)],
                        capture_output=True,
                    )
                else:
                    import signal

                    os.killpg(os.getpgid(_training_process.pid), signal.SIGTERM)

                try:
                    _training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _training_process.kill()
                    _training_process.wait()

                _training_log.append("")
                _training_log.append("=== Training stopped by user ===")
                _training_process = None
                close_training_log_file()
                return t("training_stopped"), get_training_log()
            except Exception as e:
                close_training_log_file()
                return f"Error stopping training: {e}", get_training_log()
        else:
            return "No training process running", get_training_log()


def get_training_log():
    return "\n".join(_training_log[-100:])


def refresh_training_log():
    global _training_process
    status = t("ready")
    if _training_process is not None:
        if _training_process.poll() is None:
            status = t("training_running")
        else:
            status = f"Training finished (exit code: {_training_process.returncode})"
    return status, get_training_log()


def start_validation(
    model_type,
    config_name,
    checkpoint_name,
    valid_paths,
    store_dir,
    device_ids_str,
    num_workers,
    pin_memory,
    extension,
    use_tta,
    metrics,
):
    global _validation_process, _validation_log

    with _validation_lock:
        if _validation_process is not None and _validation_process.poll() is None:
            return t("validation_running"), get_validation_log()

    if not config_name:
        return t("error_no_config"), ""
    if not checkpoint_name:
        return t("error_no_checkpoint"), ""
    if not valid_paths:
        return "Please provide validation paths", ""

    try:
        paths = get_settings_paths()
        configs_dir = paths["configs_dir"]
        models_dir = paths["models_dir"]

        config_path = os.path.join(configs_dir, config_name)
        if not os.path.exists(config_path):
            return f"Config not found: {config_path}", ""

        checkpoint_path = os.path.join(models_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            return f"Checkpoint not found: {checkpoint_path}", ""

        try:
            device_ids = [
                int(x.strip()) for x in device_ids_str.split(",") if x.strip()
            ]
        except ValueError:
            device_ids = [0]

        cmd = [
            "python",
            "valid.py",
            "--model_type",
            model_type,
            "--config_path",
            config_path,
            "--start_check_point",
            checkpoint_path,
            "--num_workers",
            str(int(num_workers)),
            "--extension",
            extension,
            "--device_ids",
        ] + [str(d) for d in device_ids]

        valid_path_list = [p.strip() for p in valid_paths.split(";") if p.strip()]
        if valid_path_list:
            cmd.extend(["--valid_path"] + valid_path_list)

        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            cmd.extend(["--store_dir", store_dir])

        if pin_memory:
            cmd.append("--pin_memory")

        if use_tta:
            cmd.append("--use_tta")

        if metrics:
            cmd.extend(["--metrics"] + list(metrics))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"valid_{timestamp}.log")
        _validation_log_file = open(log_filename, "w", encoding="utf-8")

        _validation_log = [
            f"Starting validation with command:",
            " ".join(cmd),
            f"Log file: {log_filename}",
            "",
        ]
        _validation_log_file.write(
            f"Starting validation with command:\n{' '.join(cmd)}\n\n"
        )
        _validation_log_file.flush()

        _validation_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        output_thread = threading.Thread(
            target=read_process_output,
            args=(_validation_process, _validation_log, _validation_log_file),
            daemon=True,
        )
        output_thread.start()

        return t("validation_started"), get_validation_log()

    except Exception as e:
        import traceback

        error_msg = f"{t('validation_error')}: {str(e)}\n{traceback.format_exc()}"
        return error_msg, ""


def stop_validation():
    global _validation_process, _validation_log

    with _validation_lock:
        if _validation_process is not None and _validation_process.poll() is None:
            try:
                import platform

                if platform.system() == "Windows":
                    import subprocess as sp

                    sp.run(
                        ["taskkill", "/F", "/T", "/PID", str(_validation_process.pid)],
                        capture_output=True,
                    )
                else:
                    import signal

                    os.killpg(os.getpgid(_validation_process.pid), signal.SIGTERM)

                try:
                    _validation_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _validation_process.kill()
                    _validation_process.wait()

                _validation_log.append("")
                _validation_log.append("=== Validation stopped by user ===")
                _validation_process = None
                close_validation_log_file()
                return t("validation_stopped"), get_validation_log()
            except Exception as e:
                close_validation_log_file()
                return f"Error stopping validation: {e}", get_validation_log()
        else:
            return "No validation process running", get_validation_log()


def get_validation_log():
    return "\n".join(_validation_log[-100:])


def refresh_validation_log():
    global _validation_process
    status = t("ready")
    if _validation_process is not None:
        if _validation_process.poll() is None:
            status = t("validation_running")
        else:
            status = (
                f"Validation finished (exit code: {_validation_process.returncode})"
            )
    return status, get_validation_log()


def close_training_log_file():
    global _training_log_file
    if _training_log_file is not None:
        try:
            _training_log_file.close()
        except Exception:
            pass
        _training_log_file = None


def close_validation_log_file():
    global _validation_log_file
    if _validation_log_file is not None:
        try:
            _validation_log_file.close()
        except Exception:
            pass
        _validation_log_file = None


DATASET_GUIDE_EN = """
# Dataset Preparation Guide

## Dataset Types

### Type 1 - MUSDB Format (Recommended)
Each song in a separate folder with stem files:
```
dataset/
├── song1/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── song2/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
```
**Note:** All stems in the same folder must have the same duration (exact sample count).

### Type 2 - Stems in Separate Folders
```
dataset/
├── vocals/
│   ├── song1.wav
│   ├── song2.wav
├── drums/
│   ├── song1.wav
│   ├── song2.wav
├── bass/
│   ├── song1.wav
│   ├── song2.wav
└── other/
    ├── song1.wav
    ├── song2.wav
```
**Note:** Duration matching is NOT required for this type.

### Type 3 & 4 - Advanced Formats
Refer to the [MSST documentation](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md).

---

## Validation Dataset Requirements

Validation datasets **must** use Type 1 (MUSDB) format with an additional `mixture.wav` file:
```
valid_dataset/
├── song1/
│   ├── mixture.wav  # Required! Sum of all stems
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
```

**Recommendations:**
- Minimum 20 songs, ideally 50+
- Cover diverse languages, genres, and styles

---

## Training Recommendations

### Dataset Size
- Minimum: 100-200 songs
- Recommended: 500-1000+ songs
- Validation: 20-50 songs

### BS_Roformer Batch Size Guide

| chunk_size | dim | depth | A6000 48GB | 3090/4090 24GB | 16GB |
|------------|-----|-------|------------|----------------|------|
| 131584 | 128 | 6 | 10 | 5 | 3 |
| 131584 | 256 | 6 | 8 | 4 | 2 |
| 131584 | 384 | 6 | 7 | 3 | 2 |
| 131584 | 512 | 6 | 6 | 3 | 2 |
| 263168 | 128 | 6 | 4 | 2 | 1 |
| 263168 | 256 | 6 | 3 | 1 | 1 |
| 352800 | 128 | 6 | 2 | 1 | - |

### Training Tips
1. **Don't stop training midway!** The learning rate is not saved. If you must resume, manually set the LR in config.
2. Training duration: 50-300 epochs typically, or 50-70 hours for good results with 500+ songs.
3. If SDR is very low (negative or near zero) in first few epochs, consider stopping - your data or model choice may be wrong.
4. MSST saves only improved models, so watch for new checkpoints as progress indicator.

---

## Config File Key Parameters

```yaml
audio:
  chunk_size: 352800  # Audio chunk length in samples (larger = more VRAM, better quality)
  sample_rate: 44100
  num_channels: 2

model:
  dim: 128           # Network dimension
  depth: 4           # Network depth
  num_stems: 4       # Number of stems to separate

training:
  batch_size: 5
  instruments: ['drums', 'bass', 'other', 'vocals']
  lr: 1.0e-05        # Learning rate
  num_epochs: 1000
  num_steps: 1000    # Steps per epoch
  optimizer: prodigy # Options: adam, adamw, radam, rmsprop, prodigy, adamw8bit, sgd
  use_amp: true      # Mixed precision (recommended)
```

---

## Data Augmentation

Add to your config file to enable augmentations:

```yaml
augmentations:
  enable: true
  loudness: true
  loudness_min: 0.5
  loudness_max: 1.5
  mixup: true
  mixup_probs: [0.2, 0.02]
  
  all:
    channel_shuffle: 0.5
    random_inverse: 0.1
    random_polarity: 0.5
  
  vocals:
    pitch_shift: 0.1
    pitch_shift_min_semitones: -5
    pitch_shift_max_semitones: 5
```

Set `enable: false` or remove the section to disable all augmentations.
"""

DATASET_GUIDE_ZH = """
# 数据集制作指南

## 数据集类型

### 类型1 - MUSDB格式（推荐）
每首歌放在单独的文件夹中，包含各个音轨文件：
```
dataset/
├── song1/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── song2/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
```
**注意：** 同一文件夹中所有音轨的时长（精确到采样点）必须一致。

### 类型2 - 音轨分别存放
```
dataset/
├── vocals/
│   ├── song1.wav
│   ├── song2.wav
├── drums/
│   ├── song1.wav
│   ├── song2.wav
├── bass/
│   ├── song1.wav
│   ├── song2.wav
└── other/
    ├── song1.wav
    ├── song2.wav
```
**注意：** 此类型不需要时长一致。

### 类型3和4 - 高级格式
请参考 [MSST文档](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md)。

---

## 验证集要求

验证集**必须**使用类型1（MUSDB）格式，并额外包含 `mixture.wav` 文件：
```
valid_dataset/
├── song1/
│   ├── mixture.wav  # 必需！所有音轨的混合
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
```

**建议：**
- 最少20首歌，建议50首以上
- 尽量覆盖多种语言、曲风、类型

---

## 训练建议

### 数据集规模
- 最少：100-200首歌
- 推荐：500-1000首以上
- 验证集：20-50首歌

### BS_Roformer 批次大小参考

| chunk_size | dim | depth | A6000 48GB | 3090/4090 24GB | 16GB |
|------------|-----|-------|------------|----------------|------|
| 131584 | 128 | 6 | 10 | 5 | 3 |
| 131584 | 256 | 6 | 8 | 4 | 2 |
| 131584 | 384 | 6 | 7 | 3 | 2 |
| 131584 | 512 | 6 | 6 | 3 | 2 |
| 263168 | 128 | 6 | 4 | 2 | 1 |
| 263168 | 256 | 6 | 3 | 1 | 1 |
| 352800 | 128 | 6 | 2 | 1 | - |

### 训练技巧
1. **不要中途停止训练！** 学习率不会保存。如果必须继续训练，请手动在配置文件中设置lr值。
2. 训练时长：通常50-300个epoch，500首以上数据量约需50-70小时。
3. 如果前几个epoch的SDR值很低（负数或接近零），考虑停止训练——可能是数据或模型选择有问题。
4. MSST只保存更优的模型，观察是否有新检查点保存作为进度指标。

---

## 配置文件关键参数

```yaml
audio:
  chunk_size: 352800  # 音频切片长度（采样点），越大显存占用越多但质量越好
  sample_rate: 44100
  num_channels: 2

model:
  dim: 128           # 网络维度
  depth: 4           # 网络深度
  num_stems: 4       # 分离的音轨数量

training:
  batch_size: 5
  instruments: ['drums', 'bass', 'other', 'vocals']
  lr: 1.0e-05        # 学习率
  num_epochs: 1000
  num_steps: 1000    # 每个epoch的步数
  optimizer: prodigy # 可选：adam, adamw, radam, rmsprop, prodigy, adamw8bit, sgd
  use_amp: true      # 混合精度（推荐开启）
```

---

## 数据增强

在配置文件中添加以下内容启用数据增强：

```yaml
augmentations:
  enable: true
  loudness: true
  loudness_min: 0.5
  loudness_max: 1.5
  mixup: true
  mixup_probs: [0.2, 0.02]
  
  all:
    channel_shuffle: 0.5
    random_inverse: 0.1
    random_polarity: 0.5
  
  vocals:
    pitch_shift: 0.1
    pitch_shift_min_semitones: -5
    pitch_shift_max_semitones: 5
```

设置 `enable: false` 或删除该部分可禁用所有数据增强。
"""


def get_dataset_guide():
    from webui.i18n import get_lang

    if get_lang() == "zh":
        return DATASET_GUIDE_ZH
    return DATASET_GUIDE_EN


def create_training_tab():
    with gr.Tab(t("tab_training")) as tab:
        with gr.Tabs():
            with gr.TabItem(t("tab_train")):
                gr.Markdown(f"## {t('training_title')}")
                gr.Markdown(t("training_desc"))

                with gr.Row():
                    with gr.Column():
                        model_type = gr.Dropdown(
                            label=t("model_type"),
                            choices=MODEL_TYPES,
                            value="mel_band_roformer",
                        )

                        with gr.Row():
                            config_dropdown = gr.Dropdown(
                                label=t("config_file"),
                                choices=get_config_choices(),
                                allow_custom_value=True,
                            )
                            refresh_config_btn = gr.Button("🔄", scale=0, min_width=40)

                        with gr.Row():
                            checkpoint_dropdown = gr.Dropdown(
                                label=t("start_checkpoint"),
                                choices=get_checkpoint_choices(),
                                value="",
                                allow_custom_value=True,
                            )
                            refresh_ckpt_btn = gr.Button("🔄", scale=0, min_width=40)

                        data_path = gr.Textbox(
                            label=t("data_paths"),
                            placeholder="/path/to/data1;/path/to/data2",
                            lines=2,
                        )

                        valid_path = gr.Textbox(
                            label=t("valid_paths"),
                            placeholder="/path/to/valid1;/path/to/valid2",
                            lines=2,
                        )

                        results_path = gr.Textbox(
                            label=t("results_path"),
                            value="results",
                            placeholder="/path/to/results",
                        )

                        dataset_type = gr.Dropdown(
                            label=t("dataset_type"),
                            choices=[
                                ("1 - MUSDB", 1),
                                ("2 - Stems folders", 2),
                                ("3 - CSV", 3),
                                ("4 - Custom", 4),
                            ],
                            value=1,
                        )

                        with gr.Row():
                            num_workers = gr.Number(
                                label=t("num_workers"), value=4, precision=0
                            )
                            batch_size = gr.Number(
                                label=t("batch_size"), value=2, precision=0
                            )
                            seed = gr.Number(label=t("seed"), value=42, precision=0)

                        device_ids_list = get_device_ids()
                        device_ids_default = (
                            ",".join(str(d) for d in device_ids_list)
                            if device_ids_list
                            else "0"
                        )

                        device_ids = gr.Textbox(
                            label=t("device_ids"),
                            value=device_ids_default,
                            placeholder="0,1,2",
                        )

                        with gr.Row():
                            pin_memory = gr.Checkbox(label=t("pin_memory"), value=True)
                            pre_valid = gr.Checkbox(label=t("pre_valid"), value=False)

                        with gr.Row():
                            use_multistft_loss = gr.Checkbox(
                                label=t("use_multistft_loss"), value=False
                            )
                            use_mse_loss = gr.Checkbox(
                                label=t("use_mse_loss"), value=False
                            )
                            use_l1_loss = gr.Checkbox(
                                label=t("use_l1_loss"), value=False
                            )

                        metrics = gr.CheckboxGroup(
                            label=t("metrics"),
                            choices=TRAIN_METRICS_LIST,
                            value=["sdr"],
                        )

                        metric_for_scheduler = gr.Dropdown(
                            label=t("metric_for_scheduler"),
                            choices=[""] + TRAIN_METRICS_LIST,
                            value="sdr",
                        )

                        with gr.Row():
                            start_btn = gr.Button(
                                t("start_training"), variant="primary"
                            )
                            stop_btn = gr.Button(t("stop_training"), variant="stop")
                            refresh_log_btn = gr.Button(
                                "🔄 " + t("refresh_list"), variant="secondary"
                            )

                    with gr.Column():
                        training_status = gr.Textbox(
                            label=t("training_status"),
                            value=t("ready"),
                            interactive=False,
                        )

                        training_log = gr.Code(
                            label=t("training_log"),
                            language=None,
                            lines=25,
                            interactive=False,
                        )

                refresh_config_btn.click(
                    fn=refresh_configs, inputs=[], outputs=[config_dropdown]
                )
                refresh_ckpt_btn.click(
                    fn=refresh_checkpoints, inputs=[], outputs=[checkpoint_dropdown]
                )

                start_btn.click(
                    fn=start_training,
                    inputs=[
                        model_type,
                        config_dropdown,
                        checkpoint_dropdown,
                        data_path,
                        valid_path,
                        results_path,
                        dataset_type,
                        num_workers,
                        batch_size,
                        device_ids,
                        seed,
                        pin_memory,
                        use_multistft_loss,
                        use_mse_loss,
                        use_l1_loss,
                        pre_valid,
                        metrics,
                        metric_for_scheduler,
                    ],
                    outputs=[training_status, training_log],
                )

                stop_btn.click(
                    fn=stop_training, inputs=[], outputs=[training_status, training_log]
                )
                refresh_log_btn.click(
                    fn=refresh_training_log,
                    inputs=[],
                    outputs=[training_status, training_log],
                )

                training_timer = gr.Timer(2)
                training_timer.tick(
                    fn=refresh_training_log,
                    inputs=[],
                    outputs=[training_status, training_log],
                )

            with gr.TabItem(t("tab_valid")):
                gr.Markdown(f"## {t('valid_title')}")
                gr.Markdown(t("valid_desc"))

                with gr.Row():
                    with gr.Column():
                        valid_model_type = gr.Dropdown(
                            label=t("model_type"),
                            choices=MODEL_TYPES,
                            value="mel_band_roformer",
                        )

                        with gr.Row():
                            valid_config_dropdown = gr.Dropdown(
                                label=t("config_file"),
                                choices=get_config_choices(),
                                allow_custom_value=True,
                            )
                            valid_refresh_config_btn = gr.Button(
                                "🔄", scale=0, min_width=40
                            )

                        with gr.Row():
                            valid_checkpoint_dropdown = gr.Dropdown(
                                label=t("checkpoint_file"),
                                choices=get_checkpoint_choices()[1:],
                                allow_custom_value=True,
                            )
                            valid_refresh_ckpt_btn = gr.Button(
                                "🔄", scale=0, min_width=40
                            )

                        valid_valid_path = gr.Textbox(
                            label=t("valid_paths"),
                            placeholder="/path/to/valid1;/path/to/valid2",
                            lines=2,
                        )

                        valid_store_dir = gr.Textbox(
                            label=t("store_dir"),
                            value="valid_results",
                            placeholder="/path/to/output",
                        )

                        with gr.Row():
                            valid_num_workers = gr.Number(
                                label=t("num_workers"), value=4, precision=0
                            )
                            valid_extension = gr.Dropdown(
                                label=t("extension"),
                                choices=["wav", "flac"],
                                value="wav",
                            )

                        valid_device_ids = gr.Textbox(
                            label=t("device_ids"),
                            value=device_ids_default,
                            placeholder="0,1,2",
                        )

                        with gr.Row():
                            valid_pin_memory = gr.Checkbox(
                                label=t("pin_memory"), value=True
                            )
                            valid_use_tta = gr.Checkbox(label=t("use_tta"), value=False)

                        valid_metrics = gr.CheckboxGroup(
                            label=t("metrics"),
                            choices=VALID_METRICS_LIST,
                            value=["sdr"],
                        )

                        with gr.Row():
                            valid_start_btn = gr.Button(
                                t("start_validation"), variant="primary"
                            )
                            valid_stop_btn = gr.Button(
                                t("stop_validation"), variant="stop"
                            )
                            valid_refresh_log_btn = gr.Button(
                                "🔄 " + t("refresh_list"), variant="secondary"
                            )

                    with gr.Column():
                        validation_status = gr.Textbox(
                            label=t("validation_status"),
                            value=t("ready"),
                            interactive=False,
                        )

                        validation_log = gr.Code(
                            label=t("validation_log"),
                            language=None,
                            lines=25,
                            interactive=False,
                        )

                valid_refresh_config_btn.click(
                    fn=refresh_configs, inputs=[], outputs=[valid_config_dropdown]
                )
                valid_refresh_ckpt_btn.click(
                    fn=lambda: gr.update(choices=get_checkpoint_choices()[1:]),
                    inputs=[],
                    outputs=[valid_checkpoint_dropdown],
                )

                valid_start_btn.click(
                    fn=start_validation,
                    inputs=[
                        valid_model_type,
                        valid_config_dropdown,
                        valid_checkpoint_dropdown,
                        valid_valid_path,
                        valid_store_dir,
                        valid_device_ids,
                        valid_num_workers,
                        valid_pin_memory,
                        valid_extension,
                        valid_use_tta,
                        valid_metrics,
                    ],
                    outputs=[validation_status, validation_log],
                )

                valid_stop_btn.click(
                    fn=stop_validation,
                    inputs=[],
                    outputs=[validation_status, validation_log],
                )
                valid_refresh_log_btn.click(
                    fn=refresh_validation_log,
                    inputs=[],
                    outputs=[validation_status, validation_log],
                )

                validation_timer = gr.Timer(2)
                validation_timer.tick(
                    fn=refresh_validation_log,
                    inputs=[],
                    outputs=[validation_status, validation_log],
                )

            with gr.TabItem(t("tab_guide")):
                gr.Markdown(get_dataset_guide())

    return tab
