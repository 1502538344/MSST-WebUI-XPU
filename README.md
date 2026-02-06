
# MSST-WebUI-XPU

支持 **Intel XPU (Arc 显卡)** 的音乐源分离训练工具，带有 Gradio 网页界面。

本项目基于 [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 开发，增加了 Intel XPU 支持和现代化网页界面。

本项目使用了AI进行辅助操作

## 系统要求

- Windows 10/11 或 Linux
- Intel Arc 显卡（推荐 A770、A750 等，16GB+ 显存）
- Python 3.11
- Intel oneAPI Base Toolkit

## 安装步骤

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/1502538344/MSST-WebUI-XPU.git
cd MSST-WebUI-XPU
pip install -r requirements.txt
```
## 安装 ffmpeg

1. 下载 ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. 解压到项目根目录的 `ffmpeg/` 文件夹

### 2. 安装 Intel oneAPI 和 IPEX

参考 [Intel Extension for PyTorch 安装指南](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=windows&package=pip)

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.8.10+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## 使用方法

### 网页界面模式（推荐）

```bash
python webui.py
```

在浏览器中打开 http://localhost:7860

### 命令行训练

```bash
python train.py \
    --model_type bs_roformer \
    --config_path configs/config_bs_roformer.yaml \
    --data_path datasets/train \
    --valid_path datasets/valid \
    --results_path results/ \
    --device_ids 0
```

### 命令行推理

```bash
python inference.py \
    --model_type bs_roformer \
    --config_path configs/config_bs_roformer.yaml \
    --start_check_point checkpoints/model.ckpt \
    --input_folder input/ \
    --store_dir output/
```


## 数据集准备

### 训练数据集（类型1 - MUSDB 格式）

```
dataset/
├── song1/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── song2/
│   └── ...
```

**注意：** 同一文件夹中所有音轨的时长（精确到采样点）必须一致。

### 验证数据集

```
valid_dataset/
├── song1/
│   ├── mixture.wav  # 必需！所有音轨的混合
│   ├── vocals.wav
│   └── ...
```

## XPU 使用注意事项

- **Batch Size**: 16GB 显存（A770）建议使用 batch_size 2-3，chunk_size 131584
- **num_workers**: Windows 上设置为 0 以避免 DataLoader 问题
- **评估指标**: 所有指标在 CPU 上运行以保证 XPU 兼容性

## 验证指标说明

| 指标 | 说明 |
|------|------|
| sdr | 信号失真比 |
| si_sdr | 尺度不变 SDR |
| l1_freq | L1 频率损失 |
| neg_log_wmse | 负对数加权均方误差 |
| aura_stft | Aura STFT 损失 |
| aura_mrstft | Aura 多分辨率 STFT |
| bleedless | 无泄漏指标 |
| fullness | 完整性指标 |

## 训练建议

### 数据集规模
- 最少：100-200 首歌
- 推荐：500-1000 首以上
- 验证集：20-50 首歌

### BS_Roformer 批次大小参考

| chunk_size | dim | depth | A6000 48GB | 3090/4090 24GB | 16GB |
|------------|-----|-------|------------|----------------|------|
| 131584 | 128 | 6 | 10 | 5 | 3 |
| 131584 | 256 | 6 | 8 | 4 | 2 |
| 263168 | 128 | 6 | 4 | 2 | 1 |

### 训练技巧

1. **不要中途停止训练！** 学习率不会保存。如果必须继续训练，请手动在配置文件中设置 lr 值。
2. 训练时长：通常 50-300 个 epoch，500 首以上数据量约需 50-70 小时。
3. 如果前几个 epoch 的 SDR 值很低（负数或接近零），考虑停止训练——可能是数据或模型选择有问题。
4. MSST 只保存更优的模型，观察是否有新检查点保存作为进度指标。

## 致谢

- 原始 MSST 代码来自 [ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [MVSep.com](https://mvsep.com) 提供原始项目
- Intel 提供 IPEX 和 XPU 支持
- 参考了 [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI)

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 引用

```bibtex
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

