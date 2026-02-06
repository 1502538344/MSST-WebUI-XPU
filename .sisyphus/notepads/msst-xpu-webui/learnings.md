# MSST XPU WebUI - Learnings

## 2026-02-03: Initial Implementation

### XPU Adaptation Patterns

1. **Device Detection Priority**: XPU > CPU
   - Use `torch.xpu.is_available()` after importing IPEX
   - Safe import pattern for IPEX:
   ```python
   try:
       import intel_extension_for_pytorch as ipex
       IPEX_AVAILABLE = True
   except ImportError:
       IPEX_AVAILABLE = False
   ```

2. **AMP (Automatic Mixed Precision)**:
   - Replace `torch.cuda.amp.autocast(enabled=use_amp)` with `torch.amp.autocast(device_type='xpu', enabled=use_amp)`
   - GradScaler: Use `torch.amp.GradScaler()` instead of `torch.cuda.amp.GradScaler()`

3. **DDP (Distributed Data Parallel)**:
   - Replace `nccl` backend with `ccl` (Intel oneCCL)
   - Import: `import oneccl_bindings_for_pytorch`
   - Device count: `torch.xpu.device_count()` instead of `torch.cuda.device_count()`
   - Set device: `torch.xpu.set_device(rank)` instead of `torch.cuda.set_device(rank)`

4. **Memory Management**:
   - `torch.xpu.empty_cache()` instead of `torch.cuda.empty_cache()`
   - XPU doesn't have `set_per_process_memory_fraction`, skip this call

5. **Seed Setting**:
   - `torch.xpu.manual_seed(seed)` and `torch.xpu.manual_seed_all(seed)`

### WebUI Implementation

1. **Gradio Blocks Structure**:
   - Use `gr.Blocks()` as main container
   - Use `gr.Tabs()` with `gr.Tab()` for multi-tab interface
   - Each tab in separate module for maintainability

2. **Command Line Arguments**:
   - `--server_port` for port configuration
   - `--server_name` for hostname (0.0.0.0 for remote access)
   - `--share` for public Gradio link

### Files Modified

- `utils/device_utils.py` (NEW) - XPU device utilities
- `inference.py` - Device detection, DataParallel
- `train.py` - AMP, GradScaler, DDP device, memory management
- `train_ddp.py` - Device count
- `valid.py` - Device detection, multi-GPU validation
- `valid_ddp.py` - DDP device handling
- `utils/model_utils.py` - AMP autocast, initialize_model_and_device
- `utils/settings.py` - DDP setup with CCL, seed setting
- `requirements.txt` - Added IPEX, oneccl, gradio
- `webui.py` (NEW) - Main WebUI entry
- `webui/` (NEW) - WebUI modules

### Verification Commands

```bash
# Check no CUDA references
grep -r "\.cuda()" inference.py train.py utils/ --include="*.py" | wc -l  # Should be 0

# Check XPU availability
python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.is_available())"

# Start WebUI
python webui.py --server_port 7860
```
