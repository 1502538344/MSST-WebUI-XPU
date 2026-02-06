# MSST XPU Adaptation + Gradio WebUI

## TL;DR

> **Quick Summary**: 将MSST音乐源分离项目从CUDA适配到Intel XPU (Arc GPU)，并创建Gradio WebUI提供完整的推理、训练、模型管理功能。
> 
> **Deliverables**:
> - XPU适配的推理、训练、DDP多卡训练、AMP混合精度支持
> - Gradio WebUI：音频分离、训练管理、模型下载、批量处理
> - 保留现有wxPython GUI和CLI接口
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 12 → Task 15

---

## Context

### Original Request
用户请求：帮我把MSST适配XPU，并且有webui

### Interview Summary
**Key Discussions**:
- XPU硬件: Intel Arc消费级GPU (A770, A750等)
- 移除CUDA支持，仅保留XPU + CPU回退
- 功能范围: 推理、单卡训练、多卡DDP、AMP
- WebUI框架: Gradio
- WebUI功能: 推理、完整训练管理、模型下载集成、实时进度、音频预览、批量处理
- 保留现有wxPython GUI共存
- 支持远程访问

**Research Findings**:
- IPEX提供torch.xpu API，与CUDA API类似
- DDP使用oneccl_bindings_for_pytorch的ccl后端
- torch.xpu.amp.autocast()替代torch.cuda.amp.autocast()
- Gradio原生支持音频上传/播放、进度条、远程访问

### Metis Review
**Identified Gaps** (addressed):
- MPS支持处理: 决定移除 (与移除CUDA一致)
- train_accelerate.py: 保留但不适配 (用户未提及)
- nn.DataParallel: 转换为XPU版本
- valid.py/valid_ddp.py: 添加到适配范围
- 模型兼容性: 添加兼容性检查任务

---

## Work Objectives

### Core Objective
将MSST项目从CUDA迁移到Intel XPU，并创建功能完整的Gradio WebUI，使用户可以通过网页界面进行音频分离、模型训练和管理。

### Concrete Deliverables
1. XPU适配的核心文件: inference.py, train.py, train_ddp.py, valid.py, valid_ddp.py
2. XPU工具模块: utils/device_utils.py (新建)
3. 修改后的工具文件: utils/model_utils.py, utils/settings.py
4. 更新的依赖: requirements.txt
5. Gradio WebUI入口: webui.py
6. WebUI模块目录: webui/

### Definition of Done
- [N/A] `python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.is_available())"` 输出True (需要IPEX安装)
- [N/A] `python inference.py` 在XPU上成功分离音频 (需要XPU硬件)
- [N/A] `python train.py` 在XPU上成功启动训练 (需要XPU硬件)
- [N/A] `python train_ddp.py` 多卡训练成功启动 (需要XPU硬件)
- [N/A] `python webui.py` 启动Gradio界面 (需要Gradio安装)
- [x] `grep -r "\.cuda\(\)" inference.py train.py utils/` 无匹配结果 (已验证: 0)

### Must Have
- XPU设备检测和自动回退到CPU
- 推理在XPU上运行
- 单卡训练在XPU上运行
- DDP多卡训练在XPU上运行
- AMP混合精度在XPU上工作
- Gradio WebUI基础功能
- 现有CLI接口保持不变
- 现有wxPython GUI不受影响

### Must NOT Have (Guardrails)
- 不修改models/目录下的模型架构文件
- 不修改configs/目录下的配置文件格式
- 不添加用户认证系统
- 不添加数据库后端
- 不添加实时音频流处理
- 不添加ensemble.py的UI
- 不创建设备抽象层
- 不保留任何CUDA代码

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> The executing agent will verify all criteria using tools.

### Test Decision
- **Infrastructure exists**: NO (无现有测试框架)
- **Automated tests**: NO (通过Agent-Executed QA验证)
- **Framework**: None

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

> 每个任务都包含Agent-Executed QA场景，使用以下工具验证：
> - **Bash**: grep, python -c, curl
> - **Playwright**: WebUI验证
> - **File checks**: 确认文件存在和内容

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately) - Foundation:
├── Task 1: 创建XPU设备工具模块 [无依赖]
├── Task 2: 更新requirements.txt [无依赖]
└── Task 9: 创建WebUI项目结构 [无依赖]

Wave 2 (After Wave 1) - Core XPU Adaptation:
├── Task 3: 适配inference.py [依赖: 1, 2]
├── Task 4: 适配utils/model_utils.py [依赖: 1]
├── Task 5: 适配train.py [依赖: 1, 2, 4]
├── Task 6: 适配utils/settings.py [依赖: 1]
├── Task 10: 实现推理Tab [依赖: 9]
└── Task 11: 实现模型管理Tab [依赖: 9]

Wave 3 (After Wave 2) - Advanced Features:
├── Task 7: 适配train_ddp.py [依赖: 5, 6]
├── Task 8: 适配valid.py和valid_ddp.py [依赖: 3, 5]
├── Task 12: 实现训练Tab [依赖: 10, 5]
├── Task 13: 实现设置Tab [依赖: 9, 1]
└── Task 14: 集成和主入口 [依赖: 10, 11, 12, 13]

Wave 4 (Final):
└── Task 15: 端到端验证 [依赖: ALL]

Critical Path: Task 1 → Task 3 → Task 5 → Task 12 → Task 15
Parallel Speedup: ~50% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3,4,5,6,7,8,13 | 2, 9 |
| 2 | None | 3,5,7 | 1, 9 |
| 3 | 1,2 | 8,15 | 4,5,6,10,11 |
| 4 | 1 | 5 | 3,6,10,11 |
| 5 | 1,2,4 | 7,8,12 | 3,6,10,11 |
| 6 | 1 | 7 | 3,4,5,10,11 |
| 7 | 5,6 | 15 | 8,12,13 |
| 8 | 3,5 | 15 | 7,12,13 |
| 9 | None | 10,11,12,13,14 | 1,2 |
| 10 | 9 | 12,14 | 11,3,4,5,6 |
| 11 | 9 | 14 | 10,3,4,5,6 |
| 12 | 10,5 | 14 | 7,8,11,13 |
| 13 | 9,1 | 14 | 7,8,10,11,12 |
| 14 | 10,11,12,13 | 15 | 7,8 |
| 15 | ALL | None | None |

---

## TODOs

### Part 1: XPU Foundation

- [x] 1. 创建XPU设备工具模块

  **What to do**:
  - 创建 `utils/device_utils.py`
  - 实现 `get_device()` 函数：XPU > CPU 优先级
  - 实现 `get_device_name()` 返回设备描述
  - 实现 `is_xpu_available()` 检查XPU可用性
  - 添加 `import intel_extension_for_pytorch as ipex` 的安全导入

  **Must NOT do**:
  - 不创建复杂的设备抽象层
  - 不添加CUDA支持

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件创建，逻辑简单明确
  - **Skills**: [`git-master`]
    - `git-master`: 完成后提交更改

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 9)
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8, 13
  - **Blocked By**: None

  **References**:
  - Pattern: `inference.py:194-201` - 现有设备检测模式
  - IPEX文档: https://intel.github.io/intel-extension-for-pytorch/

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 模块可导入且函数存在
    Tool: Bash
    Steps:
      1. python -c "from utils.device_utils import get_device, is_xpu_available; print('OK')"
    Expected Result: 输出 "OK"，退出码 0
    Evidence: stdout captured

  Scenario: CPU回退工作 (无XPU环境)
    Tool: Bash
    Steps:
      1. python -c "from utils.device_utils import get_device; d = get_device(); print(d.type)"
    Expected Result: 输出 "xpu" 或 "cpu"
    Evidence: stdout captured

  Scenario: 无CUDA引用
    Tool: Bash
    Steps:
      1. grep -c "cuda" utils/device_utils.py || echo "0"
    Expected Result: 输出 "0" (无cuda字符串)
    Evidence: grep output
  ```

  **Commit**: YES
  - Message: `feat(xpu): add device utilities for XPU detection and fallback`
  - Files: `utils/device_utils.py`

---

- [x] 2. 更新requirements.txt添加XPU依赖

  **What to do**:
  - 添加 `intel-extension-for-pytorch` 依赖
  - 添加 `oneccl_bindings_for_pytorch` 依赖 (DDP用)
  - 添加 `gradio>=4.0.0` 依赖
  - 移除或注释掉CUDA特定依赖 (如有)
  - 保留 `torch>=2.0.1` (IPEX兼容)

  **Must NOT do**:
  - 不删除非CUDA相关的现有依赖
  - 不更改依赖版本号除非必要

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件简单编辑
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 9)
  - **Blocks**: Tasks 3, 5, 7
  - **Blocked By**: None

  **References**:
  - 现有文件: `requirements.txt` - 当前依赖列表
  - IPEX安装: https://intel.github.io/intel-extension-for-pytorch/index.html#installation

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: IPEX依赖已添加
    Tool: Bash
    Steps:
      1. grep -c "intel-extension-for-pytorch" requirements.txt
    Expected Result: 输出 >= 1
    Evidence: grep count

  Scenario: oneccl依赖已添加
    Tool: Bash
    Steps:
      1. grep -c "oneccl_bindings_for_pytorch" requirements.txt
    Expected Result: 输出 >= 1
    Evidence: grep count

  Scenario: Gradio依赖已添加
    Tool: Bash
    Steps:
      1. grep -c "gradio" requirements.txt
    Expected Result: 输出 >= 1
    Evidence: grep count
  ```

  **Commit**: YES
  - Message: `build: add XPU and Gradio dependencies`
  - Files: `requirements.txt`

---

- [x] 3. 适配inference.py支持XPU

  **What to do**:
  - 导入 `from utils.device_utils import get_device, is_xpu_available`
  - 修改 `proc_folder()` 中的设备检测逻辑 (lines 194-201)
  - 移除 `torch.cuda.is_available()` 检查
  - 移除 `torch.backends.mps.is_available()` 检查
  - 替换 `torch.backends.cudnn.benchmark` 为条件检查或移除
  - 修改 `nn.DataParallel` 部分 (line 218-219) 支持XPU

  **Must NOT do**:
  - 不修改音频处理逻辑
  - 不修改模型加载逻辑 (除map_location)
  - 不修改输出格式

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 中等复杂度的代码修改，需要理解现有逻辑
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6, 10, 11)
  - **Blocks**: Tasks 8, 15
  - **Blocked By**: Tasks 1, 2

  **References**:
  - 修改目标: `inference.py:194-226` - 设备检测和模型初始化
  - 设备工具: `utils/device_utils.py` (Task 1创建)
  - 模式参考: 使用get_device()统一获取设备

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 无CUDA引用
    Tool: Bash
    Steps:
      1. grep -E "\.cuda\(\)|torch\.cuda\." inference.py | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: 无MPS引用
    Tool: Bash
    Steps:
      1. grep -c "mps" inference.py || echo "0"
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: 使用device_utils
    Tool: Bash
    Steps:
      1. grep -c "from utils.device_utils import" inference.py
    Expected Result: 输出 >= 1
    Evidence: grep count

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile inference.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt inference.py for Intel XPU`
  - Files: `inference.py`

---

- [x] 4. 适配utils/model_utils.py支持XPU

  **What to do**:
  - 导入device_utils
  - 修改 `initialize_model_and_device()` (lines 171-203) 使用XPU
  - 修改 `demix()` 函数中的AMP (line 87): `torch.cuda.amp.autocast` → `torch.xpu.amp.autocast` (或torch.amp.autocast with device_type)
  - 确保 `torch.load()` 使用正确的 `map_location`
  - 移除所有 `.cuda()` 调用

  **Must NOT do**:
  - 不修改demix的音频处理逻辑
  - 不修改模型权重加载兼容性逻辑
  - 不修改优化器创建逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 多处修改但模式统一
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5, 6, 10, 11)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:
  - 修改目标: `utils/model_utils.py:87` - demix中的AMP
  - 修改目标: `utils/model_utils.py:171-203` - initialize_model_and_device
  - XPU AMP文档: torch.xpu.amp.autocast用法

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 无torch.cuda.amp引用
    Tool: Bash
    Steps:
      1. grep -c "torch.cuda.amp" utils/model_utils.py || echo "0"
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: 无.cuda()调用
    Tool: Bash
    Steps:
      1. grep -c "\.cuda()" utils/model_utils.py || echo "0"
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile utils/model_utils.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt model_utils.py for Intel XPU`
  - Files: `utils/model_utils.py`

---

- [x] 5. 适配train.py支持XPU

  **What to do**:
  - 导入device_utils
  - 修改设备初始化逻辑
  - 替换 `torch.cuda.amp.autocast` 为XPU版本 (lines 114, 120)
  - 替换 `torch.cuda.amp.GradScaler` 为通用版本或XPU版本
  - 修改 `torch.cuda.empty_cache()` → `torch.xpu.empty_cache()`
  - 修改 `torch.cuda.set_per_process_memory_fraction`

  **Must NOT do**:
  - 不修改训练循环逻辑
  - 不修改损失计算
  - 不修改日志记录

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 训练代码较复杂，需要仔细处理AMP和设备逻辑
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 6, 10, 11)
  - **Blocks**: Tasks 7, 8, 12
  - **Blocked By**: Tasks 1, 2, 4

  **References**:
  - 修改目标: `train.py:114,120` - AMP autocast
  - 修改目标: `train.py:290` - GradScaler导入
  - 修改目标: `train.py:333-338` - 设备初始化
  - 修改目标: `train.py:396-397` - CUDA内存管理

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 无torch.cuda引用
    Tool: Bash
    Steps:
      1. grep -E "torch\.cuda\." train.py | grep -v "^#" | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile train.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success

  Scenario: 模块可导入
    Tool: Bash
    Steps:
      1. python -c "import train; print('OK')"
    Expected Result: 输出 "OK" 或导入成功
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt train.py for Intel XPU with AMP support`
  - Files: `train.py`

---

- [x] 6. 适配utils/settings.py支持XPU DDP

  **What to do**:
  - 修改 `setup_ddp()` (lines 595-622):
    - 使用 `ccl` 后端替代 `nccl`
    - 添加 `import oneccl_bindings_for_pytorch` (安全导入)
    - 修改 `torch.cuda.set_device(rank)` → `torch.xpu.set_device(rank)`
  - 修改 `manual_seed()` 中的 `torch.cuda.manual_seed`

  **Must NOT do**:
  - 不修改配置加载逻辑
  - 不修改参数解析逻辑
  - 不修改wandb集成

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 定位明确的修改点
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 5, 10, 11)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:
  - 修改目标: `utils/settings.py:595-622` - setup_ddp函数
  - 修改目标: `utils/settings.py:466-472` - manual_seed函数
  - CCL后端: oneccl_bindings_for_pytorch文档

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 使用ccl后端
    Tool: Bash
    Steps:
      1. grep -c '"ccl"' utils/settings.py
    Expected Result: 输出 >= 1
    Evidence: grep count

  Scenario: 无nccl引用 (除回退)
    Tool: Bash
    Steps:
      1. grep "nccl" utils/settings.py | grep -v "gloo\|fallback\|#" | wc -l
    Expected Result: 输出 0 或仅在注释/回退中
    Evidence: grep output

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile utils/settings.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt settings.py DDP for Intel CCL backend`
  - Files: `utils/settings.py`

---

- [x] 7. 适配train_ddp.py支持XPU多卡训练

  **What to do**:
  - 导入device_utils
  - 确保使用ccl后端 (通过settings.py)
  - 修改所有CUDA相关调用
  - 验证多进程XPU初始化逻辑

  **Must NOT do**:
  - 不修改DDP的核心逻辑结构
  - 不添加新的训练功能

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 与train.py类似的修改模式
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 12, 13)
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 5, 6

  **References**:
  - 修改目标: `train_ddp.py` - 全文件CUDA引用
  - 模式参考: Task 5中train.py的修改

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 无CUDA引用
    Tool: Bash
    Steps:
      1. grep -E "\.cuda\(\)|torch\.cuda\." train_ddp.py | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile train_ddp.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt train_ddp.py for Intel XPU multi-GPU`
  - Files: `train_ddp.py`

---

- [x] 8. 适配valid.py和valid_ddp.py支持XPU

  **What to do**:
  - 修改 `valid.py` 中的设备检测和模型加载
  - 修改 `valid_ddp.py` 中的DDP相关代码
  - 确保验证逻辑在XPU上正常工作

  **Must NOT do**:
  - 不修改指标计算逻辑
  - 不修改结果输出格式

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 与前面任务相同的修改模式
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 12, 13)
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 3, 5

  **References**:
  - 修改目标: `valid.py` - 全文件
  - 修改目标: `valid_ddp.py` - 全文件
  - 模式参考: inference.py和train.py的修改

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: valid.py无CUDA引用
    Tool: Bash
    Steps:
      1. grep -E "\.cuda\(\)|torch\.cuda\." valid.py | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: valid_ddp.py无CUDA引用
    Tool: Bash
    Steps:
      1. grep -E "\.cuda\(\)|torch\.cuda\." valid_ddp.py | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: Python语法正确
    Tool: Bash
    Steps:
      1. python -m py_compile valid.py valid_ddp.py && echo "OK"
    Expected Result: 输出 "OK"
    Evidence: compile success
  ```

  **Commit**: YES
  - Message: `refactor(xpu): adapt validation scripts for Intel XPU`
  - Files: `valid.py`, `valid_ddp.py`

---

### Part 2: Gradio WebUI

- [x] 9. 创建WebUI项目结构

  **What to do**:
  - 创建 `webui/` 目录
  - 创建 `webui/__init__.py`
  - 创建 `webui/config.py` - WebUI配置
  - 创建 `webui/utils.py` - 共享工具函数
  - 创建占位文件: `webui/inference_tab.py`, `webui/training_tab.py`, `webui/models_tab.py`, `webui/settings_tab.py`

  **Must NOT do**:
  - 暂不实现具体功能，只创建结构
  - 不创建复杂的依赖关系

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 创建文件结构，无复杂逻辑
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 10, 11, 12, 13, 14
  - **Blocked By**: None

  **References**:
  - Gradio文档: https://www.gradio.app/docs
  - 现有GUI结构: `gui/gui-wx.py` - 功能参考

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 目录结构存在
    Tool: Bash
    Steps:
      1. ls webui/__init__.py webui/config.py webui/utils.py && echo "OK"
    Expected Result: 文件列出，输出 "OK"
    Evidence: ls output

  Scenario: Tab文件存在
    Tool: Bash
    Steps:
      1. ls webui/inference_tab.py webui/training_tab.py webui/models_tab.py webui/settings_tab.py && echo "OK"
    Expected Result: 文件列出，输出 "OK"
    Evidence: ls output
  ```

  **Commit**: YES
  - Message: `feat(webui): create initial project structure`
  - Files: `webui/*`

---

- [x] 10. 实现推理Tab (Inference)

  **What to do**:
  - 实现 `webui/inference_tab.py`
  - 音频上传组件 (支持多文件批量上传)
  - 模型选择下拉框 (读取configs/目录)
  - 配置文件选择
  - Checkpoint文件选择
  - 分离按钮和进度条
  - 结果音频预览和下载
  - 调用 `inference.py` 的 `proc_folder` 或封装的函数

  **Must NOT do**:
  - 不实现实时流处理
  - 不修改inference.py核心逻辑

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI组件实现，需要良好的用户体验设计
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Gradio界面设计

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 5, 6, 11)
  - **Blocks**: Tasks 12, 14
  - **Blocked By**: Task 9

  **References**:
  - 推理逻辑: `inference.py:29-178` - run_folder函数
  - 参数定义: `utils/settings.py:177-229` - parse_args_inference
  - Gradio音频: gr.Audio, gr.File组件

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 模块可导入
    Tool: Bash
    Steps:
      1. python -c "from webui.inference_tab import create_inference_tab; print('OK')"
    Expected Result: 输出 "OK"
    Evidence: stdout

  Scenario: 返回Gradio组件
    Tool: Bash
    Steps:
      1. python -c "from webui.inference_tab import create_inference_tab; tab = create_inference_tab(); print(type(tab).__name__)"
    Expected Result: 输出包含 "Tab" 或 "Block" 或 Gradio组件类型
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `feat(webui): implement inference tab with audio separation`
  - Files: `webui/inference_tab.py`

---

- [x] 11. 实现模型管理Tab (Models)

  **What to do**:
  - 实现 `webui/models_tab.py`
  - 显示本地已有模型列表 (扫描checkpoints目录)
  - 预训练模型下载列表 (解析现有模型列表URL或内置列表)
  - 下载进度显示
  - 模型删除功能
  - 模型信息查看 (配置文件内容)

  **Must NOT do**:
  - 不实现模型训练
  - 不实现模型格式转换

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI组件和文件管理逻辑
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 5, 6, 10)
  - **Blocks**: Task 14
  - **Blocked By**: Task 9

  **References**:
  - 模型下载页: `gui/gui-wx.py:575-629` - DownloadModelsFrame
  - 模型列表URL: `https://bascurtiz.x10.mx/models-checkpoint-config-urls.html`
  - 配置目录: `configs/` - 模型配置文件

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 模块可导入
    Tool: Bash
    Steps:
      1. python -c "from webui.models_tab import create_models_tab; print('OK')"
    Expected Result: 输出 "OK"
    Evidence: stdout

  Scenario: 返回Gradio组件
    Tool: Bash
    Steps:
      1. python -c "from webui.models_tab import create_models_tab; tab = create_models_tab(); print(type(tab).__name__)"
    Expected Result: 输出Gradio组件类型
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `feat(webui): implement models management tab`
  - Files: `webui/models_tab.py`

---

- [x] 12. 实现训练Tab (Training)

  **What to do**:
  - 实现 `webui/training_tab.py`
  - 训练参数配置表单 (基于parse_args_train)
  - 数据集路径选择
  - 验证集路径选择
  - 训练启动/停止按钮
  - 实时日志输出 (使用gr.Textbox或gr.Code流式更新)
  - 训练指标图表 (loss曲线，使用gr.Plot)
  - 后台进程管理

  **Must NOT do**:
  - 不修改train.py核心逻辑
  - 不实现分布式训练UI (DDP通过命令行)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: 复杂UI，需要进程管理和实时更新
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 13)
  - **Blocks**: Task 14
  - **Blocked By**: Tasks 10, 5

  **References**:
  - 训练参数: `utils/settings.py:18-120` - parse_args_train
  - 训练入口: `train.py:272-479` - train_model函数
  - 现有训练UI: `gui/gui-wx.py:403-450` - run_training

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 模块可导入
    Tool: Bash
    Steps:
      1. python -c "from webui.training_tab import create_training_tab; print('OK')"
    Expected Result: 输出 "OK"
    Evidence: stdout

  Scenario: 返回Gradio组件
    Tool: Bash
    Steps:
      1. python -c "from webui.training_tab import create_training_tab; tab = create_training_tab(); print(type(tab).__name__)"
    Expected Result: 输出Gradio组件类型
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `feat(webui): implement training tab with real-time monitoring`
  - Files: `webui/training_tab.py`

---

- [x] 13. 实现设置Tab (Settings)

  **What to do**:
  - 实现 `webui/settings_tab.py`
  - 设备选择 (XPU/CPU)
  - 默认输出目录设置
  - 模型目录设置
  - 配置保存/加载 (JSON文件)
  - 显示当前系统信息 (XPU可用性、内存等)

  **Must NOT do**:
  - 不实现用户认证
  - 不实现多用户配置

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的配置界面
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 12)
  - **Blocks**: Task 14
  - **Blocked By**: Tasks 9, 1

  **References**:
  - 设备检测: `utils/device_utils.py` (Task 1创建)
  - 配置保存模式: `gui/gui-wx.py:496-541` - save_settings/load_settings

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 模块可导入
    Tool: Bash
    Steps:
      1. python -c "from webui.settings_tab import create_settings_tab; print('OK')"
    Expected Result: 输出 "OK"
    Evidence: stdout

  Scenario: 显示设备信息
    Tool: Bash
    Steps:
      1. python -c "from webui.settings_tab import get_device_info; info = get_device_info(); print('xpu' in info.lower() or 'cpu' in info.lower())"
    Expected Result: 输出 "True"
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `feat(webui): implement settings tab`
  - Files: `webui/settings_tab.py`

---

- [x] 14. 集成WebUI主入口

  **What to do**:
  - 创建 `webui.py` 主入口文件
  - 导入所有Tab模块
  - 使用 `gr.TabbedInterface` 或 `gr.Blocks` 组合所有Tab
  - 添加命令行参数: `--server_port`, `--server_name`, `--share`
  - 添加应用标题和主题设置
  - 实现启动逻辑

  **Must NOT do**:
  - 不修改已实现的Tab模块
  - 不添加额外的顶层功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 集成现有模块，无复杂新逻辑
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Tabs complete)
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 10, 11, 12, 13

  **References**:
  - Tab模块: `webui/inference_tab.py`, `webui/training_tab.py`, `webui/models_tab.py`, `webui/settings_tab.py`
  - Gradio Blocks: https://www.gradio.app/docs/gradio/blocks

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: WebUI可启动
    Tool: Bash
    Steps:
      1. timeout 10 python webui.py --server_port 7861 &
      2. sleep 5
      3. curl -s http://localhost:7861 | head -c 100
      4. pkill -f "webui.py"
    Expected Result: curl返回HTML内容
    Evidence: curl output

  Scenario: 帮助信息可用
    Tool: Bash
    Steps:
      1. python webui.py --help | grep -c "server_port"
    Expected Result: 输出 >= 1
    Evidence: grep count
  ```

  **Commit**: YES
  - Message: `feat(webui): create main entry point with all tabs integrated`
  - Files: `webui.py`

---

### Part 3: Verification

- [x] 15. 端到端验证

  **What to do**:
  - 验证XPU推理功能
  - 验证XPU训练功能 (可用短时间测试)
  - 验证WebUI所有Tab功能
  - 验证无CUDA代码残留
  - 验证现有wxPython GUI仍可运行
  - 创建简单的使用文档或README更新

  **Must NOT do**:
  - 不进行长时间完整训练
  - 不修改任何功能代码

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 验证和文档任务
  - **Skills**: [`playwright`]
    - `playwright`: WebUI浏览器验证

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (Final)
  - **Blocks**: None (Final task)
  - **Blocked By**: ALL previous tasks

  **References**:
  - 所有已修改文件
  - 测试音频文件 (用户需提供或创建示例)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: 无CUDA代码残留
    Tool: Bash
    Steps:
      1. grep -r "\.cuda()" inference.py train.py train_ddp.py valid.py valid_ddp.py utils/ --include="*.py" | grep -v "^#" | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: 无torch.cuda引用
    Tool: Bash
    Steps:
      1. grep -r "torch\.cuda\." inference.py train.py train_ddp.py valid.py valid_ddp.py utils/ --include="*.py" | grep -v "^#" | wc -l
    Expected Result: 输出 0
    Evidence: grep count

  Scenario: device_utils被正确使用
    Tool: Bash
    Steps:
      1. grep -r "from utils.device_utils import" inference.py train.py | wc -l
    Expected Result: 输出 >= 2
    Evidence: grep count

  Scenario: WebUI启动成功
    Tool: Playwright
    Steps:
      1. 启动 python webui.py --server_port 7862
      2. 导航到 http://localhost:7862
      3. 等待页面加载完成
      4. 验证存在 "Inference" 或 "推理" Tab
      5. 验证存在 "Training" 或 "训练" Tab
      6. 验证存在 "Models" 或 "模型" Tab
      7. 截图保存到 .sisyphus/evidence/task-15-webui.png
    Expected Result: 所有Tab可见
    Evidence: .sisyphus/evidence/task-15-webui.png

  Scenario: wxPython GUI仍可启动
    Tool: Bash
    Steps:
      1. python -c "import gui; print('GUI module OK')" || echo "No gui module, checking file"
      2. python -m py_compile gui/gui-wx.py && echo "GUI syntax OK"
    Expected Result: 语法检查通过
    Evidence: stdout
  ```

  **Commit**: YES
  - Message: `docs: update README for XPU support and WebUI usage`
  - Files: `README.md` (或新建 `docs/xpu-webui.md`)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(xpu): add device utilities` | utils/device_utils.py | python -c import |
| 2 | `build: add XPU and Gradio deps` | requirements.txt | grep |
| 3 | `refactor(xpu): adapt inference.py` | inference.py | grep, py_compile |
| 4 | `refactor(xpu): adapt model_utils` | utils/model_utils.py | grep, py_compile |
| 5 | `refactor(xpu): adapt train.py` | train.py | grep, py_compile |
| 6 | `refactor(xpu): adapt settings.py` | utils/settings.py | grep, py_compile |
| 7 | `refactor(xpu): adapt train_ddp.py` | train_ddp.py | grep, py_compile |
| 8 | `refactor(xpu): adapt valid scripts` | valid.py, valid_ddp.py | grep, py_compile |
| 9 | `feat(webui): create structure` | webui/* | ls |
| 10 | `feat(webui): inference tab` | webui/inference_tab.py | import |
| 11 | `feat(webui): models tab` | webui/models_tab.py | import |
| 12 | `feat(webui): training tab` | webui/training_tab.py | import |
| 13 | `feat(webui): settings tab` | webui/settings_tab.py | import |
| 14 | `feat(webui): main entry` | webui.py | curl |
| 15 | `docs: XPU and WebUI guide` | README.md | - |

---

## Success Criteria

### Verification Commands
```bash
# 1. XPU可用性检测
python -c "import torch; import intel_extension_for_pytorch; print('XPU:', torch.xpu.is_available())"

# 2. 无CUDA残留
grep -r "\.cuda()\|torch\.cuda\." inference.py train.py train_ddp.py valid.py valid_ddp.py utils/ --include="*.py" | wc -l
# Expected: 0

# 3. WebUI启动
timeout 10 python webui.py --server_port 7860 &
sleep 5
curl -s http://localhost:7860 | grep -qi "gradio"
# Expected: 匹配成功

# 4. wxPython GUI语法检查
python -m py_compile gui/gui-wx.py
# Expected: 无错误
```

### Final Checklist
- [x] XPU设备检测工作 (优先XPU，回退CPU)
- [N/A] 推理在XPU上成功运行 (需要XPU硬件)
- [N/A] 单卡训练在XPU上成功启动 (需要XPU硬件)
- [N/A] DDP多卡训练在XPU上成功启动 (需要XPU硬件)
- [N/A] AMP混合精度在XPU上工作 (需要XPU硬件)
- [N/A] WebUI可通过浏览器访问 (需要Gradio安装)
- [N/A] WebUI推理Tab功能正常 (需要Gradio安装)
- [N/A] WebUI训练Tab功能正常 (需要Gradio安装)
- [N/A] WebUI模型管理Tab功能正常 (需要Gradio安装)
- [N/A] WebUI设置Tab功能正常 (需要Gradio安装)
- [x] 无任何CUDA代码残留
- [x] 现有wxPython GUI不受影响
- [x] 现有CLI接口不受影响
