<img width="475" height="467" alt="Alexandria Logo" src="https://github.com/user-attachments/assets/fa2c36d3-a5f3-49ab-9dfe-30933359dfbd" />

# Alexandria 有声书生成器

[English](README.md) | 中文

> **致新用户：** 感谢大家的关注！Alexandria 近期突然获得了大量关注，新用户涌入的速度远超预期。作为一个小型项目，我们可能无法及时回复每一个 Issue。在提交问题之前，请先仔细阅读本文档和 [Wiki](https://github.com/Finrandojin/alexandria-audiobook/wiki)，其中涵盖了大部分常见问题的解答。感谢大家的耐心与理解！

利用 AI 驱动的脚本标注和文本转语音技术，将任何书籍或小说转化为全配音有声书。内置 Qwen3-TTS 引擎，支持批量处理，并提供浏览器端编辑器，可逐行精调后导出。

## 示例音频：[sample.mp3](https://github.com/user-attachments/files/25276110/sample.mp3)

## 截图

<img src="https://github.com/user-attachments/assets/874b5e30-56d2-4292-b754-4408fc53f5d6" width="30%"></img> <img src="https://github.com/user-attachments/assets/488cde02-6b93-47fa-874b-97a618ae482c" width="30%"></img> <img src="https://github.com/user-attachments/assets/4c0805a6-bb9d-42c1-a9ff-79bb29d0613c" width="30%"></img> <img src="https://github.com/user-attachments/assets/8e58a5bf-ed8f-4864-8545-1e3d9681b0cf" width="30%"></img> <img src="https://github.com/user-attachments/assets/531830da-8668-4189-a0dc-020e6661bfb6" width="30%"></img>

## 主要功能

### AI 驱动流水线
- **本地与云端 LLM 支持** — 兼容任何 OpenAI 兼容 API（LM Studio、Ollama、OpenAI 等）
- **自动脚本标注** — LLM 将文本解析为包含说话人、对话和 TTS 指令的 JSON 格式
- **LLM 脚本审校** — 可选的二次 LLM 校验，修复常见标注错误
- **智能分块** — 按说话人连续分组（最多 500 字符），保持自然语流
- **上下文保持** — 在分块间传递角色名单和最后 3 条脚本条目，确保角色和风格连贯

### 语音生成
- **内置 TTS 引擎** — Qwen3-TTS 本地运行，无需外部服务器
- **多语言支持** — 中文、英语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语，或自动检测
- **预置声音** — 9 种预训练声音，支持基于指令的情感/语调控制
- **声音克隆** — 仅需 5-15 秒参考音频即可克隆任何声音
- **声音设计器** — 通过文字描述创建新声音（例如："温暖、低沉的男性声音，语调沉稳"）
- **LoRA 声音训练** — 在自定义语音数据集上微调 Base 模型，创建持久的声音身份
- **内置 LoRA 预设** — 开箱即用的预训练声音适配器
- **数据集构建器** — 交互式工具，逐条创建训练数据集，支持预览
- **批量处理** — 同时生成数十个语音块，吞吐量达实时速度的 3-6 倍
- **编解码器编译** — 可选的 `torch.compile` 优化，批量解码速度提升 3-4 倍

### Web UI 编辑器
- **8 标签页界面** — 设置、脚本生成、声音配置、声音设计、LoRA 训练、数据集构建、编辑器、结果
- **分块编辑** — 编辑任意行的说话人、文本和指令
- **选择性重新生成** — 单独重新渲染某一分块
- **实时进度** — 所有操作的实时日志和状态跟踪
- **音频预览** — 单独播放或按顺序预览整本有声书

### 导出选项
- **合并有声书** — 包含所有声音和自然停顿的单个 MP3 文件
- **单独语音行** — 每行单独导出 MP3，方便在 DAW 中编辑
- **Audacity 导出** — 一键导出 ZIP，包含按说话人分轨的 WAV 文件、LOF 项目文件和标签

---

## 系统要求

- [Pinokio](https://pinokio.computer/)
- LLM 服务器（以下任选其一）：
  - [LM Studio](https://lmstudio.ai/)（本地）— 推荐使用 Qwen3 或类似模型
  - [Ollama](https://ollama.ai/)（本地）
  - [OpenAI API](https://platform.openai.com/)（云端）
  - 任何 OpenAI 兼容 API
- **GPU：** 最低 8 GB 显存，推荐 16 GB 以上 — 详见下方兼容性表格
  - 每个 TTS 模型占用约 3.4 GB 显存；剩余显存决定批量大小
  - 所有平台均可使用 CPU 模式，但速度明显较慢
- **内存：** 推荐 16 GB（最低 8 GB）
- **磁盘：** 约 20 GB（8 GB venv/PyTorch + 约 7 GB 模型权重 + 音频工作空间）

### GPU 兼容性

| GPU | 操作系统 | 状态 | 驱动要求 | 备注 |
|-----|---------|------|---------|------|
| **NVIDIA** | Windows | 完全支持 | 驱动 550+（CUDA 12.8） | 包含 Flash Attention 加速编码 |
| **NVIDIA** | Linux | 完全支持 | 驱动 550+（CUDA 12.8） | 包含 Flash Attention + Triton |
| **AMD** | Linux | 完全支持 | ROCm 6.3 | 自动应用 ROCm 优化 |
| **AMD** | Windows | 仅 CPU | 不适用 | 不支持 GPU 加速 — 如需 AMD GPU 加速请使用 Linux |
| **Apple Silicon** | macOS | 仅 CPU | 不适用 | 暂不支持 MPS 加速，可运行但速度较慢 |

> **提示：** 无需外部 TTS 服务器。Alexandria 内置 Qwen3-TTS 引擎，模型权重在首次使用时自动从 Hugging Face 下载（每个模型变体约 3.5 GB）。

---

## 安装

1. 安装 [Pinokio](https://pinokio.computer/)（如尚未安装）
2. 在 Pinokio 中点击 **Download**，粘贴以下 URL：
   ```
   https://github.com/Finrandojin/alexandria-audiobook
   ```
3. 点击 **Install** 安装依赖
4. 点击 **Start** 启动 Web 界面

---

## 首次启动 — 必读

如果你是第一次运行 Alexandria，请在操作前仔细阅读本节。

### 1. 必须先启动 LLM 服务器

Alexandria **不包含** LLM — 它通过 API 连接到外部 LLM。在生成脚本之前，你必须先启动以下任一服务：

| 服务器 | 默认 URL | 安装方式 |
|--------|---------|---------|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` | 下载安装，加载模型，启动服务器 |
| [Ollama](https://ollama.ai/) | `http://localhost:11434/v1` | 安装后运行 `ollama run qwen3` |
| [OpenAI API](https://platform.openai.com/) | `https://api.openai.com/v1` | 获取 API Key |

如果在点击"Generate Script"时 LLM 服务器未运行，生成将会失败。请查看 Pinokio 终端获取错误详情。

### 2. 首次 TTS 生成会下载约 3.5 GB 模型

TTS 模型**不包含在安装中**，首次生成音频时会自动从 Hugging Face 下载：

- **每个模型变体约 3.5 GB**（CustomVoice、Base/克隆、VoiceDesign）
- 只有你使用的变体才会下载（大多数用户从 CustomVoice 开始）
- 下载在后台进行 — **请在 Pinokio 终端中查看进度**
- 此时 Web UI 可能看起来没有响应，这是正常的 — 它在等待下载完成
- 首次下载后，模型将缓存在本地，后续加载只需几秒钟

> **提示：** 如果下载似乎卡住了，请检查网络连接。如果失败，重启应用再试 — 会从断点处继续下载。

### 3. 首批生成需要额外预热时间

每个会话中的首次批量生成比后续生成更慢：

- **MIOpen 自动调优**（AMD GPU）：GPU 核心优化器每个会话运行一次，增加约 30-60 秒
- **编解码器编译**（如已启用）：一次性约 30-60 秒预热，之后所有批次速度提升 3-4 倍
- **这是正常现象。** 首批之后，生成速度会稳定下来

### 4. 显存决定你能做什么

| 可用显存 | 可行操作 |
|---------|---------|
| 8 GB | 一次只能加载一个模型，小批量（2-5 个语音块），可能需要 CPU 卸载 |
| 16 GB | 大多数用例都能舒适运行，批量 10-20 个语音块 |
| 24 GB+ | 全速运行，批量 40-60 个语音块，搭配编解码器编译 |

- 如果显存不足，请在设置标签页中降低 **Parallel Workers** 或 **Max Chars/Batch**
- 生成前关闭其他 GPU 应用程序（游戏、其他 AI 工具）
- 切换声音类型（Custom → Clone → LoRA）时会卸载并重新加载模型，暂时释放显存

### 5. 出问题时去哪里查看

Web UI 显示的是高层状态，**详细日志在 Pinokio 终端中**：

- 点击 Pinokio 侧栏中的 **Terminal** 查看实时输出
- 模型加载、下载进度、显存估算和错误信息都会显示在这里
- 如果 UI 中生成静默失败，终端会显示原因

---

## 快速入门

1. **设置标签页** — 配置 LLM 和 TTS：
   - **LLM Base URL**：`http://localhost:1234/v1`（LM Studio）或 `http://localhost:11434/v1`（Ollama）
   - **LLM API Key**：你的 API 密钥（本地服务器使用 `local`）
   - **LLM Model Name**：要使用的模型（例如 `qwen2.5-14b`）
   - **TTS Mode**：`local`（内置引擎，推荐）或 `external`（Gradio 服务器）

2. **脚本标签页** — 上传书籍文件（.txt 或 .md），点击"Generate Annotated Script"

3. **声音标签页** — 点击"Refresh Voices"，然后配置每个说话人：
   - 选择 Custom Voice、Clone Voice、LoRA Voice 或 Voice Design
   - 设置声音参数和角色风格，然后保存
   - 详见 [Voice Types](https://github.com/Finrandojin/alexandria-audiobook/wiki/Voice-Types)

4. **（可选）设计器标签页** — 通过文字描述创建新声音，用作克隆参考

5. **（可选）训练标签页** — 在自定义语音数据集上训练 LoRA 适配器

6. **（可选）数据集构建器标签页** — 交互式构建训练数据集，支持逐条预览

7. **编辑器标签页** — 审查和编辑分块：
   - 选择"Batch (Fast)"模式，点击"Batch Render Pending"以获得最快生成速度
   - 编辑任何分块的文本/指令/说话人，可单独重新生成
   - 满意后点击"Merge All"

8. **结果标签页** — 下载完成的有声书

---

## 常见问题

### 脚本生成失败
- 确认 LLM 服务器正在运行且可访问
- 验证模型名称与已加载模型一致
- 尝试使用其他模型 — 某些模型在 JSON 输出方面表现不佳
- 思维链模型（DeepSeek-R1、GLM4 等）可能干扰 JSON 输出。如需使用，请在设置中的 **Banned Tokens** 字段添加 `<think>` 以禁用思考模式

### TTS 生成失败
- 查看 Pinokio 终端中的模型加载错误
- 确保有足够的显存（推荐 16 GB 以上 bfloat16）
- 检查 voice_config.json 中所有说话人的设置是否有效
- 克隆声音时，确认参考音频存在且转录文本准确

### 生成速度慢
- 在设置中启用 **Compile Codec**（首次预热后速度提升 3-4 倍）
- 如果显存允许，增加 **Parallel Workers**（批量大小）
- 使用 **Batch (Fast)** 渲染模式而非 Standard
- 首批生成较慢是正常现象（参见上方"首次启动"部分）

### 显存不足 / OOM 错误
- 在设置中降低 **Max Chars/Batch**（特别是使用克隆/LoRA 声音且参考音频较长时）
- 降低 **Parallel Workers**（批量大小）
- 关闭其他 GPU 密集型应用程序
- 如果仍然不行，尝试 `device: cpu`（速度会慢很多）

### MP3 文件损坏或很小（428 字节）
Conda 自带的 ffmpeg 在 Windows 上通常缺少 MP3 编码器（libmp3lame）。Alexandria 会自动检测并回退到 WAV 格式。如需 MP3 输出：
- 安装带 MP3 支持的 ffmpeg：`conda install -c conda-forge ffmpeg`
- 或移除 conda 的 ffmpeg 以使用系统自带的：`conda remove ffmpeg`

### 中文书籍处理提示
- 在设置标签页的 **Language** 下拉菜单中选择"Chinese"或"Auto"
- 默认 LLM 提示是为英文编写的 — 处理中文书籍时，建议在设置标签页的"Prompt Customization"部分修改提示，使其适配中文对话约定（如使用「」引号等）
- 提示文件 `default_prompts.txt` 和 `review_prompts.txt` 可永久修改，更改即时生效无需重启

---

## 推荐 LLM 模型

用于脚本生成，非思维链模型效果最佳：
- **Qwen3-next**（80B-A3B-instruct）— JSON 输出和指令方向优秀
- **Gemma3**（推荐 27B）— JSON 输出和指令方向出色
- **Qwen2.5**（任意大小）— JSON 输出稳定
- **Qwen3**（非思维链变体）
- **Llama 3.1/3.2** — 角色区分能力强
- **Mistral/Mixtral** — 速度快，稳定可靠

---

## 更多文档

完整文档请参阅：
- [English README](README.md) — 完整英文文档，包含 API 参考和项目结构
- [Wiki](https://github.com/Finrandojin/alexandria-audiobook/wiki) — 详细指南：声音类型、LoRA 训练、批量生成等

## 许可证

MIT
