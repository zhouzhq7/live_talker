# Live Talker - 实时语音对话系统

## 项目概述

Live Talker 是一个完整的实时语音对话系统，基于 Eva 项目的 `perception/audio` 模块设计，实现了从语音输入到智能回复的完整流程。

## 核心功能

- 🎤 **实时语音识别 (ASR)** - 支持 Whisper、FunASR、FireRedASR
- 🔊 **语音合成 (TTS)** - 支持 Edge-TTS、Pyttsx3
- 🎯 **语音活动检测 (VAD)** - 自动分段、打断检测
- 🤖 **智能对话 (LLM)** - Deepseek API 集成
- ⚡ **低延迟** - 优化的实时处理流程

## 快速开始

### 安装依赖

```bash
# 创建conda环境
conda create -n live_talker python=3.10
conda activate live_talker

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

在项目根目录创建 `.env` 文件（或复制 `.env.example`）：

```bash
# 必需：设置 Deepseek API Key
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# 可选：自定义模型缓存目录
MODEL_CACHE_DIR=D:\models
```

更多配置选项请参考 `.env.example` 文件。

### 运行示例

```bash
# 基础演示
python examples/basic_demo.py

# 完整功能演示
python examples/full_demo.py

# 主程序
python main.py
```

## 项目结构

```
live_talker/
├── audio/          # 音频处理模块
├── asr/            # ASR语音识别模块
├── tts/            # TTS语音合成模块
├── llm/            # LLM对话模块
├── core/           # 核心对话引擎
└── examples/       # 示例代码
```

## 配置说明

编辑 `config.py` 或设置环境变量：

```bash
# Deepseek API Key
export DEEPSEEK_API_KEY="your-api-key"

# ASR引擎选择
export ASR_ENGINE="funasr"  # funasr, whisper, fireredasr

# TTS引擎选择
export TTS_ENGINE="edge"    # edge, pyttsx3

# 模型缓存目录（默认：D:\models）
export MODEL_CACHE_DIR="D:\\models"
```

### 模型下载路径

所有模型文件将下载到指定的缓存目录：
- **ModelScope (FunASR)**: `D:\models\modelscope`
- **HuggingFace (Whisper)**: `D:\models\huggingface`
- **Torch Hub (Silero VAD)**: `D:\models\torch`

可以通过环境变量 `MODEL_CACHE_DIR` 自定义路径。

## 使用示例

```python
from core.talker import LiveTalker

# 初始化
talker = LiveTalker(
    asr_engine="funasr",
    tts_engine="edge",
    llm_provider="deepseek"
)

# 启动对话
talker.start()

# 自动处理：
# 用户说话 → ASR识别 → LLM生成回复 → TTS合成 → 播放
```

## 技术栈

- **ASR**: FunASR, Whisper, FireRedASR
- **TTS**: Edge-TTS, Pyttsx3
- **VAD**: Silero, WebRTC, Energy-based
- **LLM**: Deepseek API
- **音频**: PyAudio, NumPy

## 参考

- Eva项目: `perception/audio` 模块
- Voice Benchmark: ASR/TTS对比项目

## License

MIT License

