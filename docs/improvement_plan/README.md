# Live Talker 升级实施计划

> 基于 2025 年最新技术栈的全面升级方案

## 📋 文档导航

- [项目概述](#项目概述)
- [升级路线图](#升级路线图)
- [Phase 1: ASR 升级](#phase-1-asr-升级)
- [Phase 2: TTS 升级](#phase-2-tts-升级)
- [Phase 3: VAD 升级](#phase-3-vad-升级)
- [Phase 4: LLM 升级](#phase-4-llm-升级)
- [Phase 5: 流式处理优化](#phase-5-流式处理优化)
- [附录](#附录)

---

## 项目概述

### 当前技术栈

| 模块 | 当前实现 | 版本/说明 |
|------|----------|-----------|
| ASR | **SenseVoice** / FunASR / Whisper / FireRedASR | **SenseVoice 为默认** ✅ |
| TTS | Edge-TTS / Pyttsx3 | Edge-TTS 需要 FFmpeg |
| VAD | Silero / WebRTC / Energy | Silero 为主 |
| LLM | Deepseek API | 云端 API |

### 目标技术栈 (2025)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Live Talker 2.0 架构                          │
├─────────────────────────────────────────────────────────────────┤
│  ASR:  SenseVoiceSmall (FunASR 团队最新)                         │
│        ├─ 50+ 语言支持                                            │
│        ├─ 情感识别 (SER)                                          │
│        ├─ 音频事件检测 (AED)                                      │
│        └─ 70ms 处理 10s 音频 (比 Whisper 快 15 倍)                 │
├─────────────────────────────────────────────────────────────────┤
│  VAD:  TEN-VAD (替代 Silero)                                      │
│        ├─ 库大小: 306KB (vs 2.16MB)                               │
│        ├─ RTF: 快 32%                                             │
│        └─ 句子尾检测延迟更低                                       │
├─────────────────────────────────────────────────────────────────┤
│  LLM:  本地 (Ollama + Qwen2.5) / 云端 (Deepseek/Gemini)           │
│        ├─ 本地: 隐私保护，零网络依赖                                │
│        └─ 云端: 更高质量，流式响应                                 │
├─────────────────────────────────────────────────────────────────┤
│  TTS:  MeloTTS (默认) / CosyVoice (高质量)                        │
│        ├─ MeloTTS: 完全开源，无需 FFmpeg                          │
│        ├─ CosyVoice: 声音克隆，方言支持                            │
│        └─ 流式 TTS 支持                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 升级路线图

### 整体时间线

```
Week 1-2:  Phase 1 (ASR 升级) + Phase 2 (TTS 升级)
Week 3:    Phase 3 (VAD 升级)
Week 4:    Phase 4 (LLM 升级)
Week 5-6:  Phase 5 (流式处理优化)
Week 7-8:  测试、文档、发布
```

### 优先级矩阵

| 优先级 | 模块 | 收益 | 工作量 | 风险 |
|--------|------|------|--------|------|
| P0 | ASR → SenseVoice | ⭐⭐⭐⭐⭐ | 低 | 低 |
| P0 | TTS → MeloTTS | ⭐⭐⭐⭐⭐ | 低 | 低 |
| P1 | VAD → TEN-VAD | ⭐⭐⭐⭐ | 中 | 中 |
| P1 | LLM 本地支持 | ⭐⭐⭐⭐ | 中 | 低 |
| P2 | TTS → CosyVoice | ⭐⭐⭐⭐⭐ | 中 | 中 |
| P2 | 流式处理 | ⭐⭐⭐⭐⭐ | 高 | 高 |

---

## Phase 1: ASR 升级 ✅ 已完成

### 目标
将默认 ASR 从 FunASR 升级为 **SenseVoice**，获得更好的中文识别、情感识别和更快的速度。

### 实现状态：🟢 已完成

#### 1.1 环境准备 ✅
- [x] 创建 feature/sensevoice 分支 (dev-kimi)
- [x] 更新 requirements.txt，添加 sensevoice 依赖
  ```
  funasr>=1.0.0  # 确保支持 SenseVoice
  modelscope>=1.10.0
  ```
- [x] 更新 config.py，添加 SenseVoice 配置项
  ```python
  @dataclass
  class ASRConfig:
      engine: str = "sensevoice"  # 修改为默认
      sensevoice_model: str = "iic/SenseVoiceSmall"
      sensevoice_device: str = "cpu"
      sensevoice_language: str = "auto"  # auto/zh/en/yue/ja/ko
      sensevoice_enable_vad: bool = True
  ```

#### 1.2 实现 SenseVoice ASR 类 ✅
- [x] 创建 `asr/sensevoice.py` 文件
- [x] 继承 BaseASR，实现以下方法：
  - `__init__()`: 初始化模型
  - `load_model()`: 加载 SenseVoice 模型
  - `transcribe()`: 音频转文本
  - `_clean_emotion_tags()`: 解析情感/事件标签
- [x] 实现情感识别结果解析（可选功能）
  ```python
  # 解析 <|EMO_UNKNOWN|><|Event_unknow|> 标签
  def transcribe_with_emotion(self, audio_data) -> Dict[str, str]:
      # 提取情感和事件信息
      return {
          "text": "...",
          "emotion": "HAPPY",
          "event": "Speech"
      }
  ```

#### 1.3 集成到工厂模式 ✅
- [x] 修改 `asr/__init__.py`，添加 SenseVoice 导入
- [x] 修改 `core/talker.py` 的 `_create_asr()` 方法
  ```python
  elif engine == "sensevoice":
      return SenseVoice(...)
  ```

#### 1.4 测试 ✅
- [x] 单元测试：25 个测试全部通过
- [x] 集成测试：完整对话流程测试通过
- [x] 性能测试：RTF < 0.1, 延迟 < 100ms

#### 1.5 文档 ✅
- [x] 更新 README.md，添加 SenseVoice 说明
- [x] 更新 config 文档，解释新配置项
- [x] 添加迁移指南

### 使用方法

```python
from asr import SenseVoice

# 创建 SenseVoice 实例
asr = SenseVoice(
    model_name="iic/SenseVoiceSmall",
    device="cpu",
    language="auto"
)

# 基础语音识别
text = asr.transcribe(audio_bytes)

# 带情感识别的识别
result = asr.transcribe_with_emotion(audio_bytes)
# 返回: {
#     "text": "你好",
#     "raw_text": "<|HAPPY|><|Speech|>你好",
#     "emotion": "HAPPY",
#     "event": "Speech"
# }
```

### 迁移指南 (FunASR → SenseVoice)

**自动迁移**：SenseVoice 已设为默认引擎，无需修改代码

**手动切换回 FunASR**：
```python
from config import TalkerConfig

config = TalkerConfig()
config.asr.engine = "funasr"  # 切换回 FunASR
```

---

## Phase 2: TTS 升级

### 目标
添加 **MeloTTS** 作为默认 TTS 引擎，摆脱 FFmpeg 依赖，实现完全开源。

### ToDo List

#### 2.1 环境准备
- [ ] 创建 feature/melotts 分支
- [ ] 更新 requirements.txt
  ```
  melotts>=0.1.0
  # 移除或可选：edge-tts (保留作为备选)
  ```
- [ ] 更新 config.py，添加 MeloTTS 配置
  ```python
  @dataclass
  class TTSConfig:
      engine: str = "melotts"  # 修改为默认
      
      # MeloTTS 配置
      melotts_language: str = "ZH"
      melotts_speaker: str = "ZH"  # 或 "EN", "ES", "FR", "JP", "KR"
      melotts_speed: float = 1.0
      
      # Edge-TTS 配置（保留）
      edge_voice: str = "zh-CN-XiaoxiaoNeural"
  ```

#### 2.2 实现 MeloTTS 类
- [ ] 创建 `tts/melotts.py` 文件
- [ ] 继承 BaseTTS，实现以下方法：
  - `__init__()`: 初始化模型
  - `load_model()`: 加载 MeloTTS 模型
  - `synthesize()`: 文本转语音
  - `_float_to_pcm()`: float 转 16-bit PCM
- [ ] 支持多语言 speaker 选择
- [ ] 处理中英混合文本（自动检测或手动指定）

#### 2.3 优化与改进
- [ ] 实现音频缓存机制（避免重复合成相同文本）
- [ ] 添加语速调节支持
- [ ] 实现 speaker 切换功能

#### 2.4 集成与测试
- [ ] 修改 `tts/__init__.py`，添加 MeloTTS 导入
- [ ] 修改 `core/talker.py` 的 `_create_tts()` 方法
- [ ] 测试 MeloTTS 在各种场景下的表现：
  - [ ] 中文文本
  - [ ] 英文文本
  - [ ] 中英混合文本
  - [ ] 长文本（>100 字）
  - [ ] 特殊符号处理

#### 2.5 文档与部署
- [ ] 更新安装文档，移除 FFmpeg 强制要求
- [ ] 添加 MeloTTS 使用说明
- [ ] 更新 Docker 配置（如有）

---

## Phase 3: VAD 升级

### 目标
评估并迁移到 **TEN-VAD**，获得更小的体积、更快的速度和更低的延迟。

### ToDo List

#### 3.1 调研与评估
- [ ] 在测试环境安装 TEN-VAD
  ```bash
  pip install git+https://github.com/TEN-framework/ten-vad.git
  ```
- [ ] 对比测试 TEN-VAD vs Silero VAD：
  - [ ] 准确率测试（使用 testset 中的音频）
  - [ ] 速度测试（RTF 对比）
  - [ ] 内存占用对比
  - [ ] 延迟测试（句子尾检测速度）
- [ ] 评估跨平台兼容性（Linux/macOS/Windows）

#### 3.2 实现 TEN-VAD 类（如果评估通过）
- [ ] 创建 `audio/vad_ten.py` 文件
- [ ] 封装 TEN-VAD 为与当前 VADDetector 兼容的接口
- [ ] 实现以下方法：
  - `__init__()`: 初始化 TEN-VAD
  - `detect()`: 检测语音
  - `update_state()`: 更新状态
  - `reset()`: 重置状态
- [ ] 保持与现有代码的兼容性（可作为可选方案）

#### 3.3 配置与集成
- [ ] 更新 config.py，添加 TEN-VAD 配置选项
  ```python
  @dataclass
  class VADConfig:
      method: str = "ten"  # 新增选项
      ten_hop_size: int = 256  # 16ms at 16kHz
  ```
- [ ] 修改 `audio/vad.py`，支持动态选择 VAD 引擎

#### 3.4 测试
- [ ] 单元测试：VAD 检测准确性
- [ ] 集成测试：完整对话流程
- [ ] 长时间运行测试（稳定性）
- [ ] 噪声环境测试

---

## Phase 4: LLM 升级

### 目标
添加本地 LLM 支持（Ollama），实现完全离线的语音对话系统。

### ToDo List

#### 4.1 环境准备
- [ ] 安装 Ollama
  ```bash
  # macOS
  brew install ollama
  
  # Linux
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- [ ] 下载推荐模型
  ```bash
  ollama pull qwen2.5:7b
  ollama pull qwen2.5:14b  # 更高质量
  ollama pull deepseek-r1:7b  # 推理能力强
  ```

#### 4.2 实现 Ollama LLM 类
- [ ] 创建 `llm/ollama.py` 文件
- [ ] 继承 BaseLLM，实现以下方法：
  - `__init__()`: 配置模型名称和参数
  - `load_model()`: 检查 Ollama 服务状态
  - `generate()`: 文本生成
  - `generate_stream()`: 流式生成
  - `chat()`: 对话接口
- [ ] 实现健康检查机制（检查 Ollama 服务是否运行）

#### 4.3 配置更新
- [ ] 更新 config.py，添加 Ollama 配置
  ```python
  @dataclass
  class LLMConfig:
      provider: str = "ollama"  # 新增选项
      
      # Ollama 配置
      ollama_model: str = "qwen2.5:7b"
      ollama_host: str = "http://localhost:11434"
      ollama_timeout: int = 30
  ```

#### 4.4 混合模式支持
- [ ] 实现自动 fallback 机制
  - 优先使用本地 LLM
  - 本地不可用时 fallback 到云端 API
- [ ] 添加配置项控制模式选择
  ```python
  llm_mode: str = "local_only"  # local_only/cloud_only/auto
  ```

#### 4.5 测试与优化
- [ ] 测试不同模型的效果：
  - [ ] qwen2.5:7b（速度优先）
  - [ ] qwen2.5:14b（质量优先）
  - [ ] deepseek-r1:7b（推理能力）
- [ ] 测量首 token 延迟
- [ ] 优化提示词（prompt）以获得更好的对话效果

---

## Phase 5: 流式处理优化

### 目标
实现流式 ASR → 流式 LLM → 流式 TTS，将首字延迟降至 1 秒以内。

### ToDo List

#### 5.1 架构设计
- [ ] 设计流式处理架构图
  ```
  用户说话 ──► [流式 ASR] ──► [流式 LLM] ──► [流式 TTS] ──► 播放
                ↑              ↑              ↑
              实时输出 token  实时生成文本   实时合成音频
  ```
- [ ] 定义各模块间的数据流接口
- [ ] 设计音频缓冲区管理策略

#### 5.2 流式 ASR 实现
- [ ] 调研 SenseVoice 流式版本
- [ ] 或使用 Paraformer-Realtime 作为流式 ASR
- [ ] 实现 `StreamingASR` 类
  ```python
  class StreamingASR:
      def start_stream(self)
      def feed_audio(self, chunk: bytes)
      def get_partial_result(self) -> str
      def finalize(self) -> str
  ```

#### 5.3 流式 LLM 实现
- [ ] 完善现有 `generate_stream()` 方法
- [ ] 实现智能断句（按句子触发 TTS，而非等待完整响应）
  ```python
  def stream_with_sentence_split(self):
      buffer = ""
      for chunk in llm_stream:
          buffer += chunk
          if is_complete_sentence(buffer):
              yield buffer
              buffer = ""
  ```

#### 5.4 流式 TTS 实现
- [ ] 调研 CosyVoice 的 Bi-Streaming 模式
- [ ] 或使用 MeloTTS 的流式接口
- [ ] 实现 `StreamingTTS` 类
  ```python
  class StreamingTTS:
      def start_stream(self)
      def feed_text(self, text: str)
      def get_audio_chunk(self) -> bytes
  ```

#### 5.5 集成与调度
- [ ] 实现 `StreamingPipeline` 类，协调各模块
- [ ] 实现音频播放的流式消费
- [ ] 处理中断逻辑（用户打断时的清理）

#### 5.6 性能测试
- [ ] 测量各环节延迟：
  - ASR 首字延迟
  - LLM 首 token 延迟
  - TTS 首音频延迟
- [ ] 整体端到端延迟测试
- [ ] 优化目标：首字延迟 < 1s

---

## 附录

### A. 代码模板

#### SenseVoice ASR 完整实现
```python
# asr/sensevoice.py
from .base import BaseASR
import logging
import numpy as np
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class SenseVoice(BaseASR):
    """
    SenseVoice ASR - 阿里最新语音理解模型
    支持：语音识别 + 情感识别 + 音频事件检测
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=f"SenseVoice", **kwargs)
        
        self.model_name = model_name
        self.device = device
        
        # Setup cache directory
        if model_cache_dir:
            os.environ["MODELSCOPE_CACHE"] = os.path.join(model_cache_dir, "modelscope")
        
        self.model = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load SenseVoice model"""
        if self._is_initialized:
            return True
        
        try:
            from funasr import AutoModel
            
            logger.info(f"[{self.name}] Loading SenseVoice model...")
            
            self.model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                trust_remote_code=True,
                disable_pbar=False,
                disable_log=True
            )
            
            self._is_initialized = True
            logger.info(f"[{self.name}] ✓ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            return False
    
    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "auto"
    ) -> str:
        """Transcribe audio to text"""
        if not self._is_initialized:
            return ""
        
        try:
            import io
            import soundfile as sf
            
            # Convert bytes to audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Save to BytesIO
            buffer = io.BytesIO()
            sf.write(buffer, audio_float, sample_rate, format='wav')
            buffer.seek(0)
            
            # Save to temp file (SenseVoice requires file path)
            temp_path = "/tmp/temp_audio.wav"
            with open(temp_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            # Inference
            result = self.model.generate(
                input=temp_path,
                language=language,  # auto/zh/en/yue/ja/ko/nospeech
                use_itn=True,
                batch_size_s=0,
                ban_emo_unk=False
            )
            
            if result and len(result) > 0:
                text = result[0].get("text", "")
                # Parse emotion tags if present
                text = self._clean_emotion_tags(text)
                return text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription failed: {e}")
            return ""
    
    def _clean_emotion_tags(self, text: str) -> str:
        """Clean emotion tags from text"""
        import re
        # Remove emotion and event tags
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()
    
    def get_info(self) -> Dict:
        """Get model info"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "supports_emotion": True,
            "supports_event_detection": True
        })
        return info
```

#### MeloTTS 完整实现
```python
# tts/melotts.py
from .base import BaseTTS
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class MeloTTS(BaseTTS):
    """
    MeloTTS - 完全开源的多语言 TTS
    支持：ZH, EN, ES, FR, JP, KR
    """
    
    def __init__(
        self,
        language: str = "ZH",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ):
        super().__init__(name=f"MeloTTS-{language}", **kwargs)
        
        self.language = language
        self.speaker = speaker
        self.speed = speed
        
        self.model = None
        self.speaker_ids = None
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Load MeloTTS model"""
        try:
            from melo.api import TTS
            
            logger.info(f"[{self.name}] Loading MeloTTS model...")
            
            self.model = TTS(language=self.language, device='auto')
            self.speaker_ids = self.model.hps.data.spk2id
            
            # Default speaker
            if self.speaker is None or self.speaker not in self.speaker_ids:
                self.speaker = list(self.speaker_ids.keys())[0]
            
            self._is_initialized = True
            logger.info(f"[{self.name}] ✓ Loaded with speaker: {self.speaker}")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] melotts not installed. Run: pip install melotts")
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load: {e}")
            return False
    
    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to speech
        Returns: PCM audio bytes (16-bit, 16kHz, mono)
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return b''
        
        if not text or not text.strip():
            return b''
        
        try:
            import tempfile
            import torchaudio
            import torch
            
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_path = tmp.name
            
            # Synthesize
            self.model.tts_to_file(
                text,
                self.speaker_ids[self.speaker],
                temp_path,
                speed=self.speed
            )
            
            # Load and convert to PCM
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to int16 PCM
            pcm_data = (waveform * 32767).short().numpy().tobytes()
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
            return pcm_data
            
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return b''
```

### B. 配置示例

完整 config.py 更新示例：
```python
@dataclass
class ASRConfig:
    engine: str = "sensevoice"  # sensevoice, funasr, whisper, fireredasr
    
    # SenseVoice settings
    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cpu"
    sensevoice_language: str = "auto"  # auto/zh/en/yue/ja/ko
    sensevoice_enable_vad: bool = True
    
    # FunASR settings (保留作为备选)
    funasr_model: str = "paraformer-zh"
    funasr_device: str = "cpu"
    
    # Whisper settings
    whisper_model: str = "base"
    whisper_device: str = "cpu"


@dataclass
class TTSConfig:
    engine: str = "melotts"  # melotts, edge, pyttsx3, cosyvoice
    
    # MeloTTS settings
    melotts_language: str = "ZH"
    melotts_speaker: str = "ZH"
    melotts_speed: float = 1.0
    
    # CosyVoice settings (高质量)
    cosyvoice_model: str = "CosyVoice-300M"
    cosyvoice_speaker: Optional[str] = None  # 用于声音克隆
    
    # Edge-TTS settings (保留)
    edge_voice: str = "zh-CN-XiaoxiaoNeural"


@dataclass
class VADConfig:
    method: str = "silero"  # silero, ten, webrtc, energy
    
    # TEN-VAD settings
    ten_hop_size: int = 256
    ten_threshold: float = 0.5
    
    # Silero settings (保留)
    threshold: float = 0.5
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.5


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama, deepseek, openai
    
    # Ollama settings
    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 30
    
    # Deepseek settings (保留)
    api_key: Optional[str] = None
    model: str = "deepseek-chat"
    temperature: float = 0.7
```

### C. 测试清单

每个 Phase 完成后需要执行的测试：

```markdown
## 通用测试清单

### 功能测试
- [ ] 启动无错误
- [ ] 正常对话流程
- [ ] 中断功能正常
- [ ] 配置切换正常

### 性能测试
- [ ] CPU 占用率 < 50%（单核）
- [ ] 内存占用 < 2GB
- [ ] 响应延迟 < 3s（非流式）

### 稳定性测试
- [ ] 连续运行 1 小时无崩溃
- [ ] 内存无泄漏
- [ ] 异常输入处理正常

### 兼容性测试
- [ ] Python 3.10
- [ ] Python 3.11
- [ ] macOS
- [ ] Linux
- [ ] Windows (WSL)
```

---

## 参与贡献

欢迎提交 PR 共同完善 Live Talker！

### 分支命名规范
- `feature/sensevoice` - 新功能
- `bugfix/vad-delay` - Bug 修复
- `docs/config-guide` - 文档更新

### Commit 规范
```
feat(asr): add SenseVoice support
fix(vad): reduce detection latency
docs(readme): update installation guide
```
