# Live Talker 实时语音对话系统 - 需求文档

**版本**: v1.0
**作者**: Claude Code
**日期**: 2026-02-07
**状态**: 待评审

---

## 1. 项目概述

### 1.1 项目名称
Live Talker - 实时语音对话系统 v2.0

### 1.2 项目背景
Live Talker v1.x 采用串行处理架构，端到端延迟高达 3-8 秒，用户体验不佳。本项目旨在通过引入完整的流式处理架构，将延迟降低至 0.8-1.5 秒，实现真正的实时语音对话。

### 1.3 项目目标
| 指标 | 当前值 | 目标值 | 提升幅度 |
|------|--------|--------|---------|
| 端到端延迟 | 3-8s | 0.8-1.5s | 4-5x |
| ASR 首字延迟 | N/A (非流式) | 50-100ms | 新增 |
| TTS 首字延迟 | 800ms | 100-200ms | 4-8x |
| LLM 首字延迟 | 500ms | 150-300ms | 2-3x |
| 流式支持 | ❌ | ✅ | N/A |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            用户端层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │  麦克风输入  │  │  扬声器输出  │  │           打断检测                   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────────────────┘  │
└─────────┼────────────────┼────────────────┼───────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          音频处理层                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                    Audio Pipeline Manager                              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐ │   │
│  │  │ 采集调度 │  │ 缓冲管理 │  │ 格式转换 │  │    AEC 回声消除        │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          流式处理层                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│   │   VAD    │────▶│   ASR    │────▶│   LLM    │────▶│   TTS    │        │
│   │  端点检测 │     │  流式识别 │     │  流式推理 │     │ 流式合成 │        │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘        │
│        │                │                │                │                │
│        ▼                ▼                ▼                ▼                │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│   │ 打断判断 │   │ 增量文本 │   │ 增量Token│   │ 增量音频 │              │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Stream Orchestrator                                │   │
│   │   - 流水线调度                                                       │   │
│   │   - 状态管理                                                         │   │
│   │   - 错误恢复                                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          对话管理层                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  会话状态管理    │  │  上下文记忆     │  │      打断处理                │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
[1] 用户语音输入
    ────────────────────────────────────────────────────────────────────────▶

    Mic → Audio Interface → Buffer → VAD
                                    │
                                    ▼
                             [VAD 状态判断]
                                    │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
         [静音]              [语音开始]           [语音结束]
              │                    │                    │
              ▼                    ▼                    ▼
         继续采集         添加预录音           输出完整音频
              │            到识别队列              块
              │                    │                    │
              │                    └──────────┬─────────┘
              │                               │
              │                               ▼
              │                    ┌────────────────────┐
              │                    │   ASR 流式识别     │
              │                    │   (Sherpa-ONNX)    │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  增量文本输出      │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  LLM 流式推理      │
              │                    │  (Deepseek V3)     │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │   增量 Token       │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  TTS 流式合成      │
              │                    │  (RealtimeTTS)     │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  增量 PCM 音频     │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  Audio Player      │
              │                    │  (边接收边播放)     │
              │                    └─────────┬──────────┘
              │                              │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │  扬声器输出        │
              │                    └────────────────────┘
              │
              ▼
    [打断检测]
    ────────────────────────────────────────────────────────────────────────▶

    AEC 处理 → VAD 二次检测 → 打断确认 → 停止当前播放 → 清空管道 → 重新开始
```

### 2.3 核心组件

| 组件 | 职责 | 技术选型 |
|------|------|---------|
| **AudioCollector** | 音频采集，缓冲管理 | Sounddevice |
| **VADDetector** | 语音端点检测 | TEN-VAD / Silero VAD |
| **ASREngine** | 流式语音识别 | Sherpa-ONNX (Paraformer) |
| **LLMEngine** | 流式语言模型 | Deepseek V3 |
| **TTSEngine** | 流式语音合成 | RealtimeTTS + Kokoro |
| **AECProcessor** | 回声消除 | WebRTC AEC / SpeexAEC |
| **StreamOrchestrator** | 流水线协调 | 自定义 Async Pipeline |
| **SessionManager** | 对话状态管理 | 状态机 |

---

## 3. 功能需求

### 3.1 音频采集与处理

#### FR-AUD-001: 实时音频采集
**描述**: 系统必须能够从麦克风实时采集音频数据
**优先级**: P0
**验收标准**:
- 支持 16kHz 采样率，16-bit PCM 格式
- 采集延迟 < 10ms
- 支持动态切换输入设备
- 支持音量调节和静音控制

#### FR-AUD-002: 音频缓冲管理
**描述**: 系统必须实现高效的音频缓冲管理
**优先级**: P0
**验收标准**:
- 支持动态缓冲区大小调整
- 预录音缓冲：至少 1 秒
- 环形缓冲支持，防止内存溢出
- 缓冲状态可监控

#### FR-AUD-003: 回声消除 (AEC)
**描述**: 系统必须在播放时检测用户打断
**优先级**: P1
**验收标准**:
- 支持 WebRTC AEC 或 SpeexAEC
- 打断检测延迟 < 300ms
- 准确率 > 90%
- 支持麦克风-扬声器回声消除

#### FR-AUD-004: 音频格式转换
**描述**: 系统必须支持多种音频格式
**优先级**: P1
**验收标准**:
- MP3 → PCM 转换
- WAV → PCM 转换
- 采样率自动转换 (8kHz/16kHz/44.1kHz)
- 单声道/立体道自动转换

### 3.2 语音端点检测 (VAD)

#### FR-VAD-001: 语音活动检测
**描述**: 系统必须实时检测语音活动
**优先级**: P0
**验收标准**:
- 检测延迟 < 30ms
- 准确率 > 95%
- 支持多种检测模式切换
- 支持灵敏度调节

#### FR-VAD-002: 语音端点判定
**描述**: 系统必须准确判定语音开始和结束
**优先级**: P0
**验收标准**:
- 语音开始检测延迟 < 100ms
- 语音结束检测延迟 300-800ms
- 支持静音阈值调节
- 支持最小语音时长过滤

#### FR-VAD-003: 打断检测
**描述**: 系统必须在播放时检测用户打断
**优先级**: P0
**验收标准**:
- 打断检测延迟 < 300ms
- 支持持续打断检测 (300ms+ 语音)
- 打断后 100ms 内停止播放
- 支持打断后快速恢复

### 3.3 语音识别 (ASR)

#### FR-ASR-001: 流式语音识别
**描述**: 系统必须支持实时流式语音识别
**优先级**: P0
**验收标准**:
- 首字延迟 < 100ms
- 实时因子 (RTF) < 0.3
- 支持增量文本输出
- 支持识别置信度输出

#### FR-ASR-002: 中文语音识别
**描述**: 系统必须准确识别中文语音
**优先级**: P0
**验收标准**:
- 中文识别准确率 > 95%
- 支持标点符号自动恢复
- 支持数字/日期/时间规范化
- 支持常见口语化表达

#### FR-ASR-003: 识别结果处理
**描述**: 系统必须有效处理识别结果
**优先级**: P1
**验收标准**:
- 支持实时文本流输出
- 支持句子边界检测
- 支持识别超时处理
- 支持说话人分离 (可选)

### 3.4 语言模型 (LLM)

#### FR-LLM-001: 流式文本生成
**描述**: 系统必须支持 LLM 流式输出
**优先级**: P0
**验收标准**:
- 首 Token 延迟 < 300ms
- Token 间延迟 < 50ms
- 支持流式响应中断
- 支持停止词检测

#### FR-LLM-002: 对话上下文
**描述**: 系统必须维护对话上下文
**优先级**: P1
**验收标准**:
- 支持多轮对话记忆
- 支持上下文长度配置
- 支持历史摘要
- 支持上下文压缩

#### FR-LLM-003: 提示词优化
**描述**: 系统必须生成适合语音播放的文本
**优先级**: P1
**验收标准**:
- 自动添加自然停顿
- 避免长难句
- 避免生僻字词
- 语气自然流畅

### 3.5 语音合成 (TTS)

#### FR-TTS-001: 流式语音合成
**描述**: 系统必须支持流式语音合成和播放
**优先级**: P0
**验收标准**:
- 首字节延迟 < 200ms
- 支持边合成边播放
- 支持句子边界检测
- 音频输出：16kHz, 16-bit PCM

#### FR-TTS-002: 中文语音合成
**描述**: 系统必须支持高质量中文语音合成
**优先级**: P0
**验收标准**:
- 支持多种中文音色
- 自然流畅的语音
- 支持语速调节
- 支持音量调节

#### FR-TTS-003: 情感表达
**描述**: 系统应支持情感化的语音合成
**优先级**: P2
**验收标准**:
- 支持情感标签
- 支持语气变化
- 支持停顿和强调

### 3.6 对话管理

#### FR-DIA-001: 对话状态管理
**描述**: 系统必须管理对话状态
**优先级**: P0
**验收标准**:
- 支持空闲/倾听/思考/播放 四种状态
- 支持状态转换追踪
- 支持状态可视化

#### FR-DIA-002: 打断处理
**描述**: 系统必须正确处理用户打断
**优先级**: P0
**验收标准**:
- 检测到打断立即停止播放
- 清空未完成的管道数据
- 快速恢复到倾听状态
- 保留对话上下文

#### FR-DIA-003: 错误恢复
**描述**: 系统必须有完善的错误处理机制
**优先级**: P1
**验收标准**:
- ASR 失败重试
- LLM 超时处理
- TTS 失败回退
- 自动状态恢复

---

## 4. 非功能需求

### 4.1 性能需求

| 指标 | 要求 | 测量方法 |
|------|------|---------|
| **端到端延迟** | < 1.5s (P95) | 用户说完到听到回应的第95百分位延迟 |
| **ASR RTF** | < 0.3 | 处理时间/音频时长 |
| **TTS 首字延迟** | < 200ms | 开始合成到首字节输出的时间 |
| **LLM TTFT** | < 300ms | 发送请求到收到首Token的时间 |
| **CPU 占用** | < 60% (单核) | 稳态运行时的CPU使用率 |
| **内存占用** | < 2GB | 稳态运行时的内存使用 |
| **音频采集延迟** | < 10ms | 音频数据从麦克风到缓冲区的延迟 |

### 4.2 可用性需求

| 指标 | 要求 |
|------|------|
| **启动时间** | < 10s (含模型加载) |
| **首次响应延迟** | < 3s (冷启动) |
| **故障恢复时间** | < 5s |
| **持续运行时间** | > 8小时 (无内存泄漏) |

### 4.3 可扩展性需求

| 指标 | 要求 |
|------|------|
| **模块化设计** | 各模块可独立替换 |
| **引擎切换** | 支持运行时切换 ASR/TTS 引擎 |
| **配置热更新** | 支持部分配置不重启生效 |

### 4.4 兼容性需求

| 平台 | 要求 |
|------|------|
| **Python** | 3.10+ |
| **操作系统** | macOS 12+, Ubuntu 20.04+, Windows 10+ |
| **依赖** | 详见 requirements.txt |

### 4.5 安全性需求

| 需求 | 说明 |
|------|------|
| **API Key 安全** | 通过环境变量或配置文件加载，不硬编码 |
| **音频隐私** | 本地处理，不上传到云端 (除 LLM API) |
| **日志脱敏** | 对话内容不记录到日志 |

---

## 5. 技术选型

### 5.1 核心依赖

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| **编程语言** | Python | 3.10+ | 生态丰富，开发效率高 |
| **ASR** | Sherpa-ONNX | 1.10+ | 真正流式，跨平台，低延迟 |
| **TTS** | RealtimeTTS + Kokoro | 最新 | 流式合成，低首字延迟 |
| **VAD** | TEN-VAD / Silero | 最新 | 高准确率，ONNX 支持 |
| **LLM** | Deepseek V3 | 最新 | OpenAI 兼容，快速流式 |
| **音频 IO** | Sounddevice | 最新 | 跨平台，低延迟 |
| **异步框架** | asyncio | 标准库 | 原生支持，无需额外依赖 |

### 5.2 备选方案

| 组件 | 主选 | 备选1 | 备选2 |
|------|------|-------|-------|
| **ASR** | Sherpa-ONNX | Paraformer-Streaming | Whisper (非流式) |
| **TTS** | RealtimeTTS | Edge-TTS (缓存) | Kokoro-TTS |
| **VAD** | TEN-VAD | Silero VAD | WebRTC VAD |
| **LLM** | Deepseek V3 | OpenAI GPT-4 | Ollama 本地 |

### 5.3 模型选择

| 模型 | 大小 | 用途 | 下载地址 |
|------|------|------|---------|
| paraformer-zh-streaming | ~400MB | ASR | ModelScope / HuggingFace |
| sherpa-onnx-streaming-zh-14M | 14MB | 轻量 ASR | Sherpa-ONNX Releases |
| Kokoro-TTS | ~100MB | TTS | HuggingFace |
| deepseek-chat | N/A | LLM | Deepseek API |

---

## 6. 接口设计

### 6.1 核心接口定义

#### 6.1.1 音频采集器接口

```python
from abc import ABC, abstractmethod
from typing import Optional, Callable, AsyncGenerator
import numpy as np

class AudioCollector(ABC):
    """音频采集器抽象接口"""

    @abstractmethod
    async def start(self) -> None:
        """开始采集音频"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止采集音频"""
        pass

    @abstractmethod
    async def stream(self) -> AsyncGenerator[bytes, None]:
        """音频流生成器

        Yields:
            bytes: 16-bit PCM 音频数据
        """
        pass

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """设置采集音量 (0.0 - 1.0)"""
        pass

    @abstractmethod
    def get_devices(self) -> list[dict]:
        """获取可用输入设备列表"""
        pass

    @abstractmethod
    def set_device(self, device_id: int) -> bool:
        """设置输入设备

        Args:
            device_id: 设备ID

        Returns:
            bool: 设置是否成功
        """
        pass
```

#### 6.1.2 VAD 检测器接口

```python
from abc import ABC, abstractmethod
from typing import NamedTuple

class VADState(NamedTuple):
    """VAD 状态"""
    is_speech: bool          # 当前是否检测到语音
    speech_started: bool     # 语音是否刚开始
    speech_ended: bool       # 语音是否刚结束
    confidence: float        # 置信度 0-1

class VADetector(ABC):
    """语音端点检测抽象接口"""

    @abstractmethod
    def reset(self) -> None:
        """重置 VAD 状态"""
        pass

    @abstractmethod
    def detect(self, audio_chunk: bytes) -> VADState:
        """检测音频片段

        Args:
            audio_chunk: 16kHz, 16-bit PCM 音频数据

        Returns:
            VADState: 检测状态
        """
        pass

    @abstractmethod
    def set_sensitivity(self, threshold: float) -> None:
        """设置检测灵敏度

        Args:
            threshold: 阈值 0-1 (越高越灵敏)
        """
        pass
```

#### 6.1.3 ASR 引擎接口

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, NamedTuple

class ASRResult(NamedTuple):
    """ASR 识别结果"""
    text: str               # 识别的文本
    is_final: bool          # 是否是最终结果
    confidence: float       # 置信度 0-1
    timestamp_start: int    # 开始时间戳 (ms)
    timestamp_end: int      # 结束时间戳 (ms)

class ASREngine(ABC):
    """语音识别抽象接口"""

    @abstractmethod
    async def load_model(self) -> bool:
        """加载模型

        Returns:
            bool: 加载是否成功
        """
        pass

    @abstractmethod
    def recognize(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ASRResult, None]:
        """流式识别

        Args:
            audio_stream: 音频流生成器

        Yields:
            ASRResult: 识别结果
        """
        pass

    @abstractmethod
    def recognize_once(self, audio_data: bytes) -> ASRResult:
        """单次识别 (非流式)

        Args:
            audio_data: 完整音频数据

        Returns:
            ASRResult: 识别结果
        """
        pass
```

#### 6.1.4 LLM 引擎接口

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, NamedTuple

class LLMResult(NamedTuple):
    """LLM 生成结果"""
    token: str              # 生成的 token
    is_final: bool          # 是否是结束
    finish_reason: str      # 结束原因 (stop, length, etc.)

class LLMEngine(ABC):
    """语言模型抽象接口"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        stream: bool = True
    ) -> AsyncGenerator[LLMResult, None]:
        """对话生成

        Args:
            messages: 对话历史消息 [{"role": "user"|"assistant", "content": "..."}]
            stream: 是否流式输出

        Yields:
            LLMResult: 生成结果
        """
        pass

    @abstractmethod
    def set_temperature(self, temperature: float) -> None:
        """设置生成温度 (0-2)"""
        pass

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> None:
        """设置最大输出长度"""
        pass
```

#### 6.1.5 TTS 引擎接口

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, NamedTuple

class TTSResult(NamedTuple):
    """TTS 合成结果"""
    audio_chunk: bytes      # PCM 音频片段
    is_final: bool          # 是否是最后一段
    text: str              # 对应的文本

class TTSEngine(ABC):
    """语音合成抽象接口"""

    @abstractmethod
    async def synthesize(
        self,
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[TTSResult, None]:
        """流式合成

        Args:
            text_stream: 文本流生成器

        Yields:
            TTSResult: 合成结果
        """
        pass

    @abstractmethod
    def synthesize_once(self, text: str) -> bytes:
        """单次合成 (非流式)

        Args:
            text: 要合成的文本

        Returns:
            bytes: PCM 音频数据
        """
        pass

    @abstractmethod
    def set_voice(self, voice: str) -> None:
        """设置音色"""
        pass

    @abstractmethod
    def set_rate(self, rate: str) -> None:
        """设置语速 (+/-XX%)"""
        pass

    @abstractmethod
    def list_voices(self) -> list[str]:
        """获取可用音色列表"""
        pass
```

#### 6.1.6 流水线协调器接口

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple

class PipelineState(Enum):
    """流水线状态"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"

class PipelineEvent(NamedTuple):
    """流水线事件"""
    event_type: str         # 事件类型
    data: dict             # 事件数据
    timestamp: float       # 时间戳

class StreamOrchestrator(ABC):
    """流式处理协调器"""

    @abstractmethod
    async def start(self) -> None:
        """启动流水线"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止流水线"""
        pass

    @abstractmethod
    def get_state(self) -> PipelineState:
        """获取当前状态"""
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """触发打断"""
        pass

    @abstractmethod
    async def pipeline_events(self) -> AsyncGenerator[PipelineEvent, None]:
        """流水线事件流

        Yields:
            PipelineEvent: 流水线事件
        """
        pass
```

### 6.2 配置文件结构

```yaml
# config.yaml - 完整配置示例

# ========== 通用配置 ==========
app:
  name: "Live Talker v2"
  version: "2.0.0"
  log_level: "INFO"
  log_file: "logs/livetalker.log"

# ========== 音频配置 ==========
audio:
  input:
    sample_rate: 16000
    channels: 1
    chunk_size: 1024  # ~64ms 音频
    device: null      # null 表示默认设备
    buffer_size: 30   # 环形缓冲大小
  output:
    sample_rate: 16000
    channels: 1
    buffer_size: 4096
  aec:
    enabled: true
    engine: "webrtc"  # webrtc, speex
    filter_length: 4096

# ========== VAD 配置 ==========
vad:
  engine: "silero"    # silero, ten_vad, webrtc
  threshold: 0.5
  min_speech_duration: 0.25  # 秒
  min_silence_duration: 0.5  # 秒
  # 打断检测配置
  interruption:
    enabled: true
    threshold: 0.3     # 300ms 语音触发打断
    sensitivity: 0.5

# ========== ASR 配置 ==========
asr:
  engine: "sherpa_onnx"  # sherpa_onnx, paraformer, whisper
  model:
    name: "paraformer-zh-streaming"
    path: null           # null 表示自动下载
  decoding:
    method: "greedy"     # greedy, beam_search
    max_active_paths: 4
  latency:
    first_chunk_timeout: 0.5  # 首块超时 (秒)
    max_latency: 2.0          # 最大延迟 (秒)

# ========== LLM 配置 ==========
llm:
  provider: "deepseek"   # deepseek, openai, ollama
  api:
    base_url: "https://api.deepseek.com"
    api_key: null       # 从环境变量读取
  model: "deepseek-chat"
  generation:
    temperature: 0.7
    max_tokens: 2000
    stream: true
  prompt:
    system: "你是一个友好的AI语音助手..."
    history_max_turns: 10

# ========== TTS 配置 ==========
tts:
  engine: "realtime"     # realtime, edge, kokoro
  primary:
    engine: "realtime"
    voice: "zh-CN-XiaoxiaoNeural"
    rate: "+0%"
    volume: "+0%"
  fallback:
    engine: "edge"
    voice: "zh-CN-XiaoxiaoNeural"
  streaming:
    chunk_size: 100      # 字符数阈值
    min_sentence_length: 5  # 最小句子长度

# ========== 对话配置 ==========
conversation:
  max_history: 10
  auto_summarize: true
  summarize_threshold: 5  # 超过N轮自动摘要
  welcome_message: "你好，我是你的AI助手，有什么可以帮你？"

# ========== 性能配置 ==========
performance:
  max_concurrent_requests: 1
  timeout:
    asr: 30.0
    llm: 60.0
    tts: 30.0
  cache:
    enabled: true
    max_size: 1000
```

### 6.3 事件类型定义

```python
# 流水线事件类型常量

# VAD 事件
VAD_SPEECH_STARTED = "vad.speech_started"
VAD_SPEECH_ENDED = "vad.speech_ended"
VAD_INTERRUPTION_DETECTED = "vad.interruption"

# ASR 事件
ASR_PARTIAL_RESULT = "asr.partial"
ASR_FINAL_RESULT = "asr.final"
ASR_TIMEOUT = "asr.timeout"
ASR_ERROR = "asr.error"

# LLM 事件
LLM_TOKEN_GENERATED = "llm.token"
LLM_RESPONSE_COMPLETE = "llm.complete"
LLM_ERROR = "llm.error"

# TTS 事件
TTS_AUDIO_CHUNK = "tts.audio_chunk"
TTS_SENTENCE_END = "tts.sentence_end"
TTS_COMPLETE = "tts.complete"
TTS_ERROR = "tts.error"

# 系统事件
PIPELINE_STARTED = "pipeline.started"
PIPELINE_STOPPED = "pipeline.stopped"
PIPELINE_INTERRUPTED = "pipeline.interrupted"
PIPELINE_ERROR = "pipeline.error"
```

---

## 7. 目录结构

```
live_talker/
├── config/
│   ├── __init__.py
│   ├── config.py           # 配置加载和管理
│   ├── schema.py           # 配置 Schema 定义
│   └── defaults.py         # 默认配置
│
├── audio/
│   ├── __init__.py
│   ├── collector.py        # 音频采集器
│   ├── player.py           # 音频播放器
│   ├── buffer.py           # 音频缓冲管理
│   ├── aec.py              # 回声消除
│   └── device.py           # 设备管理
│
├── vad/
│   ├── __init__.py
│   ├── base.py             # VAD 接口定义
│   ├── silero.py           # Silero VAD 实现
│   ├── ten_vad.py          # TEN-VAD 实现
│   └── webrtc.py           # WebRTC VAD 实现
│
├── asr/
│   ├── __init__.py
│   ├── base.py             # ASR 接口定义
│   ├── sherpa_onnx.py      # Sherpa-ONNX 实现
│   ├── paraformer.py       # Paraformer 实现
│   └── whisper.py          # Whisper 实现 (非流式备选)
│
├── llm/
│   ├── __init__.py
│   ├── base.py             # LLM 接口定义
│   ├── deepseek.py         # Deepseek 实现
│   ├── openai.py           # OpenAI 兼容实现
│   └── ollama.py           # Ollama 本地实现
│
├── tts/
│   ├── __init__.py
│   ├── base.py             # TTS 接口定义
│   ├── realtime.py         # RealtimeTTS 实现
│   ├── edge.py             # Edge-TTS 实现
│   └── kokoro.py           # Kokoro TTS 实现
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py      # 流水线协调器
│   ├── state.py            # 状态机
│   ├── events.py           # 事件处理
│   └── error_handler.py    # 错误处理
│
├── core/
│   ├── __init__.py
│   ├── talker.py           # 主对话引擎
│   ├── session.py          # 会话管理
│   └── interrupt.py        # 打断处理
│
├── utils/
│   ├── __init__.py
│   ├── logger.py           # 日志工具
│   ├── timer.py            # 计时工具
│   └── metrics.py           # 性能指标
│
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_vad.py
│   ├── test_asr.py
│   ├── test_llm.py
│   ├── test_tts.py
│   └── test_pipeline.py
│
├── requirements.txt
├── requirements-dev.txt
├── config.yaml
├── CLAUDE.md
└── README.md
```

---

## 8. 实施计划

### 8.1 开发阶段划分

#### Phase 1: 基础设施 (第1周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| 项目脚手架搭建 | - | 1天 | 目录结构, 配置框架 |
| 音频采集模块 | - | 2天 | AudioCollector 实现 |
| 音频缓冲管理 | - | 1天 | RingBuffer 实现 |
| 配置系统完善 | - | 1天 | config.yaml, 配置加载 |
| 单元测试框架 | - | 1天 | 测试用例, CI 流程 |

#### Phase 2: VAD 模块 (第1-2周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| VAD 接口定义 | - | 1天 | VADetector 接口 |
| Silero VAD 集成 | - | 2天 | silero.py |
| TEN-VAD 集成 | - | 3天 | ten_vad.py |
| 打断检测实现 | - | 2天 | InterruptionDetector |
| VAD 测试优化 | - | 2天 | 测试用例, 性能调优 |

#### Phase 3: ASR 模块 (第2-3周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| ASR 接口定义 | - | 1天 | ASREngine 接口 |
| Sherpa-ONNX 集成 | - | 3天 | sherpa_onnx.py |
| 流式识别管道 | - | 2天 | 增量文本输出 |
| ASR 测试优化 | - | 2天 | 测试用例, 准确率测试 |

#### Phase 4: LLM 模块 (第3周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| LLM 接口定义 | - | 1天 | LLMEngine 接口 |
| Deepseek 流式集成 | - | 2天 | deepseek.py |
| 对话历史管理 | - | 1天 | SessionManager |
| Prompt 优化 | - | 1天 | 语音友好 Prompt |

#### Phase 5: TTS 模块 (第3-4周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| TTS 接口定义 | - | 1天 | TTSEngine 接口 |
| RealtimeTTS 集成 | - | 2天 | realtime.py |
| Kokoro 集成 | - | 2天 | kokoro.py |
| 流式合成管道 | - | 2天 | 边合成边播放 |
| TTS 测试优化 | - | 1天 | 延迟测试, 音质评估 |

#### Phase 6: 流水线集成 (第4-5周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| 流水线协调器 | - | 3天 | StreamOrchestrator |
| 状态机实现 | - | 2天 | PipelineState |
| 事件系统 | - | 1天 | PipelineEvent |
| 打断处理 | - | 2天 | interrupt.py |
| 错误恢复 | - | 2天 | error_handler.py |
| 端到端测试 | - | 2天 | 集成测试 |

#### Phase 7: 优化与测试 (第5-6周)

| 任务 | 负责人 | 工期 | 交付物 |
|------|--------|------|--------|
| 性能调优 | - | 3天 | 延迟优化, 内存优化 |
| 稳定性测试 | - | 2天 | 24小时压力测试 |
| 文档完善 | - | 1天 | README, API 文档 |
| Bug 修复 | - | 3天 | 修复发现的问题 |
| 验收测试 | - | 2天 | 验收用例通过 |

### 8.2 里程碑

| 里程碑 | 时间 | 验收标准 |
|--------|------|---------|
| **M1: 基础设施完成** | 第1周末 | 音频采集正常, 配置加载正常 |
| **M2: VAD 模块完成** | 第2周末 | VAD 准确率 > 95%, 打断检测正常 |
| **M3: ASR 模块完成** | 第3周末 | ASR 延迟 < 100ms, 准确率 > 95% |
| **M4: 核心流水线完成** | 第5周末 | 端到端延迟 < 2s |
| **M5: 版本发布** | 第6周末 | 所有验收用例通过 |

---

## 9. 验收标准

### 9.1 功能验收

| 用例编号 | 用例名称 | 前置条件 | 输入 | 预期输出 |
|----------|---------|---------|------|---------|
| FV-001 | 音频采集 | 系统启动 | 麦克风输入 | 正确采集 16kHz PCM |
| FV-002 | VAD 检测 | 安静环境 | 用户说话 | 准确检测语音起止 |
| FV-003 | 打断检测 | 系统播放 | 用户说话 | 300ms 内停止播放 |
| FV-004 | ASR 识别 | 麦克风输入 | "你好" | 识别文本 "你好" |
| FV-005 | LLM 生成 | ASR 结果 | "你好" | 生成自然回复 |
| FV-006 | TTS 合成 | LLM 输出 | 回复文本 | 合成语音并播放 |
| FV-007 | 完整流程 | 系统启动 | 完整对话 | 端到端延迟 < 1.5s |
| FV-008 | 多轮对话 | 完成1轮 | 继续对话 | 保持上下文 |

### 9.2 性能验收

| 指标 | 测试方法 | 目标值 | 验收标准 |
|------|---------|--------|---------|
| 端到端延迟 | 10次完整对话 | < 1.5s | P95 < 1.5s |
| ASR 首字延迟 | 10次测试 | < 100ms | 平均 < 100ms |
| TTS 首字延迟 | 10次测试 | < 200ms | 平均 < 200ms |
| LLM TTFT | 10次测试 | < 300ms | 平均 < 300ms |
| 打断检测 | 10次打断 | > 90% | 准确率 > 90% |
| ASR 准确率 | 100句测试 | > 95% | 字准确率 > 95% |
| 内存占用 | 长时间运行 | < 2GB | 稳态 < 2GB |
| CPU 占用 | 稳态运行 | < 60% | 单核 < 60% |

### 9.3 稳定性验收

| 测试场景 | 测试方法 | 目标值 | 验收标准 |
|---------|---------|--------|---------|
| 8小时连续运行 | 持续对话测试 | 0 崩溃 | 无内存泄漏, 无崩溃 |
| 网络中断恢复 | 断网测试 | 自动恢复 | 30s 内恢复 |
| 快速打断测试 | 连续打断 | 正确处理 | 5/5 次正确 |
| 并发请求 | 多用户测试 | 正常处理 | 单用户模式稳定 |

---

## 10. 风险与应对

### 10.1 技术风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| Sherpa-ONNX 流式不稳定 | ASR 延迟增加 | 中 | 准备 Paraformer 作为备选 |
| RealtimeTTS 兼容性 | TTS 无法流式 | 低 | 保留 Edge-TTS 备选 |
| AEC 回声消除效果差 | 打断误触发 | 中 | 调整灵敏度, 备用 WebRTC |
| 模型下载失败 | 系统无法启动 | 低 | 提供离线模型加载 |
| API 限流 | LLM 响应失败 | 中 | 实现退避重试 |

### 10.2 进度风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| TEN-VAD 集成困难 | VAD 延迟增加 | 中 | 降低优先级, 继续用 Silero |
| Sherpa-ONNX 性能不达标 | ASR 延迟高 | 低 | 调整参数, 换用更大模型 |
| 回声消除实现复杂 | 打断功能延迟 | 高 | 简化 AEC, 先实现基础打断 |
| 端到端延迟不达标 | 需求无法满足 | 中 | 分阶段优化, 接受 2s 目标 |

### 10.3 依赖风险

| 依赖 | 版本 | 风险 | 应对措施 |
|------|------|------|---------|
| Sherpa-ONNX | 1.10+ | API 变更 | 锁定版本, 准备降级 |
| RealtimeTTS | 最新 | 功能变化 | 封装适配层 |
| Deepseek API | V3 | 兼容性 | 保持 OpenAI 兼容接口 |
| Sounddevice | 最新 | 平台兼容 | 准备 PyAudio 备选 |

---

## 11. 附录

### 11.1 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| ASR | Automatic Speech Recognition | 自动语音识别 |
| TTS | Text-to-Speech | 语音合成 |
| VAD | Voice Activity Detection | 语音端点检测 |
| AEC | Acoustic Echo Cancellation | 声学回声消除 |
| RTF | Real-Time Factor | 实时因子 |
| TTFT | Time To First Token | 首 Token 延迟 |
| P95 | 95th Percentile | 第95百分位 |
| PCM | Pulse Code Modulation | 脉冲编码调制 |

### 11.2 参考资料

- [Sherpa-ONNX GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [RealtimeTTS GitHub](https://github.com/KoljaB/RealtimeTTS)
- [TEN-VAD GitHub](https://github.com/TEN-framework/ten-vad)
- [Deepseek API 文档](https://api.deepseek.com/docs)
- [LiveKit Agents](https://github.com/livekit/agents)

### 11.3 版本历史

| 版本 | 日期 | 作者 | 变更 |
|------|------|------|------|
| v1.0 | 2026-02-07 | Claude Code | 初始版本 |

---

**文档结束**
