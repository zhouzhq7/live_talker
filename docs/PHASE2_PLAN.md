# Phase 2: TTS 升级到 MeloTTS - 实施计划

> 目标：添加 MeloTTS 作为默认 TTS 引擎，摆脱 FFmpeg 依赖，实现完全开源

---

## 📋 任务清单

### 2.1 环境准备

| 任务 | 状态 | 优先级 | 预估工时 | 说明 |
|------|------|--------|----------|------|
| 更新 requirements.txt | 🔵 | P0 | 0.5h | 添加 melotts 依赖 |
| 更新 config.py | 🔵 | P0 | 1h | 添加 MeloTTS 配置项 |

**详细步骤：**

1. **更新 requirements.txt**
   ```bash
   # 添加 MeloTTS
   melotts>=0.1.0
   torch>=2.0.0
   torchaudio>=2.0.0
   
   # Edge-TTS 变为可选
   # edge-tts>=6.0.0  # 可选，保留作为备选
   ```

2. **更新 config.py TTSConfig**
   ```python
   @dataclass
   class TTSConfig:
       """TTS configuration"""
       engine: str = "melotts"         # melotts, edge, pyttsx3
       
       # MeloTTS settings (default)
       melotts_language: str = "ZH"    # ZH, EN, ES, FR, JP, KR
       melotts_speaker: str = "ZH"     # Speaker ID
       melotts_speed: float = 1.0      # Speed: 0.5 - 2.0
       
       # Edge-TTS settings (保留)
       edge_voice: str = "zh-CN-XiaoxiaoNeural"
       edge_rate: str = "+0%"
       edge_volume: str = "+0%"
       
       # Pyttsx3 settings
       pyttsx3_rate: int = 200
       pyttsx3_volume: float = 1.0
   ```

---

### 2.2 实现 MeloTTS 类

| 任务 | 状态 | 优先级 | 预估工时 | 说明 |
|------|------|--------|----------|------|
| 创建 tts/melotts.py | 🔵 | P0 | 3h | 实现 MeloTTS 类 |
| 实现 __init__ | 🔵 | P0 | 0.5h | 初始化和配置 |
| 实现 load_model | 🔵 | P0 | 1h | 模型加载 |
| 实现 synthesize | 🔵 | P0 | 2h | 核心合成方法 |
| 支持多语言 | 🔵 | P1 | 1h | ZH/EN/ES/FR/JP/KR |

**实现要点：**

```python
# tts/melotts.py

import logging
import numpy as np
import tempfile
import os
from typing import Optional
from .base import BaseTTS

logger = logging.getLogger(__name__)


class MeloTTS(BaseTTS):
    """
    MeloTTS - 完全开源的多语言 TTS
    
    Features:
    - 完全开源，无需 FFmpeg
    - 支持 ZH, EN, ES, FR, JP, KR
    - 中英混合合成
    - 语速调节
    """
    
    def __init__(
        self,
        language: str = "ZH",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MeloTTS
        
        Args:
            language: Language code (ZH, EN, ES, FR, JP, KR)
            speaker: Speaker ID (None = use default for language)
            speed: Speech speed (0.5 - 2.0)
            model_cache_dir: Model cache directory
        """
        super().__init__(name=f"MeloTTS-{language}", **kwargs)
        
        self.language = language.upper()
        self.speaker = speaker
        self.speed = speed
        self.model_cache_dir = model_cache_dir
        
        self.model = None
        self.speaker_ids = None
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Load MeloTTS model"""
        if self._is_initialized:
            return True
        
        try:
            from melo.api import TTS
            
            logger.info(f"[{self.name}] Loading MeloTTS model...")
            
            # Load model
            device = 'auto'  # auto, cpu, cuda
            self.model = TTS(language=self.language, device=device)
            
            # Get available speakers
            self.speaker_ids = self.model.hps.data.spk2id
            
            # Set default speaker if not specified
            if self.speaker is None or self.speaker not in self.speaker_ids:
                self.speaker = list(self.speaker_ids.keys())[0]
            
            self._is_initialized = True
            logger.info(f"[{self.name}] Loaded with speaker: {self.speaker}")
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
        
        Returns:
            PCM audio bytes (16-bit, 16kHz, mono)
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return b''
        
        if not text or not text.strip():
            return b''
        
        try:
            import torchaudio
            
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_path = tmp.name
            
            try:
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
                
                return pcm_data
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    def get_info(self) -> dict:
        """Get MeloTTS-specific information"""
        info = super().get_info()
        info.update({
            "language": self.language,
            "speaker": self.speaker,
            "speed": self.speed,
            "available_speakers": list(self.speaker_ids.keys()) if self.speaker_ids else []
        })
        return info
```

---

### 2.3 集成

| 任务 | 状态 | 优先级 | 预估工时 | 说明 |
|------|------|--------|----------|------|
| 更新 tts/__init__.py | 🔵 | P0 | 0.5h | 导出 MeloTTS |
| 更新 core/talker.py | 🔵 | P0 | 1h | 工厂方法支持 |
| 设置默认引擎 | 🔵 | P0 | 0.5h | melotts 为默认 |

**详细步骤：**

1. **更新 tts/__init__.py**
   ```python
   from .base import BaseTTS
   from .melotts import MeloTTS
   from .edge_tts import EdgeTTS
   from .pyttsx3_tts import Pyttsx3TTS
   
   __all__ = [
       "BaseTTS",
       "MeloTTS",
       "EdgeTTS",
       "Pyttsx3TTS",
   ]
   ```

2. **更新 core/talker.py _create_tts()**
   ```python
   def _create_tts(self) -> BaseTTS:
       """Create TTS engine based on config"""
       engine = self.config.tts.engine.lower()
       
       if engine == "melotts":
           return MeloTTS(
               language=self.config.tts.melotts_language,
               speaker=self.config.tts.melotts_speaker,
               speed=self.config.tts.melotts_speed,
               model_cache_dir=self.config.model_cache_dir
           )
       elif engine == "edge":
           return EdgeTTS(
               voice=self.config.tts.edge_voice,
               rate=self.config.tts.edge_rate,
               volume=self.config.tts.edge_volume
           )
       elif engine == "pyttsx3":
           return Pyttsx3TTS(
               rate=self.config.tts.pyttsx3_rate,
               volume=self.config.tts.pyttsx3_volume
           )
       else:
           logger.warning(f"Unknown TTS engine: {engine}, using MeloTTS")
           return MeloTTS()
   ```

---

### 2.4 测试

| 任务 | 状态 | 优先级 | 预估工时 | 说明 |
|------|------|--------|----------|------|
| 单元测试 | 🔵 | P0 | 3h | 25 个测试 |
| 集成测试 | 🔵 | P0 | 2h | 完整流程 |
| 性能测试 | 🔵 | P0 | 2h | 延迟和吞吐量 |

**测试场景：**

```python
# 测试用例清单
1. 中文文本合成
2. 英文文本合成
3. 中英混合文本
4. 长文本 (>100字)
5. 特殊符号处理
6. 语速调节 (0.5x, 1.0x, 2.0x)
7. 多语言切换
8. 模型加载失败处理
9. 空文本处理
10. 音频格式验证 (16-bit PCM)
```

---

### 2.5 文档更新

| 任务 | 状态 | 优先级 | 预估工时 | 说明 |
|------|------|--------|----------|------|
| 更新 README.md | 🔵 | P1 | 1h | MeloTTS 说明 |
| 移除 FFmpeg 依赖说明 | 🔵 | P1 | 0.5h | 安装文档更新 |
| 更新 CHANGELOG.md | 🔵 | P1 | 0.5h | 版本记录 |
| 迁移指南 | 🔵 | P2 | 1h | Edge-TTS → MeloTTS |

---

## 🎯 性能目标

基于 CHECKLIST.md 的要求：

| 指标 | 目标 | 验收标准 |
|------|------|----------|
| 首包延迟 | < 500ms | 首次合成响应时间 |
| 10字合成 | < 1s | 短文本合成时间 |
| 100字合成 | < 3s | 长文本合成时间 |
| 内存占用 | < 1GB | 模型加载后内存 |
| 输出格式 | 16-bit PCM | 16kHz, mono |
| 依赖 | 无需 FFmpeg | 完全开源 |

---

## 📊 与 Edge-TTS 对比

| 特性 | Edge-TTS | MeloTTS |
|------|----------|---------|
| 开源 | ❌ 依赖微软服务 | ✅ 完全开源 |
| FFmpeg | ❌ 需要 | ✅ 不需要 |
| 离线 | ❌ 需要网络 | ✅ 完全离线 |
| 速度 | ⚡ 快 | ⚡ 快 |
| 质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 多语言 | 50+ | 6 (ZH, EN, ES, FR, JP, KR) |
| 自定义 | ❌ 有限 | ✅ 可训练 |

---

## 🚀 实施步骤

```bash
# 1. 安装依赖
pip install melotts torch torchaudio

# 2. 运行测试
pytest tests/unit/phase2_tts/ -v
pytest tests/integration/phase2_tts/ -v

# 3. 验证性能
pytest tests/performance/phase2_tts/ -v

# 4. 功能验证
python examples/basic_demo.py
```

---

## 📝 迁移指南 (Edge-TTS → MeloTTS)

### 自动迁移
MeloTTS 已设为默认引擎，无需修改代码

### 手动切换回 Edge-TTS
```python
from config import TalkerConfig

config = TalkerConfig()
config.tts.engine = "edge"  # 切换回 Edge-TTS
config.tts.edge_voice = "zh-CN-XiaoxiaoNeural"
```

### MeloTTS 高级用法
```python
from tts import MeloTTS

# 创建 MeloTTS 实例
tts = MeloTTS(
    language="ZH",      # ZH, EN, ES, FR, JP, KR
    speaker="ZH",       # Speaker ID
    speed=1.2           # 1.2x 速度
)

# 合成语音
audio = tts.synthesize("你好，世界")

# 获取可用 speakers
info = tts.get_info()
print(info["available_speakers"])  # ['ZH', 'EN', ...]
```

---

## ⏱️ 时间估算

| 阶段 | 预估工时 | 实际工时 |
|------|----------|----------|
| 环境准备 | 1.5h | - |
| 核心实现 | 5h | - |
| 集成 | 2h | - |
| 测试 | 7h | - |
| 文档 | 3h | - |
| **总计** | **18.5h** | **~2-3 天** |

---

## ✅ 验收标准

- [ ] 25 个单元测试全部通过
- [ ] 集成测试通过
- [ ] 首包延迟 < 500ms
- [ ] 10字合成 < 1s
- [ ] 100字合成 < 3s
- [ ] 无需 FFmpeg 即可运行
- [ ] 支持 ZH/EN 混合文本
- [ ] README 文档已更新
