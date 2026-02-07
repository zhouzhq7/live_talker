# Live Talker 升级快速参考

> 开发人员快速上手指南

---

## 🚀 5 分钟快速开始

### 1. 克隆项目并切换分支

```bash
# 克隆项目
git clone https://github.com/zhouzhq7/live_talker.git
cd live_talker

# 创建并切换到你的 feature 分支
git checkout -b feature/your-feature-name
```

### 2. 安装依赖

```bash
# 创建 conda 环境
conda create -n live_talker python=3.10
conda activate live_talker

# 安装基础依赖
pip install -r requirements.txt

# 根据你开发的模块安装额外依赖

# ASR (SenseVoice)
pip install "funasr>=1.0.0" modelscope

# TTS (MeloTTS)
pip install melotts

# VAD (TEN-VAD)
pip install git+https://github.com/TEN-framework/ten-vad.git

# LLM (Ollama 本地)
# 先安装 Ollama: https://ollama.com/
ollama pull qwen2.5:7b
```

### 3. 配置环境变量

```bash
# 创建 .env 文件
cp .env.example .env

# 编辑 .env，根据你的开发模块配置
nano .env
```

示例配置：
```bash
# ASR 引擎: sensevoice, funasr, whisper
ASR_ENGINE=sensevoice

# TTS 引擎: melotts, edge, pyttsx3
TTS_ENGINE=melotts

# VAD 引擎: silero, ten, webrtc
VAD_METHOD=silero

# LLM 引擎: ollama, deepseek
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b

# 模型缓存目录
MODEL_CACHE_DIR=D:\models  # Windows
# MODEL_CACHE_DIR=/Users/xxx/models  # macOS
# MODEL_CACHE_DIR=/home/xxx/models  # Linux
```

### 4. 运行测试

```bash
# 运行基础 demo
python examples/basic_demo.py

# 运行完整 demo
python examples/full_demo.py

# 运行主程序
python main.py
```

---

## 📁 项目结构速览

```
live_talker/
├── asr/                    # ASR 模块
│   ├── base.py            # 抽象基类
│   ├── funasr.py          # FunASR 实现
│   ├── whisper.py         # Whisper 实现
│   └── sensevoice.py      # 【新增】SenseVoice 实现
├── tts/                    # TTS 模块
│   ├── base.py            # 抽象基类
│   ├── edge_tts.py        # Edge-TTS 实现
│   ├── pyttsx3_tts.py     # Pyttsx3 实现
│   └── melotts.py         # 【新增】MeloTTS 实现
├── audio/                  # 音频处理
│   ├── recorder.py        # 录音器
│   ├── player.py          # 播放器
│   ├── vad.py             # VAD 检测器
│   └── vad_ten.py         # 【新增】TEN-VAD 实现
├── llm/                    # LLM 模块
│   ├── base.py            # 抽象基类
│   ├── deepseek.py        # Deepseek 实现
│   └── ollama.py          # 【新增】Ollama 本地实现
├── core/                   # 核心引擎
│   └── talker.py          # 主对话引擎
├── config.py              # 配置管理
└── main.py                # 主入口
```

---

## 🔧 开发一个新模块

以添加新的 ASR 引擎为例：

### 1. 创建实现文件

```bash
touch asr/news_asr.py
```

### 2. 继承基类并实现方法

```python
# asr/news_asr.py
from .base import BaseASR
import logging

logger = logging.getLogger(__name__)


class NewsASR(BaseASR):
    """你的 ASR 实现"""
    
    def __init__(self, **kwargs):
        super().__init__(name="NewsASR", **kwargs)
        self.model = None
        self.load_model()
    
    def load_model(self) -> bool:
        """加载模型"""
        try:
            # 你的模型加载代码
            self._is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return False
    
    def transcribe(self, audio_data: bytes, sample_rate: int = 16000, language: str = "zh") -> str:
        """语音识别"""
        if not self._is_initialized:
            return ""
        
        try:
            # 你的识别代码
            text = "识别结果"
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
```

### 3. 导出模块

```python
# asr/__init__.py
from .base import BaseASR
from .funasr import FunASR
from .whisper import Whisper
from .fireredasr import FireRedASR
from .sensevoice import SenseVoice  # 【新增】
from .news_asr import NewsASR       # 【新增】
```

### 4. 添加到工厂方法

```python
# core/talker.py

def _create_asr(self) -> BaseASR:
    engine = self.config.asr.engine.lower()
    
    if engine == "sensevoice":
        return SenseVoice(...)
    elif engine == "news":               # 【新增】
        return NewsASR(...)
    # ... 其他引擎
```

### 5. 更新配置

```python
# config.py

@dataclass
class ASRConfig:
    engine: str = "news"  # 设置为默认或添加为新选项
    
    # News ASR 配置
    news_model_path: str = "path/to/model"
    news_device: str = "cpu"
```

### 6. 编写测试

```python
# tests/test_asr_news.py
import unittest
from asr.news_asr import NewsASR


class TestNewsASR(unittest.TestCase):
    def setUp(self):
        self.asr = NewsASR()
    
    def test_initialization(self):
        self.assertTrue(self.asr.is_available())
    
    def test_transcribe(self):
        # 准备测试音频数据
        audio_data = b"..."  # 16-bit PCM
        text = self.asr.transcribe(audio_data)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
```

---

## 🐛 常见问题

### Q: SenseVoice 下载模型很慢？

A: 设置 ModelScope 镜像
```bash
export MODELSCOPE_CACHE=/path/to/cache
# 或使用国内镜像
export MODELSCOPE_ENDPOINT=https://www.modelscope.cn
```

### Q: MeloTTS 合成中文有杂音？

A: 确保输入文本编码为 UTF-8
```python
text = text.encode('utf-8').decode('utf-8')
```

### Q: TEN-VAD 编译失败？

A: 确保安装依赖
```bash
# Ubuntu/Debian
sudo apt-get install libc++1

# macOS
brew install llvm
```

### Q: Ollama 连接失败？

A: 检查服务状态
```bash
# 启动 Ollama 服务
ollama serve

# 检查模型是否存在
ollama list

# 测试连接
curl http://localhost:11434/api/tags
```

### Q: 音频播放没有声音？

A: 检查音频格式
```python
import pyaudio

# 检查设备
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))
```

---

## 📊 性能优化技巧

### 1. ASR 优化

```python
# 使用 GPU
SenseVoice(device="cuda:0")

# 批处理（如果支持）
audio_batches = [audio1, audio2, audio3]
results = model.batch_transcribe(audio_batches)
```

### 2. TTS 优化

```python
# 音频缓存
class CachedTTS:
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.cache = {}
    
    def synthesize(self, text):
        if text in self.cache:
            return self.cache[text]
        
        audio = self.tts.synthesize(text)
        self.cache[text] = audio
        return audio
```

### 3. VAD 优化

```python
# 调整阈值适应环境
VADDetector(
    threshold=0.3,  # 嘈杂环境降低阈值
    min_speech_duration=0.1,  # 快速响应
    min_silence_duration=0.3  # 快速断句
)
```

### 4. 整体延迟优化

```python
# 预热模型
talker.asr.warmup()
talker.tts.synthesize("预热文本")

# 使用流式处理
# 参考 Phase 5 流式优化文档
```

---

## 🔍 调试技巧

### 开启详细日志

```python
# main.py 开头
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 检查音频数据

```python
import numpy as np

# 检查音频格式
audio_array = np.frombuffer(audio_data, dtype=np.int16)
print(f"Sample count: {len(audio_array)}")
print(f"Duration: {len(audio_array) / 16000:.2f}s")
print(f"Max amplitude: {np.max(np.abs(audio_array))}")
```

### 性能分析

```python
import time

# 测量各环节耗时
start = time.time()
text = asr.transcribe(audio)
print(f"ASR: {time.time() - start:.2f}s")

start = time.time()
response = llm.chat(text)
print(f"LLM: {time.time() - start:.2f}s")

start = time.time()
audio = tts.synthesize(response)
print(f"TTS: {time.time() - start:.2f}s")
```

---

## 📚 相关资源

### 官方文档
- [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice)
- [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- [TEN-VAD GitHub](https://github.com/TEN-framework/ten-vad)
- [Ollama 官网](https://ollama.com/)

### 模型下载
- [ModelScope (国内)](https://www.modelscope.cn)
- [HuggingFace](https://huggingface.co)

### 社区支持
- GitHub Issues: https://github.com/zhouzhq7/live_talker/issues

---

## ✅ 提交前检查清单

```markdown
## 代码检查
- [ ] 代码遵循 PEP 8 规范
- [ ] 添加必要的注释和文档字符串
- [ ] 没有硬编码的敏感信息
- [ ] 错误处理完善

## 测试检查
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过

## 文档检查
- [ ] README.md 已更新
- [ ] 配置文档已更新
- [ ] CHANGELOG.md 已更新

## 提交检查
- [ ] Commit message 符合规范
- [ ] 分支命名符合规范
- [ ] 无不必要的文件提交
```

---

## 🆘 获取帮助

1. **查看文档**: `docs/improvement_plan/README.md`
2. **查看 TODO**: `docs/improvement_plan/TODO_TRACKING.md`
3. **提交 Issue**: GitHub Issues 页面
4. **查看日志**: 开启 DEBUG 级别日志

---

**最后更新**: 2025-02-07
