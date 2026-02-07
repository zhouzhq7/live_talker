# Live Talker 测试文档

## 测试架构

测试按照 Phase 和类型组织：

```
tests/
├── unit/                    # 单元测试
│   ├── phase1_asr/         # ASR (SenseVoice) 单元测试
│   ├── phase2_tts/         # TTS (MeloTTS) 单元测试
│   ├── phase3_vad/         # VAD (TEN-VAD) 单元测试
│   ├── phase4_llm/         # LLM (Ollama) 单元测试
│   └── phase5_streaming/   # 流式处理单元测试
├── integration/            # 集成测试
│   ├── phase1_asr/
│   ├── phase2_tts/
│   ├── phase3_vad/
│   ├── phase4_llm/
│   └── phase5_streaming/
├── performance/            # 性能测试
│   ├── phase1_asr/
│   ├── phase2_tts/
│   ├── phase3_vad/
│   ├── phase4_llm/
│   └── phase5_streaming/
├── acceptance/             # 验收测试
│   ├── phase1_asr/
│   ├── phase2_tts/
│   ├── phase3_vad/
│   ├── phase4_llm/
│   └── phase5_streaming/
├── test_data/              # 测试数据
├── utils/                  # 测试工具
├── conftest.py            # pytest fixtures
└── README.md              # 本文档
```

## 快速开始

### 安装测试依赖

```bash
cd live_talker
pip install -r requirements-test.txt
```

### 运行所有测试

```bash
pytest
```

### 运行特定 Phase 的测试

```bash
# Phase 1: ASR
pytest -m phase1

# Phase 2: TTS
pytest -m phase2

# Phase 3: VAD
pytest -m phase3

# Phase 4: LLM
pytest -m phase4

# Phase 5: Streaming
pytest -m phase5
```

### 运行特定类型的测试

```bash
# 仅单元测试
pytest -m unit

# 仅集成测试
pytest -m integration

# 仅性能测试
pytest -m performance

# 仅验收测试
pytest -m acceptance
```

### 排除慢速测试

```bash
pytest -m "not slow"
```

### 排除需要本地 LLM 服务的测试

```bash
pytest -m "not local_only"
```

## 测试标记说明

| 标记 | 说明 |
|------|------|
| `phase1` | Phase 1: ASR (SenseVoice) 相关测试 |
| `phase2` | Phase 2: TTS (MeloTTS) 相关测试 |
| `phase3` | Phase 3: VAD (TEN-VAD) 相关测试 |
| `phase4` | Phase 4: LLM (Ollama) 相关测试 |
| `phase5` | Phase 5: 流式处理相关测试 |
| `unit` | 单元测试 |
| `integration` | 集成测试 |
| `performance` | 性能测试 |
| `acceptance` | 验收测试 |
| `slow` | 慢速测试（需要加载模型）|
| `gpu` | 需要 GPU 的测试 |
| `network` | 需要网络的测试 |
| `local_only` | 需要本地 LLM 服务的测试 |

## 性能测试目标

### Phase 1: ASR (SenseVoice)
- RTF (实时率) < 0.1
- 10s 音频处理时间 < 100ms
- 内存占用 < 2GB
- 中文准确率 > 95%

### Phase 2: TTS (MeloTTS)
- 首包延迟 < 500ms
- 10 字文本合成 < 1s
- 100 字文本合成 < 3s
- 输出格式: 16-bit PCM, 16kHz, mono

### Phase 3: VAD (TEN-VAD)
- 检测延迟 < 100ms
- RTF < 0.01
- 比 Silero 快 30%+
- 库大小 < 306KB

### Phase 4: LLM (Ollama)
- 首 token 延迟 < 500ms
- 生成速度 > 10 tokens/s
- 支持 qwen2.5:7b/14b, deepseek-r1:7b

### Phase 5: 流式处理
- 端到端延迟 < 1s
- ASR 首字延迟 < 200ms
- LLM 首 token 延迟 < 500ms
- TTS 首音频延迟 < 500ms

## 编写新测试

### 单元测试示例

```python
import pytest

@pytest.mark.phase1
@pytest.mark.unit
@pytest.mark.asr
def test_sensevoice_init():
    """Test SenseVoice initialization"""
    from asr.sensevoice import SenseVoice
    
    asr = SenseVoice(model_name="iic/SenseVoiceSmall")
    assert asr.name == "SenseVoice"
```

### 性能测试示例

```python
import pytest

@pytest.mark.phase1
@pytest.mark.performance
def test_sensevoice_rtf(performance_monitor):
    """Test SenseVoice RTF"""
    from asr.sensevoice import SenseVoice
    
    engine = SenseVoice()
    engine.load_model()
    
    monitor = performance_monitor()
    monitor.start()
    
    result = engine.transcribe(audio_data)
    
    metrics = monitor.stop()
    
    # Assert performance
    assert metrics["rtf"] < 0.1
```

## 测试报告

运行测试并生成报告：

```bash
# HTML 报告
pytest --html=report.html

# JSON 报告
pytest --json-report --json-report-file=report.json

# 覆盖率报告
pytest --cov=live_talker --cov-report=html
```

## 持续集成

示例 GitHub Actions 配置：

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run unit tests
        run: pytest -m unit -v
      
      - name: Run integration tests
        run: pytest -m integration -v --timeout=300
```

## 调试技巧

### 查看详细输出

```bash
pytest -vvs
```

### 在第一个失败处停止

```bash
pytest -x
```

### 重新运行上次失败的测试

```bash
pytest --lf
```

### 进入 PDB 调试

```bash
pytest --pdb
```

## 贡献指南

1. 为每个新功能编写对应的测试
2. 测试应该独立运行，不依赖其他测试
3. 使用 fixtures 共享测试资源
4. 标记慢速测试和需要特殊环境的测试
5. 保持测试简单和可读
