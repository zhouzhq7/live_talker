# Qt客户端架构变更图

## 当前架构 vs 目标架构

### 当前架构（原有）

```
┌─────────────────────────────────────────┐
│              MainWindow                 │
│  ┌─────────────────────────────────┐   │
│  │         VoiceSphere             │   │
│  │      (5种基础状态颜色)           │   │
│  └─────────────────────────────────┘   │
│              StatusLabel                │
│           "正在听..."                    │
│  ┌─────────────────────────────────┐   │
│  │        ControlButtons           │   │
│  │      [🎤] [⏹]                  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    │
                    │ signals/slots
                    ▼
┌─────────────────────────────────────────┐
│           TalkerThread                  │
│  ┌─────────────────────────────────┐   │
│  │         LiveTalker              │   │
│  │    - DeepseekLLM (单一)          │   │
│  │    - 标准回调模式                 │   │
│  │    - 完整utterance处理           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 目标架构（升级后）

```
┌─────────────────────────────────────────┐
│              MainWindow                 │
│  ┌─────────────────────────────────┐   │
│  │        ProviderPanel   [v]      │   │  ← 新增
│  │  🔵 DeepSeek [🟢智谱][⚪OpenAI]  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │         VoiceSphere             │   │
│  │    (10+种细粒度动画状态)          │   │  ← 增强
│  │   旋转/脉冲/闪烁/波动...         │   │
│  └─────────────────────────────────┘   │
│              StatusLabel                │
│     "识别中: 你好..."                    │  ← 实时
│  ┌─────────────────────────────────┐   │
│  │       ConversationView          │   │  ← 新增
│  │  ┌──────────┐ ┌──────────┐     │   │
│  │  │ User     │ │ AI       │     │   │
│  │  │ 你好     │ │ 你好！...│     │   │
│  │  └──────────┘ └──────────┘     │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │        ControlButtons           │   │
│  │   [⚙️] [🎤] [⏹] [📊]           │   │  ← 新增设置/监控
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    │
                    │ 新信号:
                    │ - provider_changed
                    │ - asr_partial
                    │ - llm_stream_chunk
                    ▼
┌─────────────────────────────────────────┐
│      StreamingTalkerThread              │  ← 新增/替换
│  ┌─────────────────────────────────┐   │
│  │      StreamingPipeline          │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │   ConversationManager   │   │   │
│  │  │  (多Provider + 故障转移)  │   │   │
│  │  └─────────────────────────┘   │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │   IncrementalASR        │   │   │
│  │  │   (500ms滑动窗口)         │   │   │
│  │  └─────────────────────────┘   │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │   StreamingTTS          │   │   │
│  │  │   (边合成边播放)          │   │   │
│  │  └─────────────────────────┘   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 数据流变化

### 当前数据流

```
用户说话 → [等待完整utterance] → ASR → [等待完整文本] → LLM → [等待完整音频] → TTS → 播放

延迟: ~1.2秒 (串行处理)
```

### 目标数据流

```
用户说话 
    ↓
[增量ASR] ─────┬───→ "你" ──→ LLM ──→ "你好" ──→ TTS ──→ [播放片段1]
    ↓          ├───→ "好" ──→ LLM ──→ "！"   ──→ TTS ──→ [播放片段2]
[增量ASR] ─────┘    ... (流式并行)

延迟: ~0.45秒 (62%减少)
```

---

## 信号流对比

### 当前信号流

```
TalkerThread                     MainWindow
────────────                     ──────────
     │                               │
     │ user_speech_detected          │
     │──────────────────────────────>│
     │                               │
     │ asr_result("你好")            │
     │──────────────────────────────>│ 显示完整文本
     │                               │
     │ llm_thinking                  │
     │──────────────────────────────>│ 显示"思考中..."
     │                               │
     │ llm_response("你好！...")     │
     │──────────────────────────────>│ 显示完整回复
     │                               │
     │ audio_playing                 │
     │──────────────────────────────>│ 显示"说话中..."
     │                               │
     │ audio_finished                │
     │──────────────────────────────>│ 回到"监听中"
```

### 目标信号流

```
StreamingTalkerThread            MainWindow
─────────────────────            ──────────
     │                               │
     │ provider_changed("deepseek")  │
     │──────────────────────────────>│ 更新Provider显示
     │                               │
     │ asr_partial("你")             │
     │──────────────────────────────>│ 实时显示"识别中: 你"
     │                               │
     │ asr_partial("你好")           │
     │──────────────────────────────>│ 实时显示"识别中: 你好"
     │                               │
     │ llm_stream_chunk("你")        │
     │──────────────────────────────>│ 打字机效果"你"
     │                               │
     │ llm_stream_chunk("好")        │
     │──────────────────────────────>│ 打字机效果"好"
     │                               │
     │ llm_stream_chunk("！")        │
     │──────────────────────────────>│ 打字机效果"！"
     │                               │
     │ tts_chunk_ready               │
     │──────────────────────────────>│ 播放音频片段
     │                               │
     │ failover_occurred("ds","zh")  │
     │──────────────────────────────>│ 提示"已切换到智谱"
```

---

## 组件依赖图

### 当前依赖

```
main.py
  └── MainWindow
       ├── VoiceSphere
       ├── ControlButtons
       └── TalkerThread
            └── LiveTalker
                 ├── SenseVoice
                 ├── MeloTTS
                 └── DeepseekLLM
```

### 目标依赖

```
main.py
  └── MainWindow
       ├── ProviderPanel        [新增]
       ├── VoiceSphere          [增强]
       ├── ConversationView     [新增]
       ├── ControlButtons       [增强]
       ├── SettingsDialog       [新增]
       └── StreamingTalkerThread [替换]
            └── StreamingPipeline
                 ├── ConversationManager
                 │    └── ProviderPool
                 │         ├── DeepSeekProvider
                 │         ├── ZhipuProvider
                 │         ├── OpenAIProvider
                 │         └── MoonshotProvider
                 ├── IncrementalASR
                 │    └── SenseVoice
                 └── StreamingTTS
                      ├── SentenceSplitter
                      └── MeloTTS
```

---

## 修改影响范围

### 无影响（无需修改）
- ✅ `main.py` - 入口文件保持不变
- ✅ `utils/` - 工具函数
- ✅ `audio/` - 音频模块（VAD/录音/播放）

### 轻微修改
- ⚠️ `talker_thread.py` - 使用新的LLM Manager
- ⚠️ `main_window.py` - 添加新信号连接

### 中等修改
- 🔧 `widgets/voice_sphere.py` - 增加动画状态
- 🔧 `widgets/control_buttons.py` - 可能添加设置按钮

### 新建文件（推荐）
- 🆕 `streaming_talker_thread.py` - 流式线程
- 🆕 `widgets/provider_panel.py` - Provider面板
- 🆕 `widgets/conversation_view.py` - 对话视图
- 🆕 `widgets/performance_panel.py` - 性能监控
- 🆕 `dialogs/settings_dialog.py` - 设置对话框

---

## 向后兼容方案

```python
# 在 main_window.py 中

class MainWindow(QMainWindow):
    def __init__(self):
        # 配置选择使用哪种模式
        self.use_streaming = True  # 可配置
        
        if self.use_streaming:
            from streaming_talker_thread import StreamingTalkerThread
            self.talker_thread = StreamingTalkerThread()
            self._setup_streaming_connections()
        else:
            from talker_thread import TalkerThread
            self.talker_thread = TalkerThread()
            self._setup_legacy_connections()
```

---

## 性能对比预期

| 指标 | 当前 | 目标 | 改进 |
|------|------|------|------|
| 首帧播放延迟 | 1200ms | 450ms | **-62%** |
| ASR响应 | 300ms | 150ms | **-50%** |
| LLM首token | 500ms | 200ms | **-60%** |
| TTS首帧 | 400ms | 100ms | **-75%** |
| UI帧率 | 30fps | 60fps | **+100%** |

---

*图表版本: 1.0*  
*更新日期: 2025-02-07*
