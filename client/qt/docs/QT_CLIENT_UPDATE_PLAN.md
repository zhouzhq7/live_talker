# Qt客户端更新计划

> 基于Phase 4-5后端改动的Qt客户端升级方案

---

## 📋 后端改动概览

### Phase 4: 多LLM Provider支持
- 支持DeepSeek、智谱、OpenAI、Moonshot多Provider
- 自动故障转移机制
- Provider健康状态监控
- Token使用统计

### Phase 5: 流式处理优化
- IncrementalASR: 增量语音识别（500ms窗口）
- SentenceSplitter: 句子级分割
- StreamingTTS: 边合成边播放
- StreamingPipeline: 统一流式Pipeline
- 端到端延迟优化至<1秒

---

## 🎯 修改目标

1. **集成多Provider支持** - 显示当前Provider，支持切换
2. **集成流式Pipeline** - 更低延迟，实时反馈
3. **增强状态显示** - 细粒度状态（ASR/LLM/TTS子状态）
4. **实时文本显示** - 流式输出实时展示
5. **性能监控** - 延迟统计和优化分析

---

## 📅 实施计划

### Phase A: 多Provider支持 (2-3天)

#### A.1 新增Provider管理模块
**文件**: `widgets/provider_panel.py` (新建)

```python
class ProviderPanel(QWidget):
    """Provider选择和管理面板"""
    
    功能:
    - 显示当前主Provider
    - 显示备用Provider列表
    - Provider健康状态指示器
    - 点击切换Provider（可选）
    
    UI组件:
    - 当前Provider标签（带颜色标识）
    - 备用Provider列表（小圆点状态）
    - 故障转移计数显示
```

**文件**: `models/provider_model.py` (新建)

```python
class ProviderModel(QObject):
    """Provider数据模型"""
    
    信号:
    - primary_changed(str)
    - fallback_changed(list)
    - health_updated(dict)
    - cost_updated(float)
```

#### A.2 修改TalkerThread
**文件**: `talker_thread.py`

```python
# 新增信号
provider_changed = pyqtSignal(str)      # Provider切换
provider_health = pyqtSignal(dict)      # 健康状态
failover_occurred = pyqtSignal(str, str) # 故障转移 (from, to)
token_usage = pyqtSignal(int, int)      # token使用量 (prompt, completion)

# 修改run()方法
- 使用ConversationManager替代直接LiveTalker
- 监听Provider切换事件
- 定期获取Provider健康状态
```

#### A.3 修改MainWindow集成
**文件**: `main_window.py`

```python
# 新增UI元素
self.provider_panel = ProviderPanel()

# 新增槽函数
@pyqtSlot(str)
def onProviderChanged(self, provider: str):
    '''Provider切换'''
    
@pyqtSlot(str, str)
def onFailoverOccurred(self, from_provider: str, to_provider: str):
    '''故障转移通知'''
    # 显示临时提示："已切换到XXX"
```

---

### Phase B: 流式Pipeline集成 (3-4天)

#### B.1 创建StreamingTalkerThread
**文件**: `streaming_talker_thread.py` (新建)

```python
class StreamingTalkerThread(QThread):
    """
    流式处理版本的TalkerThread
    替代原有的TalkerThread
    """
    
    # 新增流式信号
    asr_partial = pyqtSignal(str)       # ASR部分结果
    llm_stream_chunk = pyqtSignal(str)  # LLM流式chunk
    llm_sentence_complete = pyqtSignal(str)  # LLM完整句子
    tts_chunk_ready = pyqtSignal()      # TTS片段就绪
    
    # 细粒度状态
    state_asr_recognizing = pyqtSignal()
    state_llm_generating = pyqtSignal() 
    state_tts_synthesizing = pyqtSignal()
    
    def run(self):
        # 使用StreamingPipeline替代LiveTalker
        from core.streaming import StreamingPipeline
        
        self.pipeline = StreamingPipeline(...)
        
        # 连接流式回调
        self.pipeline.on_partial_asr = self._on_asr_partial
        self.pipeline.on_partial_llm = self._on_llm_chunk
        
    def _on_asr_partial(self, text: str):
        self.asr_partial.emit(text)
        
    def _on_llm_chunk(self, text: str):
        self.llm_stream_chunk.emit(text)
```

#### B.2 创建实时对话显示组件
**文件**: `widgets/conversation_view.py` (新建)

```python
class ConversationView(QScrollArea):
    """
    实时对话显示区域
    类似聊天界面
    """
    
    功能:
    - 用户消息气泡（右侧）
    - AI消息气泡（左侧）
    - 流式文本实时更新（打字机效果）
    - 消息时间戳
    - 复制消息功能
    
    UI:
    - QListView或自定义QWidget
    - 气泡样式（圆角矩形）
    - 自动滚动到底部
```

#### B.3 增强VoiceSphere状态
**文件**: `widgets/voice_sphere.py`

```python
# 新增细粒度状态
STATE_ASR = "asr_recognizing"      # 识别中（旋转动画）
STATE_LLM_FIRST = "llm_first"      # LLM首token（闪烁）
STATE_LLM_STREAM = "llm_stream"    # LLM流式（脉冲）
STATE_TTS_CHUNK = "tts_chunk"      # TTS分段（波动）

# 动画增强
def setDetailedState(self, state: str, progress: float = 0):
    '''
    设置详细状态
    progress: 0-1 进度（用于流式显示）
    '''
```

#### B.4 修改MainWindow集成流式功能
**文件**: `main_window.py`

```python
def setupUI(self):
    # 新增对话显示区域
    self.conversation_view = ConversationView()
    
    # 替换TalkerThread为StreamingTalkerThread
    # self.talker_thread = TalkerThread()  # 旧
    self.talker_thread = StreamingTalkerThread()  # 新

def setupConnections(self):
    # 流式信号连接
    self.talker_thread.asr_partial.connect(self.onASRPartial)
    self.talker_thread.llm_stream_chunk.connect(self.onLLMStreamChunk)
    
@pyqtSlot(str)
def onASRPartial(self, text: str):
    '''ASR部分结果 - 实时显示识别中'''
    self.status_label.setText(f"识别中: {text}...")
    
@pyqtSlot(str)
def onLLMStreamChunk(self, text: str):
    '''LLM流式chunk - 实时更新AI回复'''
    self.conversation_view.appendAIChunk(text)
```

---

### Phase C: 性能监控面板 (1-2天)

#### C.1 创建性能监控面板
**文件**: `widgets/performance_panel.py` (新建)

```python
class PerformancePanel(QWidget):
    """性能监控面板（可折叠）"""
    
    显示内容:
    - 当前延迟指标（ASR/LLM/TTS）
    - Token使用统计
    - Provider成功率
    - 实时RTF图表（可选）
    
    UI:
    - 小字体显示
    - 可折叠/展开
    - 默认隐藏（高级用户可用）
```

---

### Phase D: 配置界面 (2天)

#### D.1 创建设置对话框
**文件**: `dialogs/settings_dialog.py` (新建)

```python
class SettingsDialog(QDialog):
    """应用设置对话框"""
    
    设置项:
    - Provider选择（下拉框）
    - 备用Provider勾选
    - API Key输入（密码框）
    - 模型选择
    - 温度参数调节
    - 语音选择（TTS）
    
    功能:
    - 测试API Key按钮
    - 保存到配置文件
    - 实时生效/重启生效提示
```

---

## 📁 文件变更清单

### 新建文件
```
client/qt/
├── models/
│   ├── __init__.py
│   └── provider_model.py          # Provider数据模型
├── widgets/
│   ├── provider_panel.py          # Provider管理面板
│   ├── conversation_view.py       # 实时对话显示
│   └── performance_panel.py       # 性能监控面板
├── dialogs/
│   ├── __init__.py
│   └── settings_dialog.py         # 设置对话框
├── streaming_talker_thread.py     # 流式TalkerThread
└── docs/
    └── QT_CLIENT_UPDATE_PLAN.md   # 本文档
```

### 修改文件
```
client/qt/
├── talker_thread.py               # 支持多Provider（或废弃，使用新版）
├── main_window.py                 # 集成新组件
├── widgets/voice_sphere.py        # 增强动画状态
└── main.py                        # 可能无需修改
```

---

## 🎨 UI布局更新

### 新布局设计

```
┌─────────────────────────────────────┐
│  Live Talker                    [≡] │  ← 标题栏 + 菜单按钮
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │  Provider: [🔵 DeepSeek]    │   │  ← 新增Provider面板
│  │  备用: [⚪ Zhipu] [⚪ OpenAI]  │   │
│  └─────────────────────────────┘   │
├─────────────────────────────────────┤
│                                     │
│         ╭───────────╮              │
│        ╱   Voice    ╲             │  ← VoiceSphere
│       │   Sphere    │             │    (增强动画)
│        ╲____________╱             │
│                                     │
│    "识别中: 你好..."                 │  ← 状态文字
│                                     │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │ User: 你好                   │   │  ← 新增对话区域
│  │                              │   │    (滚动区域)
│  │ AI: 你好！很高兴见到你...     │   │
│  │      [正在输入...]           │   │
│  └─────────────────────────────┘   │
├─────────────────────────────────────┤
│  [🎤] [⏹]                          │  ← 控制按钮
└─────────────────────────────────────┘
```

---

## 📊 优先级与排期

| 阶段 | 功能 | 优先级 | 工时 | 依赖 |
|------|------|--------|------|------|
| A1 | Provider管理模块 | P0 | 1d | - |
| A2 | TalkerThread改造 | P0 | 1d | A1 |
| A3 | Provider面板UI | P0 | 0.5d | A2 |
| B1 | StreamingTalkerThread | P0 | 2d | - |
| B2 | ConversationView | P1 | 1d | B1 |
| B3 | VoiceSphere增强 | P1 | 0.5d | B1 |
| B4 | MainWindow集成 | P0 | 1d | B1-B3 |
| C1 | 性能面板 | P2 | 1d | B4 |
| D1 | 设置对话框 | P1 | 1.5d | A1 |

**总工时**: ~9.5天

**推荐开发顺序**:
1. Week 1: Phase A (多Provider) + B1 (流式Thread)
2. Week 2: Phase B2-B4 (流式UI) + D1 (配置)
3. Week 3: Phase C (性能监控) + 测试优化

---

## 🔄 向后兼容

### 兼容性策略
- 保留原有`TalkerThread`作为备选
- 通过配置切换流式/非流式模式
- 原有信号保持兼容（新增信号扩展）

### 配置切换
```python
# config.py
QT_CLIENT_CONFIG = {
    "use_streaming": True,  # True: 使用StreamingTalkerThread
                            # False: 使用原有TalkerThread
    "show_provider_panel": True,
    "show_conversation_view": True,
    "enable_performance_monitor": False,
}
```

---

## ✅ 验收标准

### 功能验收
- [ ] 显示当前Provider和健康状态
- [ ] Provider故障时自动切换并提示用户
- [ ] ASR实时显示部分识别结果
- [ ] LLM流式输出实时显示（打字机效果）
- [ ] 对话历史可滚动查看
- [ ] 可配置Provider和API Key

### 性能验收
- [ ] UI响应流畅（无卡顿）
- [ ] 流式延迟 < 1秒（从说话到听到回复）
- [ ] 内存占用稳定（无泄漏）

### 兼容性验收
- [ ] 支持PyQt6 6.6.0+
- [ ] 支持Windows/macOS/Linux
- [ ] 向后兼容非流式模式

---

## 📝 注意事项

1. **线程安全**: 所有UI更新必须在主线程执行（使用信号-槽）
2. **资源管理**: 确保线程和定时器正确清理
3. **错误处理**: Provider失败时优雅降级
4. **性能考虑**: 流式更新频率控制（避免UI刷新过频）

---

## 🚀 快速开始（开发）

```bash
# 1. 创建新分支
git checkout -b feature/qt-streaming

# 2. 安装依赖
cd client/qt
pip install PyQt6>=6.6.0

# 3. 按Phase A开始开发
# ...

# 4. 测试
python main.py
```

---

*文档版本: 1.0*  
*更新日期: 2025-02-07*  
*状态: 待开发*
