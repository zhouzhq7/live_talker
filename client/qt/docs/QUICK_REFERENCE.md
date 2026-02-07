# Qt客户端修改快速参考

## 🔑 关键修改点（一句话总结）

| 后端改动 | Qt客户端对应修改 |
|---------|----------------|
| **多Provider支持** | 新增Provider面板，显示当前Provider和健康状态 |
| **故障转移** | 显示故障转移提示，记录切换历史 |
| **流式ASR** | 实时显示"识别中: xxx..." |
| **流式LLM** | 打字机效果显示AI回复 |
| **流式TTS** | 逐句播放，显示播放进度 |
| **延迟优化** | 整体响应更快，无需特别修改 |

---

## 📋 最小修改方案（MVP）

如果只需要基本功能，最少修改：

### 1. 修改 `talker_thread.py` (30分钟)
```python
# 新增信号
provider_changed = pyqtSignal(str)  # 当前使用的Provider名称

# 在run()中
# 使用新的ConversationManager
from llm import ConversationManager, ConversationConfig
config = ConversationConfig(...)
self.conversation = ConversationManager(config)
# 获取当前provider并发射信号
```

### 2. 修改 `main_window.py` (30分钟)
```python
# 在状态栏显示Provider
@pyqtSlot(str)
def onProviderChanged(self, provider: str):
    self.setWindowTitle(f"Live Talker - {provider}")
```

**总用时**: ~1小时

---

## 🎨 完整功能修改清单

### 必须修改（P0）
- [ ] `talker_thread.py` - 使用新的LLM Manager
- [ ] `main_window.py` - 显示Provider信息

### 强烈建议（P1）
- [ ] 新建 `streaming_talker_thread.py` - 流式支持
- [ ] 新建 `conversation_view.py` - 对话显示
- [ ] 修改 `voice_sphere.py` - 更多动画状态

### 可选增强（P2）
- [ ] 新建 `provider_panel.py` - Provider管理面板
- [ ] 新建 `settings_dialog.py` - 设置界面
- [ ] 新建 `performance_panel.py` - 性能监控

---

## 🔄 信号对照表

### 原有信号（保持不变）
```python
user_speech_detected = pyqtSignal()  # 用户开始说话
asr_result = pyqtSignal(str)         # ASR最终结果
llm_thinking = pyqtSignal()          # LLM开始思考
llm_response = pyqtSignal(str)       # LLM完整回复
tts_synthesizing = pyqtSignal()      # TTS合成中
audio_playing = pyqtSignal()         # 开始播放
audio_finished = pyqtSignal()        # 播放完成
error_occurred = pyqtSignal(str)     # 错误
state_changed = pyqtSignal(str)      # 状态变化
```

### 新增信号（Phase 4-5）
```python
# Phase 4: 多Provider
provider_changed = pyqtSignal(str)          # Provider切换
provider_health = pyqtSignal(dict)          # 健康状态
failover_occurred = pyqtSignal(str, str)    # 故障转移 (from, to)
token_usage = pyqtSignal(int, int)          # Token使用

# Phase 5: 流式
asr_partial = pyqtSignal(str)               # ASR部分结果
llm_stream_chunk = pyqtSignal(str)          # LLM流式chunk
llm_sentence_complete = pyqtSignal(str)     # LLM完整句子
```

---

## 📊 状态对照表

### 原有状态
```python
"idle"          # 待机
"initializing"  # 初始化
"listening"     # 监听中
"thinking"      # 思考中
"speaking"      # 说话中
```

### 新增状态（流式）
```python
"asr_recognizing"   # ASR识别中（实时）
"llm_generating"    # LLM生成中（流式）
"tts_chunk"         # TTS分段合成
```

---

## 🎨 UI颜色对照表

### Provider颜色标识
```python
{
    "deepseek": "#2196F3",  # 蓝色
    "zhipu": "#4CAF50",     # 绿色
    "openai": "#FF9800",    # 橙色
    "moonshot": "#9C27B0",  # 紫色
}
```

### 健康状态颜色
```python
{
    "healthy": "#4CAF50",     # 绿色
    "degraded": "#FFC107",    # 黄色
    "unavailable": "#F44336", # 红色
}
```

---

## 💡 常见修改场景

### 场景1: 只显示当前Provider
```python
# main_window.py
@pyqtSlot(str)
def onProviderChanged(self, provider: str):
    self.status_label.setText(f"使用: {provider}")
```

### 场景2: 流式显示LLM输出
```python
# main_window.py
@pyqtSlot(str)
def onLLMStreamChunk(self, chunk: str):
    # 追加到当前显示
    current = self.status_label.text()
    self.status_label.setText(current + chunk)
```

### 场景3: 故障转移提示
```python
# main_window.py
@pyqtSlot(str, str)
def onFailoverOccurred(self, from_p: str, to_p: str):
    QMessageBox.information(self, "Provider切换", 
        f"已从 {from_p} 切换到 {to_p}")
```

---

## ⚡ 性能优化提示

1. **流式更新频率控制**
```python
# 避免每帧都更新UI
self._update_counter += 1
if self._update_counter % 3 == 0:  # 每3帧更新一次
    self.status_label.setText(text)
```

2. **文本长度限制**
```python
# 避免过长文本影响性能
if len(text) > 500:
    text = text[:500] + "..."
```

3. **使用update()而非repaint()**
```python
# 正确
self.voice_sphere.update()

# 避免
self.voice_sphere.repaint()
```

---

## 🧪 测试检查清单

- [ ] 启动正常，无报错
- [ ] Provider显示正确
- [ ] 对话流程完整（听->识别->思考->回复）
- [ ] 打断功能正常
- [ ] 关闭窗口时资源正确释放
- [ ] 长时间运行无内存泄漏

---

## 📚 相关文档

- `QT_CLIENT_UPDATE_PLAN.md` - 详细实施计划
- `../../../docs/API_KEY_SETUP.md` - API Key配置
- `../../../docs/improvement_plan/phase4_llm_implementation_plan.md` - Phase 4详情
- `../../../docs/improvement_plan/phase5_streaming_optimization.md` - Phase 5详情

---

## 🆘 故障排查

### 问题1: UI卡顿
**原因**: 在主线程执行耗时操作
**解决**: 确保所有耗时操作在TalkerThread中执行

### 问题2: Provider不显示
**原因**: 信号未连接
**解决**: 检查`setupConnections()`中是否连接了`provider_changed`

### 问题3: 流式文本不更新
**原因**: 跨线程更新UI
**解决**: 使用信号-槽机制，不要直接操作UI

---

*最后更新: 2025-02-07*
