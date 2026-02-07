# Phase 4: LLM多提供商支持 - 实施计划

> 基于市场调研的实施方案
> 状态: 待开发

---

## 1. 目标调整

原目标: Ollama本地LLM支持  
**新目标**: 多云端LLM API提供商支持 + 故障转移

理由:
- 无需强力硬件
- 成本可控 (¥0-20/月)
- 稳定性高
- 集成快速

---

## 2. 实施范围

### 2.1 核心功能
- [ ] 多LLM Provider支持 (DeepSeek/智谱/OpenAI/Gemini)
- [ ] OpenAI兼容接口封装
- [ ] 自动故障转移机制
- [ ] 配置热更新

### 2.2 支持的提供商
| 优先级 | 提供商 | 模型 | 原因 |
|--------|--------|------|------|
| P0 | DeepSeek | deepseek-chat | 当前主力，性价比高 |
| P0 | 智谱 | glm-4-flash | 免费备用 |
| P1 | 智谱 | glm-4-air | 低价高质量备用 |
| P1 | OpenAI | gpt-4o-mini | 国际备选 |
| P2 | Google | gemini-1.5-flash | 免费国际备选 |
| P2 | 讯飞 | 星火Lite | 免费国产备选 |

---

## 3. 架构设计

### 3.1 类图

```python
                    ┌─────────────────┐
                    │   LLMProvider   │
                    │   (抽象基类)     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ DeepSeekProvider│ │  ZhipuProvider  │ │  OpenAIProvider │
│                 │ │                 │ │                 │
│ - chat()        │ │ - chat()        │ │ - chat()        │
│ - stream_chat() │ │ - stream_chat() │ │ - stream_chat() │
│ - check_health()│ │ - check_health()│ │ - check_health()│
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ LLMProviderPool │
                    │                 │
                    │ - providers[]   │
                    │ - fallback_chain│
                    │ - route_request()│
                    └─────────────────┘
```

### 3.2 故障转移流程

```
用户请求
    │
    ▼
┌─────────────┐
│ DeepSeek    │───失败?───┐
│ (主)        │           │
└─────────────┘           ▼
    成功          ┌─────────────┐
    │             │ 智谱Air     │───失败?───┐
    ▼             │ (备用1)     │           │
返回结果          └─────────────┘           ▼
                      成功          ┌─────────────┐
                      │             │ 智谱Flash   │───失败?───┐
                      ▼             │ (备用2)     │           │
                  返回结果          └─────────────┘           ▼
                                        成功          ┌─────────────┐
                                        │             │ 本地Mock    │
                                        ▼             │ (最终备用)  │
                                    返回结果          └─────────────┘
```

---

## 4. 任务分解

### 4.1 Phase 4-1: 基础架构 (2-3天)
| # | 任务 | 工时 | 状态 |
|---|------|------|------|
| 4.1.1 | 创建 `llm/base.py` 抽象基类 | 2h | 🔵 |
| 4.1.2 | 实现 `DeepSeekProvider` | 2h | 🔵 |
| 4.1.3 | 实现 `LLMProviderPool` | 3h | 🔵 |
| 4.1.4 | 更新 `config.py` LLM配置 | 1h | 🔵 |
| 4.1.5 | 修改 `core/conversation.py` 集成 | 3h | 🔵 |

### 4.2 Phase 4-2: 智谱支持 (1-2天)
| # | 任务 | 工时 | 状态 |
|---|------|------|------|
| 4.2.1 | 实现 `ZhipuProvider` | 2h | 🔵 |
| 4.2.2 | 测试 GLM-4-Flash (免费) | 1h | 🔵 |
| 4.2.3 | 测试 GLM-4-Air | 1h | 🔵 |
| 4.2.4 | 故障转移测试 | 2h | 🔵 |

### 4.3 Phase 4-3: OpenAI兼容 (1-2天)
| # | 任务 | 工时 | 状态 |
|---|------|------|------|
| 4.3.1 | 实现 `OpenAIProvider` | 2h | 🔵 |
| 4.3.2 | 实现 `GeminiProvider` (OpenAI适配) | 2h | 🔵 |
| 4.3.3 | 测试 GPT-4o-mini | 1h | 🔵 |
| 4.3.4 | 测试 Gemini Flash | 1h | 🔵 |

### 4.4 Phase 4-4: 测试与优化 (2天)
| # | 任务 | 工时 | 状态 |
|---|------|------|------|
| 4.4.1 | 单元测试 | 4h | 🔵 |
| 4.4.2 | 故障转移集成测试 | 3h | 🔵 |
| 4.4.3 | 性能基准测试 | 2h | 🔵 |
| 4.4.4 | 文档更新 | 2h | 🔵 |

**总工时估算**: 6-9 天

---

## 5. 关键代码示例

### 5.1 配置示例

```python
# config.py
@dataclass
class LLMConfig:
    """LLM配置 - 支持多提供商"""
    
    # 主提供商
    primary_provider: str = "deepseek"
    
    # 故障转移链 (按优先级)
    fallback_chain: List[str] = field(default_factory=lambda: [
        "deepseek",
        "zhipu_air",
        "zhipu_flash",
    ])
    
    # DeepSeek配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 30
    
    # 智谱配置
    zhipu_api_key: str = ""
    zhipu_model_air: str = "glm-4-air"
    zhipu_model_flash: str = "glm-4-flash"
    zhipu_timeout: int = 30
    
    # OpenAI配置 (可选)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout: int = 30
```

### 5.2 使用示例

```python
# 初始化
config = LLMConfig(
    deepseek_api_key="sk-xxx",
    zhipu_api_key="xxx.xxx",
)

provider_pool = LLMProviderPool(config)

# 自动选择可用提供商
response = provider_pool.chat(
    messages=[
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7
)

# 流式输出
for chunk in provider_pool.stream_chat(messages):
    print(chunk.content, end="", flush=True)
```

---

## 6. 成本预估

### 6.1 开发成本
| 项目 | 估算 |
|------|------|
| 开发工时 | 6-9 天 |
| 测试成本 | ¥10-20 |
| 总人力 | 1人 |

### 6.2 运营成本 (月度)
| 场景 | 配置 | 成本 |
|------|------|------|
| 最小成本 | 智谱Flash(免费) | ¥0 |
| 推荐配置 | DeepSeek + Flash备用 | ¥5-10 |
| 高可用 | 多厂商备份 | ¥10-20 |

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| API变更 | 中 | 中 | 封装抽象层 |
| 免费服务限速 | 高 | 低 | 准备付费备用 |
| 网络不稳定 | 中 | 中 | 多厂商备份 |
| 额度耗尽 | 低 | 高 | 监控告警 |

---

## 8. 验收标准

- [ ] 支持至少3个LLM提供商
- [ ] 主提供商故障时自动切换 < 5秒
- [ ] 所有提供商通过单元测试
- [ ] 故障转移场景集成测试通过
- [ ] 文档更新完整
- [ ] 成本控制在 ¥20/月以内

---

## 9. 下一步

1. 确认实施计划
2. 申请各平台API Key
3. 开始Phase 4-1开发

---

*文档版本: 1.0*  
*创建时间: 2025-02-07*  
*状态: 待评审*
