# API Key 申请指南

> 配置 Live Talker 多 LLM Provider 支持

---

## 快速配置

在项目根目录创建 `.env` 文件：

```bash
# 推荐：DeepSeek (主力，性价比高)
DEEPSEEK_API_KEY=your_deepseek_key_here

# 推荐：智谱 GLM-4-Flash (免费备用)
ZHIPU_API_KEY=your_zhipu_key_here

# 可选：OpenAI (国际备用)
OPENAI_API_KEY=your_openai_key_here

# 可选：Moonshot/Kimi (超长文本)
MOONSHOT_API_KEY=your_moonshot_key_here
```

---

## 详细申请步骤

### 1. 智谱 AI (GLM-4-Flash 免费) ⭐ 推荐优先申请

**特点：**
- GLM-4-Flash 完全免费
- 128K 上下文
- 中文能力强

**申请步骤：**

1. 访问 https://open.bigmodel.cn
2. 点击右上角「登录」，用手机号注册
3. 完成实名认证（需要身份证）
4. 进入「API Keys」页面
5. 点击「添加新的 API key"
6. 复制生成的 key（格式：`xxx.xxx`）

**免费额度：**
- GLM-4-Flash：无限制免费
- GLM-4-Air：新用户有赠送额度

---

### 2. DeepSeek (价格低) ⭐ 推荐主力

**特点：**
- 价格最低：¥1/1M tokens
- 中文能力强
- 推理能力好

**申请步骤：**

1. 访问 https://platform.deepseek.com
2. 点击「去登录」，支持手机号/邮箱
3. 完成注册后自动赠送 **¥10** 额度
4. 进入左侧「API Keys」菜单
5. 点击「创建 API key"
6. 复制生成的 key（格式：`sk-xxx`）

**价格：**
- 输入：¥1/1M tokens
- 输出：¥2/1M tokens

---

### 3. OpenAI (国际稳定)

**特点：**
- GPT-4o-mini 性价比高
- 全球 CDN，延迟低
- 需要海外支付方式

**申请步骤：**

1. 访问 https://platform.openai.com
2. 注册账号（需要海外手机号验证）
3. 绑定支付方式（信用卡）
4. 进入 Settings → API keys
5. 创建新的 secret key
6. 复制生成的 key（格式：`sk-xxx`）

**注意：**
- 新账号有 $5-18 免费额度
- 国内访问需要代理
- 需要海外支付方式

---

### 4. Moonshot / Kimi (超长上下文)

**特点：**
- 128K 超长上下文
- 注册送 ¥15 额度
- 支持联网搜索

**申请步骤：**

1. 访问 https://platform.moonshot.cn
2. 点击「立即登录」，手机号注册
3. 自动获得 ¥15 赠送额度
4. 进入「API Key 管理"
5. 点击「新建"
6. 复制生成的 key

**价格：**
- moonshot-v1-8k: ¥12/12 per 1M tokens
- moonshot-v1-32k: ¥24/24 per 1M tokens
- moonshot-v1-128k: ¥60/60 per 1M tokens

---

## 测试配置

### 环境变量方式

```bash
export DEEPSEEK_API_KEY="sk-your-key"
export ZHIPU_API_KEY="your.zhipu.key"
export OPENAI_API_KEY="sk-your-openai-key"
export MOONSHOT_API_KEY="your-moonshot-key"
```

### Python 代码方式

```python
from llm import ConversationManager, ConversationConfig

config = ConversationConfig(
    primary_provider="deepseek",
    deepseek_api_key="sk-your-key",
    zhipu_api_key="your.zhipu.key",
    enable_fallback=True,
    fallback_providers=["zhipu"]
)

manager = ConversationManager(config)
```

---

## 验证 API Key

运行测试脚本：

```bash
# 测试所有配置的Provider
python tests/integration/phase4_llm/test_api_keys.py
```

---

## 故障排查

### 智谱 API 返回 401
- 检查 key 格式是否正确（包含点号）
- 确认已完成实名认证

### DeepSeek 返回 401
- 检查 key 是否以 `sk-` 开头
- 确认账户还有余额

### OpenAI 返回 429
- 超出速率限制，等待一会儿再试
- 或切换到其他 Provider

### 所有 Provider 都失败
- 检查网络连接
- 检查 `.env` 文件格式
- 查看日志 `logs/live_talker.log`

---

## 成本优化建议

### 方案 1：完全免费
```python
config = ConversationConfig(
    primary_provider="zhipu",
    zhipu_model="glm-4-flash",  # 免费
    enable_fallback=False
)
```

### 方案 2：低成本高可用
```python
config = ConversationConfig(
    primary_provider="deepseek",
    enable_fallback=True,
    fallback_providers=["zhipu"],
    deepseek_api_key="sk-xxx",
    zhipu_api_key="xxx.xxx",  # 免费备用
    zhipu_model="glm-4-flash"
)
```

### 方案 3：国际用户
```python
config = ConversationConfig(
    primary_provider="openai",
    enable_fallback=True,
    fallback_providers=["zhipu"],
    openai_model="gpt-4o-mini",
    openai_api_key="sk-xxx",
    zhipu_api_key="xxx.xxx"  # 国内备用
)
```

---

## 安全提示

⚠️ **永远不要：**
- 将 API Key 提交到 Git
- 在公开代码中硬编码 Key
- 分享你的 `.env` 文件

✅ **正确做法：**
- 使用 `.env` 文件（已添加到 `.gitignore`）
- 定期轮换 API Key
- 使用最小权限原则

---

*最后更新: 2025-02-07*
