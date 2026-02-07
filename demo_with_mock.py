#!/usr/bin/env python3
"""
多 Provider LLM 演示 - 使用 Mock 数据

展示系统架构和故障转移功能，无需真实 API Key
"""

import os
import sys
import time
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(__file__))

from llm.provider import LLMProvider, LLMProviderPool, LLMResponse


class MockProvider(LLMProvider):
    """Mock provider for demonstration"""
    
    def __init__(self, name, model, should_fail=False, latency_ms=100):
        super().__init__(name=name, model=model)
        self.should_fail = should_fail
        self.latency_ms = latency_ms
        self._is_initialized = True
    
    def initialize(self):
        return True
    
    def chat_completion(self, messages, stream=False, **kwargs):
        if self.should_fail:
            raise Exception(f"{self.name} is down")
        
        time.sleep(self.latency_ms / 1000)  # Simulate latency
        
        user_msg = messages[-1].get("content", "") if messages else ""
        
        responses = {
            "deepseek": f"[DeepSeek] 收到你的消息: '{user_msg}'。我是DeepSeek，价格低且中文能力强！",
            "zhipu": f"[智谱AI] 收到: '{user_msg}'。我是GLM-4，支持128K上下文！",
            "openai": f"[OpenAI] Received: '{user_msg}'. I'm GPT-4o-mini, fast and reliable!",
        }
        
        response_text = responses.get(self.name, f"[{self.name}] Echo: {user_msg}")
        
        return LLMResponse(
            text=response_text,
            provider=self.name,
            model=self.model,
            latency_ms=self.latency_ms,
            prompt_tokens=len(user_msg),
            completion_tokens=len(response_text),
            total_tokens=len(user_msg) + len(response_text)
        )
    
    def chat_completion_stream(self, messages, **kwargs):
        response = self.chat_completion(messages, stream=True)
        words = response.text.split()
        for word in words:
            yield word + " "


def demo_basic():
    """Basic provider pool demo"""
    print("="*70)
    print("🎯 Demo 1: 基础 Provider Pool")
    print("="*70)
    
    # Create providers
    providers = [
        MockProvider("deepseek", "deepseek-chat", latency_ms=150),
        MockProvider("zhipu", "glm-4-flash", latency_ms=200),
        MockProvider("openai", "gpt-4o-mini", latency_ms=100),
    ]
    
    # Create pool
    pool = LLMProviderPool(
        providers=providers,
        fallback_chain=["deepseek", "zhipu", "openai"]
    )
    
    print("\n📋 Provider 列表:")
    for name, provider in pool.providers.items():
        print(f"   • {name}: {provider.model}")
    
    print(f"\n🔗 故障转移链: {' -> '.join(pool.fallback_chain)}")
    
    # Test chat
    messages = [{"role": "user", "content": "你好，请介绍一下自己"}]
    
    print("\n💬 发送消息...")
    response = pool.chat_completion(messages)
    
    print(f"\n✅ 响应:")
    print(f"   Provider: {response.provider}")
    print(f"   Model: {response.model}")
    print(f"   Latency: {response.latency_ms:.0f}ms")
    print(f"   Content: {response.text}")


def demo_failover():
    """Failover demonstration"""
    print("\n" + "="*70)
    print("🔄 Demo 2: 自动故障转移")
    print("="*70)
    
    # Create providers with primary failing
    providers = [
        MockProvider("deepseek", "deepseek-chat", should_fail=True),
        MockProvider("zhipu", "glm-4-flash"),
        MockProvider("openai", "gpt-4o-mini"),
    ]
    
    pool = LLMProviderPool(
        providers=providers,
        fallback_chain=["deepseek", "zhipu", "openai"]
    )
    
    print("\n📋 配置:")
    print("   • DeepSeek: ❌ 模拟故障")
    print("   • Zhipu: ✅ 正常")
    print("   • OpenAI: ✅ 正常")
    
    messages = [{"role": "user", "content": "测试故障转移"}]
    
    print("\n💬 发送消息（DeepSeek 应该会失败）...")
    response = pool.chat_completion(messages)
    
    print(f"\n✅ 响应来自: {response.provider}")
    print(f"   故障转移次数: {pool.stats['failover_count']}")
    print(f"   内容: {response.text}")


def demo_health_check():
    """Health check demonstration"""
    print("\n" + "="*70)
    print("🏥 Demo 3: 健康检查")
    print("="*70)
    
    providers = [
        MockProvider("deepseek", "deepseek-chat"),
        MockProvider("zhipu", "glm-4-flash", should_fail=True),
        MockProvider("openai", "gpt-4o-mini"),
    ]
    
    pool = LLMProviderPool(providers=providers)
    
    print("\n📊 运行健康检查...")
    for name, provider in pool.providers.items():
        health = provider.check_health()
        status_icon = "✅" if health.status.value == "healthy" else "❌"
        print(f"   {status_icon} {name}: {health.status.value}")


def demo_stats():
    """Statistics demonstration"""
    print("\n" + "="*70)
    print("📈 Demo 4: 统计信息")
    print("="*70)
    
    providers = [
        MockProvider("deepseek", "deepseek-chat"),
        MockProvider("zhipu", "glm-4-flash"),
    ]
    
    pool = LLMProviderPool(providers=providers)
    
    # Simulate some requests
    messages = [{"role": "user", "content": "测试"}]
    
    for i in range(5):
        pool.chat_completion(messages)
    
    # Simulate a failure
    pool.providers["deepseek"].should_fail = True
    pool.chat_completion(messages)  # Should failover
    
    print("\n📊 Pool 统计:")
    print(f"   总请求: {pool.stats['total_requests']}")
    print(f"   成功: {pool.stats['successful_requests']}")
    print(f"   失败: {pool.stats['failed_requests']}")
    print(f"   故障转移: {pool.stats['failover_count']}")
    
    print("\n📊 Provider 详情:")
    stats = pool.get_stats()
    for name, info in stats['providers'].items():
        print(f"   • {name}:")
        print(f"      状态: {info['health']['status']}")
        print(f"      成功率: {info['health']['success_rate']:.1%}")


def demo_streaming():
    """Streaming demonstration"""
    print("\n" + "="*70)
    print("🌊 Demo 5: 流式响应")
    print("="*70)
    
    providers = [MockProvider("deepseek", "deepseek-chat", latency_ms=50)]
    pool = LLMProviderPool(providers=providers)
    
    messages = [{"role": "user", "content": "你好"}]
    
    print("\n💬 流式接收响应...")
    print("🤖 AI: ", end="", flush=True)
    
    for chunk in pool.chat_completion_stream(messages):
        print(chunk, end="", flush=True)
        time.sleep(0.05)  # Simulate streaming delay
    
    print("\n")


def demo_cost_comparison():
    """Cost comparison"""
    print("\n" + "="*70)
    print("💰 Demo 6: 成本对比（基于 100万 tokens）")
    print("="*70)
    
    costs = {
        "智谱 GLM-4-Flash": {"input": 0, "output": 0, "note": "完全免费"},
        "DeepSeek": {"input": 1, "output": 2, "note": "价格最低"},
        "智谱 GLM-4-Air": {"input": 1, "output": 1, "note": "性价比高"},
        "OpenAI GPT-4o-mini": {"input": 1.09, "output": 4.35, "note": "国际稳定"},
        "Moonshot": {"input": 12, "output": 12, "note": "超长上下文"},
    }
    
    print(f"\n{'Provider':<25} {'Input':>10} {'Output':>10} {'Total':>10} 备注")
    print("-"*70)
    
    for name, cost in costs.items():
        total = cost["input"] + cost["output"]
        print(f"{name:<25} ¥{cost['input']:>8.2f} ¥{cost['output']:>8.2f} ¥{total:>8.2f}  {cost['note']}")
    
    print("\n💡 推荐配置:")
    print("   • 免费方案: 智谱 GLM-4-Flash")
    print("   • 高性价比: DeepSeek + 智谱备用")
    print("   • 高可用: DeepSeek + 智谱 + OpenAI")


def main():
    """Run all demos"""
    print("\n" + "🚀"*35)
    print("\n   Live Talker - 多 LLM Provider 架构演示\n")
    print("   无需 API Key，使用 Mock 数据展示功能")
    print("\n" + "🚀"*35)
    
    demo_basic()
    demo_failover()
    demo_health_check()
    demo_stats()
    demo_streaming()
    demo_cost_comparison()
    
    print("\n" + "="*70)
    print("✅ 所有演示完成！")
    print("="*70)
    print("\n📚 下一步:")
    print("   1. 申请 API Key (详见 docs/API_KEY_SETUP.md)")
    print("   2. 配置 .env 文件")
    print("   3. 运行: python demo_multi_provider.py")
    print("\n")


if __name__ == "__main__":
    main()
