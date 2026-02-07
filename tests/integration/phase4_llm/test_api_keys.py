#!/usr/bin/env python3
"""
API Key 验证和 Provider 测试脚本

测试所有配置的 LLM Provider 是否可用
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from llm import (
    DeepSeekProvider,
    ZhipuProvider,
    OpenAIProvider,
    MoonshotProvider,
    ConversationManager,
    ConversationConfig
)


def test_provider(name: str, provider) -> Tuple[bool, str, float]:
    """
    Test a single provider
    
    Returns:
        (success, response_text, latency_ms)
    """
    print(f"\n{'='*60}")
    print(f"🧪 测试 {name}")
    print(f"{'='*60}")
    
    # Check initialization
    if not provider._is_initialized:
        print(f"❌ {name} 初始化失败")
        return False, "", 0
    
    print(f"✅ {name} 初始化成功")
    print(f"   - Model: {provider.model}")
    print(f"   - Base URL: {provider.base_url}")
    
    # Test chat completion
    test_messages = [
        {"role": "system", "content": "你是一个友好的助手。请用一句话回答。"},
        {"role": "user", "content": "你好！请介绍一下自己。"}
    ]
    
    try:
        start_time = time.time()
        response = provider.chat_completion(test_messages, stream=False)
        latency_ms = (time.time() - start_time) * 1000
        
        if response.text:
            print(f"✅ {name} 响应成功")
            print(f"   - 延迟: {latency_ms:.0f}ms")
            print(f"   - Tokens: {response.total_tokens}")
            print(f"   - 响应: {response.text[:100]}...")
            return True, response.text, latency_ms
        else:
            print(f"❌ {name} 返回空响应")
            return False, "", latency_ms
            
    except Exception as e:
        print(f"❌ {name} 调用失败: {e}")
        return False, "", 0


def test_deepseek() -> Tuple[bool, str, float]:
    """Test DeepSeek provider"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("\n⚠️  未设置 DEEPSEEK_API_KEY，跳过测试")
        print("   申请地址: https://platform.deepseek.com")
        return False, "", 0
    
    provider = DeepSeekProvider(
        api_key=api_key,
        model="deepseek-chat"
    )
    
    return test_provider("DeepSeek", provider)


def test_zhipu() -> Tuple[bool, str, float]:
    """Test Zhipu (智谱) provider"""
    api_key = os.getenv("ZHIPU_API_KEY")
    
    if not api_key:
        print("\n⚠️  未设置 ZHIPU_API_KEY，跳过测试")
        print("   申请地址: https://open.bigmodel.cn")
        print("   💡 推荐: GLM-4-Flash 完全免费！")
        return False, "", 0
    
    # Test GLM-4-Flash (free)
    print("\n📝 测试 GLM-4-Flash (免费模型)...")
    provider_flash = ZhipuProvider(
        api_key=api_key,
        model="glm-4-flash"
    )
    result_flash = test_provider("Zhipu (GLM-4-Flash)", provider_flash)
    
    # Test GLM-4-Air (paid, cheaper)
    print("\n📝 测试 GLM-4-Air (高性价比)...")
    provider_air = ZhipuProvider(
        api_key=api_key,
        model="glm-4-air"
    )
    result_air = test_provider("Zhipu (GLM-4-Air)", provider_air)
    
    return result_flash or result_air, "", 0


def test_openai() -> Tuple[bool, str, float]:
    """Test OpenAI provider"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  未设置 OPENAI_API_KEY，跳过测试")
        print("   申请地址: https://platform.openai.com")
        print("   注意: 需要海外支付方式")
        return False, "", 0
    
    provider = OpenAIProvider(
        api_key=api_key,
        model="gpt-4o-mini"
    )
    
    return test_provider("OpenAI (GPT-4o-mini)", provider)


def test_moonshot() -> Tuple[bool, str, float]:
    """Test Moonshot/Kimi provider"""
    api_key = os.getenv("MOONSHOT_API_KEY")
    
    if not api_key:
        print("\n⚠️  未设置 MOONSHOT_API_KEY，跳过测试")
        print("   申请地址: https://platform.moonshot.cn")
        print("   新用户赠送 ¥15 额度")
        return False, "", 0
    
    provider = MoonshotProvider(
        api_key=api_key,
        model="moonshot-v1-8k"
    )
    
    return test_provider("Moonshot (Kimi)", provider)


def test_conversation_manager():
    """Test ConversationManager with multiple providers"""
    print("\n" + "="*60)
    print("🧪 测试 ConversationManager (多Provider故障转移)")
    print("="*60)
    
    # Build config from environment
    config = ConversationConfig(
        primary_provider="deepseek",
        enable_fallback=True,
        fallback_providers=["zhipu", "openai"],
        temperature=0.7,
        max_tokens=500,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        zhipu_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        moonshot_api_key=os.getenv("MOONSHOT_API_KEY"),
        zhipu_model="glm-4-flash"  # Use free model as fallback
    )
    
    manager = ConversationManager(config)
    
    if not manager.is_ready():
        print("❌ ConversationManager 未就绪，没有可用的 Provider")
        return False
    
    available = manager.get_available_providers()
    print(f"✅ ConversationManager 就绪")
    print(f"   - 可用 Providers: {available}")
    print(f"   - 主 Provider: {config.primary_provider}")
    print(f"   - 故障转移: {'启用' if config.enable_fallback else '禁用'}")
    
    # Test conversation
    print("\n📝 测试对话...")
    test_messages = [
        "你好！请用一句话介绍自己。",
        "中国的首都是哪里？",
        "谢谢，再见！"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n💬 消息 {i}/{len(test_messages)}: {msg}")
        start_time = time.time()
        response = manager.chat(message=msg)
        latency = (time.time() - start_time) * 1000
        
        if response:
            print(f"✅ 响应 ({latency:.0f}ms): {response[:100]}...")
        else:
            print(f"❌ 无响应")
    
    # Print stats
    stats = manager.get_stats()
    print(f"\n📊 会话统计:")
    print(f"   - 发送消息: {stats['messages_sent']}")
    print(f"   - 收到响应: {stats['messages_received']}")
    print(f"   - 平均延迟: {stats['avg_latency_ms']:.0f}ms")
    print(f"   - 总 Tokens: {stats['total_tokens']}")
    
    return True


def print_summary(results: Dict[str, Tuple[bool, str, float]]):
    """Print test summary"""
    print("\n" + "="*60)
    print("📋 测试结果汇总")
    print("="*60)
    
    for name, (success, _, latency) in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        latency_str = f"({latency:.0f}ms)" if success else ""
        print(f"   {status} {name} {latency_str}")
    
    total = len(results)
    passed = sum(1 for s, _, _ in results.values() if s)
    
    print(f"\n总计: {passed}/{total} 个 Provider 可用")
    
    if passed == 0:
        print("\n⚠️  警告: 没有可用的 Provider！")
        print("   请按照以下步骤申请 API Key:")
        print("   1. 智谱 (免费): https://open.bigmodel.cn")
        print("   2. DeepSeek (低价): https://platform.deepseek.com")
        print("   3. 在 .env 文件中配置 API Key")
        print("   4. 重新运行测试")
    elif passed < total:
        print("\n💡 提示: 部分 Provider 未配置")
        print("   建议至少配置 2 个 Provider 以实现故障转移")


def main():
    """Main test function"""
    print("="*60)
    print("🚀 Live Talker - LLM Provider API 测试")
    print("="*60)
    print("\n此脚本将测试所有配置的 LLM Provider")
    print("确保已设置环境变量或在 .env 文件中配置 API Key")
    
    # Load .env if exists
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"\n✅ 已加载 .env 文件")
    except ImportError:
        pass
    
    results = {}
    
    # Test individual providers
    results["DeepSeek"] = test_deepseek()
    results["Zhipu"] = test_zhipu()
    results["OpenAI"] = test_openai()
    results["Moonshot"] = test_moonshot()
    
    # Test ConversationManager
    manager_ok = test_conversation_manager()
    
    # Print summary
    print_summary(results)
    
    # Final status
    print("\n" + "="*60)
    if manager_ok:
        print("✅ 所有测试完成，系统可以正常使用！")
    else:
        print("❌ 测试未通过，请检查配置")
    print("="*60)
    
    return 0 if manager_ok else 1


if __name__ == "__main__":
    sys.exit(main())
