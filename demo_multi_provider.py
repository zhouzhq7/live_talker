#!/usr/bin/env python3
"""
多 Provider LLM 对话演示

演示 Live Talker 的多 LLM Provider 支持和故障转移功能
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from llm import ConversationManager, ConversationConfig
from config import load_config_from_env


def print_banner():
    """Print demo banner"""
    print("\n" + "="*70)
    print("🚀 Live Talker - 多 LLM Provider 对话演示")
    print("="*70)
    print("\n支持的 Provider:")
    print("   • DeepSeek - 价格低，中文强")
    print("   • 智谱 GLM-4-Flash - 完全免费")
    print("   • OpenAI GPT-4o-mini - 国际稳定")
    print("   • Moonshot/Kimi - 超长上下文")
    print("\n功能特性:")
    print("   ✅ 自动故障转移")
    print("   ✅ 健康状态检查")
    print("   ✅ Token 使用统计")
    print("   ✅ 流式响应支持")
    print("="*70 + "\n")


def get_provider_emoji(provider_name: str) -> str:
    """Get emoji for provider"""
    emojis = {
        "deepseek": "🔷",
        "zhipu": "🔶",
        "openai": "🟢",
        "moonshot": "🌙",
    }
    return emojis.get(provider_name.lower(), "🤖")


def chat_session(config: ConversationConfig):
    """Run interactive chat session"""
    
    print("🔄 正在初始化 LLM Providers...")
    
    manager = ConversationManager(config)
    
    if not manager.is_ready():
        print("\n❌ 错误: 没有可用的 LLM Provider")
        print("\n请配置至少一个 API Key:")
        print("   1. 智谱 (免费): https://open.bigmodel.cn")
        print("   2. DeepSeek: https://platform.deepseek.com")
        print("\n在 .env 文件中添加:")
        print('   ZHIPU_API_KEY="your-key-here"')
        print('   DEEPSEEK_API_KEY="sk-your-key-here"')
        return
    
    available = manager.get_available_providers()
    print(f"✅ 就绪！可用 Providers: {', '.join(available)}\n")
    
    # System prompt
    system_prompt = """你是一个友好的AI助手。请用简洁、自然的语言回答问题。
重要：不要使用emoji或特殊符号，因为回答将被转换为语音。"""
    
    print("📝 系统提示: " + system_prompt.replace('\n', ' ')[:60] + "...\n")
    print("💡 输入 'quit' 或 'exit' 退出，输入 'stats' 查看统计，输入 'clear' 清空历史\n")
    
    message_count = 0
    
    while True:
        # Get user input
        try:
            user_input = input("👤 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 再见！")
            break
        
        if user_input.lower() == 'stats':
            stats = manager.get_stats()
            print("\n" + "-"*70)
            print("📊 会话统计:")
            print(f"   消息数: {stats['messages_sent']}")
            print(f"   响应数: {stats['messages_received']}")
            print(f"   总Tokens: {stats['total_tokens']}")
            print(f"   平均延迟: {stats['avg_latency_ms']:.0f}ms")
            if 'providers' in stats:
                print("\n   Provider 状态:")
                for name, info in stats['providers'].items():
                    emoji = get_provider_emoji(name)
                    status = info['health']['status']
                    print(f"      {emoji} {name}: {status}")
            print("-"*70 + "\n")
            continue
        
        if user_input.lower() == 'clear':
            manager.clear_history()
            print("\n🗑️  对话历史已清空\n")
            continue
        
        # Send message
        message_count += 1
        print()
        
        start_time = time.time()
        
        # Use streaming for better UX
        print(f"🤖 AI: ", end="", flush=True)
        
        response_text = ""
        current_provider = ""
        
        try:
            # Try to get response with streaming
            for chunk in manager.chat_stream(user_input, system_prompt=system_prompt):
                print(chunk, end="", flush=True)
                response_text += chunk
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Get provider info
            available = manager.get_available_providers()
            if available:
                current_provider = available[0]
                emoji = get_provider_emoji(current_provider)
            else:
                emoji = "🤖"
            
            print(f"\n\n   [{emoji} {current_provider} | {elapsed_ms:.0f}ms]")
            print()
            
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")
            continue
    
    # Final stats
    print("\n" + "="*70)
    print("📊 最终统计")
    print("="*70)
    
    stats = manager.get_stats()
    print(f"总消息数: {stats['messages_sent']}")
    print(f"总响应数: {stats['messages_received']}")
    print(f"总Tokens: {stats['total_tokens']}")
    print(f"平均延迟: {stats['avg_latency_ms']:.0f}ms")
    
    if stats['total_tokens'] > 0:
        # Estimate cost
        deepseek_cost = stats['total_tokens'] * 1.5 / 1_000_000  # ¥1 input + ¥2 output avg
        zhipu_cost = stats['total_tokens'] * 1.0 / 1_000_000  # ¥1 per 1M for Air
        
        print(f"\n预估成本:")
        print(f"   DeepSeek: ¥{deepseek_cost:.4f}")
        print(f"   智谱 GLM-4-Air: ¥{zhipu_cost:.4f}")
        print(f"   智谱 GLM-4-Flash: ¥0 (免费)")
    
    print("="*70 + "\n")


def quick_test(config: ConversationConfig):
    """Quick non-interactive test"""
    print("\n🔄 快速测试模式\n")
    
    manager = ConversationManager(config)
    
    if not manager.is_ready():
        print("❌ 没有可用的 Provider")
        return False
    
    available = manager.get_available_providers()
    print(f"✅ 可用 Providers: {available}\n")
    
    test_questions = [
        "你好！请用一句话介绍自己。",
        "1+1等于几？",
        "中国的首都是哪里？",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n💬 问题 {i}/{len(test_questions)}: {question}")
        
        start_time = time.time()
        response = manager.chat(question)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if response:
            print(f"✅ 回答 ({elapsed_ms:.0f}ms): {response}")
        else:
            print(f"❌ 无响应")
    
    # Stats
    stats = manager.get_stats()
    print("\n📊 统计:")
    print(f"   平均延迟: {stats['avg_latency_ms']:.0f}ms")
    print(f"   总Tokens: {stats['total_tokens']}")
    
    return True


def main():
    """Main entry"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Talker 多Provider演示')
    parser.add_argument('--test', action='store_true', help='快速测试模式（非交互）')
    parser.add_argument('--provider', type=str, default='deepseek', 
                       help='主Provider (deepseek/zhipu/openai/moonshot)')
    args = parser.parse_args()
    
    print_banner()
    
    # Load config
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print("✅ 已加载 .env 文件\n")
    except ImportError:
        pass
    
    # Check API keys
    has_key = any([
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("ZHIPU_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("MOONSHOT_API_KEY"),
    ])
    
    if not has_key:
        print("⚠️  警告: 未检测到任何 API Key\n")
        print("请先申请 API Key:")
        print("   1. 智谱 (免费): https://open.bigmodel.cn")
        print("      export ZHIPU_API_KEY='your-key'")
        print("   2. DeepSeek: https://platform.deepseek.com")
        print("      export DEEPSEEK_API_KEY='sk-your-key'")
        print()
        return 1
    
    # Build config
    config = ConversationConfig(
        primary_provider=args.provider,
        enable_fallback=True,
        fallback_providers=["zhipu", "openai"],
        temperature=0.7,
        max_tokens=500,
        system_prompt="你是一个友好的AI助手。回答简洁自然。",
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        zhipu_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        moonshot_api_key=os.getenv("MOONSHOT_API_KEY"),
        zhipu_model="glm-4-flash"  # Free model
    )
    
    # Run demo
    if args.test:
        success = quick_test(config)
    else:
        chat_session(config)
        success = True
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
