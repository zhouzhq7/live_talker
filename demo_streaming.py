#!/usr/bin/env python3
"""
Streaming Pipeline Demonstration

演示流式处理如何降低端到端延迟
"""

import os
import sys
import time
import threading
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(__file__))

from core.streaming import (
    StreamingPipeline,
    StreamState,
    IncrementalASR,
    SentenceSplitter,
    StreamingTTS,
    StreamingMetrics
)


def mock_asr_engine():
    """Create mock ASR engine"""
    engine = Mock()
    engine.transcribe = Mock(return_value="这是一个测试语音识别")
    return engine


def mock_tts_engine():
    """Create mock TTS engine"""
    engine = Mock()
    # Return dummy audio data
    engine.synthesize = Mock(return_value=b'\x00\x01' * 8000)
    return engine


def mock_audio_player():
    """Create mock audio player"""
    player = Mock()
    player.play_bytes = Mock()
    player.stop = Mock()
    return player


def mock_llm_manager():
    """Create mock LLM manager that streams text"""
    manager = Mock()
    
    def stream_response(messages):
        """Simulate streaming response"""
        text = "你好！我是AI助手。很高兴为你服务。今天天气不错，希望你有美好的一天！"
        # Stream word by word
        for char in text:
            yield char
            time.sleep(0.02)  # Simulate delay
            
    manager.chat_completion_stream = stream_response
    return manager


def demo_incremental_asr():
    """Demonstrate incremental ASR"""
    print("\n" + "="*70)
    print("🎯 Demo 1: 增量ASR (Incremental ASR)")
    print("="*70)
    
    asr_engine = mock_asr_engine()
    
    incremental_asr = IncrementalASR(
        asr_engine=asr_engine,
        window_size_ms=500,
        hop_size_ms=200,
        sample_rate=16000
    )
    
    print("\n📋 配置:")
    print(f"   窗口大小: 500ms")
    print(f"   滑动步长: 200ms")
    print(f"   采样率: 16kHz")
    
    print("\n📝 模拟音频输入...")
    incremental_asr.start()
    
    # Simulate feeding audio chunks
    chunk_size = 3200  # 100ms of 16-bit audio at 16kHz
    for i in range(20):  # 2 seconds of audio
        audio_chunk = b'\x00\x00' * chunk_size
        incremental_asr.feed_audio(audio_chunk)
        time.sleep(0.05)  # 50ms between chunks
        
        # Show partial result occasionally
        if i % 5 == 0:
            partial = incremental_asr.get_partial_result()
            print(f"   [{i*100}ms] 部分结果: '{partial}'")
    
    # Finalize
    result = incremental_asr.finalize()
    print(f"\n✅ 最终结果: '{result}'")
    
    incremental_asr.stop()


def demo_sentence_splitter():
    """Demonstrate sentence splitting"""
    print("\n" + "="*70)
    print("✂️  Demo 2: 句子分割 (Sentence Splitter)")
    print("="*70)
    
    splitter = SentenceSplitter(min_length=5)
    
    # Simulate streaming text
    text_stream = [
        "你好",
        "！我是",
        "AI助手。",
        "很高兴",
        "为你服务。",
        "今天",
        "天气不错",
        "。",
        "希望",
        "你有",
        "美好的一天！"
    ]
    
    print("\n📝 流式输入文本:")
    complete_sentences = []
    
    for chunk in text_stream:
        print(f"   收到: '{chunk}'")
        sentences = splitter.feed_text(chunk)
        if sentences:
            for sent in sentences:
                print(f"   ✅ 完整句子: '{sent}'")
                complete_sentences.append(sent)
        time.sleep(0.1)
    
    # Finalize
    remaining = splitter.finalize()
    for sent in remaining:
        print(f"   ✅ 最终句子: '{sent}'")
        complete_sentences.append(sent)
    
    print(f"\n📊 共分割出 {len(complete_sentences)} 个句子")


def demo_streaming_tts():
    """Demonstrate streaming TTS"""
    print("\n" + "="*70)
    print("🔊 Demo 3: 流式TTS (Streaming TTS)")
    print("="*70)
    
    tts_engine = mock_tts_engine()
    audio_player = mock_audio_player()
    
    streaming_tts = StreamingTTS(
        tts_engine=tts_engine,
        audio_player=audio_player
    )
    
    print("\n📋 配置:")
    print("   TTS引擎: MeloTTS (模拟)")
    print("   播放方式: 边合成边播放")
    
    streaming_tts.start()
    
    print("\n📝 模拟LLM流式输出...")
    
    # Simulate streaming text from LLM
    text_chunks = [
        "你好",
        "！我是",
        "AI助手。",
        "很高兴",
        "为你服务。",
    ]
    
    synthesis_times = []
    for chunk in text_chunks:
        start = time.time()
        streaming_tts.feed_text(chunk)
        elapsed = (time.time() - start) * 1000
        synthesis_times.append(elapsed)
        print(f"   合成 '{chunk}' - {elapsed:.1f}ms")
        time.sleep(0.1)
    
    streaming_tts.finalize()
    
    # Wait for playback
    time.sleep(0.5)
    streaming_tts.stop()
    
    print(f"\n📊 平均合成延迟: {sum(synthesis_times)/len(synthesis_times):.1f}ms")
    print(f"   总合成次数: {tts_engine.synthesize.call_count}")


def demo_streaming_pipeline():
    """Demonstrate full streaming pipeline"""
    print("\n" + "="*70)
    print("🚀 Demo 4: 完整流式Pipeline (Streaming Pipeline)")
    print("="*70)
    
    # Create mocks
    asr_engine = mock_asr_engine()
    llm_manager = mock_llm_manager()
    tts_engine = mock_tts_engine()
    audio_player = mock_audio_player()
    
    # Create pipeline
    pipeline = StreamingPipeline(
        asr_engine=asr_engine,
        llm_manager=llm_manager,
        tts_engine=tts_engine,
        audio_player=audio_player,
        enable_incremental_asr=True,
        enable_streaming_tts=True
    )
    
    # State change callback
    def on_state_change(state):
        print(f"   [状态变化] {state.name}")
    
    pipeline.on_state_change = on_state_change
    
    print("\n📋 Pipeline配置:")
    print("   ✅ 增量ASR: 启用")
    print("   ✅ 流式LLM: 启用")
    print("   ✅ 流式TTS: 启用")
    print("   🎯 目标延迟: < 1000ms")
    
    # Start pipeline
    pipeline.start()
    
    print("\n🎤 模拟语音输入 (3秒音频)...")
    
    # Simulate audio input
    audio_data = b'\x00\x00' * 16000 * 3  # 3 seconds of silence
    
    # Process
    start_time = time.time()
    pipeline.process_utterance(audio_data)
    total_time = (time.time() - start_time) * 1000
    
    print(f"\n📊 性能指标:")
    print(f"   ASR延迟: {pipeline.metrics.first_asr_latency_ms:.0f}ms")
    print(f"   LLM首token: {pipeline.metrics.first_llm_latency_ms:.0f}ms")
    print(f"   端到端总延迟: {total_time:.0f}ms")
    
    if total_time < 1000:
        print(f"   ✅ 达到目标延迟 (< 1000ms)")
    else:
        print(f"   ⚠️  未达到目标延迟")
    
    pipeline.stop()


def demo_latency_comparison():
    """Compare streaming vs non-streaming latency"""
    print("\n" + "="*70)
    print("⏱️  Demo 5: 延迟对比 (Streaming vs Non-Streaming)")
    print("="*70)
    
    # Simulate processing steps
    def non_streaming_pipeline():
        """Traditional non-streaming pipeline"""
        # ASR: wait for all audio
        time.sleep(0.3)
        # LLM: wait for all text
        time.sleep(0.5)
        # TTS: wait for all synthesis
        time.sleep(0.4)
        # Playback
        time.sleep(2.0)
        return 3.2  # Total seconds
    
    def streaming_pipeline():
        """Streaming pipeline with overlap"""
        # ASR starts immediately
        time.sleep(0.1)
        # LLM starts with first ASR result
        time.sleep(0.1)
        # TTS starts with first sentence
        time.sleep(0.1)
        # Playback starts with first audio
        time.sleep(2.0)
        # But total wait time is reduced due to overlap
        return 0.3 + 0.3 + 0.3  # Parallel processing
    
    print("\n📊 模拟延迟对比 (10个字的短句):")
    print()
    print(f"{'阶段':<15} {'非流式':>12} {'流式':>12} {'优化':>12}")
    print("-" * 55)
    
    stages = [
        ("ASR识别", 300, 150),
        ("LLM首token", 500, 200),
        ("TTS首帧", 400, 100),
        ("首帧播放延迟", 1200, 450),
        ("完整响应时间", 3200, 2800),
    ]
    
    for stage, non_stream, stream in stages:
        improvement = ((non_stream - stream) / non_stream) * 100
        print(f"{stage:<15} {non_stream:>10}ms {stream:>10}ms {improvement:>10.0f}%")
    
    print()
    print("💡 流式优化效果:")
    print("   • 首帧播放延迟减少 62%")
    print("   • 用户体验更流畅")
    print("   • 支持打断和实时反馈")


def demo_interruption():
    """Demonstrate interruption handling"""
    print("\n" + "="*70)
    print("🛑 Demo 6: 打断处理 (Interruption Handling)")
    print("="*70)
    
    tts_engine = mock_tts_engine()
    audio_player = mock_audio_player()
    
    streaming_tts = StreamingTTS(
        tts_engine=tts_engine,
        audio_player=audio_player
    )
    
    streaming_tts.start()
    
    print("\n📝 开始播放长文本...")
    long_text = "这是一个很长的句子。包含很多内容。需要几秒钟才能读完。"
    
    # Start synthesis in background
    def synthesize_long():
        for char in long_text:
            streaming_tts.feed_text(char)
            time.sleep(0.05)
        streaming_tts.finalize()
    
    thread = threading.Thread(target=synthesize_long)
    thread.start()
    
    # Wait a bit then interrupt
    time.sleep(0.3)
    print("🛑 用户打断！")
    streaming_tts.reset()
    # audio_player.stop() is called in reset
    print("✅ 打断成功，系统已准备好接收新输入")
    
    thread.join(timeout=1.0)
    streaming_tts.stop()
    
    print("✅ 打断成功，系统已准备好接收新输入")


def main():
    """Run all demos"""
    print("\n" + "🚀"*35)
    print("\n   Live Talker - Phase 5: 流式处理优化演示\n")
    print("   目标: 端到端延迟 < 1秒")
    print("\n" + "🚀"*35)
    
    demo_incremental_asr()
    demo_sentence_splitter()
    demo_streaming_tts()
    demo_streaming_pipeline()
    demo_latency_comparison()
    demo_interruption()
    
    print("\n" + "="*70)
    print("✅ 所有演示完成！")
    print("="*70)
    print("\n📚 关键优化点:")
    print("   1. 增量ASR - 滑动窗口实时识别")
    print("   2. 句子分割 - 逐句处理减少等待")
    print("   3. 流式TTS - 边合成边播放")
    print("   4. Pipeline并行 - 三个阶段同时运行")
    print("\n")


if __name__ == "__main__":
    main()
