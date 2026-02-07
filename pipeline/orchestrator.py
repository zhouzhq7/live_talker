"""
Stream Orchestrator
流式处理流水线协调器

协调 ASR -> LLM -> TTS 的流式处理流程
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime

from .state import (
    PipelineState,
    PipelineEvent,
    PipelineContext,
    StateValidator,
)

logger = logging.getLogger(__name__)


class PipelineConfig(NamedTuple):
    """流水线配置"""
    enable_parallel: bool = True  # 是否启用并行处理
    buffer_pre_speech: float = 1.0  # 预录音时长(秒)
    max_latency: float = 10.0  # 最大延迟(秒)


class PipelineResult(NamedTuple):
    """流水线处理结果"""
    success: bool
    text: str  # ASR 识别文本
    response: str  # LLM 回复
    audio: bytes  # TTS 音频
    latency: float  # 端到端延迟
    error: Optional[str] = None


class StreamOrchestrator:
    """流式处理协调器

    核心职责：
    1. 协调 ASR -> LLM -> TTS 的流式处理
    2. 管理流水线状态
    3. 处理用户打断
    4. 收集性能指标
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        asr_engine=None,
        llm_engine=None,
        tts_engine=None,
        vad_engine=None,
    ):
        """初始化流式协调器

        Args:
            config: 流水线配置
            asr_engine: ASR 引擎
            llm_engine: LLM 引擎
            tts_engine: TTS 引擎
            vad_engine: VAD 引擎
        """
        self.config = config or PipelineConfig()
        self.asr = asr_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.vad = vad_engine

        self._state = PipelineState.IDLE
        self._context = PipelineContext()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._interrupted = False

        # 回调函数
        self._on_state_change = None
        self._on_result = None
        self._on_error = None

    @property
    def state(self) -> PipelineState:
        """获取当前状态"""
        return self._state

    def set_callbacks(
        self,
        on_state_change=None,
        on_result=None,
        on_error=None,
    ):
        """设置回调函数

        Args:
            on_state_change: 状态变化回调 (state: PipelineState)
            on_result: 结果回调 (result: PipelineResult)
            on_error: 错误回调 (error: str)
        """
        self._on_state_change = on_state_change
        self._on_result = on_result
        self._on_error = on_error

    async def start(self) -> None:
        """启动流水线"""
        if self._running:
            logger.warning("[StreamOrchestrator] Already running")
            return

        self._running = True
        self._interrupted = False
        await self._transition(PipelineEvent.START)
        logger.info("[StreamOrchestrator] Started")

    async def stop(self) -> None:
        """停止流水线"""
        if not self._running:
            return

        self._running = False
        self._interrupted = False
        await self._transition(PipelineEvent.STOP)
        logger.info("[StreamOrchestrator] Stopped")

    async def interrupt(self) -> bool:
        """触发打断

        Returns:
            bool: 是否成功触发打断
        """
        if self._state in (PipelineState.IDLE, PipelineState.INTERRUPTED):
            return False

        self._interrupted = True
        await self._transition(PipelineEvent.INTERRUPTION_DETECTED)
        logger.info("[StreamOrchestrator] Interrupted")
        return True

    def reset(self) -> None:
        """重置流水线状态"""
        self._context.reset()
        self._interrupted = False

    async def process(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> PipelineResult:
        """处理音频数据

        Args:
            audio_data: 16kHz, 16-bit PCM 音频数据
            sample_rate: 采样率

        Returns:
            PipelineResult: 处理结果
        """
        if not self._running:
            await self.start()

        self._context.reset()
        self._context.start_time = datetime.now()
        self._context.audio_data = audio_data

        try:
            # 状态转换到处理中
            await self._transition(PipelineEvent.SPEECH_ENDED)

            # 执行流水线处理
            asr_latency = await self._run_asr(audio_data, sample_rate)
            self._context.asr_latency = asr_latency

            if self._interrupted:
                return PipelineResult(
                    success=False,
                    text=self._context.asr_result or "",
                    response="",
                    audio=b"",
                    latency=0.0,
                    error="Interrupted"
                )

            llm_latency = await self._run_llm()
            self._context.llm_latency = llm_latency

            if self._interrupted:
                return PipelineResult(
                    success=False,
                    text=self._context.asr_result or "",
                    response=self._context.llm_response,
                    audio=b"",
                    latency=0.0,
                    error="Interrupted"
                )

            tts_latency = await self._run_tts()
            self._context.tts_latency = tts_latency

            # 计算总延迟
            if self._context.start_time:
                self._context.total_latency = (
                    datetime.now() - self._context.start_time
                ).total_seconds()

            result = PipelineResult(
                success=True,
                text=self._context.asr_result or "",
                response=self._context.llm_response,
                audio=self._context.tts_audio,
                latency=self._context.total_latency,
            )

            # 通知结果
            if self._on_result:
                self._on_result(result)

            # 状态转换到播放
            await self._transition(PipelineEvent.TTS_COMPLETE)

            # 播放完成后回到空闲
            await self._transition(PipelineEvent.STOP)

            return result

        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            logger.error(f"[StreamOrchestrator] {error_msg}")

            if self._on_error:
                self._on_error(error_msg)

            await self._transition(PipelineEvent.ERROR)
            return PipelineResult(
                success=False,
                text=self._context.asr_result or "",
                response=self._context.llm_response,
                audio=self._context.tts_audio,
                latency=self._context.total_latency,
                error=error_msg,
            )

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """流式处理音频

        Args:
            audio_stream: 音频流生成器

        Yields:
            str: 处理进度或结果
        """
        # 启动流水线
        await self.start()

        try:
            # 收集完整音频进行 ASR
            audio_chunks = []
            async for chunk in audio_stream:
                if self._interrupted:
                    break
                audio_chunks.append(chunk)
                yield f"collected: {len(audio_chunks)} chunks"

            if self._interrupted:
                yield "interrupted"
                return

            audio_data = b''.join(audio_chunks)

            # 执行处理
            result = await self.process(audio_data)

            if result.success:
                yield f"complete: {result.latency:.2f}s"
            else:
                yield f"error: {result.error}"

        finally:
            await self.stop()

    async def _run_asr(
        self,
        audio_data: bytes,
        sample_rate: int
    ) -> float:
        """执行 ASR 识别

        Returns:
            float: 延迟(秒)
        """
        start_time = datetime.now()

        if self.asr is None:
            logger.warning("[StreamOrchestrator] No ASR engine")
            self._context.asr_result = ""
            return 0.0

        try:
            # 检查是否有流式接口
            if hasattr(self.asr, 'stream_recognize'):
                results = []
                async for result in self.asr.stream_recognize(
                    self._audio_stream_generator(audio_data, sample_rate)
                ):
                    if self._interrupted:
                        break
                    results.append(result)
                    # 通知部分结果
                    await self._emit_event(PipelineEvent.ASR_PARTIAL, {
                        "text": result.text if result.text else "",
                        "is_final": result.is_final,
                    })

                # 取最终结果
                if results:
                    final_result = results[-1]
                    self._context.asr_result = final_result.text or ""
                    await self._emit_event(PipelineEvent.ASR_COMPLETE, {
                        "text": self._context.asr_result,
                    })
            else:
                # 回退到非流式
                result = self.asr.transcribe(audio_data, sample_rate)
                self._context.asr_result = result
                await self._emit_event(PipelineEvent.ASR_COMPLETE, {
                    "text": result,
                })

        except Exception as e:
            logger.error(f"[StreamOrchestrator] ASR error: {e}")
            await self._emit_event(PipelineEvent.ASR_ERROR, {"error": str(e)})
            raise

        return (datetime.now() - start_time).total_seconds()

    async def _run_llm(self) -> float:
        """执行 LLM 生成

        Returns:
            float: 延迟(秒)
        """
        start_time = datetime.now()

        if self.llm is None:
            logger.warning("[StreamOrchestrator] No LLM engine")
            self._context.llm_response = ""
            return 0.0

        try:
            # 检查是否有流式接口
            if hasattr(self.llm, 'chat_stream'):
                async for token in self.llm.chat_stream(
                    self._context.asr_result
                ):
                    if self._interrupted:
                        break
                    self._context.llm_response += token
                    await self._emit_event(PipelineEvent.LLM_TOKEN, {
                        "token": token,
                        "response": self._context.llm_response,
                    })
            else:
                # 回退到非流式
                self._context.llm_response = self.llm.chat(
                    self._context.asr_result
                )

            await self._emit_event(PipelineEvent.LLM_COMPLETE, {
                "response": self._context.llm_response,
            })

        except Exception as e:
            logger.error(f"[StreamOrchestrator] LLM error: {e}")
            await self._emit_event(PipelineEvent.LLM_ERROR, {"error": str(e)})
            raise

        return (datetime.now() - start_time).total_seconds()

    async def _run_tts(self) -> float:
        """执行 TTS 合成

        Returns:
            float: 延迟(秒)
        """
        start_time = datetime.now()

        if self.tts is None:
            logger.warning("[StreamOrchestrator] No TTS engine")
            self._context.tts_audio = b""
            return 0.0

        try:
            # 检查是否有流式接口
            if hasattr(self.tts, 'synthesize_stream'):
                audio_chunks = []
                async for result in self.tts.synthesize_stream(
                    self._text_stream_generator(self._context.llm_response)
                ):
                    if self._interrupted:
                        break
                    audio_chunks.append(result.audio_chunk)
                    await self._emit_event(PipelineEvent.TTS_AUDIO, {
                        "chunk_size": len(result.audio_chunk),
                        "is_final": result.is_final,
                    })

                self._context.tts_audio = b''.join(audio_chunks)
            else:
                # 回退到非流式
                self._context.tts_audio = self.tts.synthesize(
                    self._context.llm_response
                )

            await self._emit_event(PipelineEvent.TTS_COMPLETE, {
                "audio_size": len(self._context.tts_audio),
            })

        except Exception as e:
            logger.error(f"[StreamOrchestrator] TTS error: {e}")
            await self._emit_event(PipelineEvent.TTS_ERROR, {"error": str(e)})
            raise

        return (datetime.now() - start_time).total_seconds()

    async def _transition(self, event: PipelineEvent) -> bool:
        """执行状态转换

        Args:
            event: 触发事件

        Returns:
            bool: 是否成功转换
        """
        valid, next_state, error = StateValidator.validate_transition(
            self._state, event
        )

        if not valid:
            logger.warning(
                f"[StreamOrchestrator] Invalid transition: "
                f"{self._state.value} -> {event.value}: {error}"
            )
            return False

        old_state = self._state
        self._state = next_state

        logger.debug(
            f"[StreamOrchestrator] State: {old_state.value} -> "
            f"{self._state.value} ({event.value})"
        )

        # 通知状态变化
        if self._on_state_change:
            try:
                self._on_state_change(self._state)
            except Exception as e:
                logger.error(f"[StreamOrchestrator] State callback error: {e}")

        return True

    async def _emit_event(self, event_type: PipelineEvent, data: dict):
        """发送流水线事件

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = {
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        await self._event_queue.put(event)

    async def events(self) -> AsyncGenerator[dict, None]:
        """流水线事件流

        Yields:
            dict: 流水线事件
        """
        while self._running or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue

    def _audio_stream_generator(
        self,
        audio_data: bytes,
        sample_rate: int
    ) -> AsyncGenerator[bytes, None]:
        """音频流生成器

        将完整音频分块输出
        """
        chunk_size = sample_rate * 2  # 1秒音频
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]

    def _text_stream_generator(
        self,
        text: str
    ) -> AsyncGenerator[str, None]:
        """文本流生成器

        将文本按字符/词输出
        """
        # 按句子分段
        import re
        sentences = re.split(r'([。！？])', text)
        buffer = ""

        for i, segment in enumerate(sentences):
            if segment in '。！？':
                buffer += segment
                if buffer.strip():
                    yield buffer.strip()
                buffer = ""
            else:
                buffer += segment

        # 最后一段
        if buffer.strip():
            yield buffer.strip()

    def get_info(self) -> dict:
        """获取协调器信息"""
        return {
            "state": self._state.value,
            "running": self._running,
            "interrupted": self._interrupted,
            "config": self.config._asdict() if hasattr(self.config, '_asdict') else dict(self.config),
            "engines": {
                "asr": self.asr.__class__.__name__ if self.asr else None,
                "llm": self.llm.__class__.__name__ if self.llm else None,
                "tts": self.tts.__class__.__name__ if self.tts else None,
                "vad": self.vad.__class__.__name__ if self.vad else None,
            },
            "metrics": {
                "asr_latency": self._context.asr_latency,
                "llm_latency": self._context.llm_latency,
                "tts_latency": self._context.tts_latency,
                "total_latency": self._context.total_latency,
            }
        }
