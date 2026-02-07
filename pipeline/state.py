"""
Pipeline State Machine
定义流水线状态枚举和状态转换
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


class PipelineState(Enum):
    """流水线状态枚举"""
    IDLE = "idle"                    # 空闲
    LISTENING = "listening"          # 倾听
    PROCESSING = "processing"        # 处理中 (ASR/LLM/TTS)
    SPEAKING = "speaking"            # 播放中
    INTERRUPTED = "interrupted"      # 被打断
    ERROR = "error"                  # 错误


class PipelineEvent(Enum):
    """流水线事件枚举"""
    # VAD 事件
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    INTERRUPTION_DETECTED = "interruption_detected"

    # ASR 事件
    ASR_PARTIAL = "asr_partial"
    ASR_COMPLETE = "asr_complete"
    ASR_ERROR = "asr_error"

    # LLM 事件
    LLM_TOKEN = "llm_token"
    LLM_COMPLETE = "llm_complete"
    LLM_ERROR = "llm_error"

    # TTS 事件
    TTS_AUDIO = "tts_audio"
    TTS_SENTENCE_END = "tts_sentence_end"
    TTS_COMPLETE = "tts_complete"
    TTS_ERROR = "tts_error"

    # 系统事件
    START = "start"
    STOP = "stop"
    RESET = "reset"


@dataclass
class PipelineContext:
    """流水线上下文

    存储流水线运行时的状态信息
    """
    state: PipelineState = PipelineState.IDLE
    audio_data: Optional[bytes] = None
    asr_result: Optional[str] = None
    llm_response: str = ""
    tts_audio: bytes = b""
    error_message: Optional[str] = None

    # 性能指标
    start_time: Optional[datetime] = None
    asr_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    total_latency: float = 0.0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """重置上下文"""
        self.state = PipelineState.IDLE
        self.audio_data = None
        self.asr_result = None
        self.llm_response = ""
        self.tts_audio = b""
        self.error_message = None
        self.start_time = None
        self.asr_latency = 0.0
        self.llm_latency = 0.0
        self.tts_latency = 0.0
        self.total_latency = 0.0
        self.metadata.clear()


# 状态转换表
STATE_TRANSITIONS = {
    PipelineState.IDLE: {
        PipelineEvent.START: PipelineState.LISTENING,
        PipelineEvent.STOP: PipelineState.IDLE,
    },
    PipelineState.LISTENING: {
        PipelineEvent.SPEECH_STARTED: PipelineState.LISTENING,
        PipelineEvent.SPEECH_ENDED: PipelineState.PROCESSING,
        PipelineEvent.INTERRUPTION_DETECTED: PipelineState.INTERRUPTED,
        PipelineEvent.STOP: PipelineState.IDLE,
        PipelineEvent.ERROR: PipelineState.ERROR,
    },
    PipelineState.PROCESSING: {
        PipelineEvent.ASR_COMPLETE: PipelineState.PROCESSING,
        PipelineEvent.LLM_COMPLETE: PipelineState.PROCESSING,
        PipelineEvent.TTS_COMPLETE: PipelineState.SPEAKING,
        PipelineEvent.INTERRUPTION_DETECTED: PipelineState.INTERRUPTED,
        PipelineEvent.ERROR: PipelineState.ERROR,
    },
    PipelineState.SPEAKING: {
        PipelineEvent.INTERRUPTION_DETECTED: PipelineState.INTERRUPTED,
        PipelineEvent.TTS_COMPLETE: PipelineState.IDLE,
        PipelineEvent.STOP: PipelineState.IDLE,
    },
    PipelineState.INTERRUPTED: {
        PipelineEvent.SPEECH_STARTED: PipelineState.LISTENING,
        PipelineEvent.RESET: PipelineState.IDLE,
        PipelineEvent.STOP: PipelineState.IDLE,
    },
    PipelineState.ERROR: {
        PipelineEvent.RESET: PipelineState.IDLE,
        PipelineEvent.STOP: PipelineState.IDLE,
    },
}


class StateValidator:
    """状态转换验证器

    验证状态转换是否合法
    """

    @staticmethod
    def can_transition(
        from_state: PipelineState,
        event: PipelineEvent
    ) -> bool:
        """检查状态转换是否允许

        Args:
            from_state: 当前状态
            event: 触发事件

        Returns:
            bool: 是否允许转换
        """
        transitions = STATE_TRANSITIONS.get(from_state, {})
        return event in transitions

    @staticmethod
    def get_next_state(
        from_state: PipelineState,
        event: PipelineEvent
    ) -> Optional[PipelineState]:
        """获取下一个状态

        Args:
            from_state: 当前状态
            event: 触发事件

        Returns:
            PipelineState: 下一个状态，None 表示不允许转换
        """
        transitions = STATE_TRANSITIONS.get(from_state, {})
        return transitions.get(event)

    @staticmethod
    def validate_transition(
        from_state: PipelineState,
        event: PipelineEvent
    ) -> tuple[bool, Optional[PipelineState], Optional[str]]:
        """验证状态转换

        Args:
            from_state: 当前状态
            event: 触发事件

        Returns:
            tuple: (是否允许, 下一个状态, 错误信息)
        """
        next_state = StateValidator.get_next_state(from_state, event)

        if next_state is None:
            allowed_events = STATE_TRANSITIONS.get(from_state, {}).keys()
            return (
                False,
                None,
                f"Cannot transition from {from_state.value} on event {event.value}. "
                f"Allowed events: {[e.value for e in allowed_events]}"
            )

        return True, next_state, None
