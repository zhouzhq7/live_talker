"""
Unit Tests for Pipeline State Machine
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.state import (
    PipelineState,
    PipelineEvent,
    PipelineContext,
    StateValidator,
)


class TestPipelineState:
    """Test cases for PipelineState enum"""

    def test_state_values(self):
        """Test state enum values"""
        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.PROCESSING.value == "processing"
        assert PipelineState.SPEAKING.value == "speaking"
        assert PipelineState.INTERRUPTED.value == "interrupted"
        assert PipelineState.ERROR.value == "error"


class TestPipelineEvent:
    """Test cases for PipelineEvent enum"""

    def test_event_categories(self):
        """Test event categories"""
        # VAD events
        assert PipelineEvent.SPEECH_STARTED.value == "speech_started"
        assert PipelineEvent.SPEECH_ENDED.value == "speech_end"

        # ASR events
        assert PipelineEvent.ASR_COMPLETE.value == "asr_complete"
        assert PipelineEvent.ASR_PARTIAL.value == "asr_partial"

        # LLM events
        assert PipelineEvent.LLM_TOKEN.value == "llm_token"
        assert PipelineEvent.LLM_COMPLETE.value == "llm_complete"

        # TTS events
        assert PipelineEvent.TTS_COMPLETE.value == "tts_complete"


class TestPipelineContext:
    """Test cases for PipelineContext"""

    def test_init(self):
        """Test context initialization"""
        ctx = PipelineContext()

        assert ctx.state == PipelineState.IDLE
        assert ctx.audio_data is None
        assert ctx.asr_result is None
        assert ctx.llm_response == ""
        assert ctx.error_message is None

    def test_reset(self):
        """Test context reset"""
        ctx = PipelineContext()
        ctx.audio_data = b"test"
        ctx.asr_result = "hello"
        ctx.llm_response = "world"
        ctx.error_message = "error"

        ctx.reset()

        assert ctx.state == PipelineState.IDLE
        assert ctx.audio_data is None
        assert ctx.asr_result is None
        assert ctx.llm_response == ""
        assert ctx.error_message is None

    def test_metadata(self):
        """Test metadata storage"""
        ctx = PipelineContext()
        ctx.metadata["test_key"] = "test_value"
        ctx.metadata["latency"] = 1.5

        assert ctx.metadata["test_key"] == "test_value"
        assert ctx.metadata["latency"] == 1.5


class TestStateValidator:
    """Test cases for StateValidator"""

    def test_valid_transition_from_idle(self):
        """Test valid transitions from IDLE"""
        assert StateValidator.can_transition(
            PipelineState.IDLE,
            PipelineEvent.START
        ) is True

        assert StateValidator.can_transition(
            PipelineState.IDLE,
            PipelineEvent.STOP
        ) is True

    def test_invalid_transition_from_idle(self):
        """Test invalid transitions from IDLE"""
        # Cannot go directly to SPEAKING from IDLE
        assert StateValidator.can_transition(
            PipelineState.IDLE,
            PipelineEvent.TTS_COMPLETE
        ) is False

    def test_valid_transition_from_listening(self):
        """Test valid transitions from LISTENING"""
        assert StateValidator.can_transition(
            PipelineState.LISTENING,
            PipelineEvent.SPEECH_ENDED
        ) is True

        assert StateValidator.can_transition(
            PipelineState.LISTENING,
            PipelineEvent.INTERRUPTION_DETECTED
        ) is True

    def test_valid_transition_from_processing(self):
        """Test valid transitions from PROCESSING"""
        assert StateValidator.can_transition(
            PipelineState.PROCESSING,
            PipelineEvent.ASR_COMPLETE
        ) is True

        assert StateValidator.can_transition(
            PipelineState.PROCESSING,
            PipelineEvent.LLM_COMPLETE
        ) is True

        assert StateValidator.can_transition(
            PipelineState.PROCESSING,
            PipelineEvent.INTERRUPTION_DETECTED
        ) is True

    def test_valid_transition_from_speaking(self):
        """Test valid transitions from SPEAKING"""
        assert StateValidator.can_transition(
            PipelineState.SPEAKING,
            PipelineEvent.INTERRUPTION_DETECTED
        ) is True

        assert StateValidator.can_transition(
            PipelineState.SPEAKING,
            PipelineEvent.TTS_COMPLETE
        ) is True

    def test_get_next_state(self):
        """Test getting next state"""
        next_state = StateValidator.get_next_state(
            PipelineState.IDLE,
            PipelineEvent.START
        )

        assert next_state == PipelineState.LISTENING

    def test_validate_transition_success(self):
        """Test successful transition validation"""
        valid, next_state, error = StateValidator.validate_transition(
            PipelineState.IDLE,
            PipelineEvent.START
        )

        assert valid is True
        assert next_state == PipelineState.LISTENING
        assert error is None

    def test_validate_transition_failure(self):
        """Test failed transition validation"""
        valid, next_state, error = StateValidator.validate_transition(
            PipelineState.IDLE,
            PipelineEvent.TTS_COMPLETE
        )

        assert valid is False
        assert next_state is None
        assert error is not None
        assert "Cannot transition" in error


class TestStateTransitions:
    """Test complete state transition sequences"""

    def test_normal_conversation_flow(self):
        """Test normal conversation state flow"""
        # IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE

        # IDLE -> START -> LISTENING
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.IDLE,
            PipelineEvent.START
        )
        assert valid and next_state == PipelineState.LISTENING

        # LISTENING -> SPEECH_ENDED -> PROCESSING
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.LISTENING,
            PipelineEvent.SPEECH_ENDED
        )
        assert valid and next_state == PipelineState.PROCESSING

        # PROCESSING -> TTS_COMPLETE -> SPEAKING
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.PROCESSING,
            PipelineEvent.TTS_COMPLETE
        )
        assert valid and next_state == PipelineState.SPEAKING

        # SPEAKING -> TTS_COMPLETE -> IDLE
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.SPEAKING,
            PipelineEvent.TTS_COMPLETE
        )
        assert valid and next_state == PipelineState.IDLE

    def test_interruption_flow(self):
        """Test interruption state flow"""
        # LISTENING -> INTERRUPTION_DETECTED -> INTERRUPTED

        # LISTENING -> INTERRUPTION_DETECTED
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.LISTENING,
            PipelineEvent.INTERRUPTION_DETECTED
        )
        assert valid and next_state == PipelineState.INTERRUPTED

        # INTERRUPTED -> SPEECH_STARTED -> LISTENING
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.INTERRUPTED,
            PipelineEvent.SPEECH_STARTED
        )
        assert valid and next_state == PipelineState.LISTENING

    def test_error_flow(self):
        """Test error state flow"""
        # PROCESSING -> ERROR

        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.PROCESSING,
            PipelineEvent.ERROR
        )
        assert valid and next_state == PipelineState.ERROR

        # ERROR -> RESET -> IDLE
        valid, next_state, _ = StateValidator.validate_transition(
            PipelineState.ERROR,
            PipelineEvent.RESET
        )
        assert valid and next_state == PipelineState.IDLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
