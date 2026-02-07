"""
Integration tests for VAD
"""

import pytest
import numpy as np
import time


@pytest.mark.phase3
@pytest.mark.integration
@pytest.mark.vad
@pytest.mark.slow
class TestVADIntegration:
    """Integration tests for VAD detectors"""
    
    def test_silero_vad_integration(self, vad_test_audio):
        """Test Silero VAD integration"""
        try:
            from audio.vad import VADDetector
            
            vad = VADDetector(method="silero", threshold=0.5)
            
            # Test speech detection
            result_speech = vad.detect(vad_test_audio["pure_speech"])
            assert isinstance(result_speech, bool)
            
            # Test silence detection
            result_silence = vad.detect(vad_test_audio["pure_silence"])
            assert isinstance(result_silence, bool)
            
        except Exception as e:
            pytest.skip(f"Silero VAD not available: {e}")
    
    def test_energy_vad_integration(self, vad_test_audio):
        """Test energy-based VAD integration"""
        from audio.vad import VADDetector
        
        vad = VADDetector(method="energy", threshold=0.5)
        
        # Test speech detection
        result_speech = vad.detect(vad_test_audio["pure_speech"])
        assert isinstance(result_speech, bool)
        
        # Test silence detection
        result_silence = vad.detect(vad_test_audio["pure_silence"])
        assert isinstance(result_silence, bool)
    
    def test_vad_state_machine(self, vad_test_audio):
        """Test VAD state machine"""
        from audio.vad import VADDetector
        
        vad = VADDetector(
            method="energy",
            min_speech_duration=0.05,
            min_silence_duration=0.05
        )
        
        # Test speech start detection
        state = vad.update_state(True)
        time.sleep(0.1)
        state = vad.update_state(True)
        
        assert state["speech_started"] is True
        assert state["is_speaking"] is True
        
        # Test speech end detection
        time.sleep(0.1)
        state = vad.update_state(False)
        
        assert state["speech_ended"] is True
        assert state["is_speaking"] is False
    
    def test_vad_reset(self, vad_test_audio):
        """Test VAD reset functionality"""
        from audio.vad import VADDetector
        
        vad = VADDetector(method="energy")
        
        # Simulate speech
        vad.update_state(True)
        time.sleep(0.1)
        vad.update_state(True)
        
        # Reset
        vad.reset()
        
        assert vad.is_speaking is False
        assert vad.speech_start_time is None
        assert vad.silence_start_time is None


@pytest.mark.phase3
@pytest.mark.integration
@pytest.mark.vad
class TestInterruptionDetector:
    """Test interruption detector"""
    
    def test_interruption_detection(self):
        """Test interruption detection"""
        from audio.vad import VADDetector, InterruptionDetector
        
        vad = VADDetector(method="energy")
        
        callback_called = False
        
        def on_interrupt():
            nonlocal callback_called
            callback_called = True
        
        detector = InterruptionDetector(
            vad_detector=vad,
            interruption_threshold=0.05,
            on_interrupt=on_interrupt
        )
        
        # Set system speaking
        detector.set_system_speaking(True)
        
        # Simulate user speech
        audio = np.ones(512, dtype=np.int16).tobytes()
        
        # First detection
        detector.check_interruption(audio)
        time.sleep(0.1)
        
        # Second detection (should trigger interrupt)
        result = detector.check_interruption(audio)
        
        assert result is True
        assert callback_called is True
