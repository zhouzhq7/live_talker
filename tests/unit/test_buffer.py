"""
Unit Tests for RingBuffer
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.buffer import RingBuffer, AudioBuffer


class TestRingBuffer:
    """Test cases for RingBuffer"""

    def test_init(self):
        """Test buffer initialization"""
        buffer = RingBuffer(max_size=48000, sample_rate=16000)
        assert buffer.capacity == 48000
        assert buffer.used == 0
        assert buffer.available == 48000
        assert buffer.occupancy == 0.0

    def test_write_read(self):
        """Test basic write and read"""
        buffer = RingBuffer(max_size=48000)

        # Generate test data
        test_data = np.zeros(1000, dtype=np.int16).tobytes()

        # Write
        written = buffer.write(test_data)
        assert written == len(test_data)
        assert buffer.used == 1000

        # Read
        read_data = buffer.read(len(test_data))
        assert read_data == test_data
        assert buffer.used == 0

    def test_circular_write(self):
        """Test circular buffer wrap-around"""
        buffer = RingBuffer(max_size=1000)

        # Write 600 samples (1200 bytes for int16)
        test_data = np.zeros(600, dtype=np.int16).tobytes()
        buffer.write(test_data)
        assert buffer.used == 600  # 600 samples

        # Read 200 samples (400 bytes)
        buffer.read(400)
        assert buffer.used == 400  # 600 - 200 = 400 samples

        # Write 500 samples (1000 bytes)
        test_data2 = np.ones(500, dtype=np.int16).tobytes()
        buffer.write(test_data2)

        # Buffer should be 900 samples (400 + 500)
        assert buffer.used == 900

    def test_overwrite(self):
        """Test buffer overwrite when full"""
        buffer = RingBuffer(max_size=1000)

        # Write more than capacity
        test_data = np.zeros(1500, dtype=np.int16).tobytes()
        written = buffer.write(test_data)

        # Should only write 1000 bytes
        assert written == 2000  # 1500 samples * 2 bytes
        assert buffer.used == 1000
        assert buffer.occupancy == 1.0

    def test_clear(self):
        """Test buffer clear"""
        buffer = RingBuffer(max_size=48000)

        test_data = np.zeros(1000, dtype=np.int16).tobytes()
        buffer.write(test_data)
        assert buffer.used == 1000

        buffer.clear()
        assert buffer.used == 0

    def test_get_stats(self):
        """Test buffer statistics"""
        buffer = RingBuffer(max_size=48000)

        test_data = np.zeros(1000, dtype=np.int16).tobytes()
        buffer.write(test_data)

        stats = buffer.get_stats()
        assert stats.capacity == 48000
        assert stats.used == 1000
        assert stats.overruns == 0
        assert stats.underruns == 0

    def test_read_empty(self):
        """Test reading from empty buffer"""
        buffer = RingBuffer(max_size=48000)

        read_data = buffer.read(100)
        assert read_data == b''


class TestAudioBuffer:
    """Test cases for AudioBuffer"""

    def test_init(self):
        """Test AudioBuffer initialization"""
        buffer = AudioBuffer(duration=3.0, sample_rate=16000)
        assert buffer.sample_rate == 16000
        assert buffer.max_duration == 3.0

    def test_duration(self):
        """Test duration property"""
        buffer = AudioBuffer(duration=3.0, sample_rate=16000)

        test_data = np.zeros(8000, dtype=np.int16).tobytes()  # 0.5s
        buffer.write(test_data)

        assert abs(buffer.duration - 0.5) < 0.01

    def test_pre_record(self):
        """Test pre-record functionality"""
        buffer = AudioBuffer(
            duration=3.0,
            sample_rate=16000,
            pre_record_duration=1.0
        )

        test_data = np.zeros(8000, dtype=np.int16).tobytes()
        buffer.write(test_data)

        pre_record = buffer.get_pre_record(0.5)
        # 0.5s @ 16kHz, 16-bit = 16000 bytes
        assert len(pre_record) == 16000  # 0.5s @ 16kHz

    def test_rms_calculation(self):
        """Test RMS energy calculation"""
        buffer = AudioBuffer()

        # Silent audio
        silent = np.zeros(1000, dtype=np.int16).tobytes()
        buffer.write(silent)
        rms = buffer.get_rms()
        assert rms < 0.001

        # Audio with some energy
        audio = np.ones(1000, dtype=np.int16).tobytes()
        buffer.write(audio)
        rms = buffer.get_rms()
        assert rms > 0.001

    def test_write_samples(self):
        """Test writing numpy samples directly"""
        buffer = AudioBuffer()

        samples = np.random.randint(-1000, 1000, 1000, dtype=np.int16)
        buffer.write_samples(samples)

        assert buffer.used == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
