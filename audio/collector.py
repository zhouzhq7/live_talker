"""
Audio Collector
实时音频采集器
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    """音频设备信息"""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float


class AudioConfig(NamedTuple):
    """音频采集配置"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024  # samples
    dtype: str = "int16"
    device: Optional[int] = None
    buffer_size: int = 30  # 环形缓冲大小（秒）


class AudioCollector(ABC):
    """音频采集抽象接口

    提供统一的音频采集接口，支持：
    - 多种音频后端 (PyAudio, Sounddevice)
    - 动态设备切换
    - 音量调节
    - 预录音缓冲
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """初始化音频采集器

        Args:
            config: 音频配置
        """
        self.config = config or AudioConfig()
        self._running = False
        self._volume: float = 1.0
        self._muted: bool = False

    @abstractmethod
    async def start(self) -> bool:
        """开始采集音频

        Returns:
            bool: 是否成功开始
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止采集音频"""
        pass

    @abstractmethod
    async def stream(self) -> AsyncGenerator[bytes, None]:
        """音频流生成器

        Yields:
            bytes: 16-bit PCM 音频数据
        """
        pass

    @abstractmethod
    def get_devices(self) -> list[AudioDeviceInfo]:
        """获取可用输入设备列表

        Returns:
            list[AudioDeviceInfo]: 设备列表
        """
        pass

    @abstractmethod
    def set_device(self, device_id: int) -> bool:
        """设置输入设备

        Args:
            device_id: 设备ID

        Returns:
            bool: 设置是否成功
        """
        pass

    def set_volume(self, volume: float) -> None:
        """设置采集音量 (0.0 - 1.0)

        Args:
            volume: 音量值
        """
        self._volume = max(0.0, min(1.0, volume))
        logger.debug(f"[AudioCollector] Volume set to {self._volume}")

    def get_volume(self) -> float:
        """获取当前音量

        Returns:
            float: 音量值 (0.0 - 1.0)
        """
        return self._volume

    def mute(self) -> None:
        """静音"""
        self._muted = True
        logger.debug("[AudioCollector] Muted")

    def unmute(self) -> None:
        """取消静音"""
        self._muted = False
        logger.debug("[AudioCollector] Unmuted")

    def is_muted(self) -> bool:
        """检查是否静音

        Returns:
            bool: 是否静音
        """
        return self._muted

    def is_running(self) -> bool:
        """检查是否正在运行

        Returns:
            bool: 是否正在运行
        """
        return self._running

    def get_info(self) -> dict:
        """获取采集器信息"""
        return {
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "chunk_size": self.config.chunk_size,
            "device": self.config.device,
            "running": self._running,
            "volume": self._volume,
            "muted": self._muted,
        }


class SounddeviceCollector(AudioCollector):
    """Sounddevice 音频采集器

    基于 sounddevice 库的音频采集实现
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """初始化 Sounddevice 采集器"""
        super().__init__(config)
        self._stream = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._buffer: asyncio.Queue = asyncio.Queue()

    async def start(self) -> bool:
        """开始采集"""
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                device=self.config.device,
                callback=self._callback,
            )

            self._stream.start()
            self._running = True
            logger.info(
                f"[SounddeviceCollector] Started: "
                f"{self.config.sample_rate}Hz, "
                f"{self.config.channels}ch"
            )
            return True

        except Exception as e:
            logger.error(f"[SounddeviceCollector] Start failed: {e}")
            return False

    def stop(self) -> None:
        """停止采集"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False
        logger.info("[SounddeviceCollector] Stopped")

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """音频流生成器"""
        while self._running:
            try:
                chunk = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )

                # 应用音量
                if self._muted:
                    yield b'\x00' * len(chunk)
                elif self._volume != 1.0:
                    import numpy as np
                    audio = np.frombuffer(chunk, dtype=np.int16)
                    audio = (audio * self._volume).astype(np.int16)
                    yield audio.tobytes()
                else:
                    yield chunk

            except asyncio.TimeoutError:
                continue

    def _callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            logger.warning(f"[SounddeviceCollector] Status: {status}")

        # 转换为 bytes 并放入队列
        chunk = indata.tobytes()
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning("[SounddeviceCollector] Queue full, dropping chunk")

    def get_devices(self) -> list[AudioDeviceInfo]:
        """获取可用设备"""
        import sounddevice as sd

        devices = []
        for i, dev in enumerate(sd.devices):
            if dev['max_input_channels'] > 0:
                devices.append(AudioDeviceInfo(
                    index=i,
                    name=dev['name'],
                    max_input_channels=dev['max_input_channels'],
                    max_output_channels=dev['max_output_channels'],
                    default_sample_rate=dev['default_samplerate'],
                ))

        return devices

    def set_device(self, device_id: int) -> bool:
        """设置设备"""
        if not 0 <= device_id < len(self.get_devices()):
            logger.error(f"[SounddeviceCollector] Invalid device: {device_id}")
            return False

        self.config = AudioConfig(
            **self.config._asdict(),
            device=device_id
        )
        logger.info(f"[SounddeviceCollector] Device set to {device_id}")
        return True


class PyaudioCollector(AudioCollector):
    """PyAudio 音频采集器

    基于 PyAudio 的音频采集实现（备选方案）
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """初始化 PyAudio 采集器"""
        super().__init__(config)
        self._stream = None

    async def start(self) -> bool:
        """开始采集"""
        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._callback,
            )

            self._stream.start_stream()
            self._running = True

            logger.info(
                f"[PyaudioCollector] Started: "
                f"{self.config.sample_rate}Hz, "
                f"{self.config.channels}ch"
            )
            return True

        except Exception as e:
            logger.error(f"[PyaudioCollector] Start failed: {e}")
            return False

    def stop(self) -> None:
        """停止采集"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if hasattr(self, '_pyaudio'):
            self._pyaudio.terminate()

        self._running = False

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """音频流生成器"""
        while self._running:
            try:
                chunk = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )
                yield chunk
            except asyncio.TimeoutError:
                continue

    def _callback(self, in_data, frame_count, time_info, status):
        """音频回调"""
        if status:
            logger.warning(f"[PyaudioCollector] Status: {status}")

        try:
            self._queue.put_nowait(in_data)
        except asyncio.QueueFull:
            pass

        return (None, pyaudio.paContinue)

    def get_devices(self) -> list[AudioDeviceInfo]:
        """获取可用设备"""
        import pyaudio

        devices = []
        p = pyaudio.PyAudio()

        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append(AudioDeviceInfo(
                        index=i,
                        name=info['name'],
                        max_input_channels=info['maxInputChannels'],
                        max_output_channels=info['maxOutputChannels'],
                        default_sample_rate=info['defaultSampleRate'],
                    ))
            except Exception:
                continue

        p.terminate()
        return devices

    def set_device(self, device_id: int) -> bool:
        """设置设备"""
        self.config = AudioConfig(
            **self.config._asdict(),
            device=device_id
        )
        return True


def create_collector(config: Optional[AudioConfig] = None) -> AudioCollector:
    """创建音频采集器

    优先使用 Sounddevice，失败则使用 PyAudio

    Args:
        config: 音频配置

    Returns:
        AudioCollector: 音频采集器实例
    """
    try:
        import sounddevice
        logger.info("[AudioCollector] Using SounddeviceCollector")
        return SounddeviceCollector(config)
    except ImportError:
        logger.warning("[AudioCollector] Sounddevice not available, trying PyAudio")

    try:
        import pyaudio
        logger.info("[AudioCollector] Using PyaudioCollector")
        return PyaudioCollector(config)
    except ImportError:
        logger.error("[AudioCollector] No audio backend available")
        raise RuntimeError("No audio backend available")
