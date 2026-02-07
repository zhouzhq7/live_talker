"""
Acoustic Echo Cancellation (AEC)
声学回声消除

用于在播放音频时检测用户打断
"""

import asyncio
import logging
import numpy as np
from typing import Optional, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AECConfig:
    """AEC 配置"""
    enabled: bool = True
    filter_length: int = 4096  # 滤波器长度
    echo_delay: float = 0.05  # 回声延迟估计 (秒)
    sampling_rate: int = 16000
    block_size: int = 160  # 10ms @ 16kHz


class AECProcessor:
    """声学回声消除处理器

    用于在扬声器播放时采集麦克风音频，消除回声影响

    原理：
    1. 采集扬声器播放的参考信号
    2. 采集麦克风输入（包含回声）
    3. 使用自适应滤波器消除回声
    4. 检测剩余信号中是否有人声

    应用场景：
    - 用户打断检测
    - 全双工语音通话
    """

    def __init__(self, config: Optional[AECConfig] = None):
        """初始化 AEC 处理器"""
        self.config = config or AECConfig()
        self._enabled = self.config.enabled

        # 初始化滤波器
        self._filter_length = self.config.filter_length
        self._filter_coeffs = np.zeros(self._filter_length)

        # 缓冲区
        self._echo_buffer = np.zeros(self._filter_length + self.config.block_size)
        self._mic_buffer = np.zeros(self.config.block_size)

        # 统计
        self._echo_suppression = 0.0
        self._convergence = 0.0

    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._enabled

    def enable(self) -> None:
        """启用 AEC"""
        self._enabled = True
        logger.info("[AEC] Enabled")

    def disable(self) -> None:
        """禁用 AEC"""
        self._enabled = False
        logger.info("[AEC] Disabled")

    def process(
        self,
        mic_audio: bytes,
        playback_audio: bytes
    ) -> tuple[bytes, float]:
        """处理音频

        Args:
            mic_audio: 麦克风音频 (16-bit PCM)
            playback_audio: 播放参考音频 (16-bit PCM)

        Returns:
            tuple: (消除后的麦克风音频, 回声抑制量 dB)
        """
        if not self._enabled:
            return mic_audio, 0.0

        try:
            # 转换为 numpy 数组
            mic = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
            playback = np.frombuffer(playback_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # 确保长度一致
            if len(playback) != len(mic):
                if len(playback) > len(mic):
                    playback = playback[:len(mic)]
                else:
                    playback = np.pad(playback, (0, len(mic) - len(playback)))

            # 自适应滤波
            echo_estimate = self._adaptive_filter(mic, playback)

            # 消除回声
            mic_clean = mic - echo_estimate

            # 计算回声抑制
            echo_power = np.mean(playback ** 2)
            estimate_power = np.mean(echo_estimate ** 2)
            if estimate_power > 0:
                self._echo_suppression = 10 * np.log10(echo_power / (estimate_power + 1e-10))
            else:
                self._echo_suppression = 0.0

            # 转换回 bytes
            mic_clean = np.clip(mic_clean * 32768, -32768, 32767).astype(np.int16)
            return mic_clean.tobytes(), self._echo_suppression

        except Exception as e:
            logger.error(f"[AEC] Process error: {e}")
            return mic_audio, 0.0

    def _adaptive_filter(
        self,
        mic: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """自适应滤波器

        使用 NLMS (Normalized Least Mean Squares) 算法

        Args:
            mic: 麦克风信号
            reference: 参考信号（播放音频）

        Returns:
            np.ndarray: 估计的回声
        """
        mu = 0.5  # 学习率

        # 更新缓冲区
        self._echo_buffer = np.roll(self._echo_buffer, -len(reference))
        self._echo_buffer[-len(reference):] = reference

        # 提取参考信号片段
        if len(reference) <= self._filter_length:
            ref_segment = np.concatenate([
                self._echo_buffer[-self._filter_length:-len(reference)],
                reference
            ])
        else:
            ref_segment = reference[-self._filter_length:]

        # NLMS 更新
        norm = np.dot(ref_segment, ref_segment) + 1e-8
        error = mic[:len(reference)] - np.dot(self._filter_coeffs, ref_segment)

        # 更新滤波器系数
        self._filter_coeffs += mu * error * ref_segment / norm

        # 计算回声估计
        echo_estimate = np.convolve(
            self._filter_coeffs[:len(reference)],
            reference
        )[:len(reference)]

        return echo_estimate

    def reset(self) -> None:
        """重置滤波器状态"""
        self._filter_coeffs = np.zeros(self._filter_length)
        self._echo_buffer = np.zeros(self._filter_length + self.config.block_size)
        self._echo_suppression = 0.0
        self._convergence = 0.0
        logger.info("[AEC] Reset")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "enabled": self._enabled,
            "echo_suppression_db": self._echo_suppression,
            "convergence": self._convergence,
            "filter_length": self._filter_length,
        }


class WebRTCAEC:
    """WebRTC AEC 实现

    基于 WebRTC 的声学回声消除
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        block_size: int = 160,
        filter_length: int = 4096,
    ):
        """初始化 WebRTC AEC

        Args:
            sampling_rate: 采样率
            block_size: 块大小
            filter_length: 滤波器长度
        """
        self._enabled = True
        self._sampling_rate = sampling_rate
        self._block_size = block_size
        self._filter_length = filter_length

        # 尝试导入 webrtc-aec
        try:
            import webrtc_aec
            self._aec = webrtc_aec.Aec(
                sampling_rate=sampling_rate,
                frame_size=block_size,
                filter_length=filter_length,
            )
            self._available = True
            logger.info("[WebRTCAEC] Initialized")
        except ImportError:
            logger.warning("[WebRTCAEC] webrtc-aec not available")
            self._available = False
            self._aec = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def process(
        self,
        mic_audio: bytes,
        playback_audio: bytes
    ) -> bytes:
        """处理音频"""
        if not self._enabled or not self._available:
            return mic_audio

        try:
            mic = np.frombuffer(mic_audio, dtype=np.int16)
            playback = np.frombuffer(playback_audio, dtype=np.int16)

            # 确保长度一致
            if len(playback) != len(mic):
                if len(playback) > len(mic):
                    playback = playback[:len(mic)]
                else:
                    playback = np.pad(playback, (0, len(mic) - len(playback)))

            # AEC 处理
            self._aec.process(
                nearend=mic,
                farend=playback,
                output_length=len(mic)
            )

            # 获取处理后的音频
            processed = self._aec.get_output()

            if processed is not None:
                return processed.tobytes()

        except Exception as e:
            logger.error(f"[WebRTCAEC] Process error: {e}")

        return mic_audio

    def reset(self) -> None:
        if self._available and self._aec:
            self._aec.reset()

    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "available": self._available,
            "sampling_rate": self._sampling_rate,
            "block_size": self._block_size,
        }


class InterruptionDetector:
    """打断检测器

    检测用户在系统播放时是否说话

    特性：
    - AEC 回声消除
    - VAD 语音检测
    - 持续性检测（避免误触发）
    """

    def __init__(
        self,
        threshold: float = 0.3,  # 300ms 语音触发打断
        sensitivity: float = 0.5,  # 检测灵敏度
        use_aec: bool = True,
        vad_config: Optional[dict] = None,
        aec_config: Optional[AECConfig] = None,
    ):
        """初始化打断检测器

        Args:
            threshold: 触发打断的语音持续时长 (秒)
            sensitivity: VAD 灵敏度 (0-1)
            use_aec: 是否使用 AEC
            vad_config: VAD 配置
            aec_config: AEC 配置
        """
        self._threshold = threshold
        self._sensitivity = sensitivity
        self._enabled = True

        # VAD
        self._vad_config = vad_config or {
            "threshold": sensitivity,
            "min_speech_duration": 0.1,
            "min_silence_duration": 0.1,
        }

        # AEC
        if use_aec:
            self._aec_config = aec_config or AECConfig()
            self._aec = AECProcessor(self._aec_config)
        else:
            self._aec = None

        # 状态
        self._speech_start_time: Optional[float] = None
        self._is_interrupting = False

        # 回调
        self._on_interrupt = None

    def set_callback(self, callback) -> None:
        """设置打断回调

        Args:
            callback: 回调函数 (on_interrupt())
        """
        self._on_interrupt = callback

    def enable(self) -> None:
        """启用打断检测"""
        self._enabled = True
        logger.info("[InterruptionDetector] Enabled")

    def disable(self) -> None:
        """禁用打断检测"""
        self._enabled = False
        logger.info("[InterruptionDetector] Disabled")

    def check(
        self,
        mic_audio: bytes,
        playback_audio: Optional[bytes] = None
    ) -> tuple[bool, float]:
        """检查是否检测到打断

        Args:
            mic_audio: 麦克风音频
            playback_audio: 播放参考音频（用于 AEC）

        Returns:
            tuple: (是否打断, 检测到的语音时长)
        """
        if not self._enabled:
            return False, 0.0

        import time

        # AEC 处理
        if self._aec is not None and playback_audio is not None:
            processed_audio, _ = self._aec.process(mic_audio, playback_audio)
        else:
            processed_audio = mic_audio

        # VAD 检测
        from vad import SileroVAD
        vad = SileroVAD(self._vad_config)
        result = vad.detect(processed_audio)

        # 持续性检测
        if result.is_speech:
            if self._speech_start_time is None:
                self._speech_start_time = time.time()

            speech_duration = time.time() - self._speech_start_time

            if speech_duration >= self._threshold:
                self._is_interrupting = True

                if self._on_interrupt:
                    try:
                        self._on_interrupt()
                    except Exception as e:
                        logger.error(f"[InterruptionDetector] Callback error: {e}")

                return True, speech_duration

            return False, speech_duration

        else:
            self._speech_start_time = None
            self._is_interrupting = False
            return False, 0.0

    def reset(self) -> None:
        """重置状态"""
        self._speech_start_time = None
        self._is_interrupting = False

        if self._aec is not None:
            self._aec.reset()

        logger.info("[InterruptionDetector] Reset")

    def get_stats(self) -> dict:
        """获取统计信息"""
        import time

        stats = {
            "enabled": self._enabled,
            "threshold": self._threshold,
            "sensitivity": self._sensitivity,
            "is_interrupting": self._is_interrupting,
        }

        if self._speech_start_time is not None:
            stats["current_speech_duration"] = time.time() - self._speech_start_time

        if self._aec is not None:
            stats["aec"] = self._aec.get_stats()

        return stats
