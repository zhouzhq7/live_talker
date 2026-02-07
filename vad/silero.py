"""
Silero VAD Implementation
基于 Silero VAD 的语音端点检测实现
"""

import logging
import os
import time
from typing import Optional

import numpy as np

from .base import VADResult, VADState, VADetector

logger = logging.getLogger(__name__)


class SileroVAD(VADetector):
    """Silero VAD 检测器

    基于深度学习的语音端点检测，支持：
    - 高准确率语音检测
    - 滑动窗口检测
    - 灵敏度动态调整
    - 状态追踪
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化 Silero VAD 检测器

        Args:
            config: VAD 配置参数
                - threshold: 检测阈值 (默认 0.5)
                - min_speech_duration: 最小语音时长 (默认 0.25s)
                - min_silence_duration: 最小静音时长 (默认 0.5s)
                - sample_rate: 采样率 (默认 16000)
                - model_cache_dir: 模型缓存目录
        """
        super().__init__(config)
        self.model = None
        self._model_loaded = False
        self._model_cache_dir = self.config.get("model_cache_dir")

        # 设置 Torch Hub 缓存
        self._setup_torch_cache()

    def _setup_torch_cache(self):
        """设置 Torch Hub 缓存目录"""
        try:
            if self._model_cache_dir:
                torch_cache = os.path.join(self._model_cache_dir, "torch")
            else:
                default_cache = os.path.expanduser("~/.cache/torch")
                torch_cache = os.path.join(default_cache, "hub")

            os.makedirs(torch_cache, exist_ok=True)
            os.environ["TORCH_HOME"] = torch_cache
            logger.debug(f"[SileroVAD] Torch cache: {torch_cache}")
        except Exception as e:
            logger.warning(f"[SileroVAD] Failed to setup torch cache: {e}")

    def load_model(self) -> bool:
        """加载 Silero VAD 模型"""
        try:
            import torch

            # 从 Torch Hub 加载 Silero VAD
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False
            )

            self._model_loaded = True
            logger.info("[SileroVAD] Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"[SileroVAD] Failed to load model: {e}")
            self._model_loaded = False
            return False

    def detect(self, audio_chunk: bytes) -> VADResult:
        """检测音频片段

        Args:
            audio_chunk: 16kHz, 16-bit PCM 音频数据

        Returns:
            VADResult: 检测结果
        """
        if not self._model_loaded:
            logger.warning("[SileroVAD] Model not loaded, returning silence")
            return VADResult(
                is_speech=False,
                confidence=0.0,
                state=VADState.SILENCE
            )

        try:
            import torch

            # 转换为 float32 tensor
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Silero VAD 需要 512 样本 (16kHz) 或 256 样本 (8kHz)
            required_samples = 512 if self.sample_rate == 16000 else 256

            # 如果音频太短，填充零
            if len(audio_float) < required_samples:
                audio_float = np.pad(
                    audio_float,
                    (0, required_samples - len(audio_float)),
                    mode='constant'
                )

            # 分块处理，取最大概率
            max_prob = 0.0
            for i in range(0, len(audio_float), required_samples):
                chunk = audio_float[i:i + required_samples]

                if len(chunk) < required_samples:
                    chunk = np.pad(
                        chunk,
                        (0, required_samples - len(chunk)),
                        mode='constant'
                    )

                audio_tensor = torch.from_numpy(chunk)
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                max_prob = max(max_prob, speech_prob)

            # 判断是否检测到语音
            is_speech = max_prob > self.threshold

            # 更新状态
            return self._update_state(is_speech, max_prob)

        except Exception as e:
            logger.error(f"[SileroVAD] Detection failed: {e}")
            return VADResult(
                is_speech=False,
                confidence=0.0,
                state=VADState.SILENCE
            )

    def _update_state(
        self,
        is_speech: bool,
        confidence: float
    ) -> VADResult:
        """更新 VAD 状态

        Args:
            is_speech: 是否检测到语音
            confidence: 置信度

        Returns:
            VADResult: 更新后的检测结果
        """
        current_time = time.time()

        if is_speech:
            # 检测到语音
            if not self._is_speaking:
                # 语音刚开始
                if self._speech_start_time is None:
                    self._speech_start_time = current_time
                    logger.debug("[SileroVAD] Speech started")

                # 检查是否达到最小语音时长
                elif current_time - self._speech_start_time >= self.min_speech_duration:
                    self._is_speaking = True
                    self._silence_start_time = None
                    return VADResult(
                        is_speech=True,
                        confidence=confidence,
                        state=VADState.SPEECH_START,
                        speech_duration=current_time - self._speech_start_time
                    )

            # 持续语音中
            if self._is_speaking:
                speech_duration = None
                if self._speech_start_time:
                    speech_duration = current_time - self._speech_start_time
                return VADResult(
                    is_speech=True,
                    confidence=confidence,
                    state=VADState.SPEAKING,
                    speech_duration=speech_duration
                )

            # 语音检测中，未达到最小时长
            return VADResult(
                is_speech=False,
                confidence=confidence,
                state=VADState.SILENCE
            )

        else:
            # 未检测到语音
            if self._is_speaking:
                # 语音刚结束
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                    logger.debug("[SileroVAD] Silence started")

                # 检查是否达到最小静音时长
                elif current_time - self._silence_start_time >= self.min_silence_duration:
                    self._is_speaking = False
                    silence_duration = current_time - self._silence_start_time
                    return VADResult(
                        is_speech=False,
                        confidence=confidence,
                        state=VADState.SPEECH_END,
                        silence_duration=silence_duration
                    )

                # 静音中，未达到最小时长
                return VADResult(
                    is_speech=True,
                    confidence=confidence,
                    state=VADState.SPEAKING
                )

            # 持续静音中
            return VADResult(
                is_speech=False,
                confidence=confidence,
                state=VADState.SILENCE
            )

    def reset(self) -> None:
        """重置 VAD 状态"""
        self._speech_start_time = None
        self._silence_start_time = None
        self._is_speaking = False
        logger.debug("[SileroVAD] State reset")

    def get_info(self) -> dict:
        """获取检测器信息"""
        info = super().get_info()
        info.update({
            "model_loaded": self._model_loaded,
            "model_source": "snakers4/silero-vad"
        })
        return info

    def is_available(self) -> bool:
        """检查检测器是否可用"""
        return self._model_loaded

    def detect_probability(self, audio_chunk: bytes) -> float:
        """获取原始语音概率

        Args:
            audio_chunk: 音频数据

        Returns:
            float: 语音概率 0-1
        """
        try:
            import torch

            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            required_samples = 512
            if len(audio_float) < required_samples:
                audio_float = np.pad(
                    audio_float,
                    (0, required_samples - len(audio_float)),
                    mode='constant'
                )

            audio_tensor = torch.from_numpy(audio_float)
            return self.model(audio_tensor, self.sample_rate).item()

        except Exception as e:
            logger.error(f"[SileroVAD] Probability detection failed: {e}")
            return 0.0
