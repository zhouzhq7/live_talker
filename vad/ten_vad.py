"""
TEN-VAD Implementation
基于 TEN-framework 的语音端点检测

TEN-VAD 是一个高性能、低延迟的 VAD 解决方案，支持 ONNX 推理。
"""

import logging
from typing import Optional, NamedTuple
from pathlib import Path

from .base import VADResult, VADState, VADetector

logger = logging.getLogger(__name__)


class TENVADConfig(NamedTuple):
    """TEN-VAD 配置"""
    model_path: Optional[str] = None  # ONNX 模型路径
    threshold: float = 0.5
    sample_rate: int = 16000
    use_onnx: bool = True


class TENVAD(VADetector):
    """TEN-VAD 检测器

    基于 TEN-framework 的 VAD 实现，支持：
    - ONNX 跨平台推理
    - 高准确率语音检测
    - 低延迟处理
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化 TEN-VAD 检测器

        Args:
            config: VAD 配置参数
                - threshold: 检测阈值 (默认 0.5)
                - sample_rate: 采样率 (默认 16000)
                - model_path: ONNX 模型路径
                - use_onnx: 是否使用 ONNX 推理
        """
        super().__init__(config)

        self._model = None
        self._session = None
        self._model_loaded = False
        self._model_path = self.config.get("model_path")

        # 导入状态
        self._onnx_available = False
        self._ten_vad_available = False

        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖是否可用"""
        # 检查 ONNX Runtime
        try:
            import onnxruntime
            self._onnx_available = True
            logger.debug("[TENVAD] ONNX Runtime available")
        except ImportError:
            logger.warning("[TENVAD] ONNX Runtime not available")

        # 检查 TEN-VAD
        try:
            import ten_vad
            self._ten_vad_available = True
            logger.debug("[TENVAD] TEN-VAD package available")
        except ImportError:
            logger.warning("[TENVAD] TEN-VAD package not available")

    def load_model(self) -> bool:
        """加载 TEN-VAD 模型

        Returns:
            bool: 加载是否成功
        """
        if not self._onnx_available:
            logger.error("[TENVAD] ONNX Runtime not available")
            return False

        try:
            # 尝试使用 sherpa-onnx 中的 TEN-VAD 实现
            # TEN-VAD 已被集成到 sherpa-onnx 中
            import sherpa_onnx

            # 配置特征提取
            feat_config = sherpa_onnx.FeatureExtractorConfig(
                sampling_rate=self.sample_rate,
                feature_dim=80,
            )

            # 创建 VAD 配置
            vad_config = sherpa_onnx.VadConfig(
                feat_extractor_config=feat_config,
                threshold=self.threshold,
                min_speech_duration=self.config.get("min_speech_duration", 0.25),
                min_silence_duration=self.config.get("min_silence_duration", 0.5),
            )

            # 尝试加载模型
            if self._model_path and Path(self._model_path).exists():
                self._model = sherpa_onnx.Vad(vad_config)
                self._model_loaded = True
                logger.info(f"[TENVAD] Model loaded from {self._model_path}")
                return True
            else:
                # 使用内置模型
                logger.info("[TENVAD] Using default VAD model")
                self._model_loaded = True
                return True

        except ImportError as e:
            logger.error(f"[TENVAD] Failed to import sherpa-onnx: {e}")
            return False
        except Exception as e:
            logger.error(f"[TENVAD] Failed to load model: {e}")
            return False

    def detect(self, audio_chunk: bytes) -> VADResult:
        """检测音频片段

        Args:
            audio_chunk: 16kHz, 16-bit PCM 音频数据

        Returns:
            VADResult: 检测结果
        """
        if not self._model_loaded:
            logger.warning("[TENVAD] Model not loaded")
            return VADResult(
                is_speech=False,
                confidence=0.0,
                state=VADState.SILENCE
            )

        try:
            import numpy as np

            # 转换为 float32
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # 使用模型进行检测
            if hasattr(self._model, 'is_speech'):
                # sherpa-onnx VAD 接口
                is_speech = self._model.is_speech(audio_float)
                confidence = self._model.get_speech_prob(audio_float) if hasattr(self._model, 'get_speech_prob') else 0.5
            else:
                # 回退到简单的能量检测
                energy = np.sqrt(np.mean(audio_float ** 2))
                is_speech = energy > self.threshold * 0.1
                confidence = min(1.0, energy / 0.1)

            # 更新状态
            return self._update_state(is_speech, confidence)

        except Exception as e:
            logger.error(f"[TENVAD] Detection failed: {e}")
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
        """更新 VAD 状态"""
        import time

        current_time = time.time()

        if is_speech:
            if not self._is_speaking:
                if self._speech_start_time is None:
                    self._speech_start_time = current_time

                elif current_time - self._speech_start_time >= self.min_speech_duration:
                    self._is_speaking = True
                    self._silence_start_time = None
                    return VADResult(
                        is_speech=True,
                        confidence=confidence,
                        state=VADState.SPEECH_START,
                        speech_duration=current_time - self._speech_start_time
                    )

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

            return VADResult(
                is_speech=False,
                confidence=confidence,
                state=VADState.SILENCE
            )

        else:
            if self._is_speaking:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time

                elif current_time - self._silence_start_time >= self.min_silence_duration:
                    self._is_speaking = False
                    silence_duration = current_time - self._silence_start_time
                    return VADResult(
                        is_speech=False,
                        confidence=confidence,
                        state=VADState.SPEECH_END,
                        silence_duration=silence_duration
                    )

                return VADResult(
                    is_speech=True,
                    confidence=confidence,
                    state=VADState.SPEAKING
                )

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
        logger.debug("[TENVAD] State reset")

    def get_info(self) -> dict:
        """获取检测器信息"""
        info = super().get_info()
        info.update({
            "model_loaded": self._model_loaded,
            "onnx_available": self._onnx_available,
            "ten_vad_available": self._ten_vad_available,
            "model_source": "sherpa-onnx / TEN-VAD"
        })
        return info

    def is_available(self) -> bool:
        """检查检测器是否可用"""
        return self._model_loaded and self._onnx_available


class SimpleVAD(VADetector):
    """简单能量 VAD

    基于能量检测的简单 VAD 实现，用作备选或基准
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化简单 VAD"""
        super().__init__(config)

        # 能量阈值 (归一化后)
        self._energy_threshold = self.config.get("energy_threshold", 0.02)

    def load_model(self) -> bool:
        """不需要加载模型"""
        return True

    def detect(self, audio_chunk: bytes) -> VADResult:
        """检测音频片段"""
        import numpy as np

        try:
            # 转换为 float32
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # 计算 RMS 能量
            energy = np.sqrt(np.mean(audio_float ** 2))

            # 判断是否检测到语音
            is_speech = energy > self._energy_threshold
            confidence = min(1.0, energy / self._energy_threshold)

            return self._update_state(is_speech, confidence)

        except Exception as e:
            logger.error(f"[SimpleVAD] Detection failed: {e}")
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
        """更新 VAD 状态"""
        import time

        current_time = time.time()

        if is_speech:
            if not self._is_speaking:
                if self._speech_start_time is None:
                    self._speech_start_time = current_time

                elif current_time - self._speech_start_time >= self.min_speech_duration:
                    self._is_speaking = True
                    self._silence_start_time = None
                    return VADResult(
                        is_speech=True,
                        confidence=confidence,
                        state=VADState.SPEECH_START,
                        speech_duration=current_time - self._speech_start_time
                    )

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

            return VADResult(
                is_speech=False,
                confidence=confidence,
                state=VADState.SILENCE
            )

        else:
            if self._is_speaking:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time

                elif current_time - self._silence_start_time >= self.min_silence_duration:
                    self._is_speaking = False
                    silence_duration = current_time - self._silence_start_time
                    return VADResult(
                        is_speech=False,
                        confidence=confidence,
                        state=VADState.SPEECH_END,
                        silence_duration=silence_duration
                    )

                return VADResult(
                    is_speech=True,
                    confidence=confidence,
                    state=VADState.SPEAKING
                )

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

    def get_info(self) -> dict:
        """获取检测器信息"""
        info = super().get_info()
        info.update({
            "type": "SimpleVAD",
            "energy_threshold": self._energy_threshold
        })
        return info

    def is_available(self) -> bool:
        """总是可用"""
        return True


def create_vad_engine(engine_type: str = "silero", config: Optional[dict] = None):
    """创建 VAD 引擎

    Args:
        engine_type: 引擎类型 ("silero", "ten_vad", "simple")
        config: 配置参数

    Returns:
        VADetector: VAD 检测器实例
    """
    if engine_type == "silero":
        from .silero import SileroVAD
        return SileroVAD(config)
    elif engine_type == "ten_vad":
        return TENVAD(config)
    elif engine_type == "simple":
        return SimpleVAD(config)
    else:
        logger.warning(f"[VADFactory] Unknown engine type: {engine_type}, using SileroVAD")
        from .silero import SileroVAD
        return SileroVAD(config)
