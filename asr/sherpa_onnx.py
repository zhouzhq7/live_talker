"""
Sherpa-ONNX ASR Implementation
基于 sherpa-onnx 的流式语音识别

特点：
- 真正流式识别，低延迟
- ONNX 跨平台支持
- 多种中文模型支持
"""

import logging
import asyncio
from typing import Optional, NamedTuple, AsyncGenerator
from pathlib import Path

from .base import BaseASR, ASRResult, StreamingASRMixin

logger = logging.getLogger(__name__)


class SherpaONNXASRConfig(NamedTuple):
    """Sherpa-ONNX 配置"""
    model_path: Optional[str] = None  # ONNX 模型路径
    tokens_path: Optional[str] = None  # 词汇表路径
    num_threads: int = 1
    sample_rate: int = 16000
    feat_dim: int = 80


class SherpaONNXASR(BaseASR, StreamingASRMixin):
    """Sherpa-ONNX ASR 引擎

    支持：
    - 流式识别 (StreamingASR)
    - 非流式识别 (BaseASR)
    - 多种中文模型
    """

    def __init__(
        self,
        model_type: str = "paraformer",
        config: Optional[dict] = None,
        **kwargs
    ):
        """初始化 Sherpa-ONNX ASR

        Args:
            model_type: 模型类型 ("paraformer", "zipformer")
            config: 配置参数字典
            **kwargs: 传递给 BaseASR
        """
        super().__init__(name=f"SherpaONNX-{model_type}", **kwargs)

        self.model_type = model_type
        self.config = config or {}

        # 提取配置
        self._model_path = self.config.get("model_path")
        self._tokens_path = self.config.get("tokens_path")
        self._num_threads = self.config.get("num_threads", 1)
        self._sample_rate = self.config.get("sample_rate", 16000)

        # 运行时状态
        self._recognizer = None
        self._stream = None
        self._streaming = False

        # 检查依赖
        self._sherpa_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖是否可用"""
        try:
            import sherpa_onnx
            self._sherpa_available = True
            logger.debug("[SherpaONNXASR] sherpa-onnx available")
        except ImportError:
            logger.warning("[SherpaONNXASR] sherpa-onnx not available")

    def load_model(self) -> bool:
        """加载 Sherpa-ONNX 模型"""
        if not self._sherpa_available:
            logger.error("[SherpaONNXASR] sherpa-onnx not installed")
            return False

        try:
            import sherpa_onnx

            # 配置特征提取
            feat_config = sherpa_onnx.FeatureExtractorConfig(
                sampling_rate=self._sample_rate,
                feature_dim=80,
            )

            # 配置解码器
            decoding_config = sherpa_onnx.DecodingConfig(
                method="greedy_search",
                max_active_paths=4,
            )

            # 创建在线识别器配置
            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                feat_extractor_config=feat_config,
                decoding_config=decoding_config,
                model_path=self._model_path or "",
                tokens_path=self._tokens_path or "",
                num_threads=self._num_threads,
            )

            # 创建识别器
            self._recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)

            self._is_initialized = True
            logger.info(
                f"[SherpaONNXASR] Model loaded: "
                f"sample_rate={self._sample_rate}, "
                f"model_type={self.model_type}"
            )
            return True

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Failed to load model: {e}")
            return False

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> str:
        """非流式转写 (BaseASR 接口)

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            language: 语言

        Returns:
            str: 识别的文本
        """
        if not self._is_initialized:
            logger.warning("[SherpaONNXASR] Model not loaded")
            return ""

        try:
            import numpy as np
            import sherpa_onnx

            # 转换为 numpy 数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # 使用在线识别器
            if self._recognizer:
                # 临时创建流
                stream = self._recognizer.create_stream()

                # 输入音频
                self._recognizer.accept_waveform(
                    audio_float,
                    self._sample_rate,
                    stream
                )

                # 结束输入
                self._recognizer.input_finished(stream)

                # 解码
                result = sherpa_onnx.get_results(stream)

                if result:
                    return result.text or ""

            return ""

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Transcription failed: {e}")
            return ""

    async def stream_recognize(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ASRResult, None]:
        """流式识别 (StreamingASRMixin 接口)

        Args:
            audio_stream: 音频流生成器

        Yields:
            ASRResult: 识别结果
        """
        if not self._is_initialized:
            logger.warning("[SherpaONNXASR] Model not loaded")
            return

        self.start_stream()

        try:
            async for audio_chunk in audio_stream:
                self.accept_audio(audio_chunk)

                result = self.get_result()
                if result:
                    yield result

            # 结束流
            final_result = self.end_stream()
            if final_result:
                yield final_result

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Stream recognition failed: {e}")
        finally:
            self.reset()

    def start_stream(self) -> None:
        """开始流式识别会话"""
        if self._recognizer:
            self._stream = self._recognizer.create_stream()
            self._streaming = True
            logger.debug("[SherpaONNXASR] Stream started")

    def accept_audio(self, audio_data: bytes) -> None:
        """接收音频数据"""
        if not self._streaming or not self._recognizer or not self._stream:
            return

        try:
            import numpy as np

            # 转换为 float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # 输入音频
            self._recognizer.accept_waveform(
                audio_float,
                self._sample_rate,
                self._stream
            )

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Failed to accept audio: {e}")

    def get_result(self) -> Optional[ASRResult]:
        """获取当前结果"""
        if not self._streaming or not self._recognizer or not self._stream:
            return None

        try:
            import sherpa_onnx

            # 检查是否有部分结果
            result = sherpa_onnx.get_results(self._stream)

            if result:
                is_final = result.end_of_speech
                text = result.text or ""

                if text:
                    return ASRResult(
                        text=text,
                        is_final=is_final,
                        confidence=result.confidence or 0.0,
                    )

            return None

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Failed to get result: {e}")
            return None

    def end_stream(self) -> Optional[ASRResult]:
        """结束流式识别"""
        if not self._streaming or not self._recognizer or not self._stream:
            return None

        try:
            import sherpa_onnx

            # 标记输入结束
            self._recognizer.input_finished(self._stream)

            # 获取最终结果
            result = sherpa_onnx.get_results(self._stream)

            self._streaming = False
            self._stream = None

            if result and result.text:
                return ASRResult(
                    text=result.text,
                    is_final=True,
                    confidence=result.confidence or 0.0,
                )

            return None

        except Exception as e:
            logger.error(f"[SherpaONNXASR] Failed to end stream: {e}")
            return None

    def reset(self) -> None:
        """重置流式识别状态"""
        self._streaming = False
        self._stream = None
        logger.debug("[SherpaONNXASR] Stream reset")

    def get_info(self) -> dict:
        """获取引擎信息"""
        info = super().get_info()
        info.update({
            "model_type": self.model_type,
            "sample_rate": self._sample_rate,
            "streaming": self._streaming,
            "sherpa_available": self._sherpa_available,
        })
        return info

    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return self._is_initialized and self._sherpa_available


def create_sherpa_asr(
    model_type: str = "paraformer",
    config: Optional[dict] = None
) -> SherpaONNXASR:
    """创建 Sherpa-ONNX ASR 引擎

    Args:
        model_type: 模型类型
        config: 配置参数

    Returns:
        SherpaONNXASR: ASR 引擎实例
    """
    return SherpaONNXASR(model_type=model_type, config=config)
