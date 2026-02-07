"""
VAD Base Interface
定义语音端点检测的抽象接口
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Optional


class VADState(Enum):
    """VAD 状态枚举"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


class VADResult(NamedTuple):
    """VAD 检测结果"""
    is_speech: bool              # 当前是否检测到语音
    confidence: float            # 置信度 0-1
    state: VADState             # 当前状态
    speech_duration: Optional[float] = None   # 检测到的语音时长
    silence_duration: Optional[float] = None  # 检测到的静音时长


class VADetector(ABC):
    """语音端点检测抽象接口

    提供统一的 VAD 检测接口，支持多种检测引擎的切换。
    """

    def __init__(self, config: Optional[dict] = None):
        """初始化 VAD 检测器

        Args:
            config: VAD 配置参数
                - threshold: 检测阈值 (0-1)
                - min_speech_duration: 最小语音时长 (秒)
                - min_silence_duration: 最小静音时长 (秒)
                - sample_rate: 采样率
        """
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)
        self.min_speech_duration = self.config.get("min_speech_duration", 0.25)
        self.min_silence_duration = self.config.get("min_silence_duration", 0.5)
        self.sample_rate = self.config.get("sample_rate", 16000)

        # 状态追踪
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._is_speaking: bool = False

    @abstractmethod
    def load_model(self) -> bool:
        """加载检测模型

        Returns:
            bool: 加载是否成功
        """
        pass

    @abstractmethod
    def detect(self, audio_chunk: bytes) -> VADResult:
        """检测音频片段

        Args:
            audio_chunk: 16kHz, 16-bit PCM 音频数据

        Returns:
            VADResult: 检测结果
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置 VAD 状态"""
        pass

    def set_threshold(self, threshold: float) -> None:
        """设置检测灵敏度

        Args:
            threshold: 阈值 0-1 (越高越灵敏)
        """
        self.threshold = max(0.0, min(1.0, threshold))

    def get_info(self) -> dict:
        """获取检测器信息"""
        return {
            "type": self.__class__.__name__,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "min_speech_duration": self.min_speech_duration,
            "min_silence_duration": self.min_silence_duration,
        }

    def is_available(self) -> bool:
        """检查检测器是否可用"""
        return True
