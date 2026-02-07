"""
Ring Buffer
环形缓冲区实现
"""

import asyncio
import logging
import threading
import numpy as np
from typing import Optional, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BufferStats(NamedTuple):
    """缓冲区统计信息"""
    capacity: int           # 总容量
    available: int         # 可用空间
    used: int              # 已使用空间
    occupancy: float       # 使用率 0-1
    overruns: int          # 溢出次数
    underruns: int         # 下溢次数


class RingBuffer:
    """环形缓冲区

    用于音频数据的实时采集和处理

    特性：
    - 线程安全
    - 支持异步访问
    - 动态大小调整
    - 预录音支持
    """

    def __init__(
        self,
        max_size: int = 48000,  # 3秒 @ 16kHz
        sample_rate: int = 16000,
        pre_record_size: int = 16000,  # 1秒预录音
        dtype: str = "int16",
    ):
        """初始化环形缓冲区

        Args:
            max_size: 最大容量 (样本数)
            sample_rate: 采样率
            pre_record_size: 预录音大小 (样本数)
            dtype: 数据类型
        """
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.max_size = max_size
        self.pre_record_size = pre_record_size

        # 分配内存
        self._buffer = np.zeros(max_size, dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0

        # 统计
        self._overruns = 0
        self._underruns = 0

        # 锁
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """缓冲区容量"""
        return self.max_size

    @property
    def available(self) -> int:
        """可用空间"""
        return self.max_size - self._size

    @property
    def used(self) -> int:
        """已使用空间"""
        return self._size

    @property
    def occupancy(self) -> float:
        """使用率"""
        return self._size / self.max_size

    def write(self, data: bytes) -> int:
        """写入数据

        Args:
            data: 字节数据

        Returns:
            int: 实际写入的字节数
        """
        num_samples = len(data) // self.itemsize

        with self._lock:
            available = self.max_size - self._size

            if num_samples >= available:
                # 缓冲区满，覆盖旧数据
                self._overruns += 1
                num_samples = available

            # 写入数据
            end_pos = (self._write_pos + num_samples) % self.max_size

            if self._write_pos + num_samples <= self.max_size:
                # 不需要环形
                self._buffer[self._write_pos:self._write_pos + num_samples] = \
                    np.frombuffer(data[:num_samples * self.itemsize], dtype=self.dtype)
            else:
                # 需要环形
                first_part = self.max_size - self._write_pos
                self._buffer[self._write_pos:] = \
                    np.frombuffer(data[:first_part * self.itemsize], dtype=self.dtype)
                self._buffer[:end_pos] = \
                    np.frombuffer(data[first_part * self.itemsize:], dtype=self.dtype)

            self._write_pos = end_pos
            self._size = min(self._size + num_samples, self.max_size)

        return num_samples * self.itemsize

    async def write_async(self, data: bytes) -> int:
        """异步写入数据"""
        with self._lock:
            return self.write(data)

    def read(self, size: Optional[int] = None) -> bytes:
        """读取数据

        Args:
            size: 读取大小 (字节数)，None 表示读取全部

        Returns:
            bytes: 读取的数据
        """
        with self._lock:
            if self._size == 0:
                self._underruns += 1
                return b''

            if size is None:
                size = self._size * self.itemsize
            else:
                size = min(size, self._size * self.itemsize)

            num_samples = size // self.itemsize
            end_pos = (self._read_pos + num_samples) % self.max_size

            if self._read_pos + num_samples <= self._size:
                # 连续读取
                data = self._buffer[
                    self._read_pos:self._read_pos + num_samples
                ].tobytes()
            else:
                # 环形读取
                first_part = min(num_samples, self.max_size - self._read_pos)
                data = self._buffer[self._read_pos:].tobytes()[:first_part * self.itemsize]

                remaining = num_samples - first_part
                if remaining > 0:
                    data += self._buffer[:remaining].tobytes()

            self._read_pos = end_pos
            self._size = max(0, self._size - num_samples)

        return data

    async def read_async(self, size: Optional[int] = None) -> bytes:
        """异步读取数据"""
        with self._lock:
            return self.read(size)

    def get_pre_record(self, duration: float) -> bytes:
        """获取预录音数据

        Args:
            duration: 时长 (秒)

        Returns:
            bytes: 预录音数据
        """
        num_samples = int(duration * self.sample_rate)
        return self.read(num_samples * self.itemsize)

    def peek(self, size: int) -> bytes:
        """查看数据（不取出）

        Args:
            size: 查看大小 (字节数)

        Returns:
            bytes: 查看的数据
        """
        with self._lock:
            if self._size == 0:
                return b''

            size = min(size, self._size * self.itemsize)
            num_samples = size // self.itemsize
            end_pos = (self._read_pos + num_samples) % self.max_size

            if self._read_pos + num_samples <= self._size:
                return self._buffer[
                    self._read_pos:self._read_pos + num_samples
                ].tobytes()
            else:
                first_part = min(num_samples, self.max_size - self._read_pos)
                data = self._buffer[self._read_pos:].tobytes()[:first_part * self.itemsize]

                remaining = num_samples - first_part
                if remaining > 0:
                    data += self._buffer[:remaining].tobytes()

                return data

    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._read_pos = 0
            self._size = 0

    def get_stats(self) -> BufferStats:
        """获取缓冲区统计信息"""
        with self._lock:
            return BufferStats(
                capacity=self.max_size,
                available=self.max_size - self._size,
                used=self._size,
                occupancy=self._size / self.max_size,
                overruns=self._overruns,
                underruns=self._underruns,
            )

    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._overruns = 0
            self._underruns = 0


class AudioBuffer(RingBuffer):
    """音频环形缓冲区

    专门用于音频数据，提供音频特定的便利方法
    """

    def __init__(
        self,
        duration: float = 3.0,  # 3秒缓冲
        sample_rate: int = 16000,
        pre_record_duration: float = 1.0,  # 1秒预录音
        dtype: str = "int16",
    ):
        """初始化音频缓冲区

        Args:
            duration: 缓冲时长 (秒)
            sample_rate: 采样率
            pre_record_duration: 预录音时长 (秒)
            dtype: 数据类型
        """
        max_size = int(duration * sample_rate)
        pre_record_size = int(pre_record_duration * sample_rate)

        super().__init__(
            max_size=max_size,
            sample_rate=sample_rate,
            pre_record_size=pre_record_size,
            dtype=dtype,
        )

    @property
    def duration(self) -> float:
        """当前数据时长 (秒)"""
        return self._size / self.sample_rate

    @property
    def max_duration(self) -> float:
        """最大时长 (秒)"""
        return self.max_size / self.sample_rate

    @property
    def pre_record_duration(self) -> float:
        """预录音时长 (秒)"""
        return self.pre_record_size / self.sample_rate

    def write_samples(self, samples: np.ndarray) -> int:
        """写入样本数组

        Args:
            samples: 样本数组

        Returns:
            int: 写入的样本数
        """
        return self.write(samples.tobytes())

    def read_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        """读取样本数组

        Args:
            num_samples: 读取样本数，None 表示读取全部

        Returns:
            np.ndarray: 样本数组
        """
        if num_samples is None:
            num_samples = self._size

        data = self.read(num_samples * self.itemsize)
        return np.frombuffer(data, dtype=self.dtype)

    def get_energy(self) -> float:
        """计算当前缓冲区的能量

        Returns:
            float: RMS 能量
        """
        with self._lock:
            if self._size == 0:
                return 0.0
            samples = self._buffer[:self._size].astype(np.float32)
            return np.sqrt(np.mean(samples ** 2))

    def get_rms(self) -> float:
        """计算 RMS 值

        Returns:
            float: RMS 值
        """
        return self.get_energy()
