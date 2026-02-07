"""
Performance metrics utilities for testing
"""

import time
import numpy as np
from typing import Callable, Any, Dict
from functools import wraps


def measure_latency(
    func: Callable,
    *args,
    warmup: int = 0,
    iterations: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Measure function latency
    
    Args:
        func: Function to measure
        args: Function arguments
        warmup: Number of warmup iterations
        iterations: Number of measurement iterations
        kwargs: Function keyword arguments
    
    Returns:
        Dictionary with latency metrics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
        "iterations": iterations,
        "result": result
    }


def calculate_rtf(
    processing_time_ms: float,
    audio_duration_ms: float
) -> float:
    """
    Calculate Real-Time Factor
    
    Args:
        processing_time_ms: Processing time in milliseconds
        audio_duration_ms: Audio duration in milliseconds
    
    Returns:
        RTF (processing_time / audio_duration)
    """
    if audio_duration_ms <= 0:
        return 0.0
    return processing_time_ms / audio_duration_ms


def calculate_accuracy(
    reference: str,
    hypothesis: str,
    metric: str = "cer"  # 'cer' or 'wer'
) -> float:
    """
    Calculate accuracy between reference and hypothesis
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        metric: 'cer' (Character Error Rate) or 'wer' (Word Error Rate)
    
    Returns:
        Accuracy (1 - error_rate)
    """
    if metric == "cer":
        # Character Error Rate
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
    else:
        # Word Error Rate
        ref_chars = reference.split()
        hyp_chars = hypothesis.split()
    
    # Levenshtein distance
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )
    
    error_rate = dp[m][n] / max(len(ref_chars), 1)
    return 1 - error_rate


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using simple overlap
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Simple Jaccard similarity on character bigrams
    def get_bigrams(text):
        return set(text[i:i+2] for i in range(len(text) - 1))
    
    bigrams1 = get_bigrams(text1)
    bigrams2 = get_bigrams(text2)
    
    if not bigrams1 and not bigrams2:
        return 1.0
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    return intersection / union if union > 0 else 0.0


def check_threshold(
    value: float,
    threshold: float,
    operator: str = "<="
) -> bool:
    """
    Check if value meets threshold condition
    
    Args:
        value: Value to check
        threshold: Threshold value
        operator: Comparison operator ('<', '<=', '>', '>=', '==')
    
    Returns:
        True if condition is met
    """
    operators = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        "==": lambda x, y: x == y
    }
    
    return operators.get(operator, lambda x, y: False)(value, threshold)


def benchmark(
    func: Callable,
    *args,
    iterations: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Run benchmark on a function
    
    Args:
        func: Function to benchmark
        args: Function arguments
        iterations: Number of iterations
        kwargs: Function keyword arguments
    
    Returns:
        Benchmark results
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
        "iterations": iterations,
        "memory_delta_mb": mem_after - mem_before,
        "memory_peak_mb": mem_after,
        "result": result
    }
