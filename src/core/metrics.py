"""延迟直方图与 FPS 统计（Plan V1.0 T1 / §4）。

提供验收所需的 P50 / P95 / P99 与 FPS EMA：
- `Histogram` 采用有界环形缓冲（默认 2000 样本），避免长跑内存增长；
- 分位数使用 nearest-rank 口径，与 `bench_latency.py` 的离线统计保持一致；
- 全部操作线程安全（视觉线程与事件循环会并发打点）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "Histogram",
    "Ema",
    "Metrics",
    "get_metrics",
    "reset_metrics",
    "AdaptiveDegrader",
    "DegradeAction",
]


def _percentile(sorted_values: list[float], q: float) -> float:
    """nearest-rank 分位数（q ∈ [0,1]）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, min(len(sorted_values), int(round(q * len(sorted_values) + 0.5))))
    return sorted_values[rank - 1]


class Histogram:
    """有界延迟直方图。"""

    def __init__(self, max_samples: int = 2000) -> None:
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._count = 0
        self._total = 0.0

    def observe(self, value_ms: float) -> None:
        value = float(value_ms)
        with self._lock:
            self._samples.append(value)
            self._count += 1
            self._total += value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.observe(value)

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            values = sorted(self._samples)
            count = self._count
            total = self._total
        if not values:
            return {
                "count": count,
                "window": 0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "min": 0.0,
            }
        return {
            "count": count,
            "window": len(values),
            "mean": round(total / count, 3) if count else round(sum(values) / len(values), 3),
            "p50": round(_percentile(values, 0.50), 3),
            "p90": round(_percentile(values, 0.90), 3),
            "p95": round(_percentile(values, 0.95), 3),
            "p99": round(_percentile(values, 0.99), 3),
            "max": round(values[-1], 3),
            "min": round(values[0], 3),
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._count = 0
            self._total = 0.0


class Ema:
    """指数移动平均（用于 FPS 平滑）。"""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self._value: float | None = None
        self._lock = threading.Lock()

    def update(self, value: float) -> float:
        with self._lock:
            if self._value is None:
                self._value = float(value)
            else:
                self._value = self.alpha * float(value) + (1 - self.alpha) * self._value
            return self._value

    @property
    def value(self) -> float:
        with self._lock:
            return 0.0 if self._value is None else round(self._value, 2)

    def reset(self) -> None:
        with self._lock:
            self._value = None


class Metrics:
    """指标注册表：延迟直方图 + 计数器 + 瞬时量 + FPS EMA。"""

    def __init__(self, max_samples: int = 2000) -> None:
        self._max_samples = max_samples
        self._histograms: dict[str, Histogram] = {}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._fps = Ema(alpha=0.2)
        self._lock = threading.Lock()

    # -- 写 ---------------------------------------------------------------- #
    def histogram(self, name: str) -> Histogram:
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = Histogram(self._max_samples)
                self._histograms[name] = hist
            return hist

    def observe(self, name: str, value_ms: float) -> None:
        self.histogram(name).observe(value_ms)

    def incr(self, name: str, amount: int = 1) -> int:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount
            return self._counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe_fps(self, fps: float) -> None:
        self._fps.update(fps)

    # -- 读 ---------------------------------------------------------------- #
    @property
    def fps(self) -> float:
        return self._fps.value

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def gauges(self) -> dict[str, float]:
        with self._lock:
            return dict(self._gauges)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            histograms = dict(self._histograms)
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        return {
            "fps": self._fps.value,
            "histograms": {name: hist.stats() for name, hist in histograms.items()},
            "counters": counters,
            "gauges": gauges,
        }

    def reset(self) -> None:
        with self._lock:
            self._histograms.clear()
            self._counters.clear()
            self._gauges.clear()
        self._fps.reset()


_metrics: Metrics | None = None


def get_metrics(max_samples: int | None = None) -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics(max_samples or 2000)
    return _metrics


def reset_metrics() -> Metrics:
    global _metrics
    _metrics = Metrics(2000)
    return _metrics


# --------------------------------------------------------------------------- #
# 自适应降级（Plan V1.0 T13 / §4.1）
# --------------------------------------------------------------------------- #
@dataclass
class DegradeAction:
    """一次降级决策的完整证据（可直接进日志）。"""

    level: int
    reason: str
    p95: float
    threshold: float
    samples: int
    target_width: int
    target_height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "reason": self.reason,
            "p95_ms": round(self.p95, 3),
            "threshold_ms": self.threshold,
            "samples": self.samples,
            "target": [self.target_width, self.target_height],
        }


class AdaptiveDegrader:
    """基于单帧耗时 P95 的降级看门狗。

    幂等 + 冷却：同一 `max_level` 内最多降级一次；两次降级之间需间隔 `cooldown_s`。
    """

    def __init__(
        self,
        metrics: Metrics,
        *,
        metric_name: str = "frame_ms",
        threshold_ms: float = 25.0,
        min_samples: int = 20,
        max_level: int = 1,
        cooldown_s: float = 10.0,
        target_width: int = 480,
        target_height: int = 360,
    ) -> None:
        self.metrics = metrics
        self.metric_name = metric_name
        self.threshold_ms = float(threshold_ms)
        self.min_samples = int(min_samples)
        self.max_level = int(max_level)
        self.cooldown_s = float(cooldown_s)
        self.target_width = int(target_width)
        self.target_height = int(target_height)

        self._level = 0
        self._last_degraded_at = 0.0
        self._checks = 0
        self._history: list[DegradeAction] = []
        self._lock = threading.Lock()

    # -- 查询 -------------------------------------------------------------- #
    @property
    def level(self) -> int:
        return self._level

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            hist = self.metrics.histogram(self.metric_name).stats()
            return {
                "enabled": self.max_level > 0,
                "level": self._level,
                "max_level": self.max_level,
                "threshold_ms": self.threshold_ms,
                "min_samples": self.min_samples,
                "checks": self._checks,
                "metric": self.metric_name,
                "p95_ms": round(float(hist.get("p95", 0.0)), 3),
                "samples": int(hist.get("count", 0)),
                "last": self._history[-1].as_dict() if self._history else None,
            }

    # -- 判定 -------------------------------------------------------------- #
    def check(self, now: float | None = None) -> DegradeAction | None:
        """检查一次；返回本次新触发的降级动作，否则 None。"""
        at = time.time() if now is None else now
        with self._lock:
            self._checks += 1
            if self._level >= self.max_level:
                return None
            hist = self.metrics.histogram(self.metric_name)
            stats = hist.stats()
            samples = int(stats.get("count", 0))
            p95 = float(stats.get("p95", 0.0))
            if samples < self.min_samples:
                return None
            if p95 <= self.threshold_ms:
                return None
            if at - self._last_degraded_at < self.cooldown_s:
                return None

            self._level += 1
            self._last_degraded_at = at
            action = DegradeAction(
                level=self._level,
                reason=f"{self.metric_name} P95={p95:.1f}ms > {self.threshold_ms:.1f}ms",
                p95=p95,
                threshold=self.threshold_ms,
                samples=samples,
                target_width=self.target_width,
                target_height=self.target_height,
            )
            self._history.append(action)
            return action

    def reset(self) -> None:
        with self._lock:
            self._level = 0
            self._last_degraded_at = 0.0
            self._checks = 0
            self._history.clear()
