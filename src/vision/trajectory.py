"""空间轨迹追踪与拟合（Plan V1.0 T9 / §3.1、§5.1）。

本模块只做**几何**，不做业务仲裁：
- `TrajectoryBuffer`：容量 N=30 的环形缓冲，存食指指尖 `landmark[8]` 的归一化轨迹；
- `dominant_swipe()`：水平位移判定 G-11 左挥 / G-12 右挥；
- `circle_fit()`：累计转角 + 闭合度 + 半径稳定性 + 叉乘符号一致性判定 G-13 转圈。

关键裁决（Plan §2.1）：转圈**不再等满 30 帧**，而是按 12~24 帧自适应窗口，
取"能同时满足全部判据的最小窗口"作为触发点（越小触发 = 延迟越低）。
超过 24 帧仍不达标则标记 `expired`，由调用方丢弃重算。

坐标系：MediaPipe 归一化坐标，x 向右、y 向下、z 越小越靠近镜头。
因此图像坐标下"逆时针"与数学坐标相反，本模块的方向命名一律以**图像观感**为准。
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas import GestureCode, Point2D
from src.core.config import Thresholds, get_settings

__all__ = [
    "TrajectoryPoint",
    "TrajectoryBuffer",
    "SwipeResult",
    "CircleResult",
    "dominant_swipe",
    "dominant_swipe_vertical",
    "circle_fit",
    "fit_circle",
    "arc_trend",
]

#: 触发转圈所需的最短样本数下限（Plan §2.1：12 帧 ≈400ms @30FPS）
_MIN_CIRCLE_POINTS = 3

#: 拟合半径小于该值视为退化样本（点几乎重合），直接判失败
_MIN_RADIUS = 1e-4

#: 叉乘符号一致性比较时的浮点零阈值
_CROSS_EPS = 1e-12


class _XY(Protocol):
    x: float
    y: float


def _thresholds(thresholds: Thresholds | None) -> Thresholds:
    return thresholds if thresholds is not None else get_settings().thresholds


class TrajectoryPoint(BaseModel):
    """轨迹采样点（食指指尖）。"""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float = 0.0
    ts: float
    frame_id: int = 0


class TrajectoryBuffer(BaseModel):
    """容量固定的环形轨迹窗口（Plan §2.1：容量保留 30 帧用于回溯与绘制）。"""

    model_config = ConfigDict(validate_assignment=False)

    capacity: int = Field(default=30, ge=1)
    points: list[TrajectoryPoint] = Field(default_factory=list)

    # -- 写入 -------------------------------------------------------------- #
    def push(self, point: TrajectoryPoint) -> None:
        self.points.append(point)
        overflow = len(self.points) - self.capacity
        if overflow > 0:
            del self.points[:overflow]

    def extend(self, points: Sequence[TrajectoryPoint]) -> None:
        for point in points:
            self.push(point)

    def clear(self) -> None:
        self.points.clear()

    # -- 查询 -------------------------------------------------------------- #
    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def is_full(self) -> bool:
        return len(self.points) >= self.capacity

    def recent(self, n: int | None = None) -> list[TrajectoryPoint]:
        if n is None or n >= len(self.points):
            return list(self.points)
        return self.points[-n:]

    def trail(self, n: int | None = None) -> list[Point2D]:
        return [Point2D(x=p.x, y=p.y) for p in self.recent(n)]

    def __len__(self) -> int:  # pragma: no cover - 便捷
        return len(self.points)


# --------------------------------------------------------------------------- #
# 结果模型
# --------------------------------------------------------------------------- #
class SwipeResult(BaseModel):
    detected: bool = False
    code: GestureCode | None = None
    frames: int = 0
    dx: float = 0.0
    dy: float = 0.0
    vertical_ratio: float = 0.0
    reason: str | None = None


class CircleResult(BaseModel):
    triggered: bool = False
    frames: int = 0
    cum_angle: float = 0.0
    closure: float = 0.0
    radius_cv: float = 0.0
    sign_consistency: float = 0.0
    radius: float = 0.0
    direction: str | None = None  # "cw" / "ccw"（图像观感）
    expired: bool = False
    reason: str | None = None


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _to_points(source: TrajectoryBuffer | Sequence[TrajectoryPoint]) -> list[TrajectoryPoint]:
    if isinstance(source, TrajectoryBuffer):
        return list(source.points)
    return list(source)


def fit_circle(points: Sequence[_XY]) -> tuple[float, float, float] | None:
    """代数最小二乘圆拟合（Kåsa 法）。

    返回 ``(cx, cy, r)``；点数不足或退化（共线）时返回 None。
    """
    n = len(points)
    if n < 3:
        return None
    arr = np.asarray([[p.x, p.y] for p in points], dtype=np.float64)
    mx, my = arr[:, 0].mean(), arr[:, 1].mean()
    u = arr[:, 0] - mx
    v = arr[:, 1] - my
    suu = float(np.dot(u, u))
    svv = float(np.dot(v, v))
    suv = float(np.dot(u, v))
    suuu = float(np.dot(u * u, u))
    svvv = float(np.dot(v * v, v))
    suvv = float(np.dot(u, v * v))
    svuu = float(np.dot(v, u * u))
    det = suu * svv - suv * suv
    if abs(det) < 1e-18:
        return None  # 点共线，无法定圆
    rhs_u = (suuu + suvv) / 2.0
    rhs_v = (svvv + svuu) / 2.0
    uc = (rhs_u * svv - rhs_v * suv) / det
    vc = (suu * rhs_v - suv * rhs_u) / det
    r_sq = uc * uc + vc * vc + (suu + svv) / n
    if r_sq <= 0:
        return None
    return uc + mx, vc + my, math.sqrt(r_sq)


def _turn_metrics(points: Sequence[TrajectoryPoint]) -> tuple[float, float] | None:
    """返回 ``(累计绝对转角, 叉乘符号一致性)``；点数不足返回 None。"""
    if len(points) < _MIN_CIRCLE_POINTS:
        return None
    arr = np.asarray([[p.x, p.y] for p in points], dtype=np.float64)
    deltas = np.diff(arr, axis=0)
    if deltas.shape[0] < 2:
        return None
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    d_theta = np.diff(angles)
    d_theta = (d_theta + math.pi) % (2.0 * math.pi) - math.pi
    cum_angle = float(np.degrees(np.abs(d_theta).sum()))
    crosses = deltas[:-1, 0] * deltas[1:, 1] - deltas[:-1, 1] * deltas[1:, 0]
    positive = int(np.sum(crosses > _CROSS_EPS))
    negative = int(np.sum(crosses < -_CROSS_EPS))
    total = positive + negative
    if total == 0:
        return None
    return cum_angle, max(positive, negative) / total


def arc_trend(
    window: TrajectoryBuffer | Sequence[TrajectoryPoint],
    thresholds: Thresholds | None = None,
) -> tuple[float, float]:
    """短窗"环形趋势"探测：返回 ``(累计绝对转角, 符号一致性)``。

    用于在转圈动作**尚未闭环**时提前识别"正在画弧"，从而抑制平移类候选
    （前推/左右挥）抢跑。点数不足时返回 ``(0.0, 0.0)``。
    """
    th = _thresholds(thresholds)
    pts = _to_points(window)
    n = min(len(pts), max(th.arc_gate_min_points, th.circle_min_frames))
    if n < th.arc_gate_min_points:
        return 0.0, 0.0
    metrics = _turn_metrics(pts[-n:])
    if metrics is None:
        return 0.0, 0.0
    return round(metrics[0], 2), round(metrics[1], 4)


def _circle_metrics(points: Sequence[TrajectoryPoint]) -> dict[str, Any] | None:
    """计算一组轨迹点的圆度指标。"""
    if len(points) < _MIN_CIRCLE_POINTS:
        return None
    arr = np.asarray([[p.x, p.y] for p in points], dtype=np.float64)
    deltas = np.diff(arr, axis=0)
    if deltas.shape[0] < 2:
        return None

    turn = _turn_metrics(points)
    if turn is None:
        return None
    cum_angle, sign_consistency = turn
    positive = int(np.sum((deltas[:-1, 0] * deltas[1:, 1] - deltas[:-1, 1] * deltas[1:, 0]) > _CROSS_EPS))
    negative = int(np.sum((deltas[:-1, 0] * deltas[1:, 1] - deltas[:-1, 1] * deltas[1:, 0]) < -_CROSS_EPS))
    direction = "cw" if positive >= negative else "ccw"

    fitted = fit_circle(points)
    if fitted is None:
        return None
    cx, cy, radius = fitted
    if radius < _MIN_RADIUS:
        return None

    # ③ 闭合误差：首末点距离 / 拟合半径
    gap = float(np.hypot(arr[-1, 0] - arr[0, 0], arr[-1, 1] - arr[0, 1]))
    closure = gap / radius

    # ④ 半径变异系数：各点到拟合圆心的距离离散度
    dists = np.hypot(arr[:, 0] - cx, arr[:, 1] - cy)
    mean_d = float(dists.mean())
    radius_cv = float(dists.std() / mean_d) if mean_d > 0 else float("inf")

    return {
        "cum_angle": cum_angle,
        "closure": closure,
        "radius_cv": radius_cv,
        "sign_consistency": sign_consistency,
        "radius": radius,
        "direction": direction,
    }


# --------------------------------------------------------------------------- #
# G-11 / G-12：水平挥动
# --------------------------------------------------------------------------- #
def dominant_swipe(
    window: TrajectoryBuffer | Sequence[TrajectoryPoint],
    thresholds: Thresholds | None = None,
) -> SwipeResult:
    """判定左右挥动。

    判据（Plan §5.1）：
    - 在 **≤ ``swipe_max_frames``（8）帧** 内完成；
    - 归一化水平位移 **≥ ``swipe_min_displacement``（0.12）**；
    - 垂直位移占位移总量比例 **< ``swipe_vertical_ratio``（40%）**，用于剔除斜挥；
    - **整个 8 帧窗口内水平位移单向**（``swipe_monotonic_eps`` 容差）：
      来回摆动/折返不算"一次挥动"，其回程也不得单独判为反向挥动。
      该约束同时抑制环形轨迹在 x 极值处被误判为挥动。

    方向与位移量取"能达标的最小帧数"（最快命中），响应时延最低。
    """
    th = _thresholds(thresholds)
    pts = _to_points(window)
    if len(pts) < 2:
        return SwipeResult(reason="样本不足")

    window_limit = min(th.swipe_max_frames, len(pts))
    segment = pts[-window_limit:]
    last = segment[-1]

    # 8 帧整窗的"单向性"预判（方向无关）：同时出现正向与反向位移 = 折返摆动，
    # 其回程也不得单独计为一次反向挥动；同时抑制环形轨迹在 x 极值处的误判。
    deltas = [segment[i + 1].x - segment[i].x for i in range(len(segment) - 1)]
    has_positive = any(d > th.swipe_monotonic_eps for d in deltas)
    has_negative = any(d < -th.swipe_monotonic_eps for d in deltas)
    if has_positive and has_negative:
        return SwipeResult(
            frames=window_limit,
            dx=round(last.x - segment[0].x, 4),
            dy=round(last.y - segment[0].y, 4),
            reason="8 帧窗口内出现折返，不计为单次挥动",
        )

    best = SwipeResult(reason="未达位移阈值")
    for k in range(2, window_limit + 1):
        first = segment[-k]
        dx = last.x - first.x
        dy = last.y - first.y
        total = abs(dx) + abs(dy)
        if total <= 0:
            continue
        ratio = abs(dy) / total
        if abs(dx) < th.swipe_min_displacement:
            continue
        if ratio >= th.swipe_vertical_ratio:
            best = SwipeResult(
                frames=k,
                dx=round(dx, 4),
                dy=round(dy, 4),
                vertical_ratio=round(ratio, 4),
                reason="垂直位移占比超限",
            )
            continue
        return SwipeResult(
            detected=True,
            code=GestureCode.TURN_RIGHT if dx > 0 else GestureCode.TURN_LEFT,
            frames=k,
            dx=round(dx, 4),
            dy=round(dy, 4),
            vertical_ratio=round(ratio, 4),
            reason="ok",
        )
    return best


# --------------------------------------------------------------------------- #
# G-20 / G-21：上下挥动
# --------------------------------------------------------------------------- #
def dominant_swipe_vertical(
    window: TrajectoryBuffer | Sequence[TrajectoryPoint],
    thresholds: Thresholds | None = None,
) -> SwipeResult:
    """判定上下挥动（G-20 上挥 → 前进 / G-21 下挥 → 怠速）。

    判据与左右挥动同源，只是主轴换成竖直方向：
    - 在 ≤ ``swipe_max_frames``（8）帧内完成；
    - 归一化竖直位移 ≥ ``swipe_vertical_min_displacement``（0.10）；
    - 水平位移占比 < ``swipe_horizontal_ratio``（40%），剔除斜挥；
    - 窗口内 y 方向必须单向。

    坐标系 y 向下，因此"上挥"表现为 dy < 0（`SWIPE_UP`）。
    与左右挥动互斥使用：竖直位移必须占优，否则交给水平判定。
    """
    th = _thresholds(thresholds)
    pts = _to_points(window)
    if len(pts) < 2:
        return SwipeResult(reason="样本不足")

    window_limit = min(th.swipe_max_frames, len(pts))
    segment = pts[-window_limit:]
    last = segment[-1]

    deltas = [segment[i + 1].y - segment[i].y for i in range(len(segment) - 1)]
    has_positive = any(d > th.swipe_monotonic_eps for d in deltas)
    has_negative = any(d < -th.swipe_monotonic_eps for d in deltas)
    if has_positive and has_negative:
        return SwipeResult(
            frames=window_limit,
            dy=round(last.y - segment[0].y, 4),
            reason="8 帧窗口内出现折返，不计为单次挥动",
        )

    best = SwipeResult(reason="未达位移阈值")
    for k in range(2, window_limit + 1):
        first = segment[-k]
        dx = last.x - first.x
        dy = last.y - first.y
        total = abs(dx) + abs(dy)
        if total <= 0:
            continue
        ratio = abs(dx) / total
        if abs(dy) < th.swipe_vertical_min_displacement:
            continue
        if ratio >= th.swipe_horizontal_ratio:
            best = SwipeResult(
                frames=k,
                dx=round(dx, 4),
                dy=round(dy, 4),
                vertical_ratio=round(ratio, 4),
                reason="水平位移占比超限",
            )
            continue
        return SwipeResult(
            detected=True,
            code=GestureCode.SWIPE_UP if dy < 0 else GestureCode.SWIPE_DOWN,
            frames=k,
            dx=round(dx, 4),
            dy=round(dy, 4),
            vertical_ratio=round(ratio, 4),
            reason="ok",
        )
    return best


# --------------------------------------------------------------------------- #
# G-13：环形轨迹
# --------------------------------------------------------------------------- #
def circle_fit(
    window: TrajectoryBuffer | Sequence[TrajectoryPoint],
    thresholds: Thresholds | None = None,
) -> CircleResult:
    """自适应窗口的环形轨迹判定（Plan §2.1）。

    - 只在最近 ``circle_max_frames``（24）帧内评估；
    - 从 ``circle_min_frames``（12）帧起逐帧扩张，返回**首个**满足全部判据的窗口；
    - 判据：累计转角 ≥330°、闭合误差 ≤0.18、半径 CV ≤0.35、叉乘一致性 ≥0.85；
    - 已满 24 帧仍不达标 → ``expired=True``，调用方应清空缓冲丢弃重算。
    """
    th = _thresholds(thresholds)
    pts = _to_points(window)
    if len(pts) < th.circle_min_frames:
        return CircleResult(frames=len(pts), reason="样本不足")

    segment = pts[-th.circle_max_frames :]
    n = len(segment)
    last_reason = "未达判据"
    last_metrics: dict[str, Any] | None = None

    for k in range(th.circle_min_frames, n + 1):
        sub = segment[n - k :]
        metrics = _circle_metrics(sub)
        if metrics is None:
            last_reason = "轨迹退化（共线/重合）"
            continue
        last_metrics = metrics
        if metrics["cum_angle"] < th.circle_min_cum_angle:
            last_reason = f"累计转角不足（{metrics['cum_angle']:.1f}°）"
            continue
        if metrics["closure"] > th.circle_closure_max:
            last_reason = f"闭合误差过大（{metrics['closure']:.3f}）"
            continue
        if metrics["radius_cv"] > th.circle_max_radius_cv:
            last_reason = f"半径不稳定（CV={metrics['radius_cv']:.3f}）"
            continue
        if metrics["sign_consistency"] < th.circle_sign_consistency:
            last_reason = f"旋向不一致（{metrics['sign_consistency']:.3f}）"
            continue
        return CircleResult(
            triggered=True,
            frames=k,
            cum_angle=round(metrics["cum_angle"], 2),
            closure=round(metrics["closure"], 4),
            radius_cv=round(metrics["radius_cv"], 4),
            sign_consistency=round(metrics["sign_consistency"], 4),
            radius=round(metrics["radius"], 4),
            direction=metrics["direction"],
            reason="ok",
        )

    expired = n >= th.circle_max_frames
    result = CircleResult(
        triggered=False,
        frames=n,
        expired=expired,
        reason=last_reason if not expired else f"超过 {th.circle_max_frames} 帧未达标，丢弃重算（{last_reason}）",
    )
    if last_metrics is not None:
        result.cum_angle = round(last_metrics["cum_angle"], 2)
        result.closure = round(last_metrics["closure"], 4)
        result.radius_cv = round(last_metrics["radius_cv"], 4)
        result.sign_consistency = round(last_metrics["sign_consistency"], 4)
    return result
