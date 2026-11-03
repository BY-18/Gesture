"""虚拟手势注入器（PRD §3.4 / Plan V1.0 T4、§3）。

P0 阶段尚未接入视觉层（T6/T7），本模块直接构造 **21 点归一化坐标** 与意图候选，
使「注入链路」与「真实摄像头链路」共享同一套下游（防抖层 → 控制总线 → HUD），
从而满足门禁 G0：无摄像头亦可 100% 完成闭环演示。

设计约定：
- 注入的坐标是**几何合法**的合成手型：握拳姿态满足 Plan §5.1 的 `fist` 判据，
  手掌前推/左右挥动/环形轨迹各自带一条可视化轨迹（`trail`），便于 HUD 与 P1 复用；
- `confirm_direct()` 是 P0 的**临时确认策略**：注入即代表"已确认意图"。
  T10 落地 `debounce.py` 后，此函数将被滑窗表决策略替换，签名保持不变。
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Iterable

from pydantic import BaseModel, Field

from src.api.schemas import (
    Cmd,
    ConfirmedIntent,
    GESTURE_LABELS,
    GESTURE_TO_CMD,
    GestureCandidate,
    GestureCode,
    Landmark,
    Point2D,
    Source,
)

__all__ = [
    "HAND_LANDMARK_COUNT",
    "SyntheticPose",
    "build_pose",
    "build_candidate",
    "confirm_direct",
    "landmarks_for",
    "POSE_LIBRARY",
]

#: MediaPipe Hands 关键点数量
HAND_LANDMARK_COUNT = 21

_X_MIN, _X_MAX = 0.02, 0.98
_Y_MIN, _Y_MAX = 0.02, 0.98


# --------------------------------------------------------------------------- #
# 合成手型（归一化坐标，图像 y 轴向下）
# --------------------------------------------------------------------------- #
#: 五指张开、掌心正对摄像头（G-10 / G-11 / G-12 / G-13 的基础姿态）
_OPEN_PALM: tuple[tuple[float, float, float], ...] = (
    (0.50, 0.88, 0.00),  # 0  腕
    (0.43, 0.84, -0.01),  # 1  拇指 CMC
    (0.37, 0.79, -0.02),  # 2  拇指 MCP
    (0.32, 0.74, -0.03),  # 3  拇指 IP
    (0.28, 0.69, -0.04),  # 4  拇指 TIP
    (0.46, 0.66, -0.01),  # 5  食指 MCP
    (0.45, 0.55, -0.02),  # 6  食指 PIP
    (0.44, 0.47, -0.03),  # 7  食指 DIP
    (0.44, 0.41, -0.04),  # 8  食指 TIP（轨迹追踪点）
    (0.51, 0.64, -0.01),  # 9  中指 MCP
    (0.51, 0.52, -0.02),  # 10 中指 PIP
    (0.51, 0.44, -0.03),  # 11 中指 DIP
    (0.51, 0.38, -0.04),  # 12 中指 TIP
    (0.56, 0.66, -0.01),  # 13 无名指 MCP
    (0.57, 0.55, -0.02),  # 14 无名指 PIP
    (0.57, 0.47, -0.03),  # 15 无名指 DIP
    (0.58, 0.41, -0.04),  # 16 无名指 TIP
    (0.61, 0.69, -0.01),  # 17 小指 MCP
    (0.62, 0.60, -0.02),  # 18 小指 PIP
    (0.63, 0.53, -0.03),  # 19 小指 DIP
    (0.64, 0.47, -0.04),  # 20 小指 TIP
)

#: 握拳（G-02）：四指指尖 y 均大于其 PIP 的 y，满足 Plan §5.1 的 fist 判据
_FIST: tuple[tuple[float, float, float], ...] = (
    (0.50, 0.88, 0.00),
    (0.44, 0.84, -0.01),
    (0.39, 0.80, -0.02),
    (0.36, 0.76, -0.03),
    (0.38, 0.72, -0.05),  # 4  拇指 TIP：横搭于食指第二关节
    (0.46, 0.70, -0.01),  # 5  食指 MCP
    (0.45, 0.75, -0.02),  # 6  食指 PIP
    (0.44, 0.78, -0.03),  # 7  食指 DIP
    (0.44, 0.80, -0.04),  # 8  食指 TIP  (0.80 > 0.75)
    (0.51, 0.69, -0.01),
    (0.51, 0.75, -0.02),  # 10 中指 PIP
    (0.51, 0.79, -0.03),
    (0.51, 0.81, -0.04),  # 12 中指 TIP  (0.81 > 0.75)
    (0.56, 0.70, -0.01),
    (0.57, 0.76, -0.02),  # 14 无名指 PIP
    (0.57, 0.79, -0.03),
    (0.57, 0.81, -0.04),  # 16 无名指 TIP (0.81 > 0.76)
    (0.61, 0.72, -0.01),
    (0.62, 0.77, -0.02),  # 18 小指 PIP
    (0.62, 0.80, -0.03),
    (0.62, 0.82, -0.04),  # 20 小指 TIP  (0.82 > 0.77)
)


class SyntheticPose(BaseModel):
    """一次注入所携带的可视化数据。"""

    code: GestureCode
    label: str
    landmarks: list[Landmark]
    trail: list[Point2D] = Field(default_factory=list)
    speed: float | None = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _transform(
    base: Iterable[tuple[float, float, float]],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> list[Landmark]:
    result: list[Landmark] = []
    for x, y, z in base:
        result.append(
            Landmark(
                x=round(_clamp(x + dx, _X_MIN, _X_MAX), 4),
                y=round(_clamp(y + dy, _Y_MIN, _Y_MAX), 4),
                z=round(z + dz, 4),
            )
        )
    return result


def _circle_trail(
    center: tuple[float, float] = (0.50, 0.52),
    radius: float = 0.17,
    points: int = 22,
    start_deg: float = 0.0,
) -> list[Point2D]:
    trail: list[Point2D] = []
    for i in range(points):
        angle = math.radians(start_deg + i * (340.0 / max(1, points - 1)))
        trail.append(
            Point2D(
                x=round(center[0] + radius * math.cos(angle), 4),
                y=round(center[1] + radius * math.sin(angle), 4),
            )
        )
    return trail


def _swipe_trail(
    start: tuple[float, float], end: tuple[float, float], points: int = 9
) -> list[Point2D]:
    return [
        Point2D(
            x=round(start[0] + (end[0] - start[0]) * i / (points - 1), 4),
            y=round(start[1] + (end[1] - start[1]) * i / (points - 1), 4),
        )
        for i in range(points)
    ]


def _patched(
    base: Iterable[tuple[float, float, float]], **updates: tuple[float, float, float]
) -> tuple[tuple[float, float, float], ...]:
    """在基准手型上覆盖指定关键点，用于合成扩展手势的姿态。"""
    pts = [list(p) for p in base]
    for key, value in updates.items():
        pts[int(key)] = list(value)
    return tuple(tuple(p) for p in pts)


#: 竖拇指（G-03）：四指仍蜷起，拇指尖明显高出 IP 且远离食指 MCP
_THUMB_UP_POSE = _patched(
    _FIST,
    **{
        "2": (0.42, 0.80, -0.02),
        "3": (0.40, 0.70, -0.03),
        "4": (0.39, 0.58, -0.04),
    },
)

#: 剪刀手（G-05）：食指+中指伸，无名指+小指蜷，拇指收起
_V_SIGN_POSE = _patched(
    _OPEN_PALM,
    **{
        "4": (0.36, 0.72, -0.04),
        "15": (0.57, 0.60, -0.03),
        "16": (0.58, 0.63, -0.03),
        "19": (0.63, 0.65, -0.03),
        "20": (0.64, 0.67, -0.03),
    },
)

#: 捏合（G-06）：拇指尖贴到食指尖
_PINCH_POSE = _patched(_OPEN_PALM, **{"4": (0.43, 0.43, -0.04)})


def _build_library() -> dict[GestureCode, SyntheticPose]:
    library: dict[GestureCode, SyntheticPose] = {
        GestureCode.ESTOP: SyntheticPose(
            code=GestureCode.ESTOP,
            label=GESTURE_LABELS[GestureCode.ESTOP.value],
            landmarks=_transform(_FIST),
            trail=[],
            speed=0.0,
        ),
        GestureCode.FORWARD: SyntheticPose(
            code=GestureCode.FORWARD,
            label=GESTURE_LABELS[GestureCode.FORWARD.value],
            # 单掌前推：整手向镜头（z 减小）并小幅上抬
            landmarks=_transform(_OPEN_PALM, dy=-0.06, dz=-0.06),
            trail=_swipe_trail((0.50, 0.78), (0.50, 0.40)),
            speed=1.0,
        ),
        GestureCode.TURN_LEFT: SyntheticPose(
            code=GestureCode.TURN_LEFT,
            label=GESTURE_LABELS[GestureCode.TURN_LEFT.value],
            landmarks=_transform(_OPEN_PALM, dx=-0.10),
            trail=_swipe_trail((0.66, 0.52), (0.34, 0.52)),
            speed=1.0,
        ),
        GestureCode.TURN_RIGHT: SyntheticPose(
            code=GestureCode.TURN_RIGHT,
            label=GESTURE_LABELS[GestureCode.TURN_RIGHT.value],
            landmarks=_transform(_OPEN_PALM, dx=0.10),
            trail=_swipe_trail((0.34, 0.52), (0.66, 0.52)),
            speed=1.0,
        ),
        GestureCode.SPIN: SyntheticPose(
            code=GestureCode.SPIN,
            label=GESTURE_LABELS[GestureCode.SPIN.value],
            landmarks=_transform(_OPEN_PALM, dy=-0.04),
            # 环形轨迹：22 点约占 340°，与 Plan §2.1 的"累计转角 ≥330°"判定一致
            trail=_circle_trail(),
            speed=1.0,
        ),
        # ---------------- 静态手型包 ----------------
        GestureCode.THUMB_UP: SyntheticPose(
            code=GestureCode.THUMB_UP,
            label=GESTURE_LABELS[GestureCode.THUMB_UP.value],
            landmarks=_transform(_THUMB_UP_POSE),
            trail=[],
            speed=1.0,
        ),
        GestureCode.OPEN_PALM: SyntheticPose(
            code=GestureCode.OPEN_PALM,
            label=GESTURE_LABELS[GestureCode.OPEN_PALM.value],
            landmarks=_transform(_OPEN_PALM),
            trail=[],
            speed=0.0,
        ),
        GestureCode.V_SIGN: SyntheticPose(
            code=GestureCode.V_SIGN,
            label=GESTURE_LABELS[GestureCode.V_SIGN.value],
            landmarks=_transform(_V_SIGN_POSE),
            trail=[],
            speed=1.0,
        ),
        GestureCode.PINCH: SyntheticPose(
            code=GestureCode.PINCH,
            label=GESTURE_LABELS[GestureCode.PINCH.value],
            landmarks=_transform(_PINCH_POSE),
            trail=[],
            speed=0.0,
        ),
        # ---------------- 上下挥动包 ----------------
        GestureCode.SWIPE_UP: SyntheticPose(
            code=GestureCode.SWIPE_UP,
            label=GESTURE_LABELS[GestureCode.SWIPE_UP.value],
            landmarks=_transform(_OPEN_PALM, dy=-0.05),
            trail=_swipe_trail((0.50, 0.80), (0.50, 0.34)),
            speed=0.7,
        ),
        GestureCode.SWIPE_DOWN: SyntheticPose(
            code=GestureCode.SWIPE_DOWN,
            label=GESTURE_LABELS[GestureCode.SWIPE_DOWN.value],
            landmarks=_transform(_OPEN_PALM, dy=0.05),
            trail=_swipe_trail((0.50, 0.34), (0.50, 0.80)),
            speed=0.0,
        ),
        # ---------------- 双手组合包 ----------------
        # 注入侧只携带主手坐标（HUD 渲染用）；双手判据只作用于摄像头链路
        GestureCode.TWO_PALM_PUSH: SyntheticPose(
            code=GestureCode.TWO_PALM_PUSH,
            label=GESTURE_LABELS[GestureCode.TWO_PALM_PUSH.value],
            landmarks=_transform(_OPEN_PALM, dy=-0.06, dz=-0.06),
            trail=_swipe_trail((0.50, 0.78), (0.50, 0.40)),
            speed=1.0,
        ),
        GestureCode.TWO_FIST: SyntheticPose(
            code=GestureCode.TWO_FIST,
            label=GESTURE_LABELS[GestureCode.TWO_FIST.value],
            landmarks=_transform(_FIST),
            trail=[],
            speed=0.0,
        ),
    }
    return library


POSE_LIBRARY: dict[GestureCode, SyntheticPose] = _build_library()


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def build_pose(
    code: GestureCode,
    *,
    landmarks: list[Landmark] | None = None,
    speed: float | None = None,
) -> SyntheticPose:
    """取手势对应的合成手型；传入 landmarks 时以外部坐标为准（HUD 渲染）。"""
    base = POSE_LIBRARY[code]
    return SyntheticPose(
        code=code,
        label=base.label,
        landmarks=list(landmarks) if landmarks else list(base.landmarks),
        trail=list(base.trail),
        speed=base.speed if speed is None else speed,
    )


def landmarks_for(code: GestureCode) -> list[Landmark]:
    return list(POSE_LIBRARY[code].landmarks)


def build_candidate(
    code: GestureCode,
    *,
    score: float = 1.0,
    source: Source = Source.INJECT,
    frame_id: int | None = None,
    landmarks: list[Landmark] | None = None,
    speed: float | None = None,
    trace_id: str | None = None,
) -> tuple[GestureCandidate, SyntheticPose]:
    """构造未确认候选 + 可视化姿态。"""
    pose = build_pose(code, landmarks=landmarks, speed=speed)
    candidate = GestureCandidate(
        code=code,
        score=score,
        ts=time.time(),
        source=source,
        frame_id=frame_id,
        landmarks=pose.landmarks,
        trail=pose.trail,
        reason=None if trace_id is None else f"inject:{trace_id}",
    )
    return candidate, pose


def confirm_direct(
    candidate: GestureCandidate,
    *,
    trace_id: str | None = None,
    speed: float | None = None,
    reason: str = "P0 直接确认（分级防抖 T10 未落地）",
) -> ConfirmedIntent:
    """P0 临时确认策略：注入即视为已确认意图。

    T10 落地后由 `src.agents.debounce` 的滑窗表决替换本函数，
    `ConfirmedIntent` 契约保持不变，下游零改动。
    """
    return ConfirmedIntent(
        code=candidate.code,
        cmd=GESTURE_TO_CMD[candidate.code],
        confidence=candidate.score,
        trace_id=trace_id or uuid.uuid4().hex,
        ts=time.time(),
        source=candidate.source,
        speed=speed,
        reason=reason,
    )
