"""单帧感知流水线（Plan V1.0 T13）：一帧 21 点 → 手势候选。

这是视觉层与热路径之间的唯一接缝：`capture → hand_detector → pipeline → debounce`。
真实摄像头流、`/api/frames` 离线回放、虚拟注入三者的候选产出必须完全一致。

**候选优先级**（依据 Plan §15.5 移交事项）：

    急停 G-02 > 转圈 G-13 > 前推 G-10 > 左右挥 G-11/G-12

理由：环形轨迹在 x 极值附近的片段本身满足"单向位移"，若不做优先级仲裁，
转圈过程会被先判成一次转向；急停必须永远最高优先。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.api.schemas import (
    FrameSignals,
    GestureCandidate,
    GestureCode,
    Landmark,
    Source,
)
from src.core.config import Settings, get_settings
from src.vision.classifier import (
    PushWindow,
    extended_count,
    finger_states,
    fist_score,
    hand_size,
    is_fist,
    is_open_palm,
    is_pinch,
    is_thumb_up,
    is_v_sign,
    push_score,
)
from src.vision.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    arc_trend,
    circle_fit,
    dominant_swipe,
    dominant_swipe_vertical,
)

__all__ = ["PipelineFrame", "FramePipeline", "INDEX_TIP"]

#: 食指尖索引（轨迹追踪点）
INDEX_TIP = 8

#: 各类候选的几何置信度基线（已过几何阈值即视为"检出"）
_SCORE_ESTOP = 0.95
_SCORE_SPIN = 0.95
_SCORE_SWIPE = 0.92


@dataclass
class PipelineFrame:
    """单帧流水线输出。"""

    candidate: GestureCandidate | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    frame_id: int = 0


class FramePipeline:
    """有状态（按会话）的单帧候选提取器。非线程安全：按会话串行调用。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        th = self.settings.thresholds
        self.window = TrajectoryBuffer(capacity=th.window_capacity)
        self.push_window = PushWindow(capacity=max(2, th.push_consecutive_frames))
        self._frame_id = 0
        self._last_ts: float | None = None
        self._gap_reset_ms = float(th.gap_reset_ms)

    # -- 生命周期 ---------------------------------------------------------- #
    def reset(self) -> None:
        self.window.clear()
        self.push_window.clear()
        self._frame_id = 0
        self._last_ts = None

    def next_frame_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    def _check_gap(self, at: float) -> bool:
        """相邻帧间隔超时 → 清空轨迹与位移窗口（手势中断）。

        没有这一步，上一段手势的残留样本会与新动作拼成一条"连续轨迹"，
        导致转圈在远不足 12 帧时就"命中"（P2 实测过的串扰）。
        """
        previous = self._last_ts
        self._last_ts = at
        if previous is None:
            return False
        gap_ms = (at - previous) * 1000.0
        if gap_ms <= self._gap_reset_ms:
            return False
        self.window.clear()
        self.push_window.clear()
        return True

    # -- 逐帧 -------------------------------------------------------------- #
    def ingest(
        self,
        landmarks: Sequence[Landmark],
        *,
        ts: float | None = None,
        frame_id: int | None = None,
        score: float = 1.0,
        source: Source = Source.GESTURE,
        explicit: FrameSignals | None = None,
        second: Sequence[Landmark] | None = None,
    ) -> PipelineFrame:
        """处理一帧并返回候选。

        `explicit` 非空时跳过几何判定（回放/契约测试）；
        `second` 为同帧的第二只手（双手组合包用），为空表示单手。
        """
        started = time.perf_counter()
        th = self.settings.thresholds
        at = time.time() if ts is None else ts
        fid = self.next_frame_id() if frame_id is None else frame_id
        gap_reset = self._check_gap(at)

        if explicit is not None:
            return self._explicit_frame(explicit, at=at, fid=fid, score=score, source=source, started=started, gap_reset=gap_reset)

        # ① 静态：握拳（急停）+ 静态手型包
        fist_value = fist_score(landmarks)
        fist_hit = is_fist(landmarks)
        static = {
            "thumb_up": is_thumb_up(landmarks),
            "open_palm": is_open_palm(landmarks),
            "v_sign": is_v_sign(landmarks),
            "pinch": is_pinch(landmarks),
        }
        size = hand_size(landmarks)
        t_fist = time.perf_counter()

        # ② 时序：前推（手腕 z / y 位移）
        self.push_window.push(list(landmarks))
        push = push_score(self.push_window.frames)
        t_push = time.perf_counter()

        # ③ 时序：轨迹（指尖位移与圆拟合）
        tip = landmarks[INDEX_TIP]
        self.window.push(TrajectoryPoint(x=tip.x, y=tip.y, z=tip.z, ts=at, frame_id=fid))
        swipe = dominant_swipe(self.window)
        vertical = dominant_swipe_vertical(self.window)
        circle = circle_fit(self.window)
        arc_angle, arc_consistency = arc_trend(self.window)
        t_traj = time.perf_counter()

        # ④ 双手组合：只有确实拿到两只手时才评估
        two_fist = bool(
            second is not None
            and hand_size(second) >= th.two_hand_min_size
            and size >= th.two_hand_min_size
            and is_fist(landmarks)
            and is_fist(second)
        )
        two_push = bool(second is not None and static["open_palm"] and is_open_palm(second) and push.detected)

        # 张掌=怠速只对"静止的手"成立：手在移动时交给挥动/前推/转圈判定，
        # 否则移动过程中的每一帧都会产出 IDLE 候选，既污染表决窗口
        # （把 3 帧表决拖成 4 帧）又语义错误。
        recent_pts = self.window.recent(th.open_palm_still_frames)
        # 默认"非静止"：样本不足时无法证明手是停着的，不能据此产出怠速候选
        still = False
        if len(recent_pts) >= 2:
            xs = [p.x for p in recent_pts]
            ys = [p.y for p in recent_pts]
            still = max(max(xs) - min(xs), max(ys) - min(ys)) <= th.open_palm_still_eps

        # 环形趋势闸门：正在画弧时抑制平移类候选（见模块头注释）
        arc_active = (
            arc_angle >= self.settings.thresholds.arc_gate_angle
            and arc_consistency >= self.settings.thresholds.circle_sign_consistency
        )

        signals: dict[str, Any] = {
            "fist": fist_hit,
            "fist_score": round(fist_value, 3),
            "hand_size": round(size, 4),
            "fingers": finger_states(landmarks),
            "extended": extended_count(landmarks),
            "thumb_up": static["thumb_up"],
            "open_palm": static["open_palm"],
            "still": still,
            "v_sign": static["v_sign"],
            "pinch": static["pinch"],
            "two_hand": second is not None,
            "two_fist": two_fist,
            "two_push": two_push,
            "push": push.model_dump(),
            "swipe": swipe.model_dump(mode="json"),
            "vertical": vertical.model_dump(mode="json"),
            "circle": circle.model_dump(mode="json"),
            "arc": {"angle": arc_angle, "consistency": arc_consistency, "active": arc_active},
            "window": self.window.size,
            "gap_reset": gap_reset,
        }

        code: GestureCode | None = None
        confidence = 0.0
        reason = "no_signal"

        # 优先级：安全类（STOP）永远最前；静止手型（张掌=怠速）永远最后，
        # 否则"张开手准备做动作"会先被判成怠速。
        if two_fist:
            code, confidence, reason = GestureCode.TWO_FIST, _SCORE_ESTOP, "two_fist"
        elif fist_hit:
            code, confidence, reason = GestureCode.ESTOP, _SCORE_ESTOP, "fist"
        elif static["pinch"]:
            code, confidence, reason = GestureCode.PINCH, _SCORE_ESTOP, "pinch"
        elif circle.triggered:
            code, confidence, reason = GestureCode.SPIN, _SCORE_SPIN, "circle"
            self.window.clear()  # Plan §2.1：转圈触发后立即重置轨迹
        elif arc_active:
            # 环形轨迹进行中：不给平移类指令，等待闭环触发 G-13
            reason = "arc_in_progress"
        elif two_push:
            code, confidence, reason = GestureCode.TWO_PALM_PUSH, 0.9, "two_palm_push"
        elif static["thumb_up"]:
            code, confidence, reason = GestureCode.THUMB_UP, 0.9, "thumb_up"
        elif static["v_sign"]:
            code, confidence, reason = GestureCode.V_SIGN, 0.9, "v_sign"
        # 注意顺序：G-10 前推（掌心朝镜头 + 抬腕/靠近镜头）必须先于 G-20 上挥。
        # 两者在几何上高度重叠——本项目数据集里的 G-10 本身就是"抬腕"动作
        # （竖直位移中位数 0.19，最大 0.44），无法用位移阈值区分。
        # 二者最终都映射 CMD_FORWARD，因此让判据更严格的 push 优先即可；
        # 上挥保留为"掌心未正对镜头时"的竖直动作路径。
        elif push.detected:
            code, confidence, reason = GestureCode.FORWARD, max(0.9, push.score), "push"
        elif vertical.detected and vertical.code is not None:
            code, confidence, reason = vertical.code, _SCORE_SWIPE, "vertical_swipe"
        elif swipe.detected and swipe.code is not None:
            code, confidence, reason = swipe.code, _SCORE_SWIPE, "swipe"
        elif static["open_palm"] and still:
            code, confidence, reason = GestureCode.OPEN_PALM, 0.85, "open_palm"

        candidate = None
        if code is not None:
            candidate = GestureCandidate(
                code=code,
                score=round(min(1.0, confidence * max(0.0, min(1.0, score))), 4),
                ts=at,
                source=source,
                frame_id=fid,
                landmarks=list(landmarks),
                trail=self.window.trail(12),
                reason=reason,
            )

        timings = {
            "fist_ms": round((t_fist - started) * 1000.0, 4),
            "push_ms": round((t_push - t_fist) * 1000.0, 4),
            "trajectory_ms": round((t_traj - t_push) * 1000.0, 4),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 4),
        }
        return PipelineFrame(candidate=candidate, signals=signals, timings_ms=timings, frame_id=fid)

    # -- 显式标记（跳过几何） ---------------------------------------------- #
    def _explicit_frame(
        self,
        explicit: FrameSignals,
        *,
        at: float,
        fid: int,
        score: float,
        source: Source,
        started: float,
        gap_reset: bool = False,
    ) -> PipelineFrame:
        code: GestureCode | None = None
        confidence = _SCORE_SWIPE
        reason = "no_signal"

        if explicit.fist:
            code, confidence, reason = GestureCode.ESTOP, _SCORE_ESTOP, "fist(explicit)"
        elif explicit.spin:
            code, confidence, reason = GestureCode.SPIN, _SCORE_SPIN, "circle(explicit)"
        elif explicit.push:
            code, confidence, reason = GestureCode.FORWARD, 0.9, "push(explicit)"
        elif explicit.swipe in {"left", "right"}:
            code = GestureCode.TURN_LEFT if explicit.swipe == "left" else GestureCode.TURN_RIGHT
            confidence, reason = _SCORE_SWIPE, "swipe(explicit)"

        override = explicit.score
        if override is not None:
            confidence = override

        candidate = None
        if code is not None:
            candidate = GestureCandidate(
                code=code,
                score=round(min(1.0, confidence * max(0.0, min(1.0, score))), 4),
                ts=at,
                source=source,
                frame_id=fid,
                reason=reason,
            )
        return PipelineFrame(
            candidate=candidate,
            signals={"explicit": explicit.model_dump(), "window": self.window.size, "gap_reset": gap_reset},
            timings_ms={"total_ms": round((time.perf_counter() - started) * 1000.0, 4)},
            frame_id=fid,
        )
