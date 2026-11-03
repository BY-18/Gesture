"""Agent 状态定义（Plan V1.0 T2 / §6.1）。

分工：
- 本模块持有 **内部可变状态**（`AgentState`）；
- 轨迹环形窗口的实现下沉到视觉层 `src.vision.trajectory`（T9），此处仅复用并保留
  `TrajectoryWindow` 别名，避免 P0 既有引用失效；
- 对外只读视图一律转成 `src.api.schemas.StateSnapshot`，不直接外泄内部对象；
- 不引入任何 I/O、不 import 任何框架，保证可序列化与可单测。
"""

from __future__ import annotations

import time
import uuid
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas import (
    Cmd,
    ConfirmedIntent,
    GestureCandidate,
    GestureCode,
    Landmark,
    RunMode,
    StateSnapshot,
)
from src.vision.trajectory import TrajectoryBuffer, TrajectoryPoint

__all__ = [
    "TrajectoryPoint",
    "TrajectoryBuffer",
    "TrajectoryWindow",
    "AgentState",
    "new_session_state",
    "INDEX_FINGER_TIP",
]

#: MediaPipe Hands 食指尖索引（Plan §3.1：轨迹追踪 landmark[8]）
INDEX_FINGER_TIP = 8

#: 向后兼容别名（P0 的 `TrajectoryWindow` 即为 T9 的 `TrajectoryBuffer`）
TrajectoryWindow = TrajectoryBuffer


class AgentState(BaseModel):
    """全局共享状态（Plan §6.1）。"""

    model_config = ConfigDict(validate_assignment=False)

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = ""
    frame_id: int = 0
    ts: float = Field(default_factory=time.time)
    mode: RunMode = RunMode.INJECT_ONLY
    landmarks: list[Landmark] | None = None
    handedness: str | None = None
    candidate: GestureCandidate | None = None
    window: TrajectoryBuffer = Field(default_factory=TrajectoryBuffer)
    confirmed_intent: ConfirmedIntent | None = None
    last_cmd: Cmd | None = None
    last_cmd_ts: float | None = None
    estop_latched: bool = False
    watchdog_deadline: float | None = None
    fps_ema: float = 0.0

    # -- 推进 -------------------------------------------------------------- #
    def tick(self) -> int:
        """帧号 +1 并刷新时间戳。"""
        self.frame_id += 1
        self.ts = time.time()
        return self.frame_id

    def push_index_tip(self) -> None:
        """把当前食指指尖写入轨迹窗口（供 T9 的轨迹算法消费）。"""
        if not self.landmarks:
            return
        if len(self.landmarks) <= INDEX_FINGER_TIP:
            return
        tip = self.landmarks[INDEX_FINGER_TIP]
        self.window.push(
            TrajectoryPoint(x=tip.x, y=tip.y, z=tip.z, ts=self.ts, frame_id=self.frame_id)
        )

    # -- 写入流水线结果 ---------------------------------------------------- #
    def apply_candidate(self, candidate: GestureCandidate, trace_id: str | None = None) -> None:
        if trace_id:
            self.trace_id = trace_id
        self.candidate = candidate
        if candidate.landmarks:
            self.landmarks = candidate.landmarks
        self.push_index_tip()

    def apply_intent(self, intent: ConfirmedIntent) -> None:
        self.trace_id = intent.trace_id
        self.confirmed_intent = intent
        self.last_cmd = intent.cmd
        self.last_cmd_ts = intent.ts
        if intent.cmd == Cmd.CMD_STOP:
            self.estop_latched = True

    def arm_watchdog(self, miss_ms: int) -> None:
        self.watchdog_deadline = time.time() + miss_ms / 1000.0

    def disarm_watchdog(self) -> None:
        self.watchdog_deadline = None

    # -- 对外视图 ---------------------------------------------------------- #
    def to_snapshot(self) -> StateSnapshot:
        candidate = self.candidate
        return StateSnapshot(
            session_id=self.session_id,
            trace_id=self.trace_id,
            mode=self.mode,
            frame_id=self.frame_id,
            ts=self.ts,
            last_cmd=self.last_cmd,
            last_cmd_ts=self.last_cmd_ts,
            estop_latched=self.estop_latched,
            candidate_code=candidate.code if candidate else None,
            candidate_score=candidate.score if candidate else None,
            confirmed_code=(
                self.confirmed_intent.code if self.confirmed_intent else None
            ),
            fps_ema=self.fps_ema,
            window_size=self.window.size,
            window_capacity=self.window.capacity,
            watchdog_deadline=self.watchdog_deadline,
        )


def new_session_state(mode: RunMode = RunMode.INJECT_ONLY, capacity: int = 30) -> AgentState:
    """新建会话状态。"""
    state = AgentState(mode=mode)
    state.window = TrajectoryBuffer(capacity=capacity)
    return state


def landmarks_from_points(points: Iterable[tuple[float, float, float]]) -> list[Landmark]:
    """便捷构造：把 (x, y, z) 三元组列表转成 21 点。"""
    return [Landmark(x=x, y=y, z=z) for x, y, z in points]


def gesture_code_or_none(value: str | None) -> GestureCode | None:
    if value is None:
        return None
    try:
        return GestureCode(value)
    except ValueError:
        return None
