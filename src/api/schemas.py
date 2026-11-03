"""协议契约层（Plan V1.0 T2 / §6）。

本模块是 **唯一的跨层数据契约**：`vision → agents → execution → api` 之间只允许
传递此处的模型，禁止传裸 dict。

WebSocket 下行消息（`/ws/stream`，非 Pydantic、直接 dict，以避免 15 Hz 推送开销）：

===================================  ==========================================
type                                 结构
===================================  ==========================================
``hello``                             {type, session_id, mode, dry_run, version,
                                       frame_push_hz, gesture_labels, cmd_labels}
``frame``                             {type, seq, ts, width, height, mode, fps,
                                       frame_ms, landmarks|null, trail, code,
                                       has_video, age_ms}
``command``                           {type, frame: ControlFrame, label}
``state``                             {type, state: StateSnapshot}
``pong``                              {type, ts}
``metrics``                           {type, metrics: {...}}
===================================  ==========================================
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "GestureCode",
    "Cmd",
    "Source",
    "RunMode",
    "GESTURE_TO_CMD",
    "GESTURE_LABELS",
    "ESTOP_CODES",
    "CMD_LABELS",
    "MOTION_CMDS",
    "EMERGENCY_CMDS",
    "cmd_default_speed",
    "Landmark",
    "Point2D",
    "GestureCandidate",
    "ConfirmedIntent",
    "ControlFrame",
    "StateSnapshot",
    "InjectRequest",
    "InjectResponse",
    "FrameRequest",
    "FrameResponse",
    "FrameSignals",
    "HealthResponse",
    "MetricsResponse",
]


# --------------------------------------------------------------------------- #
# 枚举与语义映射
# --------------------------------------------------------------------------- #
class GestureCode(StrEnum):
    """手势词表。

    G-0x / G-1x 为基础手势；G-2x / G-3x 为扩展包（上下挥动 / 双手组合）。
    新增手势一律映射到**冻结的 6 条指令域**，不扩展 `Cmd`。
    """

    # --- 基础手势 ---
    ESTOP = "G-02"
    FORWARD = "G-10"
    TURN_LEFT = "G-11"
    TURN_RIGHT = "G-12"
    SPIN = "G-13"
    # --- 静态手型包 ---
    THUMB_UP = "G-03"
    OPEN_PALM = "G-04"
    V_SIGN = "G-05"
    PINCH = "G-06"
    # --- 上下挥动包 ---
    SWIPE_UP = "G-20"
    SWIPE_DOWN = "G-21"
    # --- 双手组合包 ---
    TWO_PALM_PUSH = "G-30"
    TWO_FIST = "G-31"


class Cmd(StrEnum):
    """控制指令域（Plan §6.2 冻结，不得扩展）。"""

    CMD_STOP = "CMD_STOP"
    CMD_FORWARD = "CMD_FORWARD"
    CMD_TURN_LEFT = "CMD_TURN_LEFT"
    CMD_TURN_RIGHT = "CMD_TURN_RIGHT"
    CMD_SPIN = "CMD_SPIN"
    CMD_IDLE = "CMD_IDLE"


class Source(StrEnum):
    GESTURE = "GESTURE"
    INJECT = "INJECT"
    WATCHDOG = "WATCHDOG"


class RunMode(StrEnum):
    LIVE = "LIVE"
    INJECT_ONLY = "INJECT_ONLY"


GESTURE_TO_CMD: dict[GestureCode, Cmd] = {
    GestureCode.ESTOP: Cmd.CMD_STOP,
    GestureCode.FORWARD: Cmd.CMD_FORWARD,
    GestureCode.TURN_LEFT: Cmd.CMD_TURN_LEFT,
    GestureCode.TURN_RIGHT: Cmd.CMD_TURN_RIGHT,
    GestureCode.SPIN: Cmd.CMD_SPIN,
    # 静态手型包
    GestureCode.THUMB_UP: Cmd.CMD_FORWARD,
    GestureCode.OPEN_PALM: Cmd.CMD_IDLE,
    GestureCode.V_SIGN: Cmd.CMD_SPIN,
    GestureCode.PINCH: Cmd.CMD_STOP,
    # 上下挥动包
    GestureCode.SWIPE_UP: Cmd.CMD_FORWARD,
    GestureCode.SWIPE_DOWN: Cmd.CMD_IDLE,
    # 双手组合包
    GestureCode.TWO_PALM_PUSH: Cmd.CMD_FORWARD,
    GestureCode.TWO_FIST: Cmd.CMD_STOP,
}

#: 产出 CMD_STOP 的手势集合（走 2 帧直出的急停快通道，安全一视同仁）
ESTOP_CODES: frozenset[GestureCode] = frozenset(
    {GestureCode.ESTOP, GestureCode.PINCH, GestureCode.TWO_FIST}
)

GESTURE_LABELS: dict[str, str] = {
    GestureCode.ESTOP.value: "停止/急停",
    GestureCode.FORWARD.value: "前进",
    GestureCode.TURN_LEFT.value: "左转",
    GestureCode.TURN_RIGHT.value: "右转",
    GestureCode.SPIN.value: "转圈",
    GestureCode.THUMB_UP.value: "竖拇指·前进",
    GestureCode.OPEN_PALM.value: "张掌·怠速",
    GestureCode.V_SIGN.value: "剪刀手·转圈",
    GestureCode.PINCH.value: "捏合·急停",
    GestureCode.SWIPE_UP.value: "上挥·前进",
    GestureCode.SWIPE_DOWN.value: "下挥·怠速",
    GestureCode.TWO_PALM_PUSH.value: "双掌前推·前进",
    GestureCode.TWO_FIST.value: "双手握拳·急停",
}

CMD_LABELS: dict[str, str] = {
    Cmd.CMD_STOP.value: "停止/急停",
    Cmd.CMD_FORWARD.value: "前进",
    Cmd.CMD_TURN_LEFT.value: "左转",
    Cmd.CMD_TURN_RIGHT.value: "右转",
    Cmd.CMD_SPIN.value: "转圈",
    Cmd.CMD_IDLE.value: "怠速",
}

#: 会造成物理位移的指令（急停冷却期内必须被拒）
MOTION_CMDS: frozenset[Cmd] = frozenset(
    {Cmd.CMD_FORWARD, Cmd.CMD_TURN_LEFT, Cmd.CMD_TURN_RIGHT, Cmd.CMD_SPIN}
)

#: 永不丢弃、永不去抖、永不被冷却期拦截的指令
EMERGENCY_CMDS: frozenset[Cmd] = frozenset({Cmd.CMD_STOP})


def cmd_default_speed(cmd: Cmd) -> float:
    """指令默认速度（Plan §6.2：speed 字段语义）。"""
    return 0.0 if cmd in {Cmd.CMD_STOP, Cmd.CMD_IDLE} else 1.0


# --------------------------------------------------------------------------- #
# 基础几何模型
# --------------------------------------------------------------------------- #
class Point2D(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class Landmark(BaseModel):
    """MediaPipe 归一化关键点（x/y ∈ [0,1]，z 为相对深度，可为负）。"""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


# --------------------------------------------------------------------------- #
# 流水线上的三类对象
# --------------------------------------------------------------------------- #
class GestureCandidate(BaseModel):
    """几何层产出的**未确认**候选（防抖层输入）。"""

    code: GestureCode
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    ts: float = Field(default_factory=time.time)
    source: Source = Source.GESTURE
    frame_id: int | None = None
    landmarks: list[Landmark] | None = None
    trail: list[Point2D] | None = None
    reason: str | None = None


class ConfirmedIntent(BaseModel):
    """防抖层确认后的意图（控制总线输入）。

    `code` 允许为空：看门狗产生的 `CMD_IDLE` 并不对应任何手势。
    """

    code: GestureCode | None = None
    cmd: Cmd
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trace_id: str
    ts: float = Field(default_factory=time.time)
    source: Source = Source.GESTURE
    speed: float | None = None
    reason: str | None = None


class ControlFrame(BaseModel):
    """对外控制帧（Plan §6.2：PRD 示例的超集）。"""

    cmd: Cmd
    speed: float
    timestamp: float
    seq: int
    source: Source
    trace_id: str
    dry_run: bool
    #: 被去抖/冷却期抑制时为 True（帧仍返回，便于前端与验收脚本观测抑制行为）
    suppressed: bool = False


# --------------------------------------------------------------------------- #
# 状态快照（对外只读视图）
# --------------------------------------------------------------------------- #
class StateSnapshot(BaseModel):
    session_id: str
    trace_id: str
    mode: RunMode
    frame_id: int
    ts: float
    last_cmd: Cmd | None = None
    last_cmd_ts: float | None = None
    estop_latched: bool = False
    candidate_code: GestureCode | None = None
    candidate_score: float | None = None
    confirmed_code: GestureCode | None = None
    fps_ema: float = 0.0
    window_size: int = 0
    window_capacity: int = 0
    watchdog_deadline: float | None = None
    # P2（T10~T13）运行时视图
    cooldown_ms: float = 0.0
    debounce_window: int = 0
    graph_degraded: bool = False


# --------------------------------------------------------------------------- #
# API 请求 / 响应模型
# --------------------------------------------------------------------------- #
class InjectRequest(BaseModel):
    """虚拟手势注入请求（PRD §3.4 注入器 / Plan §5.4）。"""

    code: GestureCode
    #: 一次性注入次数（高压连点压测用），上限 200
    count: int = Field(default=1, ge=1, le=200)
    #: 候选置信度；< min_confidence 时预留给 P2 的防抖层拒绝，P0 直接采用
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    #: 覆盖默认速度（None 表示按指令类型取默认值）
    speed: float | None = Field(default=None, ge=0.0, le=1.0)
    #: 两次注入之间的间隔（毫秒），用于验证 200 ms 去抖窗口
    interval_ms: int = Field(default=0, ge=0, le=2000)
    #: 透传 trace_id；None 表示服务端新建
    trace_id: str | None = None
    #: 原始 21 点（仅用于 HUD 渲染，P0 不参与几何判定；P1 起由 vision 层产出）
    landmarks: list[Landmark] | None = None

    @field_validator("landmarks")
    @classmethod
    def _check_landmarks(cls, value: list[Landmark] | None) -> list[Landmark] | None:
        if value is not None and len(value) != 21:
            raise ValueError("landmarks 必须为 21 个关键点（MediaPipe Hands）")
        return value

    @field_validator("trace_id")
    @classmethod
    def _check_trace(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class InjectResponse(BaseModel):
    accepted: bool
    requested: int
    emitted: int
    suppressed: int
    blocked: int
    latency_ms: float
    frames: list[ControlFrame]
    state: StateSnapshot
    dedupe_window_ms: int


class FrameSignals(BaseModel):
    """显式手势标记（跳过几何判定，用于契约测试与离线回放）。"""

    fist: bool | None = None
    push: bool | None = None
    swipe: str | None = None  # "left" / "right"
    spin: bool | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class FrameRequest(BaseModel):
    """热路径逐帧输入（Plan §3 / T13：真实视觉流与离线回放共用入口）。

    与 `/api/inject`（已确认意图注入）的区别：本端点送入的是**单帧候选**，
    必须经 `debounce.py` 的滑窗表决才能触发指令。
    """

    landmarks: list[Landmark]
    handedness: str = "Right"
    #: 手部检测/分类置信度（0~1），参与加权表决
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    #: 采集时刻（秒，Unix 时间）。为空则由服务端以处理时刻计。
    #: 客户端按 30 FPS 节奏填 `t0 + i/30` 时，服务端返回的 `gesture_latency_ms`
    #: 即完整包含"视觉段折算 + 处理"，对应 Plan §4.2 的口径。
    frame_ts: float | None = None
    #: 透传 trace_id
    trace_id: str | None = None
    #: 关闭几何判定、直接采用显式标记（离线回放/契约测试用）
    explicit: FrameSignals | None = None

    @field_validator("landmarks")
    @classmethod
    def _check_landmarks(cls, value: list[Landmark]) -> list[Landmark]:
        if len(value) != 21:
            raise ValueError("landmarks 必须为 21 个关键点（MediaPipe Hands）")
        return value


class FrameResponse(BaseModel):
    accepted: bool
    frame_id: int
    code: GestureCode | None = None
    cmd: Cmd | None = None
    triggered: bool = False
    reason: str | None = None
    #: 服务端处理本帧耗时（微秒级转毫秒）
    latency_ms: float = 0.0
    #: 从客户端声明的采集时刻到指令生成的总耗时（含视觉段折算）
    gesture_latency_ms: float | None = None
    frames: list[ControlFrame] = Field(default_factory=list)
    state: StateSnapshot | None = None
    signals: dict[str, Any] = Field(default_factory=dict)
    estop_latched: bool = False
    cooldown_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    mode: RunMode
    dry_run: bool
    inject_only: bool
    camera_available: bool
    camera_note: str
    uptime_s: float
    fps: float
    session_id: str
    seq: int
    ws_connections: int
    log_file: str


class MetricsResponse(BaseModel):
    mode: RunMode
    dry_run: bool
    uptime_s: float
    fps: float
    histograms: dict[str, dict[str, float | int]]
    counters: dict[str, int]
    gauges: dict[str, float]
    bus: dict[str, Any]
    ws_connections: int
    frame_budget_ms: float
    frame_p95_limit_ms: float
    # P2（T10~T13）新增观测面
    debounce: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    watchdog: dict[str, Any] = Field(default_factory=dict)
    degrade: dict[str, Any] = Field(default_factory=dict)
