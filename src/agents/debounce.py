"""热路径分级防抖内核（Plan V1.0 T10 / T11，落实 §2.2、§2.3、§2.5）。

**本模块的铁律（Plan §2.5）**：
- 禁止 import LangGraph，禁止任何网络/磁盘/日志 I/O；
- 全部逻辑为微秒级纯计算，只读写本模块定义的 `DebounceState`；
- 帧循环里唯一允许调用的就是 `push()`。

三条通道的裁决口径：

1. **急停 G-02（§2.3）**：连续 2 帧命中 → **当帧同步直出** `CMD_STOP`（旁路滑窗）；
   第 3 帧只做 `estop_latched` 锁存确认，不再重复出帧；
   触发即刻进入 500 ms 冷却期，冷却期内任何运动指令（G-10/11/12/13）一律拒绝。
2. **方向 G-10/11/12（§2.2）**：滑窗容量 5 帧，但只看**最近连续 3 帧同类**且
   **加权置信度 > 0.85** 即刻触发；第 4、5 帧用于复核——若出现反向高置信候选则回滚撤销。
3. **转圈 G-13（§2.1）**：由 `trajectory.circle_fit()` 直接确认（`preconfirmed`），
   校验非急停冷却期后立即触发 `CMD_SPIN`，不进入 3 帧滑窗。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable

from src.api.schemas import ESTOP_CODES, Cmd, GestureCode, Source
from src.core.config import DebounceConfig, get_settings

__all__ = [
    "FrameSignal",
    "DebounceResult",
    "DebounceState",
    "DebounceStats",
    "new_state",
    "push",
    "snapshot",
    "reset",
    "break_streak",
    "MOTION_CODES",
    "DIRECTION_CODES",
    "STATIC_HOLD_CODES",
]

#: 运动类手势（受急停冷却期约束）
MOTION_CODES: frozenset[GestureCode] = frozenset(
    {
        GestureCode.FORWARD,
        GestureCode.TURN_LEFT,
        GestureCode.TURN_RIGHT,
        GestureCode.SPIN,
        GestureCode.THUMB_UP,
        GestureCode.V_SIGN,
        GestureCode.SWIPE_UP,
        GestureCode.TWO_PALM_PUSH,
    }
)

#: 需要 3 帧滑窗表决的手势（方向类 + 静态手型包 + 上下挥动 + 双手组合）
DIRECTION_CODES: frozenset[GestureCode] = frozenset(
    {
        GestureCode.FORWARD,
        GestureCode.TURN_LEFT,
        GestureCode.TURN_RIGHT,
        GestureCode.THUMB_UP,
        GestureCode.OPEN_PALM,
        GestureCode.V_SIGN,
        GestureCode.SWIPE_UP,
        GestureCode.SWIPE_DOWN,
        GestureCode.TWO_PALM_PUSH,
    }
)

#: 静止保持类手势：同一手势持续存在时只触发一次，避免"手一直张开"刷屏 IDLE。
#: 松手（帧间超时 → `break_streak`）后解除。
STATIC_HOLD_CODES: frozenset[GestureCode] = frozenset(
    {GestureCode.OPEN_PALM, GestureCode.THUMB_UP, GestureCode.V_SIGN}
)

#: 手势 → 指令映射（与 schemas.GESTURE_TO_CMD 同源，此处避免热路径二次导入开销）
_CODE_TO_CMD: dict[GestureCode, Cmd] = {
    GestureCode.ESTOP: Cmd.CMD_STOP,
    GestureCode.FORWARD: Cmd.CMD_FORWARD,
    GestureCode.TURN_LEFT: Cmd.CMD_TURN_LEFT,
    GestureCode.TURN_RIGHT: Cmd.CMD_TURN_RIGHT,
    GestureCode.SPIN: Cmd.CMD_SPIN,
    GestureCode.THUMB_UP: Cmd.CMD_FORWARD,
    GestureCode.OPEN_PALM: Cmd.CMD_IDLE,
    GestureCode.V_SIGN: Cmd.CMD_SPIN,
    GestureCode.PINCH: Cmd.CMD_STOP,
    GestureCode.SWIPE_UP: Cmd.CMD_FORWARD,
    GestureCode.SWIPE_DOWN: Cmd.CMD_IDLE,
    GestureCode.TWO_PALM_PUSH: Cmd.CMD_FORWARD,
    GestureCode.TWO_FIST: Cmd.CMD_STOP,
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameSignal:
    """单帧手势候选（几何层输出）。`code=None` 表示本帧无可识别手势。"""

    code: GestureCode | None
    score: float = 0.0
    ts: float = field(default_factory=time.time)
    frame_id: int = 0
    #: True 表示"已被上游确认"，跳过滑窗表决（转圈 / 注入器 / 离线回放）
    preconfirmed: bool = False
    source: Source = Source.GESTURE

    @property
    def is_estop(self) -> bool:
        """是否属于急停类手势（G-02 握拳 / G-06 捏合 / G-31 双手握拳）。

        安全口径统一：任何产出 CMD_STOP 的手势都走 2 帧直出的快通道。
        """
        return self.code in ESTOP_CODES

    @classmethod
    def from_candidate(
        cls,
        candidate: Any,
        *,
        ts: float,
        frame_id: int = 0,
        source: Source = Source.GESTURE,
    ) -> "FrameSignal":
        """由 `GestureCandidate` 构造信号，并落实 **Plan §2.1 的转圈直通规则**。

        转圈（G-13）由 `trajectory.circle_fit()` 完成闭环确认，属于"已确认"，
        因此置 `preconfirmed=True` 跳过 3 帧滑窗。这条规则必须集中在一处，
        否则各调用点（HTTP 帧入口 / 视觉循环 / 离线基准）会各写一遍并漏掉。
        """
        code = getattr(candidate, "code", None) if candidate is not None else None
        score = getattr(candidate, "score", 0.0) if candidate is not None else 0.0
        return cls(
            code=code,
            score=score,
            ts=ts,
            frame_id=frame_id,
            preconfirmed=code is GestureCode.SPIN,
            source=source,
        )


@dataclass
class DebounceResult:
    """单帧裁决结果。字段全部为纯值，可直接进日志或状态机。"""

    now: float = 0.0
    frame_id: int = 0
    cmd: Cmd | None = None
    code: GestureCode | None = None
    confidence: float = 0.0
    estop: bool = False
    latch: bool = False
    revoked_code: GestureCode | None = None
    blocked: bool = False
    #: 被冷却期拦截的原始指令（帧路径不下发；注入通道交由总线终裁以便回传 suppressed 帧）
    blocked_cmd: Cmd | None = None
    latency_frames: int = 0
    cooldown_ms: float = 0.0
    reason: str = "no_signal"

    @property
    def triggered(self) -> bool:
        return self.cmd is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cmd": self.cmd.value if self.cmd else None,
            "code": self.code.value if self.code else None,
            "confidence": round(self.confidence, 4),
            "estop": self.estop,
            "latch": self.latch,
            "revoked": self.revoked_code.value if self.revoked_code else None,
            "blocked": self.blocked,
            "latency_frames": self.latency_frames,
            "cooldown_ms": round(self.cooldown_ms, 1),
            "reason": self.reason,
        }


@dataclass
class DebounceStats:
    frames: int = 0
    estop_triggers: int = 0
    direction_triggers: int = 0
    spin_triggers: int = 0
    blocked: int = 0
    revocations: int = 0
    latches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "estop_triggers": self.estop_triggers,
            "direction_triggers": self.direction_triggers,
            "spin_triggers": self.spin_triggers,
            "blocked": self.blocked,
            "revocations": self.revocations,
            "latches": self.latches,
        }


@dataclass
class DebounceState:
    """防抖状态（由调用方持有并串行推进；本模块不持有全局状态）。"""

    window_size: int = 5
    confirm_frames: int = 3
    estop_frames: int = 2
    min_confidence: float = 0.85
    estop_cooldown_s: float = 0.5

    window: Deque[FrameSignal] = field(default_factory=deque)
    estop_hits: int = 0
    estop_fired: bool = False
    estop_latched: bool = False
    estop_until: float = 0.0
    last_trigger: FrameSignal | None = None
    last_trigger_at: float = 0.0
    pending_review: int = 0
    #: 静止保持类手势的"已触发"标记（同手势持续存在时不再重复出帧）
    static_hold_code: GestureCode | None = None
    stats: DebounceStats = field(default_factory=DebounceStats)

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def from_config(cls, debounce: DebounceConfig, cooldown_s: float = 0.5) -> "DebounceState":
        state = cls(
            window_size=debounce.window_size,
            confirm_frames=debounce.confirm_frames,
            estop_frames=debounce.estop_confirm_frames,
            min_confidence=debounce.min_confidence,
            estop_cooldown_s=cooldown_s,
        )
        state.window = deque(maxlen=debounce.window_size)
        return state

    # -- 查询 -------------------------------------------------------------- #
    @property
    def cooldown_remaining_ms(self) -> float:
        return round(max(0.0, self.estop_until - time.time()) * 1000.0, 1)

    def in_cooldown(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) < self.estop_until


def new_state(cooldown_s: float | None = None) -> DebounceState:
    """按配置中心构造防抖状态。"""
    settings = get_settings()
    cooldown = (
        settings.safety.estop_cooldown_ms / 1000.0 if cooldown_s is None else cooldown_s
    )
    return DebounceState.from_config(settings.debounce, cooldown_s=cooldown)


def reset(state: DebounceState) -> None:
    """清空窗口与急停计数（会话重启用）。"""
    state.window.clear()
    state.estop_hits = 0
    state.estop_fired = False
    state.estop_latched = False
    state.estop_until = 0.0
    state.last_trigger = None
    state.last_trigger_at = 0.0
    state.pending_review = 0
    state.static_hold_code = None
    state.stats = DebounceStats()


def break_streak(state: DebounceState) -> None:
    """手势中断（帧间超时）：只清表决窗口与急停连击。

    与 `reset()` 的区别：**保留** `estop_until`（冷却期）与 `estop_latched`（锁存），
    否则"离手一下"就能绕过急停冷却，属于安全漏洞。
    """
    state.window.clear()
    state.estop_hits = 0
    state.estop_fired = False
    state.pending_review = 0
    # 手势中断 = "手离开画面"：解除静止保持，用户重新比同一个手型还能再触发
    state.static_hold_code = None


def _confidence_weights(n: int) -> tuple[float, ...]:
    """按"越新权重越大"生成和为 1 的权重序列（最新在前）。"""
    raw = [float(n - i) for i in range(n)]
    total = sum(raw)
    return tuple(w / total for w in raw)


def _weighted_confidence(signals: Iterable[FrameSignal]) -> float:
    items = list(signals)
    if not items:
        return 0.0
    weights = _confidence_weights(len(items))
    newest_first = list(reversed(items))
    return sum(w * s.score for w, s in zip(weights, newest_first))


# --------------------------------------------------------------------------- #
# 核心：逐帧推进
# --------------------------------------------------------------------------- #
def push(
    state: DebounceState,
    signal: FrameSignal,
    *,
    now: float | None = None,
) -> DebounceResult:
    """推进一帧并返回裁决结果（纯计算，无 I/O）。"""
    at = signal.ts if now is None else now
    result = DebounceResult(now=at, frame_id=signal.frame_id)
    state.stats.frames += 1
    result.cooldown_ms = round(max(0.0, state.estop_until - at) * 1000.0, 1)

    # ---------------- ① 急停快通道（§2.3，旁路滑窗） ---------------- #
    if signal.is_estop:
        state.estop_hits += 1
        # 预确认通道（/api/inject）：语义是"该手势已被确认"，
        # 因此单帧即视为达成 2 帧判据，保留 P0 的注入契约。
        if signal.preconfirmed and not state.estop_fired:
            state.estop_hits = max(state.estop_hits, state.estop_frames)
        if not state.estop_fired and state.estop_hits >= state.estop_frames:
            state.estop_fired = True
            state.estop_until = at + state.estop_cooldown_s
            state.window.clear()
            state.pending_review = 0
            state.last_trigger = signal
            state.last_trigger_at = at
            state.stats.estop_triggers += 1
            result.cmd = Cmd.CMD_STOP
            result.code = signal.code
            result.confidence = signal.score
            result.estop = True
            result.latency_frames = state.estop_hits
            result.cooldown_ms = state.estop_cooldown_s * 1000.0
            result.reason = "estop_fast_path"
            return result

        if state.estop_fired:
            # 第 3 帧：锁存确认（不重复出帧，避免总线重复下发）
            if state.estop_hits >= state.estop_frames + 1 and not state.estop_latched:
                state.estop_latched = True
                state.stats.latches += 1
                result.latch = True
                result.code = signal.code
                result.reason = "estop_latch_confirmed"
            else:
                result.code = signal.code
                result.reason = "estop_repeat"
            return result

        result.code = signal.code
        result.reason = "estop_arming"
        return result

    # ---------------- ①.b 静止保持类：同一手型持续存在时只触发一次 ---------- #
    if signal.code in STATIC_HOLD_CODES:
        if state.static_hold_code is signal.code:
            result.code = signal.code
            result.reason = "static_hold"
            return result
    else:
        state.static_hold_code = None

    # ---------------- ② 非急停帧：重置急停连击 ---------------- #
    state.estop_hits = 0
    state.estop_fired = False
    # 锁存解除：已松手（本帧非急停）且冷却期已过 → 解除急停锁存。
    # 冷却期内保持锁存，避免"松手一下"就绕过急停状态。
    if state.estop_latched and at >= state.estop_until:
        state.estop_latched = False

    # ---------------- ③ 触发后第 4/5 帧复核：反向回滚撤销 ---------------- #
    # 注意：预确认通道（/api/inject）的每一次注入都是**独立且已确认**的意图，
    # 不能被上一条指令的复核窗口判成"折返"，否则注入链路会丢帧。
    if state.pending_review > 0 and not signal.preconfirmed:
        state.pending_review -= 1
        last = state.last_trigger
        if (
            last is not None
            and signal.code is not None
            and signal.code is not last.code
            and signal.score >= state.min_confidence
        ):
            state.pending_review = 0
            state.stats.revocations += 1
            result.revoked_code = last.code
            result.code = signal.code
            result.reason = "rollback"
            return result

    # ---------------- ④ 急停冷却期：拒绝一切运动指令 ---------------- #
    if signal.code in MOTION_CODES and at < state.estop_until:
        state.window.clear()
        state.stats.blocked += 1
        result.blocked = True
        result.code = signal.code
        result.blocked_cmd = _CODE_TO_CMD[signal.code]
        result.cooldown_ms = round((state.estop_until - at) * 1000.0, 1)
        result.reason = "estop_cooldown"
        return result

    # ---------------- ⑤ 转圈 / 预确认信号：直接触发 ---------------- #
    if signal.code is not None and signal.preconfirmed:
        return _trigger(
            state,
            signal,
            at,
            result,
            latency_frames=1,
            reason="preconfirmed",
            review_frames=2,
        )

    # ---------------- ⑥ 方向类：3 帧滑窗表决（§2.2） ---------------- #
    state.window.append(signal)
    if signal.code is None:
        result.reason = "no_signal"
        return result

    if len(state.window) < state.confirm_frames:
        result.code = signal.code
        result.reason = "warming_up"
        return result

    tail = list(state.window)[-state.confirm_frames :]
    codes = {s.code for s in tail}
    if len(codes) != 1:
        result.code = signal.code
        result.reason = "window_mixed"
        return result

    candidate = next(iter(codes))
    confidence = _weighted_confidence(tail)
    result.confidence = round(confidence, 4)
    result.code = candidate
    if candidate not in DIRECTION_CODES:
        result.reason = "unsupported_code"
        return result
    if confidence <= state.min_confidence:
        result.reason = "low_confidence"
        return result

    return _trigger(
        state,
        signal,
        at,
        result,
        latency_frames=state.confirm_frames,
        reason="window_confirm",
        review_frames=state.window_size - state.confirm_frames,
    )


def _trigger(
    state: DebounceState,
    signal: FrameSignal,
    at: float,
    result: DebounceResult,
    *,
    latency_frames: int,
    reason: str,
    review_frames: int,
) -> DebounceResult:
    """统一触发出口：生成指令、清窗、登记复核窗口。"""
    code = signal.code
    assert code is not None  # 调用方已保证
    state.last_trigger = signal
    state.last_trigger_at = at
    state.pending_review = max(0, review_frames)
    state.window.clear()
    if code in STATIC_HOLD_CODES:
        state.static_hold_code = code
    if code is GestureCode.SPIN:
        state.stats.spin_triggers += 1
    else:
        state.stats.direction_triggers += 1
    result.cmd = _CODE_TO_CMD[code]
    result.code = code
    result.confidence = result.confidence or round(signal.score, 4)
    result.latency_frames = latency_frames
    result.reason = reason
    return result


# --------------------------------------------------------------------------- #
# 只读快照
# --------------------------------------------------------------------------- #
def snapshot(state: DebounceState) -> dict[str, Any]:
    return {
        "window": [
            {"code": s.code.value if s.code else None, "score": round(s.score, 3)}
            for s in state.window
        ],
        "estop_hits": state.estop_hits,
        "estop_fired": state.estop_fired,
        "estop_latched": state.estop_latched,
        "estop_until": round(state.estop_until, 3),
        "cooldown_ms": state.cooldown_remaining_ms,
        "window_size": state.window_size,
        "confirm_frames": state.confirm_frames,
        "estop_frames": state.estop_frames,
        "min_confidence": state.min_confidence,
        "pending_review": state.pending_review,
        "stats": state.stats.as_dict(),
    }
