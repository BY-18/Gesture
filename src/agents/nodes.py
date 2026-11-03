"""LangGraph 温路径节点（Plan V1.0 T12，落实 §2.5）。

与 `debounce.py` 的分工：
- `debounce.py`：**热路径**，逐帧微秒级纯计算；
- 本模块：**温路径**，只在意图确认（约每 100 ms 一次）或看门狗滴答时执行，
  负责优先级仲裁、指令落地、审计留痕。允许 I/O（写日志、调用控制总线）。

节点顺序（图的边）::

    START → kill_switch ─┬─(estop)────→ dispatch → audit → END
                         ├─(blocked)──→ audit → END
                         └─(continue)─→ confirm_intent → watchdog ─┬─→ dispatch → audit → END
                                                                   └─→ audit → END
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from src.api.schemas import (
    Cmd,
    ConfirmedIntent,
    GestureCode,
    MOTION_CMDS,
    Source,
)
from src.core.logger import SessionLogger
from src.core.metrics import Metrics
from src.execution.control_bus import ControlBus

__all__ = [
    "OrchestrationState",
    "kill_switch_node",
    "confirm_intent_node",
    "watchdog_node",
    "dispatch_node",
    "audit_node",
    "route_after_kill",
    "route_after_watchdog",
]

_MOTION_VALUES = frozenset(cmd.value for cmd in MOTION_CMDS)


class OrchestrationState(TypedDict, total=False):
    """图状态（LangGraph 的 state schema）。"""

    trace_id: str
    session_id: str
    now: float
    frame_id: int
    source: str
    # 意图输入
    code: str | None
    cmd: str | None
    confidence: float
    speed: float | None
    reason: str
    # 急停与冷却
    estop: bool
    estop_latched: bool
    cooldown_ms: float
    # 看门狗
    watchdog_due: bool
    watchdog_reason: str | None
    # 节点产出
    decision: str  # continue | pending | dispatch | idle | estop | blocked | none
    revoked: str | None
    node_trace: list[str]
    frame: Any
    dispatched: bool


def _trace(state: OrchestrationState, name: str) -> list[str]:
    return list(state.get("node_trace") or []) + [name]


# --------------------------------------------------------------------------- #
# ① 急停硬拦截（第一跳，Plan §2.5 / §2.3）
# --------------------------------------------------------------------------- #
def kill_switch_node(state: OrchestrationState) -> dict[str, Any]:
    """图的第一跳：急停优先、冷却期拦截，任何运动指令都要先过这一关。"""
    trace = _trace(state, "kill_switch")
    if state.get("estop"):
        return {
            "decision": "estop",
            "cmd": Cmd.CMD_STOP.value,
            "reason": "kill_switch",
            "node_trace": trace,
        }

    cmd = state.get("cmd")
    cooldown_ms = float(state.get("cooldown_ms") or 0.0)
    if cmd in _MOTION_VALUES and cooldown_ms > 0:
        return {
            "decision": "blocked",
            "cmd": None,
            "reason": "estop_cooldown",
            "node_trace": trace,
        }

    if cmd is None:
        # 无指令帧：交给看门狗决定是否需要怠速兜底
        return {"decision": "pending", "reason": state.get("reason") or "no_cmd", "node_trace": trace}

    return {"decision": "continue", "node_trace": trace}


# --------------------------------------------------------------------------- #
# ② 意图确认
# --------------------------------------------------------------------------- #
def confirm_intent_node(state: OrchestrationState) -> dict[str, Any]:
    trace = _trace(state, "confirm_intent")
    if state.get("decision") != "continue":
        return {"node_trace": trace}

    cmd = state.get("cmd")
    try:
        resolved = Cmd(cmd)  # 冻结指令域校验
    except (TypeError, ValueError):
        return {
            "decision": "none",
            "cmd": None,
            "reason": "invalid_cmd",
            "node_trace": trace,
        }
    return {
        "decision": "dispatch",
        "cmd": resolved.value,
        "confidence": float(state.get("confidence") or 0.0),
        "node_trace": trace,
    }


# --------------------------------------------------------------------------- #
# ③ 看门狗（Plan §3.2：丢失手势 >500ms 或 FPS<15 持续 300ms → CMD_IDLE）
# --------------------------------------------------------------------------- #
def watchdog_node(state: OrchestrationState) -> dict[str, Any]:
    trace = _trace(state, "watchdog")
    if state.get("decision") == "dispatch":
        return {"node_trace": trace}  # 本帧已有有效指令，不覆盖
    if not state.get("watchdog_due"):
        return {"node_trace": trace}
    return {
        "decision": "dispatch",
        "cmd": Cmd.CMD_IDLE.value,
        "code": None,
        "confidence": 1.0,
        "reason": state.get("watchdog_reason") or "gesture_miss",
        "source": Source.WATCHDOG.value,
        "node_trace": trace,
    }


# --------------------------------------------------------------------------- #
# ④ 指令落地（调用控制总线）
# --------------------------------------------------------------------------- #
def dispatch_node(
    state: OrchestrationState,
    *,
    bus: ControlBus,
    session: Any,
) -> dict[str, Any]:
    trace = _trace(state, "dispatch")
    cmd_value = state.get("cmd")
    if not cmd_value:
        return {"decision": state.get("decision") or "none", "dispatched": False, "node_trace": trace}

    code_value = state.get("code")
    code = None
    if code_value:
        try:
            code = GestureCode(code_value)
        except ValueError:
            code = None

    intent = ConfirmedIntent(
        code=code,
        cmd=Cmd(cmd_value),
        confidence=float(state.get("confidence") or 0.0),
        trace_id=str(state.get("trace_id") or ""),
        ts=float(state.get("now") or time.time()),
        source=Source(str(state.get("source") or Source.GESTURE.value)),
        speed=state.get("speed"),
        reason=state.get("reason"),
    )
    frame = bus.dispatch(intent)
    if session is not None:
        session.apply_intent(intent)
    return {
        "decision": "dispatched" if not frame.suppressed else "suppressed",
        "frame": frame,
        "dispatched": not frame.suppressed,
        "node_trace": trace,
    }


# --------------------------------------------------------------------------- #
# ⑤ 审计落盘（温路径允许 I/O）
# --------------------------------------------------------------------------- #
def audit_node(
    state: OrchestrationState,
    *,
    logger: SessionLogger,
    metrics: Metrics,
) -> dict[str, Any]:
    trace = _trace(state, "audit")
    decision = state.get("decision") or "none"
    frame = state.get("frame")
    node_trace = trace

    dur_ms = None
    if frame is not None:
        dur_ms = None  # 端到端耗时由调用方（orchestrator）统一打点
    logger.emit(
        "intent_decision",
        trace_id=str(state.get("trace_id") or "") or None,
        duration_ms=dur_ms,
        decision=decision,
        cmd=state.get("cmd"),
        code=state.get("code"),
        reason=state.get("reason"),
        confidence=round(float(state.get("confidence") or 0.0), 3),
        estop=bool(state.get("estop")),
        cooldown_ms=round(float(state.get("cooldown_ms") or 0.0), 1),
        watchdog=bool(state.get("watchdog_due")),
        frame_seq=(frame.seq if frame is not None else None),
        suppressed=(frame.suppressed if frame is not None else None),
        node_trace="→".join(node_trace),
    )
    metrics.incr(f"decision_{decision}")
    if state.get("revoked"):
        metrics.incr("rollback")
    return {"node_trace": node_trace}


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
def route_after_kill(state: OrchestrationState) -> str:
    decision = state.get("decision")
    if decision == "estop":
        return "dispatch"  # 急停旁路确认节点，直接落地
    if decision == "blocked":
        return "audit"
    return "confirm"  # continue / pending 都走确认节点


def route_after_watchdog(state: OrchestrationState) -> str:
    return "dispatch" if state.get("decision") == "dispatch" else "audit"
