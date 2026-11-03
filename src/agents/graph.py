"""LangGraph 温路径编排与异步旁路调度（Plan V1.0 T12，落实 §2.5）。

三条硬性约束：
1. **图执行永远不在帧循环里同步跑**：`submit()` 通过 `asyncio.to_thread` 把图丢到工作线程，
   热路径（`debounce.push`）只负责产出裁决结果；
2. **自动熔断**：当 `graph_ms` 的 P95 超过 15 ms（样本数 ≥ `min_samples`）时，
   永久切换为 `run_direct()`（与图同构的纯函数序列）并打点告警，保证指标不被框架开销击穿；
3. **看门狗**：由独立的后台滴答驱动（默认 10 Hz），连续动作丢失手势 >500 ms
   （或 LIVE 模式 FPS<15 持续 300 ms）即下发 `CMD_IDLE`。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from src.agents import nodes
from src.agents.debounce import DebounceResult
from src.agents.nodes import OrchestrationState
from src.api.schemas import Cmd, ControlFrame, GestureCode, MOTION_CMDS, Source
from src.core.config import Settings, get_settings
from src.core.logger import SessionLogger, new_trace_id
from src.core.metrics import Metrics
from src.execution.control_bus import ControlBus

__all__ = [
    "build_graph",
    "run_direct",
    "make_state",
    "IntentOrchestrator",
]

_MOTION_VALUES = frozenset(cmd.value for cmd in MOTION_CMDS)


# --------------------------------------------------------------------------- #
# 图构建
# --------------------------------------------------------------------------- #
def build_graph(
    *,
    bus: ControlBus,
    session: Any,
    logger: SessionLogger,
    metrics: Metrics,
) -> Any:
    """编译温路径 StateGraph。"""
    builder: StateGraph = StateGraph(OrchestrationState)
    builder.add_node("kill_switch", nodes.kill_switch_node)
    builder.add_node("confirm_intent", nodes.confirm_intent_node)
    builder.add_node("watchdog", nodes.watchdog_node)
    builder.add_node("dispatch", _bind(nodes.dispatch_node, bus=bus, session=session))
    builder.add_node("audit", _bind(nodes.audit_node, logger=logger, metrics=metrics))

    builder.add_edge(START, "kill_switch")
    builder.add_conditional_edges(
        "kill_switch",
        nodes.route_after_kill,
        {"confirm": "confirm_intent", "dispatch": "dispatch", "audit": "audit"},
    )
    builder.add_edge("confirm_intent", "watchdog")
    builder.add_conditional_edges(
        "watchdog",
        nodes.route_after_watchdog,
        {"dispatch": "dispatch", "audit": "audit"},
    )
    builder.add_edge("dispatch", "audit")
    builder.add_edge("audit", END)
    return builder.compile()


def _bind(fn: Callable[..., dict[str, Any]], **deps: Any) -> Callable[[OrchestrationState], dict[str, Any]]:
    """把带依赖的节点函数适配成 LangGraph 需要的 `state -> partial_state` 形态。"""

    def node(state: OrchestrationState) -> dict[str, Any]:
        return fn(state, **deps)

    node.__name__ = getattr(fn, "__name__", "node")
    return node


# --------------------------------------------------------------------------- #
# 降级路径：与图同构的纯函数序列（不依赖 LangGraph）
# --------------------------------------------------------------------------- #
def run_direct(
    state: OrchestrationState,
    *,
    bus: ControlBus,
    session: Any,
    logger: SessionLogger,
    metrics: Metrics,
) -> OrchestrationState:
    """熔断后的等价执行路径（节点与边顺序与图完全一致）。"""
    state.update(nodes.kill_switch_node(state))
    route = nodes.route_after_kill(state)

    if route == "dispatch":
        state.update(nodes.dispatch_node(state, bus=bus, session=session))
    elif route == "confirm":
        state.update(nodes.confirm_intent_node(state))
        state.update(nodes.watchdog_node(state))
        if nodes.route_after_watchdog(state) == "dispatch":
            state.update(nodes.dispatch_node(state, bus=bus, session=session))

    state.update(nodes.audit_node(state, logger=logger, metrics=metrics))
    return state


# --------------------------------------------------------------------------- #
# 计时包装：在线程内测量纯执行耗时，剔除线程池/事件循环排队
# --------------------------------------------------------------------------- #
def _invoke_timed(graph: Any, state: OrchestrationState) -> tuple[OrchestrationState, float]:
    started = time.perf_counter()
    result = graph.invoke(state)
    return result, (time.perf_counter() - started) * 1000.0


def _run_direct_timed(
    state: OrchestrationState,
    *,
    bus: ControlBus,
    session: Any,
    logger: SessionLogger,
    metrics: Metrics,
) -> tuple[OrchestrationState, float]:
    started = time.perf_counter()
    result = run_direct(state, bus=bus, session=session, logger=logger, metrics=metrics)
    return result, (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 状态构造
# --------------------------------------------------------------------------- #
def make_state(
    *,
    result: DebounceResult | None = None,
    trace_id: str | None = None,
    session_id: str = "",
    source: Source = Source.GESTURE,
    now: float | None = None,
    frame_id: int = 0,
    speed: float | None = None,
    speed_override: bool = False,
    watchdog_due: bool = False,
    watchdog_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> OrchestrationState:
    """把防抖裁决结果（或看门狗滴答）翻译成图状态。"""
    at = time.time() if now is None else now
    state: OrchestrationState = {
        "trace_id": trace_id or new_trace_id(),
        "session_id": session_id,
        "now": at,
        "frame_id": frame_id,
        "source": source.value,
        "code": result.code.value if result and result.code else None,
        "cmd": result.cmd.value if result and result.cmd else None,
        "confidence": float(result.confidence) if result else 0.0,
        "reason": (result.reason if result else None) or ("ok"),
        "estop": bool(result.estop) if result else False,
        "estop_latched": bool(result.latch) if result else False,
        "cooldown_ms": float(result.cooldown_ms) if result else 0.0,
        "watchdog_due": watchdog_due,
        "watchdog_reason": watchdog_reason,
        "revoked": result.revoked_code.value if result and result.revoked_code else None,
        "node_trace": [],
    }
    if speed_override:
        state["speed"] = speed
    elif result is not None and result.code is not None:
        state["speed"] = None
    if extra:
        state.update(extra)  # type: ignore[typeddict-item]
    return state


# --------------------------------------------------------------------------- #
# 编排器：异步旁路 + 熔断 + 看门狗
# --------------------------------------------------------------------------- #
class IntentOrchestrator:
    """温路径编排入口。线程安全：图本身无共享可变状态，状态全在本次 state 里。"""

    def __init__(
        self,
        *,
        settings: Settings,
        bus: ControlBus,
        session: Any,
        logger: SessionLogger,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.session = session
        self.logger = logger
        self.metrics = metrics
        self.graph = build_graph(bus=bus, session=session, logger=logger, metrics=metrics)
        self.graph_p95_limit_ms = settings.graph.p95_limit_ms
        self.graph_min_samples = settings.graph.min_samples

        self._degraded = False
        self._graph_ms = metrics.histogram("graph_ms")
        self._submits = 0
        self._last: dict[str, Any] = {}
        self._warmup_ms: float | None = None

        # 看门狗
        self._motion_armed = False
        self._last_motion_ts = 0.0
        self._low_fps_since: float | None = None
        self._watchdog_fires = 0
        self._last_idle_ts = 0.0

    # -- 熔断判据 ---------------------------------------------------------- #
    def _should_degrade(self) -> bool:
        stats = self._graph_ms.stats()
        if int(stats.get("count", 0)) < self.graph_min_samples:
            return False
        return float(stats.get("p95", 0.0)) > self.graph_p95_limit_ms

    @property
    def degraded(self) -> bool:
        return self._degraded

    # -- 预热 -------------------------------------------------------------- #
    async def warmup(self, session_id: str = "") -> float:
        """启动预热：跑一次空转编排，消除首次调用的编译/导入开销。

        实测首次 `graph.invoke` 约 94 ms（LangGraph 惰性编译），而稳态只要 2~5 ms；
        若不预热，服务启动后的第一次急停会被框架开销拖慢、直接击穿 100 ms 门禁。

        预热样本**不计入** `graph_ms` 直方图，避免污染熔断判据。
        """
        state = make_state(
            trace_id=new_trace_id(),
            session_id=session_id,
            source=Source.GESTURE,
            now=time.time(),
            extra={"cmd": None, "reason": "warmup"},
        )
        started = time.perf_counter()
        await asyncio.to_thread(self.graph.invoke, state)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._warmup_ms = elapsed_ms
        return elapsed_ms
    # -- 提交 -------------------------------------------------------------- #
    async def submit(self, state: OrchestrationState) -> OrchestrationState:
        """异步旁路执行一次编排（工作线程内跑图，绝不阻塞事件循环）。

        计时口径：
        - `graph_ms`         = **纯图执行**（线程内测量，不含线程池/事件循环排队），
                               这是 Plan §2.5 熔断判据所依据的量；
        - `orchestrator_ms`  = 含异步调度在内的总耗时，用于观察旁路开销。
        """
        started = time.perf_counter()
        self._submits += 1

        if self._degraded:
            result, graph_ms = await asyncio.to_thread(
                _run_direct_timed,
                state,
                bus=self.bus,
                session=self.session,
                logger=self.logger,
                metrics=self.metrics,
            )
        else:
            try:
                result, graph_ms = await asyncio.to_thread(_invoke_timed, self.graph, state)
            except Exception as exc:  # 图异常必须退化为直连路径，不能丢指令
                self.metrics.incr("graph_error")
                self.logger.emit(
                    "graph_error",
                    trace_id=state.get("trace_id"),
                    error=f"{type(exc).__name__}: {exc}",
                )
                result, graph_ms = await asyncio.to_thread(
                    _run_direct_timed,
                    state,
                    bus=self.bus,
                    session=self.session,
                    logger=self.logger,
                    metrics=self.metrics,
                )

        duration_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.observe("graph_ms", graph_ms)
        self.metrics.observe("orchestrator_ms", duration_ms)
        self._post_process(result, duration_ms)
        return result

    def _post_process(self, result: OrchestrationState, duration_ms: float) -> None:
        frame: ControlFrame | None = result.get("frame")
        self._last = {
            "decision": result.get("decision"),
            "cmd": result.get("cmd"),
            "reason": result.get("reason"),
            "duration_ms": round(duration_ms, 3),
            "frame_seq": frame.seq if frame is not None else None,
        }
        if frame is not None and not frame.suppressed:
            if frame.cmd.value in _MOTION_VALUES:
                self._motion_armed = True
                self._last_motion_ts = frame.timestamp
                self._low_fps_since = None
            elif frame.cmd in {Cmd.CMD_STOP, Cmd.CMD_IDLE}:
                self._motion_armed = False
                self._low_fps_since = None

        if not self._degraded and self._should_degrade():
            self._degraded = True
            stats = self._graph_ms.stats()
            self.metrics.incr("graph_degraded")
            self.metrics.set_gauge("graph_degraded", 1.0)
            self.metrics.set_gauge("graph_p95_ms", float(stats.get("p95", 0.0)))
            self.logger.emit(
                "graph_degraded",
                p95_ms=round(float(stats.get("p95", 0.0)), 3),
                limit_ms=self.graph_p95_limit_ms,
                samples=int(stats.get("count", 0)),
                action="切换为 run_direct 直连路径（Plan §2.5）",
            )

    # -- 看门狗 ------------------------------------------------------------ #
    async def watchdog_tick(
        self,
        *,
        fps: float | None = None,
        mode: str = "INJECT_ONLY",
        session_id: str = "",
    ) -> OrchestrationState | None:
        """看门狗滴答：连续动作丢失手势 >500ms（或 LIVE 下 FPS<15 持续 300ms）→ CMD_IDLE。"""
        if not self._motion_armed:
            self._low_fps_since = None
            return None

        now = time.time()
        miss_s = self.settings.safety.watchdog_miss_ms / 1000.0
        reason: str | None = None

        if now - self._last_motion_ts >= miss_s:
            reason = "gesture_miss"
        elif mode == "LIVE" and fps is not None and fps < self.settings.watchdog.fps_floor:
            if self._low_fps_since is None:
                self._low_fps_since = now
            elif (now - self._low_fps_since) * 1000.0 >= self.settings.watchdog.fps_low_ms:
                reason = "low_fps"
        else:
            self._low_fps_since = None

        if reason is None:
            return None

        state = make_state(
            trace_id=new_trace_id(),
            session_id=session_id,
            source=Source.WATCHDOG,
            now=now,
            watchdog_due=True,
            watchdog_reason=reason,
        )
        result = await self.submit(state)
        self._motion_armed = False
        self._low_fps_since = None
        self._watchdog_fires += 1
        self._last_idle_ts = now
        self.metrics.incr("watchdog_tick_fired")
        return result

    # -- 观测 -------------------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        stats = self._graph_ms.stats()
        return {
            "degraded": self._degraded,
            "p95_ms": float(stats.get("p95", 0.0)),
            "p50_ms": float(stats.get("p50", 0.0)),
            "samples": int(stats.get("count", 0)),
            "limit_ms": self.graph_p95_limit_ms,
            "min_samples": self.graph_min_samples,
            "submits": self._submits,
            "warmup_ms": round(self._warmup_ms, 3) if self._warmup_ms is not None else None,
            "watchdog_armed": self._motion_armed,
            "watchdog_fires": self._watchdog_fires,
            "watchdog_miss_ms": self.settings.safety.watchdog_miss_ms,
            "last": dict(self._last),
        }


def build_orchestrator(
    *,
    settings: Settings | None = None,
    bus: ControlBus,
    session: Any,
    logger: SessionLogger,
    metrics: Metrics,
) -> IntentOrchestrator:
    return IntentOrchestrator(
        settings=settings or get_settings(),
        bus=bus,
        session=session,
        logger=logger,
        metrics=metrics,
    )


def gesture_cmd_of(code: GestureCode | None) -> Cmd | None:
    """便利函数（供外部诊断）：手势码对应的指令。"""
    if code is None:
        return None
    from src.api.schemas import GESTURE_TO_CMD

    return GESTURE_TO_CMD.get(code)
