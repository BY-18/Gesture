"""REST 接入层（Plan V1.0 T4 / T13 / §5.4）。

端点契约：
====================  ======  ====================================================
路径                  方法    职责
====================  ======  ====================================================
/api/health          GET     健康检查（模式、FPS、相机、WS 连接数、日志路径）
/api/state           GET     当前状态机快照（含冷却期、滑窗、熔断标志）
/api/inject          POST    已确认意图注入（P0 契约；走同一套编排链路）
/api/frames          POST    热路径逐帧输入（T13：视觉流/回放 → debounce → graph → bus）
/api/metrics         GET     延迟直方图 / 计数 / 总线 / 编排 / 降级 观测面
/api/metrics/reset   POST    运维重置：清零指标直方图/计数器与总线统计（seq 保留单调性）
====================  ======  ====================================================

两条入口的分工：
- `/api/inject`：语义为"该手势已被确认"，故 `preconfirmed=True` 跳过 3 帧滑窗，
  保留门禁 G0 的既有契约；
- `/api/frames`：语义为"这是一帧原始候选"，必须经 `debounce.py` 表决。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src import __version__
from src.agents.debounce import (
    DebounceState,
    FrameSignal,
    break_streak,
    push as debounce_push,
    snapshot as debounce_snapshot,
)
from src.agents.graph import IntentOrchestrator, make_state
from src.agents.state import AgentState
from src.api.injector import build_candidate
from src.api.schemas import (
    CMD_LABELS,
    Cmd,
    FrameRequest,
    FrameResponse,
    GESTURE_LABELS,
    GESTURE_TO_CMD,
    HealthResponse,
    InjectRequest,
    InjectResponse,
    MetricsResponse,
    Source,
    StateSnapshot,
)
from src.core.config import Settings
from src.core.logger import SessionLogger, new_trace_id, trace_context
from src.core.metrics import AdaptiveDegrader, Metrics
from src.execution.control_bus import ControlBus
from src.vision.capture import CameraManager
from src.vision.pipeline import FramePipeline

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["api"])

#: 摄像头未探测时的兜底说明
_CAMERA_NOTE = "摄像头尚未探测"

#: 每帧广播 command 的最大条数（防御性上限）
_MAX_BROADCAST = 8


def _ctx(request: Request) -> tuple[Settings, SessionLogger, Metrics, ControlBus, AgentState, Any]:
    app = request.app
    return (
        app.state.settings,
        app.state.logger,
        app.state.metrics,
        app.state.bus,
        app.state.session,
        app.state.hub,
    )


def _decorate(snapshot: StateSnapshot, app_state: Any) -> StateSnapshot:
    """把 P2 运行时视图（冷却期 / 滑窗 / 熔断）补进状态快照。"""
    debouncer: DebounceState | None = getattr(app_state, "debouncer", None)
    orchestrator: IntentOrchestrator | None = getattr(app_state, "orchestrator", None)
    if debouncer is not None:
        snapshot.cooldown_ms = debouncer.cooldown_remaining_ms
        snapshot.debounce_window = len(debouncer.window)
    if orchestrator is not None:
        snapshot.graph_degraded = orchestrator.degraded
    return snapshot


def _inject_override(decision: Any, speed: float | None) -> dict[str, Any] | None:
    """注入通道的意图兜底：确保"已确认意图"必定产出一帧，由总线终裁是否放行。

    - 冷却期拦截（`blocked`）：用 `blocked_cmd` 透传，并把 `cooldown_ms` 清零，
      让编排层的 kill_switch 放行、由总线返回 `suppressed=True` 的帧；
    - 其它异常（理论上不应发生）：用该手势的原生 `cmd` 兜底。
    """
    if decision.blocked and decision.blocked_cmd is not None:
        return {"cmd": decision.blocked_cmd.value, "cooldown_ms": 0.0}
    if decision.cmd is not None:
        return None
    if decision.code is not None:
        cmd = GESTURE_TO_CMD.get(decision.code)
        if cmd is not None:
            return {"cmd": cmd.value, "confidence": max(decision.confidence, 0.9)}
    return None


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings, logger, metrics, _bus, session, hub = _ctx(request)
    camera_available = bool(getattr(request.app.state, "camera_available", False))
    camera_note = getattr(request.app.state, "camera_note", _CAMERA_NOTE)
    camera: CameraManager | None = getattr(request.app.state, "camera", None)
    if camera is not None:
        camera_available = camera.available
        camera_note = camera.note or camera_note
    return HealthResponse(
        status="ok",
        service="gesture-copilot",
        version=__version__,
        mode=session.mode,
        dry_run=settings.mode.dry_run,
        inject_only=settings.mode.inject_only,
        camera_available=camera_available,
        camera_note=str(camera_note),
        uptime_s=round(time.time() - request.app.state.started_at, 3),
        fps=metrics.fps,
        session_id=session.session_id,
        seq=session.frame_id,
        ws_connections=hub.connections,
        log_file=str(logger.path),
    )


@router.get("/state", response_model=StateSnapshot)
async def state_snapshot(request: Request) -> StateSnapshot:
    _settings, _logger, _metrics, bus, session, _hub = _ctx(request)
    session.estop_latched = bus.estop_active
    snapshot = session.to_snapshot()
    return _decorate(snapshot, request.app.state)


@router.get("/metrics", response_model=MetricsResponse)
async def metrics_snapshot(request: Request) -> MetricsResponse:
    settings, _logger, metrics, bus, session, hub = _ctx(request)
    app_state = request.app.state
    snapshot = metrics.snapshot()
    orchestrator: IntentOrchestrator | None = getattr(app_state, "orchestrator", None)
    debouncer: DebounceState | None = getattr(app_state, "debouncer", None)
    degrader: AdaptiveDegrader | None = getattr(app_state, "degrader", None)

    return MetricsResponse(
        mode=session.mode,
        dry_run=settings.mode.dry_run,
        uptime_s=round(time.time() - app_state.started_at, 3),
        fps=metrics.fps,
        histograms=snapshot["histograms"],  # type: ignore[arg-type]
        counters=snapshot["counters"],  # type: ignore[arg-type]
        gauges=snapshot["gauges"],  # type: ignore[arg-type]
        bus=bus.stats(),
        ws_connections=hub.connections,
        frame_budget_ms=settings.performance.frame_budget_ms,
        frame_p95_limit_ms=settings.performance.frame_p95_limit_ms,
        debounce=debounce_snapshot(debouncer) if debouncer is not None else {},
        graph=orchestrator.snapshot() if orchestrator is not None else {},
        watchdog={
            "armed": bool(orchestrator and orchestrator.snapshot().get("watchdog_armed")),
            "fires": int(orchestrator.snapshot().get("watchdog_fires", 0)) if orchestrator else 0,
            "miss_ms": settings.safety.watchdog_miss_ms,
            "tick_hz": settings.watchdog.tick_hz,
            "fps_floor": settings.watchdog.fps_floor,
            "fps_low_ms": settings.watchdog.fps_low_ms,
            "last_idle_ts": getattr(app_state, "last_idle_ts", None),
        },
        degrade=degrader.snapshot() if degrader is not None else {},
    )


@router.post("/metrics/reset")
async def metrics_reset(request: Request) -> dict[str, str]:
    """运维重置：清零指标直方图/计数器/瞬时量与总线统计（seq 单调性保留）。

    用途：长跑实例复用前的测量基线归零（验收脚本 / 现场演示），不影响运行状态。
    """
    _settings, logger, metrics, bus, _session, _hub = _ctx(request)
    metrics.reset()
    bus.reset_stats()
    logger.emit("metrics_reset", trace_id="ops", duration_ms=None, source="api")
    return {"status": "ok"}


async def _broadcast_frames(hub: Any, frames: list[Any]) -> float:
    """把控制帧推送给 HUD，返回广播耗时（毫秒）。"""
    started = time.perf_counter()
    for frame in frames[:_MAX_BROADCAST]:
        await hub.broadcast(
            {
                "type": "command",
                "frame": frame.model_dump(mode="json"),
                "label": CMD_LABELS.get(frame.cmd.value, frame.cmd.value),
            }
        )
    return (time.perf_counter() - started) * 1000.0


@router.post("/inject", response_model=InjectResponse)
async def inject(payload: InjectRequest, request: Request) -> InjectResponse:
    """虚拟手势注入（G0 契约：已确认意图 → 编排 → 总线 → WS）。"""
    settings, logger, metrics, bus, session, hub = _ctx(request)
    app_state = request.app.state
    orchestrator: IntentOrchestrator = app_state.orchestrator
    debouncer: DebounceState = app_state.debouncer

    started = time.perf_counter()
    trace_id = payload.trace_id or new_trace_id()
    stats_before = bus.stats()

    frames: list[Any] = []
    suppressed = 0
    emitted = 0
    debounce_ms_total = 0.0

    with trace_context(trace_id):
        for _ in range(payload.count):
            session.tick()
            candidate, pose = build_candidate(
                payload.code,
                score=payload.score,
                frame_id=session.frame_id,
                landmarks=payload.landmarks,
                speed=payload.speed,
                trace_id=trace_id,
            )
            app_state.latest_pose = {
                "landmarks": [lm.model_dump() for lm in pose.landmarks],
                "trail": [pt.model_dump() for pt in pose.trail],
                "code": pose.code.value,
                "label": pose.label,
                "source": Source.INJECT.value,
                "ts": time.time(),
            }
            session.apply_candidate(candidate, trace_id=trace_id)

            at = time.time()
            signal = FrameSignal(
                code=candidate.code,
                score=candidate.score,
                ts=at,
                frame_id=session.frame_id,
                preconfirmed=True,  # 注入语义 = 已确认
                source=Source.INJECT,
            )
            t0 = time.perf_counter()
            decision = debounce_push(debouncer, signal, now=at)
            debounce_ms_total += (time.perf_counter() - t0) * 1000.0

            state = make_state(
                result=decision,
                trace_id=trace_id,
                session_id=session.session_id,
                source=Source.INJECT,
                now=at,
                frame_id=session.frame_id,
                speed=payload.speed,
                speed_override=payload.speed is not None,
                # 注入通道的拦截由**控制总线**做终裁（返回 suppressed 帧供前端观测），
                # 不在编排层丢弃已确认的意图，否则 P0 契约（必须返回控制帧）被破坏。
                # blocked → 用 blocked_cmd；其余兜底用该手势的原生指令。
                extra=_inject_override(decision, payload.speed),
            )
            out = await orchestrator.submit(state)
            frame = out.get("frame")
            if frame is not None:
                frames.append(frame)
                if frame.suppressed:
                    suppressed += 1
                else:
                    emitted += 1
            if payload.interval_ms and payload.count > 1:
                await asyncio.sleep(payload.interval_ms / 1000.0)

        session.estop_latched = bus.estop_active
        snapshot = _decorate(session.to_snapshot(), app_state)
        await _broadcast_frames(hub, frames)
        await hub.broadcast({"type": "state", "state": snapshot.model_dump(mode="json")})

    stats_after = bus.stats()
    blocked = int(stats_after["blocked"]) - int(stats_before["blocked"])
    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics.observe("inject_total_ms", latency_ms)
    metrics.observe("debounce_ms", debounce_ms_total)
    metrics.incr("inject_requests")
    metrics.incr("inject_frames", len(frames))
    if suppressed:
        metrics.incr("inject_suppressed", suppressed)

    logger.emit(
        "inject",
        trace_id=trace_id,
        duration_ms=latency_ms,
        code=payload.code.value,
        count=payload.count,
        emitted=emitted,
        suppressed=suppressed,
        blocked=blocked,
        interval_ms=payload.interval_ms,
        score=round(payload.score, 3),
        debounce_ms=round(debounce_ms_total, 3),
        graph_degraded=orchestrator.degraded,
        ws_connections=hub.connections,
        dry_run=settings.mode.dry_run,
    )

    if not frames:  # pragma: no cover
        raise HTTPException(status_code=500, detail="注入未产生任何控制帧")

    return InjectResponse(
        accepted=True,
        requested=payload.count,
        emitted=emitted,
        suppressed=suppressed,
        blocked=blocked,
        latency_ms=round(latency_ms, 3),
        frames=frames,
        state=snapshot,
        dedupe_window_ms=settings.safety.dedupe_ms,
    )


@router.post("/frames", response_model=FrameResponse)
async def frames(payload: FrameRequest, request: Request) -> FrameResponse:
    """热路径逐帧输入：pipeline → debounce → orchestrator( graph ) → bus → WS。"""
    settings, logger, metrics, bus, session, hub = _ctx(request)
    app_state = request.app.state
    pipeline: FramePipeline = app_state.pipeline
    debouncer: DebounceState = app_state.debouncer
    orchestrator: IntentOrchestrator = app_state.orchestrator

    started = time.perf_counter()
    trace_id = payload.trace_id or new_trace_id()
    received_at = time.time()
    frame_ts = payload.frame_ts if payload.frame_ts is not None else received_at

    emitted_frames: list[Any] = []
    decision_reason: str | None = None
    code_value = None
    cmd_value = None
    triggered = False
    signals: dict[str, Any] = {}

    with trace_context(trace_id):
        session.tick()
        session.landmarks = list(payload.landmarks)

        # ① 感知流水线（几何判定）
        pf = pipeline.ingest(
            payload.landmarks,
            ts=received_at,
            frame_id=session.frame_id,
            score=payload.score,
            source=Source.GESTURE,
            explicit=payload.explicit,
        )
        metrics.observe("pipeline_ms", pf.timings_ms.get("total_ms", 0.0))
        signals = pf.signals
        candidate = pf.candidate
        # 帧间超时 = 手势中断：清表决窗口，避免上一段手势的残留帧续上新动作
        if signals.get("gap_reset"):
            break_streak(debouncer)
            metrics.incr("gesture_gap_reset")
        if candidate is not None:
            session.apply_candidate(candidate, trace_id=trace_id)

        # 姿态快照：与视觉循环保持一致——**每帧都发布**，不依赖是否识别出手势，
        # 否则 HUD 只会在指令产生的那一帧闪一下骨架。
        pose_code = candidate.code.value if candidate is not None else None
        app_state.latest_pose = {
            "landmarks": [lm.model_dump() for lm in payload.landmarks],
            "trail": [p.model_dump() for p in pipeline.window.trail(12)],
            "code": pose_code,
            "label": GESTURE_LABELS.get(pose_code) if pose_code else None,
            "source": Source.GESTURE.value,
            "ts": received_at,
        }

        # ② 热路径表决（纯函数，微秒级）
        t_debounce = time.perf_counter()
        signal = FrameSignal.from_candidate(
            candidate,
            ts=received_at,
            frame_id=session.frame_id,
            source=Source.GESTURE,
        )
        decision = debounce_push(debouncer, signal, now=received_at)
        debounce_ms = (time.perf_counter() - t_debounce) * 1000.0
        metrics.observe("debounce_ms", debounce_ms)

        decision_reason = decision.reason
        code_value = decision.code or (candidate.code if candidate else None)
        cmd_value = decision.cmd
        triggered = decision.triggered

        # ③ 回滚撤销（第 4/5 帧反向）：立即怠速
        if decision.revoked_code is not None:
            idle_state = make_state(
                trace_id=trace_id,
                session_id=session.session_id,
                source=Source.GESTURE,
                now=received_at,
                frame_id=session.frame_id,
                extra={
                    "cmd": Cmd.CMD_IDLE.value,
                    "reason": "rollback",
                    "confidence": decision.confidence,
                },
            )
            rolled = await orchestrator.submit(idle_state)
            frame = rolled.get("frame")
            if frame is not None:
                emitted_frames.append(frame)
            metrics.incr("rollback_dispatch")

        # ④ 温路径编排（触发/锁存/拦截时）
        if decision.cmd is not None or decision.latch or decision.blocked or decision.estop:
            state = make_state(
                result=decision,
                trace_id=trace_id,
                session_id=session.session_id,
                source=Source.GESTURE,
                now=received_at,
                frame_id=session.frame_id,
            )
            out = await orchestrator.submit(state)
            frame = out.get("frame")
            if frame is not None:
                emitted_frames.append(frame)

        session.estop_latched = bus.estop_active
        snapshot = _decorate(session.to_snapshot(), app_state)

    broadcast_ms = await _broadcast_frames(hub, emitted_frames)
    if emitted_frames:
        await hub.broadcast({"type": "state", "state": snapshot.model_dump(mode="json")})

    processed_at = time.time()
    latency_ms = (time.perf_counter() - started) * 1000.0
    # 负值只可能来自客户端时钟超前（frame_ts 由调用方提供），钳到 0
    gesture_latency_ms = max(0.0, (processed_at - frame_ts) * 1000.0)

    # 单帧全链路耗时：T13 的自适应降级依据
    metrics.observe("frame_ms", latency_ms)
    metrics.observe("frame_latency_ms", gesture_latency_ms)
    metrics.observe("ws_broadcast_ms", broadcast_ms)
    metrics.incr("frames_ingested")

    logger.emit(
        "frame",
        trace_id=trace_id,
        duration_ms=latency_ms,
        gesture_latency_ms=round(gesture_latency_ms, 3),
        frame_id=session.frame_id,
        code=code_value.value if code_value else None,
        cmd=cmd_value.value if cmd_value else None,
        triggered=triggered,
        reason=decision_reason,
        cooldown_ms=round(decision.cooldown_ms, 1),
        debounce_ms=round(debounce_ms, 4),
        pipeline_ms=pf.timings_ms.get("total_ms", 0.0),
        window=debouncer.window_size,
    )

    return FrameResponse(
        accepted=True,
        frame_id=session.frame_id,
        code=code_value,
        cmd=cmd_value,
        triggered=triggered,
        reason=decision_reason,
        latency_ms=round(latency_ms, 3),
        gesture_latency_ms=round(gesture_latency_ms, 3),
        frames=emitted_frames,
        state=snapshot,
        signals=signals,
        estop_latched=bus.estop_active,
        cooldown_ms=round(decision.cooldown_ms, 1),
    )
