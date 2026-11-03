"""Gesture-Copilot 启动入口（Plan V1.0 T4 / T12 / T13 / §5.4）。

启动：``python main.py``（或 ``uvicorn main:app --host 127.0.0.1 --port 8000``）

进程内单例与后台任务全部收拢在 `lifespan`：

    配置中心 → 会话日志 → 指标 → 控制总线 → WS 广播器 → 会话状态
             → 感知流水线 → 防抖状态 → 温路径编排器 → 自适应降级看门狗
             → 后台任务：看门狗滴答 / 降级巡检 / 视觉循环（无摄像头则空转退出）

任何后台任务异常都只记录日志，不阻断服务；关闭时统一取消。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src import __version__
from src.agents.debounce import DebounceState
from src.agents.debounce import break_streak as debounce_break_streak
from src.agents.debounce import new_state as new_debounce_state
from src.agents.graph import IntentOrchestrator, build_orchestrator, make_state
from src.agents.state import new_session_state
from src.api.routes import router as api_router
from src.api.schemas import CMD_LABELS, GESTURE_LABELS, MOTION_CMDS, RunMode, Source
from src.api.ws import WsHub
from src.api.ws import router as ws_router
from src.core.config import Settings, get_settings
from src.core.logger import SessionLogger, init_session_logger, new_trace_id
from src.core.metrics import AdaptiveDegrader, get_metrics
from src.execution.control_bus import build_bus_from_settings
from src.vision.capture import CameraInfo, CameraManager, CameraState
from src.vision.classifier import finger_states
from src.vision.pipeline import FramePipeline

__all__ = ["app", "create_app"]

INDEX_FILE_NAME = "index.html"

#: 视觉循环预热时长（秒）：覆盖 MediaPipe 首帧计算图构建开销
_VISION_WARMUP_S = 1.5


# --------------------------------------------------------------------------- #
# 后台任务
# --------------------------------------------------------------------------- #
async def _broadcast_command(app: FastAPI, frame: Any, logger: SessionLogger) -> None:
    hub: WsHub = app.state.hub
    await hub.broadcast(
        {
            "type": "command",
            "frame": frame.model_dump(mode="json"),
            "label": CMD_LABELS.get(frame.cmd.value, frame.cmd.value),
        }
    )
    session = app.state.session
    await hub.broadcast({"type": "state", "state": session.to_snapshot().model_dump(mode="json")})


def _publish_pose(
    app: FastAPI,
    hands: list[Any],
    trail: list[Any],
    candidate: Any | None,
) -> None:
    """把当前帧的姿态写入会话快照，供 WS 下行渲染实时骨架。

    `hands` 是"每只手一组 21 点"的列表（双手组合时长度为 2）。
    与 `/api/inject`、`/api/frames` 写入的 `latest_pose` 保持同一结构，
    关键差异是**不依赖是否识别出手势**：只要检测到手就发布。
    """
    code = candidate.code.value if candidate is not None else None
    primary = hands[0] if hands else []
    app.state.latest_pose = {
        "landmarks": [lm.model_dump() for lm in primary],
        "hands": [[lm.model_dump() for lm in group] for group in hands],
        "trail": [p.model_dump() for p in trail],
        "code": code,
        "label": GESTURE_LABELS.get(code) if code else None,
        "source": Source.GESTURE.value,
        "ts": time.time(),
    }


def _speed_from_fingers(landmarks: Any, cmd: Any) -> float | None:
    """手指计数调速：按**食指尖以外**伸出的手指数给运动指令分挡。

    轨迹追踪固定用食指尖（landmark 8），所以食指伸不伸不影响识别，
    正好拿来当"挡位数"：其余三指伸 0 根 → 低速，1 根 → 中速，≥2 根 → 高速。
    只对运动指令生效（急停/怠速恒为默认速度）。
    """
    if cmd is None or cmd not in MOTION_CMDS:
        return None
    non_index = sum(
        1 for name, extended in finger_states(landmarks).items() if name != "index" and extended
    )
    if non_index == 0:
        return 0.4
    if non_index == 1:
        return 0.7
    return 1.0


async def _watchdog_loop(app: FastAPI) -> None:
    """看门狗滴答：默认 10 Hz（Plan §3.2）。"""
    settings: Settings = app.state.settings
    orchestrator: IntentOrchestrator = app.state.orchestrator
    session = app.state.session
    metrics = app.state.metrics
    logger: SessionLogger = app.state.logger
    interval = 1.0 / max(1, settings.watchdog.tick_hz)

    while True:
        loop_started = time.perf_counter()
        try:
            outcome = await orchestrator.watchdog_tick(
                fps=metrics.fps,
                mode=session.mode.value,
                session_id=session.session_id,
            )
            if outcome is not None:
                frame = outcome.get("frame")
                app.state.last_idle_ts = time.time()
                if frame is not None:
                    await _broadcast_command(app, frame, logger)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover - 后台任务不得拖垮服务
            logger.emit("watchdog_error", error=f"{type(exc).__name__}: {exc}")
        # 周期补偿：扣掉本轮实际耗时，避免滴答漂移把 500ms 判定推到 600ms 之外
        elapsed = time.perf_counter() - loop_started
        await asyncio.sleep(max(0.0, interval - elapsed))


async def _degrade_loop(app: FastAPI) -> None:
    """自适应降级巡检：默认 1 Hz（T13）。"""
    settings: Settings = app.state.settings
    degrader: AdaptiveDegrader | None = getattr(app.state, "degrader", None)
    camera: CameraManager | None = getattr(app.state, "camera", None)
    logger: SessionLogger = app.state.logger
    metrics = app.state.metrics
    hub: WsHub = app.state.hub

    while True:
        await asyncio.sleep(1.0)
        if degrader is None:
            continue
        try:
            action = degrader.check()
            if action is None:
                continue
            applied: dict[str, Any] = {}
            if camera is not None:
                applied = camera.set_resolution(action.target_width, action.target_height)
            metrics.incr("degrade_triggered")
            metrics.set_gauge("degrade_level", float(action.level))
            metrics.set_gauge("degrade_p95_ms", float(action.p95))
            logger.emit(
                "degrade",
                duration_ms=None,
                # 字段名必须避开 `level`：logger.emit(level=...) 是控制台日志级别参数，
                # 传 int 会让 _emit_console 崩溃（历史 bug：降级成功后抛 AttributeError，
                # 导致 hub.broadcast 被跳过，HUD 收不到降级通知）。
                degrade_level=action.level,
                reason=action.reason,
                p95_ms=round(action.p95, 3),
                threshold_ms=action.threshold,
                samples=action.samples,
                applied=applied,
                note="Plan §4.1 降级：640x480 → 480x360",
            )
            await hub.broadcast({"type": "degrade", **action.as_dict(), "applied": applied})
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            logger.emit("degrade_error", error=f"{type(exc).__name__}: {exc}")


def _mirror_landmarks(landmarks: Any) -> list[Any]:
    """水平镜像归一化坐标（x → 1-x），把"观察者视角"转成"自拍视角"。

    MediaPipe 的 x 是**图像坐标**（向右为正）。用户面对摄像头时，手向自己的
    右侧移动在图像里是 x 减小，因此不翻转会让"右挥"被判成左转，同时 HUD 里的
    骨架与人手反向移动。y / z 不受镜像影响，保持原值。
    """
    return [lm.model_copy(update={"x": 1.0 - lm.x}) for lm in landmarks]


async def _vision_loop(app: FastAPI) -> None:
    """持续感知循环（T13）：capture → detect → pipeline → debounce → orchestrator。

    每帧都会通过 `_publish_pose` 发布 21 点姿态，因此 HUD 能看到实时骨架；
    但**原始视频画面不经过本进程下行**（HUD 的 `has_video` 恒为 False）。
    """
    settings: Settings = app.state.settings
    camera: CameraManager = app.state.camera
    logger: SessionLogger = app.state.logger
    metrics = app.state.metrics
    pipeline: FramePipeline = app.state.pipeline
    orchestrator: IntentOrchestrator = app.state.orchestrator
    debouncer: DebounceState = app.state.debouncer
    session = app.state.session

    while camera.state in (CameraState.UNPROBED, CameraState.PROBING):
        await asyncio.sleep(0.1)

    if not camera.available:
        logger.emit(
            "vision_loop_skipped",
            reason="无可用摄像头，视觉循环不启动（注入与 /api/frames 链路不受影响）",
            camera_state=camera.state.value,
        )
        return

    from src.vision.hand_detector import HandDetector  # 惰性导入，避免无摄像头时加载模型

    detector = await asyncio.to_thread(HandDetector)  # 初始化约 6.5s，放工作线程
    if not detector.available:  # pragma: no cover
        logger.emit("vision_loop_skipped", reason=f"检测器不可用：{detector.error}")
        return

    opened = await asyncio.to_thread(camera.open)
    if not opened:  # pragma: no cover
        logger.emit("vision_loop_skipped", reason="摄像头打开失败")
        return

    logger.emit("vision_loop_started", fps_target=settings.camera.target_fps)
    interval = 1.0 / max(1, settings.camera.target_fps)
    mirror = bool(getattr(settings.camera, "mirror_x", False))

    # 预热：MediaPipe 首次推理含计算图构建/内存分配，单帧可达数百毫秒。
    # 若把这些帧计入 frame_ms 直方图，P95 立刻超过降级阈值 25ms，
    # 会在启动 2 秒内误触发 640x480 → 480x360 降级（历史实测 P95=112ms）。
    # 与 graph_warmup 同口径：预热帧不计入直方图。
    warmup_started = time.perf_counter()
    warmup_frames = 0
    while (time.perf_counter() - warmup_started) < _VISION_WARMUP_S:
        frame = await asyncio.to_thread(camera.read)
        if frame is None:
            await asyncio.sleep(interval)
            continue
        await asyncio.to_thread(detector.detect, frame, frame_id=0)
        warmup_frames += 1
    logger.emit(
        "vision_warmup",
        duration_ms=round((time.perf_counter() - warmup_started) * 1000.0, 3),
        frames=warmup_frames,
        mirror_x=mirror,
        note="预热帧不计入 frame_ms 直方图，避免误触发自适应降级",
    )

    try:
        while True:
            started = time.perf_counter()
            frame = await asyncio.to_thread(camera.read)
            if frame is None:
                await asyncio.sleep(interval)
                continue
            session.tick()
            hand = await asyncio.to_thread(detector.detect_all, frame, frame_id=session.frame_id)
            if hand:
                # 双手组合包：最多取两只手，primary 用于单手势与 HUD 主骨架
                primary, second = hand[0], (hand[1] if len(hand) > 1 else None)
                lms = _mirror_landmarks(primary.landmarks) if mirror else list(primary.landmarks)
                lms2 = _mirror_landmarks(second.landmarks) if (mirror and second) else (
                    list(second.landmarks) if second else None
                )
                session.landmarks = list(lms)
                pf = pipeline.ingest(
                    lms,
                    ts=time.time(),
                    frame_id=session.frame_id,
                    score=primary.score,
                    source=Source.GESTURE,
                    second=lms2,
                )
                # 帧间超时 = 手势中断：同步给防抖层，解除静持与急停连击
                if pf.signals.get("gap_reset"):
                    debounce_break_streak(debouncer)
                candidate = pf.candidate
                # 无手势也要发布姿态：HUD 需要有连续骨架才能看出"手在画面里"
                _publish_pose(app, [lms] + ([lms2] if lms2 else []), pipeline.window.trail(12), candidate)
                signal_ts = time.time()
                from src.agents.debounce import FrameSignal, push as debounce_push

                decision = debounce_push(
                    debouncer,
                    FrameSignal.from_candidate(
                        candidate,
                        ts=signal_ts,
                        frame_id=session.frame_id,
                        source=Source.GESTURE,
                    ),
                    now=signal_ts,
                )
                if decision.cmd is not None or decision.latch or decision.blocked:
                    state = make_state(
                        result=decision,
                        trace_id=new_trace_id(),
                        session_id=session.session_id,
                        source=Source.GESTURE,
                        now=signal_ts,
                        frame_id=session.frame_id,
                        speed=_speed_from_fingers(primary, decision.cmd),
                    )
                    outcome = await orchestrator.submit(state)
                    emitted = outcome.get("frame")
                    if emitted is not None:
                        await _broadcast_command(app, emitted, logger)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metrics.observe("frame_ms", elapsed_ms)
            metrics.observe_fps(1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0)
            sleep_s = interval - elapsed_ms / 1000.0
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # pragma: no cover
        logger.emit("vision_loop_error", error=f"{type(exc).__name__}: {exc}")
    finally:  # pragma: no cover
        detector.close()
        camera.close()


# --------------------------------------------------------------------------- #
# lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.paths.static_dir.mkdir(parents=True, exist_ok=True)

    logger = init_session_logger(
        settings.paths.log_dir,
        level=settings.server.log_level,
        console=settings.server.log_console,
    )
    metrics = get_metrics(settings.performance.metrics_history)
    bus = build_bus_from_settings(settings, logger=logger, metrics=metrics)
    hub = WsHub()
    mode = RunMode.INJECT_ONLY if settings.mode.inject_only else RunMode.LIVE
    session = new_session_state(mode, capacity=settings.thresholds.window_capacity)

    # P2：感知流水线 / 防抖状态 / 温路径编排 / 降级看门狗
    pipeline = FramePipeline(settings)
    debouncer = new_debounce_state()
    orchestrator = build_orchestrator(
        settings=settings, bus=bus, session=session, logger=logger, metrics=metrics
    )
    degrader = AdaptiveDegrader(
        metrics,
        metric_name="frame_ms",
        threshold_ms=settings.degrade.frame_p95_ms,
        min_samples=settings.degrade.min_samples,
        max_level=1 if settings.degrade.enabled else 0,
        cooldown_s=settings.degrade.cooldown_s,
        target_width=settings.degrade.target_width,
        target_height=settings.degrade.target_height,
    )

    app.state.settings = settings
    app.state.logger = logger
    app.state.metrics = metrics
    app.state.bus = bus
    app.state.hub = hub
    app.state.session = session
    app.state.started_at = time.time()
    app.state.latest_pose = None
    app.state.pipeline = pipeline
    app.state.debouncer = debouncer
    app.state.orchestrator = orchestrator
    app.state.degrader = degrader
    app.state.last_idle_ts = None

    # 摄像头探测（Plan §12：绝不阻塞启动，失败自动降级为 INJECT_ONLY）
    camera = CameraManager(settings.camera)
    app.state.camera = camera
    app.state.camera_available = False
    app.state.camera_note = "摄像头探测未启用（CAMERA_PROBE_ENABLED=false）"

    def _on_camera_probe(info: CameraInfo) -> None:
        """探测回调（运行在探测线程内，logger 与 app.state 均线程安全）。"""
        app.state.camera_available = info.available
        app.state.camera_note = info.note or (
            "已探测到摄像头" if info.available else "未检测到可用摄像头"
        )
        logger.emit(
            "camera_probe",
            trace_id=session.session_id,
            available=info.available,
            state=info.state.value,
            index=info.index,
            width=info.width,
            height=info.height,
            fps=round(info.fps, 2),
            probe_elapsed_ms=round(info.elapsed_ms, 2),
            error=info.error,
            note=info.note,
        )

    logger.emit(
        "session_start",
        trace_id=session.session_id,
        service="gesture-copilot",
        version=__version__,
        mode=mode.value,
        dry_run=settings.mode.dry_run,
        inject_only=settings.mode.inject_only,
        log_file=str(logger.path),
        ws_port=settings.server.port,
        queue_size=bus.queue_size,
        dedupe_ms=bus.dedupe_ms,
        estop_cooldown_ms=bus.estop_cooldown_ms,
        frame_push_hz=settings.performance.frame_push_hz,
        window_capacity=settings.thresholds.window_capacity,
        camera_probe_enabled=settings.camera.probe_enabled,
        debounce={
            "window": debouncer.window_size,
            "confirm_frames": debouncer.confirm_frames,
            "estop_frames": debouncer.estop_frames,
            "min_confidence": debouncer.min_confidence,
        },
        graph={"p95_limit_ms": settings.graph.p95_limit_ms, "min_samples": settings.graph.min_samples},
        watchdog={"tick_hz": settings.watchdog.tick_hz, "miss_ms": settings.safety.watchdog_miss_ms},
        degrade={"enabled": settings.degrade.enabled, "p95_ms": settings.degrade.frame_p95_ms},
    )

    if settings.camera.probe_enabled:
        app.state.camera_note = "摄像头探测中…"
        camera.start_probe_async(_on_camera_probe)
    else:
        logger.emit(
            "camera_probe_skipped",
            trace_id=session.session_id,
            reason="CAMERA_PROBE_ENABLED=false",
        )

    # 温路径预热：消除 LangGraph 首次调用的 ~94ms 惰性编译开销（否则首个急停必被拖慢）
    warmup_ms = await orchestrator.warmup(session_id=session.session_id)
    logger.emit(
        "graph_warmup",
        trace_id=session.session_id,
        duration_ms=warmup_ms,
        note="首次 invoke 不计入 graph_ms 直方图，避免污染熔断判据",
    )

    tasks = [
        asyncio.create_task(_watchdog_loop(app), name="watchdog"),
        asyncio.create_task(_degrade_loop(app), name="degrade"),
        asyncio.create_task(_vision_loop(app), name="vision"),
    ]

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
        camera.close()
        logger.emit(
            "session_end",
            trace_id=session.session_id,
            uptime_s=round(time.time() - app.state.started_at, 3),
            bus=bus.stats(),
            camera=camera.snapshot(),
            debounce=debouncer.stats.as_dict(),
            graph=orchestrator.snapshot(),
            degrade=degrader.snapshot(),
            metrics=metrics.snapshot(),
        )
        logger.close()


def create_app() -> FastAPI:
    settings = get_settings()
    settings.paths.static_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Gesture-Copilot",
        version=__version__,
        description="纯本地实时代控手势识别 Agent（Plan V1.0 / P2 编排）",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    app.include_router(ws_router)
    app.mount(
        "/static",
        StaticFiles(directory=str(settings.paths.static_dir)),
        name="static",
    )

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> Response:
        index_path = settings.paths.static_dir / INDEX_FILE_NAME
        if not index_path.exists():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "HUD 尚未就绪：缺少 src/static/index.html",
                    "expected": str(index_path),
                },
            )
        return FileResponse(index_path)

    return app


app = create_app()


if __name__ == "__main__":
    _settings = get_settings()
    _settings.paths.ensure()
    uvicorn.run(
        app,
        host=_settings.server.host,
        port=_settings.server.port,
        log_level=_settings.server.log_level.lower(),
    )
