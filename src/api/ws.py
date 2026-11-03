"""WebSocket 推流端点（Plan V1.0 T4 / §5.4）。

实现要点：
- 每条连接一个 `asyncio.Queue`（容量 64），**只有连接主循环负责发送**，
  避免"周期推送帧"与"事件广播"两个协程并发写同一 socket；
- 帧推送频率 `FRAME_PUSH_HZ`（默认 15 Hz），仅在有新姿态时携带骨骼数据；
- 客户端可发 `{"type":"ping"}`，服务端回 `pong`（HUD 心跳）；
- 断连自愈由前端负责（指数退避重连），服务端不留残留队列。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.schemas import CMD_LABELS, GESTURE_LABELS, GestureCode
from src.agents.state import AgentState
from src.core.config import Settings
from src.core.metrics import Metrics

__all__ = ["router", "WsHub"]

router = APIRouter()

_QUEUE_CAPACITY = 64


class WsHub:
    """连接注册表与广播器。"""

    def __init__(self) -> None:
        self._queues: dict[int, asyncio.Queue[dict[str, Any]]] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()

    @property
    def connections(self) -> int:
        return len(self._queues)

    async def register(self) -> tuple[int, "asyncio.Queue[dict[str, Any]]"]:
        async with self._lock:
            self._next_id += 1
            cid = self._next_id
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_CAPACITY)
            self._queues[cid] = queue
            return cid, queue

    async def unregister(self, cid: int) -> None:
        async with self._lock:
            self._queues.pop(cid, None)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """向全部连接投递消息；队列满时丢弃最旧消息（保证最新状态可达）。"""
        async with self._lock:
            queues = list(self._queues.values())
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:  # pragma: no cover
                pass


def _put(queue: "asyncio.Queue[dict[str, Any]]", message: dict[str, Any]) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover
            pass
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:  # pragma: no cover
        pass


def _hello_message(session: AgentState, settings: Settings) -> dict[str, Any]:
    return {
        "type": "hello",
        "session_id": session.session_id,
        "mode": session.mode.value,
        "dry_run": settings.mode.dry_run,
        "version": "0.1.0",
        "frame_push_hz": settings.performance.frame_push_hz,
        "gesture_labels": GESTURE_LABELS,
        "cmd_labels": CMD_LABELS,
        "gesture_codes": [code.value for code in GestureCode],
        "width": settings.camera.width,
        "height": settings.camera.height,
    }


def _frame_message(app_state: Any, session: AgentState, metrics: Metrics, settings: Settings) -> dict[str, Any]:
    """组装一帧下行数据。

    `has_video` 恒为 False：HUD 用网格背景 + 21 点骨架渲染，不下行原始视频。
    `landmarks` 有两个来源——真实视觉循环（`source=GESTURE`）与注入/回放
    （`source=INJECT`/`GESTURE` 经 `/api/inject`、`/api/frames`），前端据此区分。
    姿态数据带 TTL（3 × watchdog_miss_ms ≈1.5 s）后自动清空，模拟"手离开画面"。
    """
    now = time.time()
    pose = getattr(app_state, "latest_pose", None)
    age_ms: float | None = None
    if pose is not None:
        age_ms = (now - float(pose.get("ts", now))) * 1000.0
        ttl_ms = settings.safety.watchdog_miss_ms * 3
        if age_ms > ttl_ms:
            pose = None

    return {
        "type": "frame",
        "seq": session.frame_id,
        "ts": now,
        "width": settings.camera.width,
        "height": settings.camera.height,
        "mode": session.mode.value,
        "fps": metrics.fps,
        "frame_ms": None,  # P1 接入视觉层后回填单帧耗时
        "has_video": False,
        "landmarks": pose.get("landmarks") if pose else None,
        # 双手组合包：每只手一组 21 点；单手势时为长度 1 的列表
        "hands": (pose.get("hands") or ([pose.get("landmarks")] if pose.get("landmarks") else []))
        if pose
        else [],
        "trail": pose.get("trail") if pose else [],
        "code": pose.get("code") if pose else None,
        "label": pose.get("label") if pose else None,
        "source": pose.get("source") if pose else None,
        "age_ms": round(age_ms, 1) if age_ms is not None else None,
        "camera_available": bool(getattr(app_state, "camera_available", False)),
        "estop_active": bool(session.estop_latched),
    }


async def _receiver(websocket: WebSocket, queue: "asyncio.Queue[dict[str, Any]]") -> None:
    """只读客户端消息（心跳），不直接发送，避免与主循环并发写。"""
    while True:
        try:
            raw = await websocket.receive_text()
        except Exception:  # 断连/取消
            return
        payload: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = {}
        if payload.get("type") == "ping":
            _put(queue, {"type": "pong", "ts": time.time()})


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    app = websocket.app
    hub: WsHub = app.state.hub
    settings: Settings = app.state.settings
    metrics: Metrics = app.state.metrics
    session: AgentState = app.state.session

    await websocket.accept()
    cid, queue = await hub.register()
    interval = 1.0 / max(1, settings.performance.frame_push_hz)
    receiver = asyncio.create_task(_receiver(websocket, queue))
    try:
        await websocket.send_json(_hello_message(session, settings))
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                message = _frame_message(app.state, session, metrics, settings)
            await websocket.send_json(message)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        receiver.cancel()
        await hub.unregister(cid)
