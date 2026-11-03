"""摄像头采集与自动降级（Plan V1.0 T6 / §5.1、§12）。

设计原则（对应 Plan §12「无摄像头/断网环境」风险项）：
1. **绝不阻塞启动**：探测在守护线程中执行，主进程直接进入 `INJECT_ONLY`；
2. **绝不崩溃**：所有 OpenCV 调用都被包在 try/except 中，失败只降级不抛异常；
3. **自动降级**：探测失败或连续取帧失败达阈值 → 释放句柄并置为 `UNAVAILABLE`，
   上层据此保持 `INJECT_ONLY`（虚拟注入链路不受影响）。

P1 阶段只做"探测 + 单帧读取"能力；持续采集循环属于 P2（T10/T12）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Callable

from src.core.config import CameraConfig, get_settings

__all__ = [
    "CameraState",
    "CameraHandle",
    "CameraInfo",
    "CameraManager",
    "open_camera",
    "read_frame",
    "is_available",
    "release_camera",
    "cv2_available",
]


# --------------------------------------------------------------------------- #
# OpenCV 惰性导入（缺失时整体降级，不影响服务启动）
# --------------------------------------------------------------------------- #
def _load_cv2() -> Any | None:
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except Exception:  # pragma: no cover - 无 opencv 的环境
        return None


_CV2 = _load_cv2()


def cv2_available() -> bool:
    return _CV2 is not None


class CameraState(StrEnum):
    UNPROBED = "UNPROBED"
    PROBING = "PROBING"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class CameraHandle:
    """已打开的摄像头句柄。"""

    index: int
    width: int
    height: int
    fps: int
    capture: Any
    backend: str = "default"
    opened_at: float = field(default_factory=time.time)


@dataclass
class CameraInfo:
    """探测结果快照（只读）。"""

    state: CameraState = CameraState.UNPROBED
    available: bool = False
    index: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    elapsed_ms: float = 0.0
    error: str | None = None
    note: str = ""
    consecutive_failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "available": self.available,
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "note": self.note,
            "consecutive_failures": self.consecutive_failures,
        }


# --------------------------------------------------------------------------- #
# 底层原语
# --------------------------------------------------------------------------- #
def open_camera(
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    *,
    backend: str | None = None,
) -> CameraHandle | None:
    """打开摄像头并做一次握手取帧。

    失败返回 None（不抛异常）。Windows 下优先走 DirectShow，避免 MSMF 的
    2~3 秒启动延迟；失败再回退到默认后端。
    """
    if _CV2 is None:
        return None

    candidates: list[tuple[str, int]] = []
    if backend == "dshow":
        candidates.append(("dshow", _CV2.CAP_DSHOW))
    elif backend == "default":
        candidates.append(("default", _CV2.CAP_ANY))
    else:
        # 自动：先 DSHOW（Windows 快），再默认
        if hasattr(_CV2, "CAP_DSHOW"):
            candidates.append(("dshow", _CV2.CAP_DSHOW))
        candidates.append(("default", _CV2.CAP_ANY))

    for name, flag in candidates:
        capture = None
        try:
            capture = _CV2.VideoCapture(index, flag)
            if not capture.isOpened():
                raise RuntimeError("isOpened() == False")
            if width > 0:
                capture.set(_CV2.CAP_PROP_FRAME_WIDTH, float(width))
            if height > 0:
                capture.set(_CV2.CAP_PROP_FRAME_HEIGHT, float(height))
            if fps > 0:
                capture.set(_CV2.CAP_PROP_FPS, float(fps))
            capture.set(_CV2.CAP_PROP_BUFFERSIZE, 1)  # 降低缓冲，避免陈旧帧
            ok, _frame = capture.read()
            if not ok:
                raise RuntimeError("首帧读取失败")
            return CameraHandle(
                index=index,
                width=int(capture.get(_CV2.CAP_PROP_FRAME_WIDTH) or width),
                height=int(capture.get(_CV2.CAP_PROP_FRAME_HEIGHT) or height),
                fps=int(capture.get(_CV2.CAP_PROP_FPS) or fps),
                capture=capture,
                backend=name,
            )
        except Exception:
            if capture is not None:
                try:
                    capture.release()
                except Exception:  # pragma: no cover
                    pass
            continue
    return None


def read_frame(handle: CameraHandle | None) -> Any | None:
    """读取一帧 BGR 图像；失败返回 None（不抛异常）。"""
    if handle is None or handle.capture is None or _CV2 is None:
        return None
    try:
        ok, frame = handle.capture.read()
        if not ok or frame is None:
            return None
        return frame
    except Exception:  # pragma: no cover
        return None


def is_available(handle: CameraHandle | None) -> bool:
    """句柄是否仍然可用。"""
    if handle is None or handle.capture is None:
        return False
    try:
        return bool(handle.capture.isOpened())
    except Exception:  # pragma: no cover
        return False


def release_camera(handle: CameraHandle | None) -> None:
    if handle is None or handle.capture is None:
        return
    try:
        handle.capture.release()
    except Exception:  # pragma: no cover
        pass
    handle.capture = None


# --------------------------------------------------------------------------- #
# 管理器（供 FastAPI lifespan 使用）
# --------------------------------------------------------------------------- #
class CameraManager:
    """线程安全的摄像头状态机：UNPROBED → PROBING → READY / UNAVAILABLE。"""

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        max_consecutive_failures: int | None = None,
        probe_timeout_ms: int | None = None,
    ) -> None:
        settings = get_settings()
        self.config = config or settings.camera
        self.max_consecutive_failures = (
            max_consecutive_failures
            if max_consecutive_failures is not None
            else getattr(settings.camera, "max_consecutive_failures", 5)
        )
        self.probe_timeout_ms = (
            probe_timeout_ms
            if probe_timeout_ms is not None
            else getattr(settings.camera, "probe_timeout_ms", 2000)
        )
        self._lock = threading.Lock()
        self._handle: CameraHandle | None = None
        self._info = CameraInfo(
            index=self.config.index,
            width=self.config.width,
            height=self.config.height,
            note="尚未探测",
        )
        self._probe_thread: threading.Thread | None = None

    # -- 查询 -------------------------------------------------------------- #
    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._info.state

    @property
    def available(self) -> bool:
        with self._lock:
            return self._info.available

    @property
    def note(self) -> str:
        with self._lock:
            return self._info.note

    def info(self) -> CameraInfo:
        with self._lock:
            return CameraInfo(**{**self._info.__dict__})

    def snapshot(self) -> dict[str, Any]:
        return self.info().as_dict()

    # -- 探测 -------------------------------------------------------------- #
    def start_probe_async(
        self,
        on_result: Callable[[CameraInfo], None] | None = None,
    ) -> bool:
        """在守护线程中探测摄像头；立即返回，绝不阻塞调用方。"""
        with self._lock:
            if self._probe_thread is not None and self._probe_thread.is_alive():
                return False
            self._info.state = CameraState.PROBING
            self._info.note = "探测中…"
            thread = threading.Thread(
                target=self._probe_worker,
                args=(on_result,),
                name="camera-probe",
                daemon=True,
            )
            self._probe_thread = thread
        thread.start()
        return True

    def _probe_worker(self, on_result: Callable[[CameraInfo], None] | None) -> None:
        info = self.probe()
        if on_result is not None:
            try:
                on_result(info)
            except Exception:  # pragma: no cover - 回调异常不影响探测状态
                pass

    def probe(self) -> CameraInfo:
        """同步探测（仅供后台线程或离线脚本调用）。"""
        started = time.perf_counter()
        handle: CameraHandle | None = None
        try:
            handle = open_camera(
                self.config.index,
                self.config.width,
                self.config.height,
                self.config.target_fps,
            )
        except Exception as exc:  # pragma: no cover
            handle = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None if handle is not None else "未能打开摄像头设备（可能无摄像头或被占用）"

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if handle is None:
            with self._lock:
                self._info = CameraInfo(
                    state=CameraState.UNAVAILABLE,
                    available=False,
                    index=self.config.index,
                    width=self.config.width,
                    height=self.config.height,
                    elapsed_ms=elapsed_ms,
                    error=error,
                    note="未检测到可用摄像头，已自动降级为 INJECT_ONLY（虚拟手势注入仍可用）",
                )
                return CameraInfo(**self._info.__dict__)

        # 采样若干帧估算实际帧率，然后释放（P1 不做持续采集）
        measured_fps = 0.0
        try:
            frames = 0
            t0 = time.perf_counter()
            while frames < 5 and (time.perf_counter() - t0) < 0.8:
                if read_frame(handle) is not None:
                    frames += 1
            dt = time.perf_counter() - t0
            measured_fps = frames / dt if dt > 0 else 0.0
        except Exception:  # pragma: no cover
            measured_fps = 0.0
        finally:
            release_camera(handle)

        with self._lock:
            self._handle = None
            self._info = CameraInfo(
                state=CameraState.READY,
                available=True,
                index=self.config.index,
                width=handle.width,
                height=handle.height,
                fps=measured_fps,
                elapsed_ms=elapsed_ms,
                error=None,
                note=(
                    "已探测到摄像头；持续采集与视觉闭环将在 P2（T10/T12）接入，"
                    "当前控制链路仍走虚拟注入"
                ),
            )
            return CameraInfo(**self._info.__dict__)

    # -- 取帧（P2 使用） --------------------------------------------------- #
    def read(self) -> Any | None:
        """读取一帧；连续失败达阈值则自动降级。"""
        with self._lock:
            handle = self._handle
        frame = read_frame(handle)
        if frame is not None:
            with self._lock:
                self._info.consecutive_failures = 0
            return frame

        with self._lock:
            self._info.consecutive_failures += 1
            failures = self._info.consecutive_failures
            should_degrade = failures >= self.max_consecutive_failures
            if should_degrade:
                self._handle = None
                self._info.state = CameraState.UNAVAILABLE
                self._info.available = False
                self._info.error = f"连续 {failures} 次取帧失败"
                self._info.note = "取帧持续失败，已自动降级为 INJECT_ONLY"
                stale = handle
            else:
                stale = None
        if stale is not None:
            release_camera(stale)
        return None

    # -- 打开/关闭（P2 持续采集使用） ------------------------------------- #
    def open(self) -> bool:
        """打开并常驻持有句柄（供视觉循环）。失败自动降级，不抛异常。"""
        handle = open_camera(
            self.config.index,
            self.config.width,
            self.config.height,
            self.config.target_fps,
        )
        with self._lock:
            if handle is None:
                self._handle = None
                self._info.state = CameraState.UNAVAILABLE
                self._info.available = False
                self._info.error = "持续采集打开失败"
                self._info.note = "取帧设备不可用，已自动降级为 INJECT_ONLY"
                return False
            self._handle = handle
            self._info.state = CameraState.READY
            self._info.available = True
            self._info.width = handle.width
            self._info.height = handle.height
            self._info.error = None
            self._info.note = "持续采集中"
            return True

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None
        release_camera(handle)

    # -- 自适应降级（T13） ------------------------------------------------- #
    def set_resolution(self, width: int, height: int) -> dict[str, Any]:
        """调整采集分辨率（对已打开的句柄即时生效，下次打开使用新值）。"""
        with self._lock:
            previous = (self.config.width, self.config.height)
            self.config = replace(self.config, width=int(width), height=int(height))
            self._info.width = int(width)
            self._info.height = int(height)
            handle = self._handle
        if handle is not None and _CV2 is not None:
            try:
                handle.capture.set(_CV2.CAP_PROP_FRAME_WIDTH, float(width))
                handle.capture.set(_CV2.CAP_PROP_FRAME_HEIGHT, float(height))
                handle.width = int(width)
                handle.height = int(height)
            except Exception:  # pragma: no cover
                pass
        return {"from": list(previous), "to": [int(width), int(height)]}
