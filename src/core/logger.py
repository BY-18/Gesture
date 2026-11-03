"""结构化会话日志（Plan V1.0 T1 / §6.3）。

设计约束：
1. 落盘格式为 **JSONL**，文件路径 `logs/session_{yyyyMMdd_HHmmss}.jsonl`；
2. 每条事件必带 `seq`（进程内严格单调递增）、`ts`、`event`；
3. 支持耗时打点 `dur_ms`，以便验收脚本（T16/T17）离线复算时延；
4. `trace_id` 通过 ContextVar 传递，跨函数无需显式传参；
5. 写入线程安全（视觉线程 / 事件循环 / 注入处理器可能并发写）。
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "SessionLogger",
    "get_logger",
    "init_session_logger",
    "new_trace_id",
    "current_trace_id",
    "trace_context",
]

_trace_var: ContextVar[str | None] = ContextVar("gesture_trace_id", default=None)

_LOG_SEQ_LOCK = threading.Lock()
_LOG_SEQ = 0

#: 保留字段：调用方不得通过 kwargs 覆盖，否则日志序会失真（例如总线把帧序号塞进来）
RESERVED_FIELDS = frozenset({"ts", "ts_epoch", "seq", "event", "trace_id", "dur_ms"})


def _next_seq() -> int:
    global _LOG_SEQ
    with _LOG_SEQ_LOCK:
        _LOG_SEQ += 1
        return _LOG_SEQ


def new_trace_id() -> str:
    """生成新的链路追踪 ID。"""
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _trace_var.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """在上下文内绑定 trace_id（缺省则新建）。"""
    tid = trace_id or new_trace_id()
    token: Token = _trace_var.set(tid)
    try:
        yield tid
    finally:
        _trace_var.reset(token)


class SessionLogger:
    """单会话 JSONL 日志器。进程内应当只创建一次。"""

    def __init__(
        self,
        log_dir: Path,
        *,
        level: str = "INFO",
        console: bool = True,
        session_id: str | None = None,
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path: Path = log_dir / f"session_{stamp}.jsonl"
        self.session_id: str = session_id or new_trace_id()
        self.level: str = level.upper()
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._write_lock = threading.Lock()
        self._console = self._build_console(console)
        self._closed = False

    # -- 控制台 ------------------------------------------------------------ #
    @staticmethod
    def _build_console(enabled: bool) -> logging.Logger:
        logger = logging.getLogger("gesture.console")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.handlers:  # 重复 init 时复用，避免重复输出
            logger.handlers.clear()
        if enabled:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())
        return logger

    # -- 写事件 ------------------------------------------------------------ #
    def emit(
        self,
        event: str,
        *,
        trace_id: str | None = None,
        duration_ms: float | None = None,
        level: str = "INFO",
        **fields: Any,
    ) -> dict[str, Any]:
        """写入一条结构化事件，并同步输出到控制台。"""
        if self._closed:
            return {}
        now = time.time()
        record: dict[str, Any] = {
            "ts": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "ts_epoch": round(now, 6),
            "seq": _next_seq(),
            "event": event,
        }
        tid = trace_id or current_trace_id()
        if tid:
            record["trace_id"] = tid
        if duration_ms is not None:
            record["dur_ms"] = round(float(duration_ms), 3)
        for key, value in fields.items():
            if key in RESERVED_FIELDS:
                # 非法覆盖：改写为 <key>_ 前缀保留可观测性，但不破坏日志序
                if value is not None:
                    record[f"{key}_payload"] = value
                continue
            if value is not None:
                record[key] = value

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._write_lock:
            self._fh.write(line + "\n")
            self._fh.flush()

        self._emit_console(record, level)
        return record

    def _emit_console(self, record: dict[str, Any], level: Any) -> None:
        # level 允许被调用方传入任意值（历史上有人传过 int 型降级级别），
        # 这里统一转为字符串，避免 `level.upper()` 抛 AttributeError 打断业务逻辑。
        numeric_level = getattr(logging, str(level).upper(), logging.INFO)
        if numeric_level < getattr(logging, self.level, logging.INFO):
            return
        parts = [str(record["event"])]
        if "dur_ms" in record:
            parts.append(f"dur={record['dur_ms']}ms")
        for key, value in record.items():
            if key in {"ts", "ts_epoch", "seq", "event", "dur_ms"}:
                continue
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, default=str)
                if len(rendered) > 160:
                    rendered = rendered[:157] + "..."
                parts.append(f"{key}={rendered}")
            else:
                parts.append(f"{key}={value}")
        parts.append(f"seq={record['seq']}")
        self._console.log(numeric_level, " ".join(parts))

    # -- 工具 -------------------------------------------------------------- #
    @staticmethod
    def now_ms() -> float:
        """单调时钟（毫秒），用于耗时打点。"""
        return time.perf_counter() * 1000.0

    @staticmethod
    def elapsed_ms(start_ms: float) -> float:
        return time.perf_counter() * 1000.0 - start_ms

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._write_lock:
            self._fh.flush()
            self._fh.close()


# --------------------------------------------------------------------------- #
# 进程内单例
# --------------------------------------------------------------------------- #
_logger: SessionLogger | None = None


def init_session_logger(
    log_dir: Path,
    *,
    level: str = "INFO",
    console: bool = True,
    session_id: str | None = None,
) -> SessionLogger:
    global _logger
    if _logger is not None:
        _logger.close()
    _logger = SessionLogger(log_dir, level=level, console=console, session_id=session_id)
    return _logger


def get_logger() -> SessionLogger:
    if _logger is None:
        raise RuntimeError("日志器尚未初始化：请先调用 init_session_logger()")
    return _logger


def try_get_logger() -> SessionLogger | None:
    return _logger
