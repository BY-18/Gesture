"""统一控制总线（Plan V1.0 T3 / §5.3）。

职责边界：
- **只做**：意图 → 标准控制帧、去抖、急停优先、发送队列、DRY_RUN 结构化日志；
- **不做**：几何判定、防抖表决（属于 `src.agents.debounce`）、HTTP 推送。

关键安全语义（Plan §2.3 / §5.3）：
1. `CMD_STOP` 永不去抖、永不丢弃、永不被冷却期拦截，且触发后立即清空队列中的
   所有运动指令；
2. 急停后 `estop_cooldown_ms` 内，一切运动指令（前进/左右转/转圈）均被拒绝；
3. 发送队列容量 32，满时**优先丢弃最旧的非急停帧**；
4. `DRY_RUN=true` 时仅落 JSONL 与控制台，不外发物理信号。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.api.schemas import (
    Cmd,
    ConfirmedIntent,
    ControlFrame,
    EMERGENCY_CMDS,
    MOTION_CMDS,
    cmd_default_speed,
)
from src.core.logger import SessionLogger, try_get_logger
from src.core.metrics import Metrics, get_metrics

__all__ = ["ControlBus", "BusStats"]


@dataclass
class BusStats:
    dispatched: int = 0  # 真正入队下发的帧数
    suppressed: int = 0  # 被抑制（去抖 / 冷却期）的帧数
    deduped: int = 0  # 其中因 200ms 去抖被抑制
    blocked: int = 0  # 其中因急停冷却期被拦截
    dropped: int = 0  # 队列满被丢弃的帧数
    estop: int = 0  # 急停触发次数

    def as_dict(self) -> dict[str, int]:
        return {
            "dispatched": self.dispatched,
            "suppressed": self.suppressed,
            "deduped": self.deduped,
            "blocked": self.blocked,
            "dropped": self.dropped,
            "estop": self.estop,
        }


class ControlBus:
    """线程安全的单例控制总线。"""

    def __init__(
        self,
        *,
        dry_run: bool = True,
        queue_size: int = 32,
        dedupe_ms: int = 200,
        estop_cooldown_ms: int = 500,
        logger: SessionLogger | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        if queue_size < 2:
            raise ValueError("queue_size 至少为 2")
        self.dry_run = dry_run
        self.queue_size = queue_size
        self.dedupe_ms = dedupe_ms
        self.estop_cooldown_ms = estop_cooldown_ms
        self._logger = logger or try_get_logger()
        self._metrics = metrics or get_metrics()

        self._lock = threading.RLock()
        self._queue: deque[ControlFrame] = deque()
        self._seq = 0
        self._last_emit_ts: dict[Cmd, float] = {}
        self._estop_until: float = 0.0
        self._stats = BusStats()
        self._dry_run_frames = 0

    # -- 查询 -------------------------------------------------------------- #
    @property
    def estop_active(self) -> bool:
        """急停锁存是否仍在冷却期内。"""
        return time.time() < self._estop_until

    @property
    def estop_remaining_ms(self) -> float:
        remaining = self._estop_until - time.time()
        return round(max(0.0, remaining) * 1000.0, 1)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            data = self._stats.as_dict()
            data.update(
                {
                    "pending": len(self._queue),
                    "queue_size": self.queue_size,
                    "estop_active": self.estop_active,
                    "estop_remaining_ms": self.estop_remaining_ms,
                    "dry_run": self.dry_run,
                    "seq": self._seq,
                }
            )
            return data

    # -- 核心 -------------------------------------------------------------- #
    def dispatch(self, intent: ConfirmedIntent) -> ControlFrame:
        """把已确认意图转成控制帧并下发（返回帧，抑制时 `suppressed=True`）。"""
        start = time.perf_counter()
        now = time.time()
        cmd = intent.cmd
        suppressed = False
        reason: str | None = None

        with self._lock:
            # ① 急停冷却期：拒绝一切运动指令
            if cmd in MOTION_CMDS and now < self._estop_until:
                suppressed = True
                reason = "estop_cooldown"
                self._stats.blocked += 1
            # ② 200ms 同指令去抖（急停除外）
            elif cmd not in EMERGENCY_CMDS:
                last = self._last_emit_ts.get(cmd)
                if last is not None and (now - last) * 1000.0 < self.dedupe_ms:
                    suppressed = True
                    reason = "dedupe"
                    self._stats.deduped += 1

            # ③ 急停：置锁 + 清空队列中的运动指令（无论是否重复触发）
            if cmd in EMERGENCY_CMDS:
                self._estop_until = now + self.estop_cooldown_ms / 1000.0
                self._drop_motion_frames()
                self._stats.estop += 1

            self._seq += 1
            frame = ControlFrame(
                cmd=cmd,
                speed=(
                    intent.speed if intent.speed is not None else cmd_default_speed(cmd)
                ),
                timestamp=now,
                seq=self._seq,
                source=intent.source,
                trace_id=intent.trace_id,
                dry_run=self.dry_run,
                suppressed=suppressed,
            )

            if suppressed:
                self._stats.suppressed += 1
            else:
                self._last_emit_ts[cmd] = now
                # DRY_RUN 没有物理消费者，入队只会把队列塞满并产生假"丢帧"；
                # 队列语义（容量 32 + 急停永不丢弃）在 dry_run=False 时完整生效。
                if self.dry_run:
                    self._dry_run_frames += 1
                else:
                    self._enqueue(frame)
                self._stats.dispatched += 1

            queue_after = len(self._queue)

        duration_ms = (time.perf_counter() - start) * 1000.0
        self._metrics.observe("dispatch_ms", duration_ms)
        self._metrics.incr(
            "bus_suppressed" if suppressed else "bus_dispatched"
        )
        if cmd in EMERGENCY_CMDS:
            self._metrics.incr("bus_estop")

        if self._logger is not None:
            self._logger.emit(
                "command" if not suppressed else "command_suppressed",
                trace_id=intent.trace_id,
                duration_ms=duration_ms,
                cmd=cmd.value,
                code=intent.code.value if intent.code is not None else None,
                speed=frame.speed,
                frame_seq=frame.seq,
                source=frame.source.value,
                dry_run=self.dry_run,
                suppressed=suppressed,
                reason=reason,
                confidence=round(intent.confidence, 3),
                queue=queue_after,
            )
        return frame

    # -- 队列 -------------------------------------------------------------- #
    def _enqueue(self, frame: ControlFrame) -> None:
        if len(self._queue) >= self.queue_size:
            self._evict_one()
        self._queue.append(frame)

    def _evict_one(self) -> None:
        """队列满：优先丢最旧的非急停帧。"""
        for index, existing in enumerate(self._queue):
            if existing.cmd not in EMERGENCY_CMDS:
                del self._queue[index]
                self._stats.dropped += 1
                return
        self._queue.popleft()
        self._stats.dropped += 1

    def _drop_motion_frames(self) -> None:
        kept = deque(f for f in self._queue if f.cmd in EMERGENCY_CMDS)
        self._stats.dropped += len(self._queue) - len(kept)
        self._queue = kept

    def flush(self) -> list[ControlFrame]:
        """取走并清空待发队列（DRY_RUN 下由调用方消费/丢弃）。"""
        with self._lock:
            frames = list(self._queue)
            self._queue.clear()
            return frames

    def clear_estop(self) -> None:
        """人工解除急停锁存（运营层操作，P2 由 HITL/看门狗接管）。"""
        with self._lock:
            self._estop_until = 0.0
        if self._logger is not None:
            self._logger.emit("estop_cleared", source="operator")

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._last_emit_ts.clear()
            self._estop_until = 0.0
            self._stats = BusStats()
            self._seq = 0

    def reset_stats(self) -> None:
        """运维重置：清零统计/队列/去抖/急停锁，但**保留 seq 单调性**（协议契约不受重置影响）。"""
        with self._lock:
            self._queue.clear()
            self._last_emit_ts.clear()
            self._estop_until = 0.0
            self._stats = BusStats()


def build_bus_from_settings(
    settings: Any,
    *,
    logger: SessionLogger | None = None,
    metrics: Metrics | None = None,
) -> ControlBus:
    """按配置中心构造总线（供 main.py 使用）。"""
    return ControlBus(
        dry_run=settings.mode.dry_run,
        queue_size=settings.safety.bus_queue_size,
        dedupe_ms=settings.safety.dedupe_ms,
        estop_cooldown_ms=settings.safety.estop_cooldown_ms,
        logger=logger,
        metrics=metrics,
    )
