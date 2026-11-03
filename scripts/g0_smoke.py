"""门禁 G0 冒烟测试（Plan V1.0 §9 / §10）。

覆盖 P0 阶段的闭环验收要点：
  1. 服务可达性（/api/health）
  2. WebSocket 握手与 hello 契约
  3. 注入 → 控制总线出帧（/api/inject 返回 ControlFrame）
  4. WS 下发 command 事件 + frame 帧（HUD 高亮所需数据）
  5. 急停优先权：CMD_STOP 后 500ms 内运动指令被拦截
  6. 200ms 同指令去抖
  7. /api/state 与 /api/metrics 契约
  8. JSONL 落盘：session_start / command / inject 事件齐全，seq 单调递增

用法：先启动服务（python main.py），再执行
    .venv\\Scripts\\python.exe scripts\\g0_smoke.py
退出码 0 表示门禁通过。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/stream"
REPO_ROOT = Path(__file__).resolve().parents[1]

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    if ok:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
    return ok


def api_get(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def drain(ws, seconds: float = 0.8) -> list[dict[str, Any]]:
    """在指定时间内收集所有 WS 消息。"""
    messages: list[dict[str, Any]] = []
    deadline = time.time() + seconds
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            break
        try:
            raw = ws.recv(timeout=min(remain, 0.4))
        except TimeoutError:
            continue
        except Exception:  # 连接关闭
            break
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main() -> int:
    print("=" * 74)
    print("G0 门禁冒烟测试：注入 → 总线 → JSONL → WebSocket → HUD")
    print("=" * 74)

    # ---- 1. 健康检查 -------------------------------------------------- #
    print("\n[1] 服务可达性 /api/health")
    try:
        health = api_get("/api/health")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [FAIL] 无法连接服务：{exc}")
        print("  提示：请先执行  .venv\\Scripts\\python.exe main.py")
        return 2
    check("health.status=ok", health.get("status") == "ok", str(health.get("status")))
    check("health.mode=INJECT_ONLY", health.get("mode") == "INJECT_ONLY", str(health.get("mode")))
    check("health.dry_run=true", health.get("dry_run") is True)
    log_file = Path(health.get("log_file", ""))
    check("日志文件已创建", log_file.exists(), str(log_file))
    log_lines_before = len(read_jsonl(log_file))

    # ---- 2. WebSocket 握手 -------------------------------------------- #
    print("\n[2] WebSocket /ws/stream 握手")
    with connect(WS_URL, open_timeout=10) as ws:
        hello = None
        deadline = time.time() + 6
        while time.time() < deadline and hello is None:
            try:
                message = json.loads(ws.recv(timeout=2))
            except TimeoutError:
                continue
            if message.get("type") == "hello":
                hello = message
        check("收到 hello", hello is not None)
        if hello:
            check("hello.mode 存在", bool(hello.get("mode")))
            check("hello.dry_run 存在", "dry_run" in hello)
            check(
                "hello 携带手势词表",
                isinstance(hello.get("gesture_codes"), list)
                and len(hello["gesture_codes"]) == 5,
                str(hello.get("gesture_codes")),
            )

        frame_messages = [m for m in drain(ws, 1.5) if m.get("type") == "frame"]
        check("周期帧推送生效", len(frame_messages) > 0, f"{len(frame_messages)} 帧/1.5s")

        # ---- 3. 注入 → 总线出帧 --------------------------------------- #
        print("\n[3] 虚拟手势注入 /api/inject（G-10 前进）")
        result = api_post("/api/inject", {"code": "G-10", "count": 1})
        frames = result.get("frames") or []
        check("注入被接受", result.get("accepted") is True)
        check("产生 1 个控制帧", len(frames) == 1, f"{len(frames)} 帧")
        if frames:
            frame = frames[0]
            check("cmd=CMD_FORWARD", frame.get("cmd") == "CMD_FORWARD", str(frame.get("cmd")))
            check("source=INJECT", frame.get("source") == "INJECT", str(frame.get("source")))
            check("dry_run=true", frame.get("dry_run") is True)
            check("trace_id 非空", bool(frame.get("trace_id")))
            check("speed=1.0", frame.get("speed") == 1.0, str(frame.get("speed")))
            trace_id = frame.get("trace_id")
        check("emitted=1", result.get("emitted") == 1)

        # ---- 4. WS 下发 command + frame -------------------------------- #
        print("\n[4] WebSocket 下行 command / frame")
        messages = drain(ws, 1.5)
        commands = [m for m in messages if m.get("type") == "command"]
        check("WS 收到 command 事件", len(commands) > 0, f"{len(commands)} 条")
        matched = [
            c for c in commands
            if (c.get("frame") or {}).get("trace_id") == trace_id
        ]
        check("command 事件携带注入的 trace_id", len(matched) > 0, str(trace_id))
        frames_ws = [m for m in messages if m.get("type") == "frame"]
        with_pose = [m for m in frames_ws if m.get("landmarks")]
        check("frame 携带 21 点骨架", len(with_pose) > 0)
        if with_pose:
            check("骨架点数为 21", len(with_pose[-1]["landmarks"]) == 21)

        # ---- 5. 急停优先权 -------------------------------------------- #
        print("\n[5] 急停优先权（Plan §2.3 / §5.3）")
        estop = api_post("/api/inject", {"code": "G-02", "count": 1})
        estop_frames = estop.get("frames") or []
        check(
            "急停产生 CMD_STOP",
            bool(estop_frames) and estop_frames[0].get("cmd") == "CMD_STOP",
            str(estop_frames[0].get("cmd") if estop_frames else None),
        )
        check(
            "急停 speed=0.0",
            bool(estop_frames) and estop_frames[0].get("speed") == 0.0,
        )
        blocked = api_post("/api/inject", {"code": "G-10", "count": 1})
        blocked_frames = blocked.get("frames") or []
        check(
            "急停冷却期内前进被拦截",
            bool(blocked_frames) and blocked_frames[0].get("suppressed") is True,
            f"suppressed={blocked_frames[0].get('suppressed') if blocked_frames else None}",
        )
        check("blocked 计数=1", blocked.get("blocked") == 1, str(blocked.get("blocked")))
        state = api_get("/api/state")
        check("state.estop_latched=true", state.get("estop_latched") is True)

        # 等待冷却期结束
        time.sleep(0.65)

        # ---- 6. 去抖 -------------------------------------------------- #
        print("\n[6] 200ms 同指令去抖（Plan §5.3）")
        first = api_post("/api/inject", {"code": "G-11", "count": 1})
        second = api_post("/api/inject", {"code": "G-11", "count": 1})
        first_frame = (first.get("frames") or [{}])[0]
        second_frame = (second.get("frames") or [{}])[0]
        check("首次左转下发", first_frame.get("suppressed") is False)
        check(
            "200ms 内重复左转被去抖",
            second_frame.get("suppressed") is True,
            f"suppressed={second_frame.get('suppressed')}",
        )
        check("dedupe 窗口=200ms", second.get("dedupe_window_ms") == 200)

        # ---- 7. 状态与指标 -------------------------------------------- #
        print("\n[7] /api/state 与 /api/metrics 契约")
        state = api_get("/api/state")
        check("state.last_cmd 已更新", state.get("last_cmd") == "CMD_TURN_LEFT", str(state.get("last_cmd")))
        check("state.window_capacity=30", state.get("window_capacity") == 30)
        metrics = api_get("/api/metrics")
        dispatch = (metrics.get("histograms") or {}).get("dispatch_ms") or {}
        check("dispatch_ms 直方图存在", bool(dispatch))
        check("dispatch_ms 含 P95", "p95" in dispatch, str(dispatch.get("p95")))
        check("inject_total_ms 直方图存在", "inject_total_ms" in (metrics.get("histograms") or {}))
        bus = metrics.get("bus") or {}
        check("bus.dispatched > 0", int(bus.get("dispatched", 0)) > 0, str(bus.get("dispatched")))
        check("bus.estop > 0", int(bus.get("estop", 0)) > 0, str(bus.get("estop")))
        check("queue_size=32", bus.get("queue_size") == 32, str(bus.get("queue_size")))

        # ---- 8. HUD 静态页 -------------------------------------------- #
        print("\n[8] HUD 单页可达")
        with urllib.request.urlopen(f"{BASE_URL}/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        check("HUD 返回 200", resp.status == 200)
        check("HUD 含画布", 'id="scene"' in html)
        check("HUD 含注入按钮组", 'data-code="G-02"' in html and 'data-code="G-13"' in html)
        check("HUD 含五态仪表盘", 'data-cmd="CMD_SPIN"' in html)
        check("HUD 零外链（内网可用）", "http://" not in html and "https://" not in html)

    # ---- 9. JSONL 落盘 ------------------------------------------------ #
    print("\n[9] JSONL 会话日志")
    records = read_jsonl(log_file)
    check("日志可解析", len(records) > 0, f"{len(records)} 条")
    events = [r.get("event") for r in records]
    check("包含 session_start", "session_start" in events)
    check("包含 command 事件", "command" in events)
    check("包含 inject 事件", "inject" in events)
    check("包含 command_suppressed 事件", "command_suppressed" in events)

    seqs = [r.get("seq") for r in records if isinstance(r.get("seq"), int)]
    check("seq 严格单调递增", all(b > a for a, b in zip(seqs, seqs[1:])), f"{len(seqs)} 条")
    check("全部事件带 ts", all("ts" in r for r in records))
    trace_ids = {r.get("trace_id") for r in records if r.get("trace_id")}
    check("trace_id 覆盖注入链路", len(trace_ids) >= 3, f"{len(trace_ids)} 个 trace")
    dur_ms = [r for r in records if "dur_ms" in r]
    check("含耗时打点 dur_ms", len(dur_ms) > 0, f"{len(dur_ms)} 条")

    new_records = records[log_lines_before:]
    check("本次运行产生了新日志", len(new_records) > 0, f"{len(new_records)} 条")

    # ---- 汇总 --------------------------------------------------------- #
    print("\n" + "=" * 74)
    total = len(PASSED) + len(FAILED)
    print(f"结果：{len(PASSED)}/{total} 通过")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        print("G0 门禁：FAIL")
        return 1
    print("G0 门禁：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
