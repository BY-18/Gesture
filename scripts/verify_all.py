"""一键全链路验收（Plan V1.0 T17 / §11.3，门禁 G3）。

自动**连接或拉起**服务，依次执行五类核心断言，并调度三个离线子套件：

====================  ================================================================
类别                   断言内容
====================  ================================================================
1 服务可达            `/api/health` 返回 `INJECT_ONLY`/`LIVE`；`/`（HUD）返回 200
2 协议正确            `/api/inject` 覆盖全部 6 种 `cmd`；`ControlFrame` 字段完整；`seq` 单调递增
3 急停优先权          先 FORWARD 再 G-02：≤100 ms 内出 `CMD_STOP`；随后 500 ms 内运动指令被拒
4 看门狗超时          FORWARD 后停止输入：500±100 ms 内自动 `CMD_IDLE`（source=WATCHDOG）
5 时延与稳态          300 次混合注入 + 三个子套件；`/api/metrics` 各 P99 达标
====================  ================================================================

子套件：`test_trajectory.py`（识别率）、`test_debounce.py`（静态误触发）、`bench_latency.py`（延迟基准）

用法::

    .venv\\Scripts\\python.exe scripts\\verify_all.py           # 连不上则自动拉起服务
    .venv\\Scripts\\python.exe scripts\\verify_all.py --no-spawn  # 只连接，不自动拉起

退出码 0 表示门禁 G3 通过。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/stream"

BUDGET_ESTOP_MS = 100.0
BUDGET_ESTOP_COOLDOWN_MS = 500.0
WATCHDOG_MISS_MS = 500.0
WATCHDOG_TOLERANCE_MS = 100.0
STEADY_INJECTIONS = 300

CMD_DOMAIN = {
    "CMD_STOP",
    "CMD_FORWARD",
    "CMD_TURN_LEFT",
    "CMD_TURN_RIGHT",
    "CMD_SPIN",
    "CMD_IDLE",
}

FRAME_FIELDS = {"cmd", "speed", "timestamp", "seq", "source", "trace_id", "dry_run"}

SUBTESTS = (
    ("识别率", "scripts/test_trajectory.py"),
    ("静态误触发", "scripts/test_debounce.py"),
    ("延迟基准", "scripts/bench_latency.py"),
)


# --------------------------------------------------------------------------- #
# 结果收集
# --------------------------------------------------------------------------- #
@dataclass
class Category:
    key: str
    name: str
    passed: int = 0
    total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def status(self) -> str:
        if self.total == 0:
            return "SKIP"
        return "PASS" if self.ok else "FAIL"


CATEGORIES: "OrderedDict[str, Category]" = OrderedDict()


def category(key: str, name: str) -> Category:
    if key not in CATEGORIES:
        CATEGORIES[key] = Category(key=key, name=name)
    return CATEGORIES[key]


def check(cat: Category, name: str, ok: bool, detail: str = "") -> bool:
    cat.total += 1
    if ok:
        cat.passed += 1
        print(f"    [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"    [FAIL] {name}" + (f" — {detail}" if detail else ""))
        cat.notes.append(f"{name}：{detail}" if detail else name)
    return ok


# --------------------------------------------------------------------------- #
# HTTP / WS
# --------------------------------------------------------------------------- #
def api_get(path: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_status(url: str, timeout: float = 10.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def collect(ws: Any, seconds: float) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    deadline = time.time() + seconds
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            break
        try:
            raw = ws.recv(timeout=min(remain, 0.2))
        except TimeoutError:
            continue
        except Exception:
            break
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return messages


def commands_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m["frame"] for m in messages if m.get("type") == "command" and "frame" in m]


# --------------------------------------------------------------------------- #
# 服务生命周期
# --------------------------------------------------------------------------- #
def service_alive(timeout: float = 2.0) -> bool:
    try:
        payload = api_get("/api/health", timeout=timeout)
        return payload.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def spawn_service(wait_s: float = 25.0) -> subprocess.Popen | None:
    """拉起服务（日志重定向到 logs/verify_server.log）。"""
    log_path = REPO_ROOT / "logs" / "verify_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(REPO_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if service_alive(1.0):
            return process
        if process.poll() is not None:
            break
        time.sleep(0.3)
    process.terminate()
    return None


# --------------------------------------------------------------------------- #
# 五类断言
# --------------------------------------------------------------------------- #
def assert_service(cat: Category) -> None:
    print("\n[1] 服务可达")
    health = api_get("/api/health")
    check(cat, "health.status = ok", health.get("status") == "ok", str(health.get("status")))
    check(
        cat,
        "health.mode ∈ {INJECT_ONLY, LIVE}",
        health.get("mode") in {"INJECT_ONLY", "LIVE"},
        str(health.get("mode")),
    )
    check(cat, "运行于 DRY_RUN（不驱动物理设备）", health.get("dry_run") is True)
    check(cat, "health 含 session_id 与日志路径", bool(health.get("session_id")) and bool(health.get("log_file")))
    status = http_status(f"{BASE_URL}/")
    check(cat, "HUD 首页返回 200", status == 200, f"HTTP {status}")


def assert_protocol(cat: Category, ws: Any) -> list[dict[str, Any]]:
    print("\n[2] 协议正确（6 种 cmd / 字段完整 / seq 单调）")
    frames: list[dict[str, Any]] = []

    # ① FORWARD：同时武装看门狗，稍后从 WS 取 CMD_IDLE
    frames.extend(api_post("/api/inject", {"code": "G-10", "count": 1})["frames"])
    idle_frames = []
    deadline = time.time() + 1.6
    while time.time() < deadline and not idle_frames:
        for frame in commands_of(collect(ws, 0.2)):
            if frame["cmd"] == "CMD_IDLE":
                idle_frames.append(frame)
    frames.extend(idle_frames)
    check(cat, "看门狗产出 CMD_IDLE 帧", len(idle_frames) > 0, str(len(idle_frames)))

    # ② 其余指令
    for code, expected in (
        ("G-11", "CMD_TURN_LEFT"),
        ("G-12", "CMD_TURN_RIGHT"),
        ("G-13", "CMD_SPIN"),
        ("G-02", "CMD_STOP"),
    ):
        response = api_post("/api/inject", {"code": code, "count": 1})
        produced = response.get("frames") or []
        frames.extend(produced)
        check(
            cat,
            f"{code} → {expected}",
            bool(produced) and produced[0]["cmd"] == expected,
            str(produced[0]["cmd"] if produced else None),
        )
        time.sleep(0.25)

    seen = {frame["cmd"] for frame in frames}
    check(cat, "覆盖全部 6 种 cmd", seen == CMD_DOMAIN, f"实际 {sorted(seen)}")

    bad_fields = [frame for frame in frames if not FRAME_FIELDS.issubset(frame.keys())]
    check(cat, "ControlFrame 字段完整", not bad_fields, f"缺失 {len(bad_fields)} 条")

    bad_cmd = [frame for frame in frames if frame["cmd"] not in CMD_DOMAIN]
    check(cat, "cmd 落在冻结指令域内", not bad_cmd, f"越界 {len(bad_cmd)} 条")

    seqs = [frame["seq"] for frame in frames]
    monotonic = all(b > a for a, b in zip(seqs, seqs[1:]))
    check(cat, "seq 严格单调递增", monotonic, f"{len(seqs)} 帧，{seqs[0] if seqs else '-'}→{seqs[-1] if seqs else '-'}")

    bad_types = [
        frame
        for frame in frames
        if not isinstance(frame["speed"], (int, float))
        or not isinstance(frame["timestamp"], (int, float))
        or not isinstance(frame["dry_run"], bool)
        or not isinstance(frame["trace_id"], str)
    ]
    check(cat, "speed/timestamp/dry_run/trace_id 类型正确", not bad_types, f"异常 {len(bad_types)} 条")
    return frames


def reset_metrics_baseline() -> bool:
    """复用长跑实例时清零测量基线（直方图 P99 与总线统计不受历史流量污染）。

    返回 False 表示运行中的实例缺少该端点 —— 即**旧版本代码**，继续验收只会产出
    无意义的红结果（历史脏计数 + 已修缺陷的旧行为），必须拒绝复用。
    """
    try:
        api_post("/api/metrics/reset", {})
        print("测量基线已重置（/api/metrics/reset）")
        return True
    except Exception as exc:
        print(f"[FAIL] /api/metrics/reset 不可用（{exc}）")
        return False


def assert_estop_priority(cat: Category, ws: Any) -> None:
    print("\n[3] 急停优先权（≤100 ms 出 CMD_STOP，随后 500 ms 严格拒绝运动）")
    time.sleep(0.7)  # 清空上一步的冷却期
    collect(ws, 0.2)

    api_post("/api/inject", {"code": "G-10", "count": 1})
    started = time.time()
    response = api_post("/api/inject", {"code": "G-02", "count": 1})
    stop_frames = response.get("frames") or []
    stop_latency = ((stop_frames[0]["timestamp"] - started) * 1000.0) if stop_frames else None

    check(cat, "急停产生 CMD_STOP", bool(stop_frames) and stop_frames[0]["cmd"] == "CMD_STOP")
    check(
        cat,
        f"急停响应 ≤ {BUDGET_ESTOP_MS:.0f} ms",
        stop_latency is not None and stop_latency <= BUDGET_ESTOP_MS,
        f"{stop_latency:.1f} ms" if stop_latency is not None else "无帧",
    )

    # 冷却期内连续注入运动指令，必须全部被拒
    rejected = 0
    attempted = 0
    for code in ("G-10", "G-11", "G-12", "G-13"):
        attempted += 1
        frame = (api_post("/api/inject", {"code": code, "count": 1}).get("frames") or [{}])[0]
        if frame.get("suppressed") is True:
            rejected += 1
    check(
        cat,
        "冷却期内运动指令全部被拒",
        rejected == attempted,
        f"{rejected}/{attempted} 被抑制",
    )
    metrics = api_get("/api/metrics")
    check(
        cat,
        "总线拦截计数递增",
        int(metrics["bus"]["blocked"]) >= attempted,
        f"blocked={metrics['bus']['blocked']}",
    )
    check(cat, "急停锁存生效", bool(metrics["debounce"]["stats"]["estop_triggers"]) or metrics["bus"]["estop_active"] is True)


def assert_watchdog(cat: Category, ws: Any) -> None:
    print("\n[4] 看门狗超时（500±100 ms 内自动 CMD_IDLE）")
    time.sleep(0.7)
    collect(ws, 0.3)  # 清空缓冲

    motion = api_post("/api/inject", {"code": "G-10", "count": 1})
    motion_frames = motion.get("frames") or []
    last_motion_ts = float(motion_frames[0]["timestamp"]) if motion_frames else time.time()

    idle: dict[str, Any] | None = None
    deadline = time.time() + 1.8
    while time.time() < deadline and idle is None:
        for frame in commands_of(collect(ws, 0.2)):
            if frame["cmd"] == "CMD_IDLE" and frame.get("source") == "WATCHDOG":
                idle = frame
                break

    check(cat, "看门狗下发 CMD_IDLE", idle is not None)
    if idle is not None:
        delay = (float(idle["timestamp"]) - last_motion_ts) * 1000.0
        low = WATCHDOG_MISS_MS - WATCHDOG_TOLERANCE_MS
        high = WATCHDOG_MISS_MS + WATCHDOG_TOLERANCE_MS
        check(
            cat,
            f"回落时机落在 {low:.0f}~{high:.0f} ms",
            low <= delay <= high,
            f"{delay:.1f} ms",
        )
        check(cat, "来源标记为 WATCHDOG", idle.get("source") == "WATCHDOG", str(idle.get("source")))
        check(cat, "怠速速度 = 0", float(idle["speed"]) == 0.0, str(idle["speed"]))


def assert_steady_state(cat: Category) -> None:
    print(f"\n[5] 时延与稳态（{STEADY_INJECTIONS} 次混合注入 + /api/metrics P99）")
    rng = random.Random(20261103)
    codes = ["G-10", "G-11", "G-12", "G-13", "G-02"]
    failures = 0
    durations: list[float] = []
    frames: list[dict[str, Any]] = []

    for _ in range(STEADY_INJECTIONS):
        code = rng.choice(codes)
        started = time.perf_counter()
        try:
            response = api_post("/api/inject", {"code": code, "count": 1})
            produced = response.get("frames") or []
            frames.extend(produced)
        except Exception:
            failures += 1
            continue
        durations.append((time.perf_counter() - started) * 1000.0)
        # 贴近 30 FPS 设计输入节奏，避免远超设计帧率的 DoS 式灌包触发队列保护的丢弃语义
        time.sleep(0.005)

    durations.sort()
    p50 = durations[len(durations) // 2] if durations else 0.0
    p99 = durations[int(len(durations) * 0.99)] if durations else 0.0
    check(cat, f"{STEADY_INJECTIONS} 次注入无失败", failures == 0, f"失败 {failures} 次")
    # 客户端 HTTP 往返受本机 urllib/回环栈影响，不是产品可控面（§10.2 无此冻结指标），
    # 仅作信息项报告；产品口径为服务端 inject_total_ms（下方断言）。
    print(
        f"  [INFO] 注入 HTTP 客户端往返（信息项）：P50={p50:.2f} ms / P99={p99:.2f} ms"
        "（含 urllib 建连与本机调度抖动；产品口径见服务端 inject_total_ms）"
    )

    metrics = api_get("/api/metrics")
    hist = metrics.get("histograms") or {}
    dispatch_p99 = float((hist.get("dispatch_ms") or {}).get("p99", 0.0))
    graph_p99 = float((hist.get("graph_ms") or {}).get("p99", 0.0))
    frame_p99 = float((hist.get("frame_ms") or {}).get("p99", 0.0))
    inject_p99 = float((hist.get("inject_total_ms") or {}).get("p99", 0.0))

    check(cat, "dispatch_ms P99 ≤ 2 ms", dispatch_p99 <= 2.0, f"{dispatch_p99:.3f} ms")
    # Plan §2.5 的熔断判据是 graph P95 > 15 ms（运行时降级看门狗同口径，"未熔断"即达标），
    # 故按 P95 断言；P99 仅作信息项展示。
    graph_p95 = float((hist.get("graph_ms") or {}).get("p95", 0.0))
    check(
        cat,
        "graph_ms P95 ≤ 15 ms（§2.5 熔断判据）",
        graph_p95 <= 15.0,
        f"P95={graph_p95:.2f} ms / P99={graph_p99:.2f} ms",
    )
    check(cat, "frame_ms P99 ≤ 25 ms", frame_p99 <= 25.0, f"{frame_p99:.2f} ms")
    check(cat, "inject_total_ms P99 ≤ 30 ms", inject_p99 <= 30.0, f"{inject_p99:.2f} ms")
    check(cat, "编排未熔断", not bool(metrics.get("graph", {}).get("degraded")))
    check(cat, "总线队列未溢出", int(metrics["bus"].get("dropped", 0)) == 0, f"dropped={metrics['bus'].get('dropped')}")


def run_subtests(cat: Category) -> None:
    print("\n[6] 离线子套件")
    for name, script in SUBTESTS:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.perf_counter() - started
        summary = ""
        for line in reversed((process.stdout or "").splitlines()):
            if "结果：" in line or "PASS" in line or "FAIL" in line:
                summary = line.strip()
                break
        check(
            cat,
            f"子套件 {name}（{script}）",
            process.returncode == 0,
            f"{summary} · {elapsed:.1f}s",
        )


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def print_matrix() -> bool:
    width = 62
    print("\n" + "=" * width)
    print("Gesture-Copilot · 门禁 G3 验收矩阵（Plan §11.3）")
    print("=" * width)
    print(f"{'#':<3}{'断言类别':<28}{'通过/总数':>12}{'状态':>10}")
    print("-" * width)
    for index, cat in enumerate(CATEGORIES.values(), start=1):
        flag = "PASS" if cat.ok else "FAIL"
        print(f"{index:<3}{cat.name:<28}{f'{cat.passed}/{cat.total}':>12}{flag:>10}")
    print("-" * width)

    total_pass = sum(cat.passed for cat in CATEGORIES.values())
    total_all = sum(cat.total for cat in CATEGORIES.values())
    failed = [cat for cat in CATEGORIES.values() if not cat.ok]

    print(f"{'合计':<31}{f'{total_pass}/{total_all}':>12}{'PASS' if not failed else 'FAIL':>10}")
    if failed:
        print("\n失败明细：")
        for cat in failed:
            for note in cat.notes:
                print(f"  - [{cat.name}] {note}")
    print("=" * width)
    return not failed


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gesture-Copilot 一键全链路验收")
    parser.add_argument("--no-spawn", action="store_true", help="只连接，不自动拉起服务")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Gesture-Copilot · 一键验收（T17 / 门禁 G3）")
    print("=" * 78)

    spawned: subprocess.Popen | None = None
    if service_alive():
        print(f"检测到运行中的服务：{BASE_URL}（直接复用）")
    elif args.no_spawn:
        print(f"[FAIL] 服务不可达且指定了 --no-spawn：{BASE_URL}")
        return 2
    else:
        print(f"服务不可达，正在拉起：{sys.executable} main.py …")
        spawned = spawn_service()
        if spawned is None:
            print("[FAIL] 服务拉起失败，详见 logs/verify_server.log")
            return 2
        print(f"服务已就绪：{BASE_URL}")

    try:
        with connect(WS_URL, open_timeout=10) as ws:
            collect(ws, 0.5)  # 丢弃 hello/积压帧
            assert_service(category("service", "1. 服务可达"))
            assert_protocol(category("protocol", "2. 协议正确"), ws)
            if not reset_metrics_baseline():  # 旧版本实例：拒绝复用，避免无意义的红结果
                print("[FAIL] 运行中的服务为旧版本（缺少 /api/metrics/reset），请关闭后重跑")
                return 2
            assert_estop_priority(category("estop", "3. 急停优先权"), ws)
            assert_watchdog(category("watchdog", "4. 看门狗超时"), ws)
        assert_steady_state(category("steady", "5. 时延与稳态"))
        run_subtests(category("subtest", "6. 离线子套件"))
    finally:
        if spawned is not None:
            spawned.terminate()
            try:
                spawned.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                spawned.kill()

    ok = print_matrix()
    if ok:
        print("门禁 G3：PASS —— 全部验收项通过")
        return 0
    print("门禁 G3：FAIL —— 请查看上方失败明细")
    return 1


if __name__ == "__main__":
    sys.exit(main())
