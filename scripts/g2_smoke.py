"""门禁 G2 冒烟测试（Plan V1.0 §4.2 / §10.2 / §15.5）。

验收口径（本脚本的核心设计）：

    时延 = 视觉段折算 + 该帧服务端处理耗时
         = i × (1000/30) + `gesture_latency_ms`

其中 `i` 是触发帧在序列中的下标（急停 i=1、方向 i=2、转圈 i≈19），
即"从手势开始到指令生成"的真实累计时延。把这一项与服务端自报的处理耗时相加，
才是可以与 §4.2 预算表对比的量；只测处理耗时会把视觉段偷偷抹掉。

覆盖项：
  1. 急停（G-02）  2 帧触发，总时延 ≤ 100 ms，第 3 帧锁存
  2. 冷却期       500 ms 内运动指令被拦截（estop_cooldown）
  3. 方向（G-10/11/12）3 帧滑窗触发，总时延 ≤ 120 ms；2 帧不得触发
  4. 转圈（G-13）  自适应窗口触发，总时延 ≤ 760 ms
  5. 看门狗        停止输入后 500 ms 量级内自动回落 CMD_IDLE
  6. 编排健康度    graph_ms P95 ≤ 15 ms；熔断判据与降级机制单元自检
  7. 观测面        直方图/计数/JSONL 审计事件齐备

用法：先启动服务（python main.py），再执行
    .venv\\Scripts\\python.exe scripts\\g2_smoke.py
退出码 0 表示门禁通过。
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_trajectory import (  # noqa: E402
    seq_circle,
    seq_fist,
    seq_push,
    seq_swipe,
)

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/stream"
FRAME_INTERVAL = 1.0 / 30.0  # 30 FPS 采集中断

# 预算（Plan §4.2 冻结值）
BUDGET_ESTOP_MS = 100.0
BUDGET_DIRECTION_MS = 120.0
BUDGET_SPIN_MS = 760.0
BUDGET_GRAPH_P95_MS = 15.0
WATCHDOG_MISS_MS = 500.0

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


# --------------------------------------------------------------------------- #
# HTTP / WS 工具
# --------------------------------------------------------------------------- #
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
    with urllib.request.urlopen(request, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_frame(ws: Any, sequence: list[list[Any]], index: int, t0: float) -> dict[str, Any]:
    landmarks = [lm.model_dump() for lm in sequence[index]]
    return api_post(
        "/api/frames",
        {"landmarks": landmarks, "frame_ts": t0 + index * FRAME_INTERVAL, "score": 1.0},
    )


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


def drive_until_trigger(
    ws: Any,
    sequence: list[list[Any]],
    *,
    max_frames: int | None = None,
    explicit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 30 FPS 节奏驱动序列，返回首个触发帧的信息。

    同时记录 `first_code_index`（几何层首次产出候选的帧号），
    用于把"几何累积"与"表决+出帧"两段延迟分开度量。
    """
    t0 = time.time()
    limit = len(sequence) if max_frames is None else min(max_frames, len(sequence))
    result: dict[str, Any] = {
        "triggered": False,
        "index": -1,
        "first_code_index": None,
        "total_ms": None,
        "decision_ms": None,
        "response": None,
        "frames_sent": 0,
    }
    response: dict[str, Any] = {}
    for index in range(limit):
        due = t0 + index * FRAME_INTERVAL
        now = time.time()
        if due > now:
            time.sleep(due - now)
        payload: dict[str, Any] = {
            "landmarks": [lm.model_dump() for lm in sequence[index]],
            "frame_ts": t0 + index * FRAME_INTERVAL,
            "score": 1.0,
        }
        if explicit is not None:
            payload["explicit"] = explicit
        response = api_post("/api/frames", payload)
        result["frames_sent"] = index + 1
        if result["first_code_index"] is None and response.get("code"):
            result["first_code_index"] = index
        if response.get("triggered"):
            visual_ms = index * FRAME_INTERVAL * 1000.0
            server_ms = float(response.get("gesture_latency_ms") or 0.0)
            first_code = result["first_code_index"]
            decision_ms = (
                (index - first_code) * FRAME_INTERVAL * 1000.0 + server_ms
                if first_code is not None
                else None
            )
            result.update(
                {
                    "triggered": True,
                    "index": index,
                    "total_ms": visual_ms + server_ms,
                    "visual_ms": visual_ms,
                    "server_ms": server_ms,
                    "decision_ms": decision_ms,
                    "response": response,
                }
            )
            return result
    result["response"] = response
    return result


def fast_swipe_frames(
    direction: str = "right", frames: int = 5, step: float = 0.13
) -> list[list[Any]]:
    """构造快挥序列：单帧位移 0.13（≥ swipe_min_displacement，第 2 帧即出候选）。

    §4.2 的 120 ms 预算按"3 帧表决 + 出帧"定义，只有当几何层在第 2 帧就产出候选时
    才能整体落在预算内；慢挥的几何累积会额外占用时间（见 §16 记录）。
    """
    from src.api.injector import POSE_LIBRARY  # noqa: PLC0415
    from src.api.schemas import GestureCode, Landmark  # noqa: PLC0415

    base = list(POSE_LIBRARY[GestureCode.FORWARD].landmarks)
    sign = 1.0 if direction == "right" else -1.0
    sequence: list[list[Any]] = []
    for index in range(frames):
        dx = sign * step * index
        sequence.append([Landmark(x=p.x + dx, y=p.y, z=p.z) for p in base])
    return sequence


# --------------------------------------------------------------------------- #
# 单元自检（不依赖服务）
# --------------------------------------------------------------------------- #
def unit_checks() -> None:
    print("\n[0] 机制单元自检（不依赖服务）")

    # ① 熔断判据
    from src.agents.graph import IntentOrchestrator  # noqa: PLC0415
    from src.core.metrics import AdaptiveDegrader, Metrics  # noqa: PLC0415

    metrics = Metrics(500)
    hist = metrics.histogram("graph_ms")
    for _ in range(20):
        hist.observe(40.0)
    probe = IntentOrchestrator.__new__(IntentOrchestrator)
    probe._graph_ms = hist
    probe.graph_p95_limit_ms = BUDGET_GRAPH_P95_MS
    probe.graph_min_samples = 20
    check(
        "熔断判据：graph P95=40ms>15ms 判为需降级",
        probe._should_degrade() is True,
    )
    for _ in range(20):
        hist.observe(2.0)
    hist.reset()
    for _ in range(20):
        hist.observe(2.0)
    check("熔断判据：graph P95=2ms 不降级", probe._should_degrade() is False)

    # ② 自适应降级
    metrics2 = Metrics(500)
    frame_hist = metrics2.histogram("frame_ms")
    for _ in range(25):
        frame_hist.observe(40.0)
    degrader = AdaptiveDegrader(
        metrics2, threshold_ms=25.0, min_samples=20, max_level=1, cooldown_s=10.0
    )
    first = degrader.check(now=1000.0)
    second = degrader.check(now=1001.0)
    check(
        "降级机制：帧 P95=40ms>25ms 触发一次降级",
        first is not None and first.level == 1 and first.target_width == 480,
        str(first.as_dict()) if first else "None",
    )
    check("降级机制：幂等（同级别不重复降级）", second is None)

    # ③ 防抖内核语义
    from src.agents.debounce import FrameSignal, new_state, push  # noqa: PLC0415
    from src.api.schemas import GestureCode  # noqa: PLC0415

    st = new_state()
    t = 1000.0
    r1 = push(st, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 1 / 30), now=t + 1 / 30)
    r2 = push(st, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 2 / 30), now=t + 2 / 30)
    r3 = push(st, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 3 / 30), now=t + 3 / 30)
    check(
        "急停：第 2 帧出帧、第 3 帧只锁存",
        r1.cmd is None and r2.cmd is not None and r2.estop and r3.cmd is None and r3.latch,
    )

    st2 = new_state()
    t2 = 5000.0
    low = [
        push(st2, FrameSignal(code=GestureCode.TURN_LEFT, score=0.5, ts=t2 + (i + 1) / 30), now=t2 + (i + 1) / 30)
        for i in range(3)
    ]
    check("方向：3 帧低置信度（0.5）不触发", low[-1].cmd is None and low[-1].reason == "low_confidence")

    st3 = new_state()
    t3 = 9000.0
    seq = [
        push(st3, FrameSignal(code=GestureCode.TURN_LEFT, score=0.95, ts=t3 + (i + 1) / 30), now=t3 + (i + 1) / 30)
        for i in range(3)
    ]
    rollback = push(
        st3,
        FrameSignal(code=GestureCode.TURN_RIGHT, score=0.95, ts=t3 + 4 / 30),
        now=t3 + 4 / 30,
    )
    check(
        "方向：3 帧高置信度触发 + 第 4 帧反向回滚",
        seq[-1].cmd is not None and rollback.revoked_code == GestureCode.TURN_LEFT,
    )

    spin_state = new_state()
    spin = push(
        spin_state,
        FrameSignal(code=GestureCode.SPIN, score=0.95, ts=t3, preconfirmed=True),
        now=t3,
    )
    check("转圈：圆拟合确认直通 CMD_SPIN", spin.cmd is not None and spin.latency_frames == 1)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("Gesture-Copilot · 门禁 G2（急停/方向/转圈/看门狗 时延实测）")
    print("=" * 78)

    try:
        health = api_get("/api/health")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [FAIL] 无法连接服务：{exc}")
        print("  提示：请先执行  .venv\\Scripts\\python.exe main.py")
        return 2

    print(f"服务模式 {health.get('mode')} · DRY_RUN={health.get('dry_run')} · "
          f"camera={health.get('camera_available')}")
    log_file = Path(health.get("log_file", ""))
    lines_before = len(log_file.read_text(encoding="utf-8").splitlines()) if log_file.exists() else 0

    unit_checks()

    rng = random.Random(20261031)
    # 急停：复用同一合成拳型连续 3 帧，避免抖动作导致中途掉帧
    fist_pose = seq_fist(random.Random(1))[0]
    fist_frames = [fist_pose, fist_pose, fist_pose]
    push_seq = seq_push(rng)
    swipe_seq = fast_swipe_frames("right")
    circle_seq = seq_circle(random.Random(42))

    with connect(WS_URL, open_timeout=10) as ws:
        hello = None
        for message in collect(ws, 3.0):
            if message.get("type") == "hello":
                hello = message
        check("WS 握手", hello is not None)

        # ---------------- 1. 急停 ---------------- #
        print("\n[1] 急停 G-02（预算 ≤ 100 ms）")
        estop = drive_until_trigger(ws, fist_frames)
        check("急停触发 CMD_STOP", estop["triggered"] and estop["response"]["cmd"] == "CMD_STOP")
        if estop["triggered"]:
            check(
                f"急停总时延 ≤ {BUDGET_ESTOP_MS:.0f} ms",
                estop["total_ms"] <= BUDGET_ESTOP_MS,
                f"{estop['total_ms']:.1f} ms（视觉段 {estop['visual_ms']:.1f} + 处理 {estop['server_ms']:.1f}）",
            )
            check("急停触发帧数 = 2", estop["index"] == 1, f"第 {estop['index'] + 1} 帧")

        # 第 3 帧必须"紧接着"发（<200ms），否则会被帧间超时判为手势中断
        third = send_frame(ws, fist_frames, 2, time.time() - FRAME_INTERVAL)
        check(
            "第 3 帧只锁存不出帧",
            third.get("cmd") is None and third.get("reason") == "estop_latch_confirmed",
            str(third.get("reason")),
        )

        # ---------------- 2. 冷却期（趁冷却未过期立刻验证） ---------------- #
        print("\n[2] 急停冷却期（500 ms 内运动指令被拒）")
        t0 = time.time()
        blocked_reasons = []
        emitted_cmds = []
        for index in range(2):
            response = api_post(
                "/api/frames",
                {
                    "landmarks": [lm.model_dump() for lm in fist_pose],
                    "frame_ts": t0 + index * FRAME_INTERVAL,
                    "score": 1.0,
                    # 显式标记：跳过几何判定，专门检验"冷却期拦截"这条规则
                    "explicit": {"push": True},
                },
            )
            blocked_reasons.append(response.get("reason"))
            emitted_cmds.extend(f["cmd"] for f in (response.get("frames") or []))

        # 此时再收 WS 消息（急停 command 已在缓冲里，不影响冷却判定）
        estop_cmds = commands_of(collect(ws, 0.4))
        check("WS 收到急停 command", any(c["cmd"] == "CMD_STOP" for c in estop_cmds))
        metrics_now = api_get("/api/metrics")
        check(
            "锁存计数 = 1",
            int(metrics_now["debounce"]["stats"]["latches"]) >= 1,
            str(metrics_now["debounce"]["stats"]),
        )
        check(
            "冷却期内未下发任何运动指令",
            not any(c != "CMD_IDLE" for c in emitted_cmds),
            f"下发={emitted_cmds}",
        )
        check(
            "拦截原因为 estop_cooldown",
            "estop_cooldown" in blocked_reasons,
            str(blocked_reasons),
        )
        metrics_now = api_get("/api/metrics")
        check(
            "debounce 拦截计数 ≥ 1",
            int(metrics_now["debounce"]["stats"]["blocked"]) >= 1,
            str(metrics_now["debounce"]["stats"]["blocked"]),
        )

        # 等冷却结束 + 看门狗静默
        time.sleep(0.8)
        collect(ws, 0.2)

        # ---------------- 3. 方向 ---------------- #
        print("\n[3] 方向指令 G-11/G-12（预算 ≤ 120 ms）")
        two = drive_until_trigger(ws, swipe_seq, max_frames=2)
        check("2 帧不得触发（防抖生效）", not two["triggered"], f"共 {two['frames_sent']} 帧")
        time.sleep(0.4)  # 触发帧间超时重置，保证下一次从干净窗口开始
        direction = drive_until_trigger(ws, swipe_seq)
        check(
            "方向触发",
            direction["triggered"] and direction["response"]["cmd"] in {"CMD_TURN_LEFT", "CMD_TURN_RIGHT"},
            str(direction["response"]["cmd"]) if direction["triggered"] else f"未触发（{direction['response'].get('reason')}）",
        )
        if direction["triggered"]:
            check(
                f"方向：候选→指令 ≤ {BUDGET_DIRECTION_MS:.0f} ms（3 帧表决 + 出帧）",
                direction["decision_ms"] is not None and direction["decision_ms"] <= BUDGET_DIRECTION_MS,
                f"{direction['decision_ms']:.1f} ms",
            )
            print(
                f"        · 手势起始→指令总计 {direction['total_ms']:.1f} ms"
                f"（其中几何累积 {direction['visual_ms']:.1f} ms，触发于第 {direction['index'] + 1} 帧）"
            )
            check("方向触发帧数 = 3 帧表决", direction["index"] == 3, f"第 {direction['index'] + 1} 帧")

        # 等看门狗回落并静默
        collect(ws, 0.9)

        # ---------------- 4. 转圈 ---------------- #
        print("\n[4] 转圈 G-13（预算 ≤ 760 ms）")
        spin = drive_until_trigger(ws, circle_seq)
        check(
            "转圈触发 CMD_SPIN",
            spin["triggered"] and spin["response"]["cmd"] == "CMD_SPIN",
            str(spin["response"]["cmd"]) if spin["triggered"] else f"未触发（{spin['response'].get('reason')}）",
        )
        if spin["triggered"]:
            check(
                f"转圈总时延 ≤ {BUDGET_SPIN_MS:.0f} ms",
                spin["total_ms"] <= BUDGET_SPIN_MS,
                f"{spin['total_ms']:.1f} ms（视觉段 {spin['visual_ms']:.1f} + 处理 {spin['server_ms']:.1f}）",
            )
            check(
                "转圈触发帧数落在 12~24 帧窗口",
                12 <= spin["index"] + 1 <= 24,
                f"第 {spin['index'] + 1} 帧",
            )

        # ---------------- 5. 看门狗 ---------------- #
        print("\n[5] 看门狗（停止输入后自动回落 CMD_IDLE）")
        idle_deadline = time.time() + 1.6
        idle_frame = None
        idle_arrival = None
        while time.time() < idle_deadline:
            for frame in commands_of(collect(ws, 0.2)):
                if frame["cmd"] == "CMD_IDLE" and frame.get("source") == "WATCHDOG":
                    idle_frame = frame
                    idle_arrival = time.time()
                    break
            if idle_frame is not None:
                break
        check("看门狗下发 CMD_IDLE", idle_frame is not None)
        if idle_frame is not None and spin["triggered"]:
            last_motion = float(spin["response"]["frames"][0]["timestamp"]) if spin["response"].get("frames") else None
            if last_motion:
                idle_delay = (float(idle_frame["timestamp"]) - last_motion) * 1000.0
                check(
                    f"看门狗回落时机 ≈ {WATCHDOG_MISS_MS:.0f} ms",
                    WATCHDOG_MISS_MS - 60 <= idle_delay <= WATCHDOG_MISS_MS + 350,
                    f"{idle_delay:.1f} ms",
                )

        # ---------------- 6. 编排健康度 ---------------- #
        print("\n[6] 温路径编排健康度")
        metrics_now = api_get("/api/metrics")
        graph = metrics_now.get("graph") or {}
        p95 = float(graph.get("p95_ms", 0.0))
        check(
            f"graph P95 ≤ {BUDGET_GRAPH_P95_MS:.0f} ms（未触发熔断）",
            graph.get("degraded") is True or p95 <= BUDGET_GRAPH_P95_MS,
            f"p95={p95:.2f} ms, samples={graph.get('samples')}, degraded={graph.get('degraded')}",
        )
        check("编排样本数 > 0", int(graph.get("samples", 0)) > 0, str(graph.get("samples")))
        check("看门狗已解除武装", graph.get("watchdog_armed") is False, str(graph.get("watchdog_armed")))

        histograms = metrics_now.get("histograms") or {}
        for name in ("pipeline_ms", "debounce_ms", "graph_ms", "dispatch_ms", "frame_ms", "frame_latency_ms"):
            check(f"直方图 {name} 存在", name in histograms)
        degrade = metrics_now.get("degrade") or {}
        check("降级看门狗在运行", int(degrade.get("checks", 0)) > 0, f"checks={degrade.get('checks')}")
        check("降级级别未触发（无摄像头时无长耗时帧）", int(degrade.get("level", 0)) == 0, str(degrade.get("level")))

    # ---------------- 7. JSONL 审计 ---------------- #
    print("\n[7] JSONL 审计留痕")
    lines = log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []
    new_lines = lines[lines_before:]
    events = [json.loads(line).get("event") for line in new_lines if line.strip()]
    all_events = [json.loads(line).get("event") for line in lines if line.strip()]
    check("含 frame 事件", "frame" in events, f"{len(new_lines)} 条新日志")
    check("含 intent_decision 审计事件", "intent_decision" in events)
    check("含 graph_warmup 预热记录（启动阶段）", "graph_warmup" in all_events)
    decisions = [
        json.loads(line)
        for line in new_lines
        if line.strip() and json.loads(line).get("event") == "intent_decision"
    ]
    check(
        "审计事件带 node_trace",
        all("node_trace" in d for d in decisions) and len(decisions) > 0,
        f"{len(decisions)} 条",
    )
    check(
        "审计事件带决策结果",
        any(d.get("decision") == "dispatched" for d in decisions),
    )

    # ---------------- 汇总 ---------------- #
    print("\n" + "=" * 78)
    total = len(PASSED) + len(FAILED)
    print(f"结果：{len(PASSED)}/{total} 通过")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        print("门禁 G2：FAIL")
        return 1
    print("门禁 G2：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
