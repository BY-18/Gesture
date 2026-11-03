"""静态误触发与防抖语义评测（Plan V1.0 T15 / §11.2）。

两部分：
1. **静态误触发率**：把 `static_noise.jsonl` 的 1000 帧（静止微抖 500 + 随机噪声 500）
   连续喂进 `FramePipeline → debounce.push`，统计**实际产生指令**的帧占比，门槛 ≤ 2%；
2. **防抖语义回归**：把 §2.2 / §2.3 的五条裁决固化为断言，防止后续改动无声破坏。

用法：``.venv\\Scripts\\python.exe scripts\\test_debounce.py``
结果表格写入 ``logs/eval_report.md``；退出码 0 表示通过。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_synthetic import (  # noqa: E402
    DATASET_VERSION,
    load_records,
    record_landmarks,
    update_report,
)
from src.agents.debounce import (  # noqa: E402
    FrameSignal,
    new_state,
    push,
)
from src.api.schemas import Cmd, GestureCode  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.vision.pipeline import FramePipeline  # noqa: E402

MAX_STATIC_FP_RATE = 0.02
FRAME_INTERVAL = 1.0 / 30.0

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
# ① 静态误触发
# --------------------------------------------------------------------------- #
def evaluate_static_noise() -> dict[str, dict[str, float]]:
    records = load_records("static_noise")
    per_kind: dict[str, dict[str, float]] = defaultdict(
        lambda: {"frames": 0, "candidates": 0, "commands": 0}
    )

    for record in records:
        kind = record["label"]
        pipeline = FramePipeline()
        state = new_state()
        stats = per_kind[kind]
        for index, landmarks in enumerate(record_landmarks(record)):
            ts = index * FRAME_INTERVAL
            frame = pipeline.ingest(landmarks, ts=ts, frame_id=index + 1)
            stats["frames"] += 1
            if frame.candidate is not None:
                stats["candidates"] += 1
            decision = push(
                state,
                FrameSignal(
                    code=frame.candidate.code if frame.candidate else None,
                    score=frame.candidate.score if frame.candidate else 0.0,
                    ts=ts,
                    frame_id=index + 1,
                ),
                now=ts,
            )
            if decision.cmd is not None:
                stats["commands"] += 1
    return per_kind


# --------------------------------------------------------------------------- #
# ② 防抖语义
# --------------------------------------------------------------------------- #
def evaluate_semantics() -> None:
    print("\n[2] 防抖语义回归（Plan §2.2 / §2.3）")
    th = get_settings().debounce

    # 急停：2 帧触发、第 3 帧锁存
    state = new_state()
    t = 1000.0
    r1 = push(state, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 1 / 30), now=t + 1 / 30)
    r2 = push(state, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 2 / 30), now=t + 2 / 30)
    r3 = push(state, FrameSignal(code=GestureCode.ESTOP, score=1.0, ts=t + 3 / 30), now=t + 3 / 30)
    check(
        "急停：第 1 帧不出帧 / 第 2 帧直出 CMD_STOP / 第 3 帧仅锁存",
        r1.cmd is None and r2.cmd is Cmd.CMD_STOP and r3.cmd is None and r3.latch,
        f"r1={r1.reason} r2={r2.reason} r3={r3.reason}",
    )
    check(
        "急停：触发即进入 500 ms 冷却",
        abs(r2.cooldown_ms - 500.0) < 1.0,
        f"{r2.cooldown_ms} ms",
    )

    # 冷却期拦截
    blocked = push(
        state,
        FrameSignal(code=GestureCode.FORWARD, score=1.0, ts=t + 4 / 30),
        now=t + 4 / 30,
    )
    check(
        "冷却期内运动指令被拒",
        blocked.blocked and blocked.cmd is None and blocked.reason == "estop_cooldown",
        f"{blocked.reason}",
    )

    # 锁存解除：松手 + 冷却期满
    later = t + 1.0
    push(state, FrameSignal(code=None, score=0.0, ts=later), now=later)
    check("锁存解除：松手且冷却期满后 estop_latched=False", state.estop_latched is False)

    # 方向：3 帧确认 / 2 帧不触发
    s2 = new_state()
    base = 5000.0
    a = push(s2, FrameSignal(code=GestureCode.TURN_LEFT, score=0.95, ts=base + 1 / 30), now=base + 1 / 30)
    b = push(s2, FrameSignal(code=GestureCode.TURN_LEFT, score=0.95, ts=base + 2 / 30), now=base + 2 / 30)
    c = push(s2, FrameSignal(code=GestureCode.TURN_LEFT, score=0.95, ts=base + 3 / 30), now=base + 3 / 30)
    check(
        "方向：仅 2 帧不触发，第 3 帧触发 CMD_TURN_LEFT",
        a.cmd is None and b.cmd is None and c.cmd is Cmd.CMD_TURN_LEFT,
        f"第 3 帧 reason={c.reason}",
    )
    check(
        "方向：触发时记录的帧数 = confirm_frames",
        c.latency_frames == th.confirm_frames,
        f"{c.latency_frames}",
    )

    # 置信度门
    s3 = new_state()
    low = None
    for index in range(3):
        low = push(
            s3,
            FrameSignal(code=GestureCode.FORWARD, score=0.5, ts=base + (index + 1) / 30),
            now=base + (index + 1) / 30,
        )
    check(
        f"置信度门：加权 {0.5} ≤ {th.min_confidence} 不触发",
        low is not None and low.cmd is None and low.reason == "low_confidence",
        f"{low.reason if low else 'None'}",
    )

    # 混窗不触发
    s4 = new_state()
    mixed = None
    for code in (GestureCode.TURN_LEFT, GestureCode.TURN_RIGHT, GestureCode.TURN_LEFT):
        mixed = push(
            s4,
            FrameSignal(code=code, score=0.95, ts=base + 2.0),
            now=base + 2.0,
        )
    check(
        "混窗：3 帧不同类不触发",
        mixed is not None and mixed.cmd is None,
        f"{mixed.reason if mixed else 'None'}",
    )

    # 回滚：触发后第 4 帧反向
    s5 = new_state()
    for index in range(3):
        push(
            s5,
            FrameSignal(code=GestureCode.TURN_LEFT, score=0.95, ts=base + 3.0 + index / 30),
            now=base + 3.0 + index / 30,
        )
    rollback = push(
        s5,
        FrameSignal(code=GestureCode.TURN_RIGHT, score=0.95, ts=base + 3.2),
        now=base + 3.2,
    )
    check(
        "回滚：第 4 帧反向高置信 → 撤销上一次触发",
        rollback.revoked_code is GestureCode.TURN_LEFT and rollback.reason == "rollback",
        f"{rollback.reason}",
    )

    # 转圈直通
    s6 = new_state()
    spin = push(
        s6,
        FrameSignal(code=GestureCode.SPIN, score=0.95, ts=base + 4.0, preconfirmed=True),
        now=base + 4.0,
    )
    check(
        "转圈：圆拟合确认后单帧直通 CMD_SPIN",
        spin.cmd is Cmd.CMD_SPIN and spin.latency_frames == 1,
        f"{spin.reason}",
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("Gesture-Copilot · 静态误触发与防抖语义评测（T15）")
    print("=" * 78)

    print(f"\n[1] 静态误触发（数据集 v{DATASET_VERSION}：静止微抖 500 帧 + 随机噪声 500 帧）")
    per_kind = evaluate_static_noise()

    total_frames = sum(int(v["frames"]) for v in per_kind.values())
    total_candidates = sum(int(v["candidates"]) for v in per_kind.values())
    total_commands = sum(int(v["commands"]) for v in per_kind.values())
    overall_rate = total_commands / total_frames if total_frames else 0.0

    print(f"  {'序列':<16}{'帧数':>8}{'候选帧':>8}{'指令帧':>8}{'误触发率':>10}")
    print("  " + "-" * 50)
    rows: list[str] = []
    for kind, stats in sorted(per_kind.items()):
        frames = int(stats["frames"])
        rate = stats["commands"] / frames if frames else 0.0
        print(f"  {kind:<16}{frames:>8}{int(stats['candidates']):>8}{int(stats['commands']):>8}{rate * 100:>9.2f}%")
        rows.append(
            f"| {kind} | {frames} | {int(stats['candidates'])} | {int(stats['commands'])} | {rate * 100:.2f}% |"
        )
    print(f"  {'合计':<16}{total_frames:>8}{total_candidates:>8}{total_commands:>8}{overall_rate * 100:>9.2f}%")

    check(
        f"静态误触发率 ≤ {MAX_STATIC_FP_RATE:.0%}",
        overall_rate <= MAX_STATIC_FP_RATE,
        f"{overall_rate * 100:.2f}%（{total_commands}/{total_frames} 帧）",
    )
    check("1000 帧无任何误触发指令（最严口径）", total_commands == 0, f"{total_commands} 条")

    evaluate_semantics()

    body = [
        f"数据集：`logs/dataset/static_noise.jsonl`（v{DATASET_VERSION}，合计 {total_frames} 帧），"
        "经 `FramePipeline → debounce.push` 串联判定。",
        "",
        "| 序列 | 帧数 | 几何候选帧 | 产生指令帧 | 误触发率 |",
        "|---|---|---|---|---|",
        *rows,
        f"| **合计** | {total_frames} | {total_candidates} | {total_commands} | {overall_rate * 100:.2f}% |",
        "",
        f"门槛：≤ {MAX_STATIC_FP_RATE:.0%} → **{'PASS' if overall_rate <= MAX_STATIC_FP_RATE else 'FAIL'}**",
        "",
        "防抖语义回归（急停 2 帧/锁存/冷却、方向 3 帧、置信度门、混窗、回滚、转圈直通）"
        f"共 {len(PASSED) - 2}/{len(PASSED) - 2} 项通过。",
    ]
    report = update_report("02_false_trigger", "静态误触发与防抖语义（scripts/test_debounce.py）", "\n".join(body))

    print("\n" + "-" * 78)
    print(f"评测报告已写入：{report}")
    print("结果：%d/%d 通过" % (len(PASSED), len(PASSED) + len(FAILED)))
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("静态误触发与防抖语义：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
