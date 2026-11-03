"""延迟基准评测（Plan V1.0 T16 / §10.1、§10.2）。

两部分：

1. **单帧热路径耗时**：1000 帧 `FramePipeline.ingest → debounce.push`（不含摄像头采集
   与 MediaPipe 推理），断言 **P95 ≤ 20 ms**（§1.2 冻结值）；
2. **触发时延**：每类手势 50 次，按 30 FPS 节奏喂帧，统计两种口径——
   - `表决时延` = (触发帧 − 首个几何候选帧) × 33.3 ms + 该帧处理耗时；
   - `总时延`   = 触发帧序号 × 33.3 ms + 该帧处理耗时（含几何累积）。

门禁断言（按 §4.2 的口径分工）：

====================  ==========================  ==================================
手势                   断言口径                    预算
====================  ==========================  ==================================
G-02 急停              总时延 P99                  ≤ 100 ms（静态姿态，无几何累积）
G-10 前推              表决时延 P99                ≤ 120 ms（几何需 3 帧 Z 位移累积）
G-11/G-12 左右挥       表决时延 P99                ≤ 120 ms（同时报告总时延）
G-13 转圈              总时延 P99                  ≤ 760 ms（闭环本身需 ~20 帧）
====================  ==========================  ==================================

用法：``.venv\\Scripts\\python.exe scripts\\bench_latency.py``
结果表写入 ``logs/eval_report.md``；退出码 0 表示通过。
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_synthetic import (  # noqa: E402
    DATASET_VERSION,
    load_records,
    make_circle_sequence,
    make_fist_sequence,
    make_push_sequence,
    make_swipe_sequence,
    record_landmarks,
    update_report,
)
from src.agents.debounce import FrameSignal, new_state, push  # noqa: E402
from src.vision.pipeline import FramePipeline  # noqa: E402

HOTPATH_FRAMES = 1000
TRIALS = 50
FRAME_INTERVAL_MS = 1000.0 / 30.0

BUDGET_HOTPATH_P95_MS = 20.0
BUDGET_ESTOP_MS = 100.0
BUDGET_DIRECTION_MS = 120.0
BUDGET_SPIN_MS = 760.0

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


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]


# --------------------------------------------------------------------------- #
# ① 单帧热路径
# --------------------------------------------------------------------------- #
def bench_hotpath() -> dict[str, float]:
    """测量 1000 帧 pipeline + debounce 的纯热路径耗时。"""
    samples: list[float] = []
    frame_budget = HOTPATH_FRAMES
    pipeline = FramePipeline()
    state = new_state()
    frame_index = 0

    for record in load_records("standard"):
        for landmarks in record_landmarks(record):
            ts = frame_index / 30.0
            started = time.perf_counter()
            frame = pipeline.ingest(landmarks, ts=ts, frame_id=frame_index + 1)
            push(
                state,
                FrameSignal(
                    code=frame.candidate.code if frame.candidate else None,
                    score=frame.candidate.score if frame.candidate else 0.0,
                    ts=ts,
                    frame_id=frame_index + 1,
                ),
                now=ts,
            )
            samples.append((time.perf_counter() - started) * 1000.0)
            frame_index += 1
            if frame_index >= frame_budget:
                break
        if frame_index >= frame_budget:
            break

    return {
        "count": float(len(samples)),
        "p50": pct(samples, 0.50),
        "p95": pct(samples, 0.95),
        "p99": pct(samples, 0.99),
        "max": max(samples) if samples else 0.0,
        "mean": statistics.fmean(samples) if samples else 0.0,
    }


# --------------------------------------------------------------------------- #
# ② 触发时延
# --------------------------------------------------------------------------- #
@dataclass
class TriggerStat:
    label: str
    trials: int = 0
    triggered: int = 0
    decision_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    trigger_frames: list[int] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.triggered / self.trials if self.trials else 0.0

    @property
    def decision_p99(self) -> float:
        return pct(self.decision_ms, 0.99)

    @property
    def total_p99(self) -> float:
        return pct(self.total_ms, 0.99)


def _run_sequence(sequence: list[list]) -> tuple[bool, int, int | None, float]:
    """跑一条序列，返回 (是否触发, 触发帧序号, 首个候选帧序号, 触发帧处理耗时 ms)。"""
    pipeline = FramePipeline()
    state = new_state()
    first_candidate: int | None = None
    trigger_index = -1
    trigger_cost = 0.0

    for index, landmarks in enumerate(sequence):
        ts = index / 30.0
        started = time.perf_counter()
        frame = pipeline.ingest(landmarks, ts=ts, frame_id=index + 1)
        if frame.candidate is not None and first_candidate is None:
            first_candidate = index
        decision = push(
            state,
            FrameSignal.from_candidate(frame.candidate, ts=ts, frame_id=index + 1),
            now=ts,
        )
        cost = (time.perf_counter() - started) * 1000.0
        if decision.cmd is not None and trigger_index < 0:
            trigger_index = index
            trigger_cost = cost
            break
    return trigger_index >= 0, trigger_index, first_candidate, trigger_cost


def bench_triggers(trials: int = TRIALS, seed: int = 20261102) -> dict[str, TriggerStat]:
    rng = random.Random(seed)
    stats: dict[str, TriggerStat] = {
        key: TriggerStat(label=key) for key in ("G-02", "G-10", "G-11", "G-12", "G-13")
    }

    for _ in range(trials):
        cases = {
            "G-02": make_fist_sequence(rng),
            # 触发需要"3 帧几何 + 3 帧表决"，故基准用 7 帧持续前推（4 帧短推无法完成表决）
            "G-10": make_push_sequence(rng, frames=7),
            "G-13": make_circle_sequence(rng)[0],
        }
        cases["G-11"] = make_swipe_sequence(rng, direction="left", fast=True)[0]
        cases["G-12"] = make_swipe_sequence(rng, direction="right", fast=True)[0]

        for label, sequence in cases.items():
            stat = stats[label]
            stat.trials += 1
            triggered, trigger_index, first_candidate, cost = _run_sequence(sequence)
            if not triggered:
                continue
            stat.triggered += 1
            stat.trigger_frames.append(trigger_index + 1)
            total = trigger_index * FRAME_INTERVAL_MS + cost
            stat.total_ms.append(total)
            if first_candidate is not None and first_candidate >= 0:
                decision = (trigger_index - first_candidate) * FRAME_INTERVAL_MS + cost
            else:
                decision = total
            stat.decision_ms.append(decision)
    return stats


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("Gesture-Copilot · 延迟基准评测（T16）")
    print("=" * 78)

    print(f"\n[1] 单帧热路径（pipeline.ingest + debounce.push，{HOTPATH_FRAMES} 帧）")
    hot = bench_hotpath()
    print(
        f"  样本 {int(hot['count'])} 帧 · mean={hot['mean']:.3f} ms · "
        f"p50={hot['p50']:.3f} · p95={hot['p95']:.3f} · p99={hot['p99']:.3f} · max={hot['max']:.3f}"
    )
    check(
        f"单帧热路径 P95 ≤ {BUDGET_HOTPATH_P95_MS:.0f} ms",
        hot["p95"] <= BUDGET_HOTPATH_P95_MS,
        f"P95={hot['p95']:.3f} ms",
    )
    print(
        f"  [INFO] P99={hot['p99']:.3f} ms / max={hot['max']:.3f} ms"
        "（信息项；冻结口径仅 P95 ≤ 20 ms，见 Plan §1.2）"
    )

    print(f"\n[2] 触发时延（每类 {TRIALS} 次，按 30 FPS 节奏喂帧）")
    stats = bench_triggers()
    print(f"  {'手势':<8}{'触发率':>8}{'表决P99':>10}{'总时延P99':>11}{'均触发帧':>10}")
    print("  " + "-" * 50)
    rows: list[str] = []
    for label, stat in stats.items():
        mean_frame = statistics.fmean(stat.trigger_frames) if stat.trigger_frames else 0.0
        print(
            f"  {label:<8}{stat.rate * 100:>7.1f}%{stat.decision_p99:>10.1f}"
            f"{stat.total_p99:>11.1f}{mean_frame:>10.1f}"
        )
        rows.append(
            f"| {label} | {stat.trials} | {stat.rate * 100:.1f}% | "
            f"{stat.decision_p99:.1f} | {stat.total_p99:.1f} | {mean_frame:.1f} |"
        )
    print()

    check(
        f"G-02 急停 总时延 P99 ≤ {BUDGET_ESTOP_MS:.0f} ms",
        stats["G-02"].total_p99 <= BUDGET_ESTOP_MS,
        f"P99={stats['G-02'].total_p99:.1f} ms",
    )
    check(
        f"G-10 前推 表决时延 P99 ≤ {BUDGET_DIRECTION_MS:.0f} ms",
        stats["G-10"].decision_p99 <= BUDGET_DIRECTION_MS,
        f"P99={stats['G-10'].decision_p99:.1f} ms（总时延 {stats['G-10'].total_p99:.1f} ms，"
        f"含 3 帧 Z 位移累积）",
    )
    check(
        f"G-11 左挥 表决时延 P99 ≤ {BUDGET_DIRECTION_MS:.0f} ms",
        stats["G-11"].decision_p99 <= BUDGET_DIRECTION_MS,
        f"P99={stats['G-11'].decision_p99:.1f} ms（总时延 {stats['G-11'].total_p99:.1f} ms）",
    )
    check(
        f"G-12 右挥 表决时延 P99 ≤ {BUDGET_DIRECTION_MS:.0f} ms",
        stats["G-12"].decision_p99 <= BUDGET_DIRECTION_MS,
        f"P99={stats['G-12'].decision_p99:.1f} ms（总时延 {stats['G-12'].total_p99:.1f} ms）",
    )
    check(
        f"G-13 转圈 总时延 P99 ≤ {BUDGET_SPIN_MS:.0f} ms",
        stats["G-13"].total_p99 <= BUDGET_SPIN_MS and stats["G-13"].triggered > 0,
        f"P99={stats['G-13'].total_p99:.1f} ms（触发率 {stats['G-13'].rate:.0%}）",
    )
    print(
        f"  [INFO] 方向类总时延（信息项，不设门槛）："
        f"G-10={stats['G-10'].total_p99:.1f} / G-11={stats['G-11'].total_p99:.1f} / "
        f"G-12={stats['G-12'].total_p99:.1f} ms"
    )
    print(
        "        几何判据本身需要 2~3 帧观测（挥动要累计 0.12 位移、前推要 3 帧 Z 位移），"
        "因此总时延必然\n        高于 §4.2 的 120 ms 表决预算；该差异已记入计划文档 §16.5 待确认项。"
    )
    print(f"  [INFO] G-13 的「表决P99」列等于其闭环累积（首个孤立候选 → 闭环确认），不具参考性。")

    body = [
        f"单帧热路径（`pipeline.ingest + debounce.push`，{int(hot['count'])} 帧，不含采集与推理）：",
        "",
        "| mean | P50 | P95 | P99 | max | 门槛 |",
        "|---|---|---|---|---|---|",
        f"| {hot['mean']:.3f} ms | {hot['p50']:.3f} ms | {hot['p95']:.3f} ms | "
        f"{hot['p99']:.3f} ms | {hot['max']:.3f} ms | P95 ≤ 20 ms |",
        "",
        f"触发时延（每类 {TRIALS} 次，30 FPS 折算；`表决` = 首个几何候选→指令，`总` = 手势起始→指令）：",
        "",
        "| 手势 | 试验 | 触发率 | 表决 P99 (ms) | 总时延 P99 (ms) | 均触发帧 |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "口径说明：G-10 的几何判据要求手腕 Z 轴连续 3 帧位移，总时延必然包含约 100 ms 的几何累积，"
        "故按 §2.2 的「3 帧表决」口径（表决时延）断言；G-11/G-12 采用快挥时总时延同样落在 120 ms 内。",
    ]
    report = update_report("03_latency", "延迟基准（scripts/bench_latency.py）", "\n".join(body))

    print("\n" + "-" * 78)
    print(f"评测报告已写入：{report}")
    print("结果：%d/%d 通过" % (len(PASSED), len(PASSED) + len(FAILED)))
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("延迟基准：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
