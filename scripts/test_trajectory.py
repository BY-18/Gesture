"""算法识别率与硬负例评测（Plan V1.0 T15 / §11.2，门禁 G1）。

数据来源：`scripts/gen_synthetic.py` 生成的 `logs/dataset/*.jsonl`（版本化）。
评测对象：**完整感知流水线** `FramePipeline`（几何判定 + 候选优先级仲裁），
而不是孤立的单函数，因此也覆盖了"转圈过程被误判成转向"这类串扰。

评测口径：
- 正例：某记录在任意一帧产出期望手势码，即记为检出；
- 硬负例：八字形 / S 形 / 往复摆动 / 半圆，**不得产出 G-13**；
- 门槛：五类识别率各 ≥ 90%，硬负例误判 ≤ 2%。

用法：``.venv\\Scripts\\python.exe scripts\\test_trajectory.py``
结果表格写入 ``logs/eval_report.md``；退出码 0 表示门禁 G1 通过。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
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
from src.api.schemas import GestureCode  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.vision.pipeline import FramePipeline  # noqa: E402

MIN_DETECTION_RATE = 0.90
MAX_HARD_NEGATIVE_RATE = 0.02
FRAME_INTERVAL = 1.0 / 30.0

GESTURE_NAMES: dict[str, str] = {
    "G-02": "握拳（急停）",
    "G-10": "前推",
    "G-11": "左挥",
    "G-12": "右挥",
    "G-13": "转圈",
}

PASSED: list[str] = []
FAILED: list[str] = []


@dataclass
class GestureStat:
    label: str
    total: int = 0
    detected: int = 0
    trigger_frames: list[int] = field(default_factory=list)
    mismatch: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.detected / self.total if self.total else 1.0

    @property
    def mean_frames(self) -> float:
        return sum(self.trigger_frames) / len(self.trigger_frames) if self.trigger_frames else 0.0


def check(name: str, ok: bool, detail: str = "") -> bool:
    if ok:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
    return ok


def run_record(record: dict) -> tuple[list[tuple[int, str]], list[str]]:
    """跑一条记录，返回 [(帧号, 手势码)] 与逐帧码序列。"""
    pipeline = FramePipeline()
    hits: list[tuple[int, str]] = []
    codes: list[str] = []
    for index, landmarks in enumerate(record_landmarks(record)):
        frame = pipeline.ingest(landmarks, ts=index * FRAME_INTERVAL, frame_id=index + 1)
        if frame.candidate is not None:
            code = frame.candidate.code.value
            codes.append(code)
            hits.append((index, code))
        else:
            codes.append("")
    return hits, codes


def main() -> int:
    settings = get_settings()
    th = settings.thresholds

    print("=" * 78)
    print("Gesture-Copilot · 门禁 G1（算法识别率与硬负例评测）")
    print("=" * 78)

    standard = load_records("standard")
    hard_negative = load_records("hard_negative")
    print(f"数据集 v{DATASET_VERSION} · 正例 {len(standard)} 条 · 硬负例 {len(hard_negative)} 条")
    print(f"阈值基线：circle ≥{th.circle_min_cum_angle}° / {th.circle_min_frames}~{th.circle_max_frames} 帧 · "
          f"swipe ≥{th.swipe_min_displacement} / {th.swipe_max_frames} 帧")

    # ---------------- ① 正例识别率 ---------------- #
    stats: dict[str, GestureStat] = defaultdict(lambda: GestureStat(label=""))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print("\n[1] 正例识别率（五类手势各 100 组）")
    for record in standard:
        label = record["label"]
        stat = stats[label]
        stat.label = label
        stat.total += 1
        hits, codes = run_record(record)
        present = {code for _, code in hits}
        if label in present:
            stat.detected += 1
            first = next(index for index, code in hits if code == label)
            stat.trigger_frames.append(first + 1)
        else:
            stat.mismatch[label] = stat.mismatch.get(label, 0) + 1
        for code in present:
            if code != label:
                confusion[label][code] += 1

    table_rows: list[str] = []
    print(f"  {'手势':<14}{'样本':>6}{'检出':>6}{'识别率':>9}{'均触发帧':>10}")
    print("  " + "-" * 48)
    for label in ("G-02", "G-10", "G-11", "G-12", "G-13"):
        stat = stats[label]
        print(
            f"  {label} {GESTURE_NAMES[label]:<8}{stat.total:>6}{stat.detected:>6}"
            f"{stat.rate * 100:>8.1f}%{stat.mean_frames:>10.1f}"
        )
        table_rows.append(
            f"| {label} {GESTURE_NAMES[label]} | {stat.total} | {stat.detected} | "
            f"{stat.rate * 100:.1f}% | {stat.mean_frames:.1f} | "
        )
    print()
    for row, label in zip(table_rows, ("G-02", "G-10", "G-11", "G-12", "G-13")):
        stat = stats[label]
        ok = check(
            f"{label} {GESTURE_NAMES[label]} 识别率 ≥ {MIN_DETECTION_RATE:.0%}",
            stat.rate >= MIN_DETECTION_RATE,
            f"{stat.rate * 100:.1f}%（{stat.detected}/{stat.total}）",
        )
        table_rows[table_rows.index(row)] = row + f"{'PASS' if ok else 'FAIL'} |"

    # ---------------- ② 硬负例 ---------------- #
    print("\n[2] 硬负例（不得误判为转圈）")
    neg_stats: dict[str, int] = defaultdict(int)
    neg_total: dict[str, int] = defaultdict(int)
    for record in hard_negative:
        kind = record["label"]
        neg_total[kind] += 1
        hits, _ = run_record(record)
        if any(code == "G-13" for _, code in hits):
            neg_stats[kind] += 1
        if any(code in ("G-11", "G-12") for _, code in hits):
            neg_stats[f"{kind}/swipe"] += 1

    neg_rows: list[str] = []
    print(f"  {'硬负例':<16}{'样本':>6}{'误判转圈':>10}{'误判挥动':>10}")
    print("  " + "-" * 44)
    for kind in ("figure_eight", "s_shape", "reciprocal", "half_circle"):
        total = neg_total[kind]
        circle_hits = neg_stats[kind]
        swipe_hits = neg_stats[f"{kind}/swipe"]
        print(f"  {kind:<16}{total:>6}{circle_hits:>10}{swipe_hits:>10}")
        rate = circle_hits / total if total else 0.0
        neg_rows.append(
            f"| {kind} | {total} | {circle_hits} | {rate * 100:.1f}% | {swipe_hits} |"
        )
    print()
    for row, kind in zip(neg_rows, ("figure_eight", "s_shape", "reciprocal", "half_circle")):
        total = neg_total[kind]
        rate = neg_stats[kind] / total if total else 0.0
        ok = check(
            f"硬负例 {kind} 不误判为转圈 ≤ {MAX_HARD_NEGATIVE_RATE:.0%}",
            rate <= MAX_HARD_NEGATIVE_RATE,
            f"{rate * 100:.1f}%（{neg_stats[kind]}/{total}）",
        )
        neg_rows[neg_rows.index(row)] = row + f" {'PASS' if ok else 'FAIL'} |"

    trajectory_total = stats["G-11"].total + stats["G-12"].total + stats["G-13"].total
    trajectory_detected = stats["G-11"].detected + stats["G-12"].detected + stats["G-13"].detected
    trajectory_rate = trajectory_detected / trajectory_total if trajectory_total else 0.0
    check(
        "轨迹类（G-11/12/13）合并识别率 ≥ 90%",
        trajectory_rate >= MIN_DETECTION_RATE,
        f"{trajectory_rate * 100:.1f}%（{trajectory_detected}/{trajectory_total}）",
    )

    circle = stats["G-13"]
    if circle.trigger_frames:
        check(
            f"转圈触发帧数落在 {th.circle_min_frames}~{th.circle_max_frames} 帧窗口",
            min(circle.trigger_frames) >= th.circle_min_frames
            and max(circle.trigger_frames) <= th.circle_max_frames,
            f"{min(circle.trigger_frames)}~{max(circle.trigger_frames)} 帧",
        )

    # ---------------- ③ 交叉触发矩阵（信息项） ---------------- #
    print("\n[3] 交叉触发矩阵（信息项，互斥仲裁见 §15.5）")
    codes = ["G-02", "G-10", "G-11", "G-12", "G-13"]
    print("  源\\判  " + "".join(f"{code:>9}" for code in codes))
    for label in codes:
        stat = stats[label]
        total = max(1, stat.total)
        cells = []
        for code in codes:
            hits = stat.detected if code == label else confusion[label].get(code, 0)
            cells.append(f"{hits / total * 100:>8.0f}%")
        print(f"  {label:<6}" + "".join(cells))

    # ---------------- ④ 写报告 ---------------- #
    body = [
        f"数据集：`logs/dataset/standard.jsonl`（v{DATASET_VERSION}，seed 固定），"
        f"噪声 = 单帧位移 5%。评测对象为完整 `FramePipeline`。",
        "",
        "| 手势 | 样本 | 检出 | 识别率 | 平均触发帧 | 结论 |",
        "|---|---|---|---|---|---|",
        *table_rows,
        "",
        f"轨迹类合并识别率：**{trajectory_rate * 100:.1f}%**（门槛 90%）",
        "",
        "| 硬负例 | 样本 | 误判为转圈 | 误判率 | 误判为挥动 | 结论 |",
        "|---|---|---|---|---|---|",
        *neg_rows,
    ]
    report = update_report("01_recognition", "五类手势识别率（scripts/test_trajectory.py）", "\n".join(body))

    print("\n" + "-" * 78)
    print(f"评测报告已写入：{report}")
    print("结果：%d/%d 通过" % (len(PASSED), len(PASSED) + len(FAILED)))
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        print("门禁 G1：FAIL")
        return 1
    print("门禁 G1：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
