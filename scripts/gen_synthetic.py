"""合成数据集生成器（Plan V1.0 T14 / §11.1）。

生成三类**归一化 21 点 + 时间戳**序列，落盘为 JSONL，供 T15/T16 离线复算：

======================  ==========================================  ============
文件                     内容                                        规模
======================  ==========================================  ============
``standard.jsonl``      五类手势正例（G-02/10/11/12/13）             各 100 组
``hard_negative.jsonl`` 八字形 / S 形 / 往复摆动 / 半圆不足 330°    各 100 组
``static_noise.jsonl``  静止微抖 500 帧 + 强随机抖动 500 帧          1000 帧
======================  ==========================================  ============

记录格式::

    {"id": "G-11-000", "label": "G-11", "kind": "positive",
     "noise_ratio": 0.05, "fps": 30,
     "frames": [{"ts": 0.0, "landmarks": [[x, y, z] × 21]}, ...]}

噪声口径：正例噪声 **幅度 = 单帧位移的 5%**（`noise_ratio=0.05`）。以相对量定义，
避免"大幅快挥被噪声淹没 / 小幅慢转被噪声扭曲"的尺度不一致问题。

用法::

    python scripts/gen_synthetic.py            # 生成到 logs/dataset/
    python scripts/gen_synthetic.py --force    # 强制重建
    python scripts/gen_synthetic.py --out D:\\data
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.injector import POSE_LIBRARY  # noqa: E402
from src.api.schemas import GestureCode, Landmark  # noqa: E402

__all__ = [
    "DATASET_DIR",
    "REPORT_PATH",
    "DATASET_VERSION",
    "make_fist_sequence",
    "make_push_sequence",
    "make_swipe_sequence",
    "make_circle_sequence",
    "make_figure_eight",
    "make_s_shape",
    "make_reciprocal",
    "make_half_circle",
    "make_static_jitter",
    "make_random_jitter",
    "generate",
    "ensure_dataset",
    "load_records",
    "record_landmarks",
    "to_landmarks",
    "update_report",
]

DATASET_DIR = REPO_ROOT / "logs" / "dataset"
REPORT_PATH = REPO_ROOT / "logs" / "eval_report.md"

#: 数据集版本（口径变更时递增，避免旧数据混入评测）
DATASET_VERSION = 3
MANIFEST_NAME = "manifest.json"

FPS = 30.0
FRAME_INTERVAL = 1.0 / FPS

#: 噪声幅度 = 单帧位移 × NOISE_RATIO
NOISE_RATIO = 0.05

#: 静止微抖的标准差（归一化坐标）
STATIC_JITTER_SIGMA = 0.0018

BASE_OPEN: list[Landmark] = list(POSE_LIBRARY[GestureCode.FORWARD].landmarks)
BASE_FIST: list[Landmark] = list(POSE_LIBRARY[GestureCode.ESTOP].landmarks)
TIP_INDEX = 8
WRIST_INDEX = 0


# --------------------------------------------------------------------------- #
# 基础变换
# --------------------------------------------------------------------------- #
def translate(
    points: Sequence[Landmark], dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
) -> list[Landmark]:
    return [Landmark(x=p.x + dx, y=p.y + dy, z=p.z + dz) for p in points]


def jitter(
    points: Sequence[Landmark], sigma: float, rng: random.Random
) -> list[Landmark]:
    return [
        Landmark(
            x=p.x + rng.gauss(0.0, sigma),
            y=p.y + rng.gauss(0.0, sigma),
            z=p.z + rng.gauss(0.0, sigma * 0.5),
        )
        for p in points
    ]


def aim_tip(points: Sequence[Landmark], tx: float, ty: float) -> list[Landmark]:
    tip = points[TIP_INDEX]
    return translate(points, tx - tip.x, ty - tip.y, 0.0)


def rotate_y(points: Sequence[Landmark], degrees: float) -> list[Landmark]:
    """绕腕关节做 Y 轴旋转（把"掌面朝镜头"转成"手侧立"）。"""
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    wrist = points[WRIST_INDEX]
    return [
        Landmark(
            x=wrist.x + (p.x - wrist.x) * cos_t + (p.z - wrist.z) * sin_t,
            y=p.y,
            z=wrist.z - (p.x - wrist.x) * sin_t + (p.z - wrist.z) * cos_t,
        )
        for p in points
    ]


def _step_sigma(step: float) -> float:
    """噪声标准差 = 单帧位移 × 5%。"""
    return max(1e-4, NOISE_RATIO * abs(step))


# --------------------------------------------------------------------------- #
# 正例生成器
# --------------------------------------------------------------------------- #
def make_fist_sequence(rng: random.Random, frames: int | None = None) -> list[list[Landmark]]:
    """G-02 握拳：3~4 帧静止拳头（静态姿态，噪声取手部尺度 5%）。"""
    count = frames or rng.choice([3, 3, 4])
    sigma = _step_sigma(0.05)
    return [jitter(BASE_FIST, sigma, rng) for _ in range(count)]


def make_push_sequence(
    rng: random.Random, frames: int | None = None
) -> list[list[Landmark]]:
    """G-10 前推：掌心朝镜头 + Z 轴前推 或 Y 轴上抬（Plan §5.1）。

    ``frames`` 默认 4~6；注意**触发**所需的最小长度是 5 帧（3 帧几何 + 3 帧表决，
    有 1 帧重叠），4 帧的极端短推只能产出候选、无法完成表决，这是 §2.2 的既有代价。
    """
    count = frames or rng.choice([4, 5, 6])
    use_z = rng.random() < 0.5
    step_z = rng.uniform(0.025, 0.045) if use_z else 0.0
    step_y = rng.uniform(0.055, 0.09) if not use_z else rng.uniform(-0.004, 0.004)
    sigma = _step_sigma(max(step_z, step_y))
    sequence: list[list[Landmark]] = []
    for index in range(count):
        pose = translate(BASE_OPEN, 0.0, -step_y * index, -step_z * index)
        sequence.append(jitter(pose, sigma, rng))
    return sequence


def make_swipe_sequence(
    rng: random.Random, *, direction: str | None = None, fast: bool = True
) -> tuple[list[list[Landmark]], GestureCode]:
    """G-11/G-12 挥动。

    ``fast=True`` 为快挥（单帧位移 ≥0.12，第 2 帧即出几何候选），对应 Plan §4.2 的
    120 ms 口径；``fast=False`` 为慢挥（0.022~0.032/帧），几何累积会额外占用 2~4 帧。
    """
    sign = 1.0
    if direction == "left":
        sign = -1.0
    elif direction is None:
        sign = 1.0 if rng.random() < 0.5 else -1.0

    if fast:
        frames = rng.choice([5, 6])
        step = rng.uniform(0.125, 0.16)
    else:
        frames = rng.choice([6, 7, 8])
        step = rng.uniform(0.022, 0.032)

    drift = rng.uniform(-0.15, 0.15) * step
    sigma = _step_sigma(step)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        pose = translate(BASE_OPEN, sign * step * index, drift * index, 0.0)
        sequence.append(jitter(pose, sigma, rng))

    code = GestureCode.TURN_RIGHT if sign > 0 else GestureCode.TURN_LEFT
    return sequence, code


def make_circle_sequence(
    rng: random.Random, *, clockwise: bool | None = None
) -> tuple[list[list[Landmark]], GestureCode]:
    """G-13 转圈：20~23 帧画满约一圈（闭环，累计转角 ≥330°）。"""
    frames = rng.randint(20, 23)
    sweep = rng.uniform(358.0, 366.0)
    radius = rng.uniform(0.15, 0.20)
    start = rng.uniform(0.0, 2.0 * math.pi)
    if clockwise is None:
        clockwise = rng.random() < 0.5
    step = 2.0 * math.pi * radius / frames
    sigma = _step_sigma(step)
    center = (0.5, 0.52)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        angle = start + math.radians(sweep) * (index / (frames - 1)) * (-1 if clockwise else 1)
        tx = center[0] + radius * math.cos(angle)
        ty = center[1] + radius * math.sin(angle)
        sequence.append(jitter(aim_tip(BASE_OPEN, tx, ty), sigma, rng))
    return sequence, GestureCode.SPIN


# --------------------------------------------------------------------------- #
# 硬负例生成器
# --------------------------------------------------------------------------- #
def make_figure_eight(rng: random.Random) -> list[list[Landmark]]:
    """八字形：转角很大但旋向反复，不得判为转圈。"""
    frames = rng.choice([26, 28, 30])
    radius = rng.uniform(0.15, 0.20)
    start = rng.uniform(0.0, 2.0 * math.pi)
    step = 2.0 * math.pi * radius / frames
    sigma = _step_sigma(step)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        t = start + 2.0 * math.pi * (index / (frames - 1))
        tx = 0.5 + radius * math.sin(t)
        ty = 0.52 + radius * 0.7 * math.sin(2.0 * t)
        sequence.append(jitter(aim_tip(BASE_OPEN, tx, ty), sigma, rng))
    return sequence


def make_s_shape(rng: random.Random) -> list[list[Landmark]]:
    """S 形：两段反向圆弧，旋向一致性不足。"""
    frames = rng.choice([20, 22, 24])
    radius = rng.uniform(0.14, 0.18)
    start = rng.uniform(-0.6, 0.6)
    step = 2.0 * math.pi * radius / frames
    sigma = _step_sigma(step)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        frac = index / (frames - 1)
        t = start + 1.3 * math.pi * frac
        tx = 0.5 + radius * math.sin(t)
        ty = 0.52 + radius * 0.55 * math.sin(2.0 * t)
        sequence.append(jitter(aim_tip(BASE_OPEN, tx, ty), sigma, rng))
    return sequence


def make_reciprocal(rng: random.Random) -> list[list[Landmark]]:
    """往复摆动：折返往复，既非单次挥动也不构成闭环。"""
    frames = rng.choice([8, 10, 12])
    amplitude = rng.uniform(0.14, 0.20)
    step = 2.0 * amplitude / frames
    sigma = _step_sigma(step)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        frac = index / (frames - 1)
        offset = amplitude * math.sin(2.0 * math.pi * frac)
        sequence.append(jitter(translate(BASE_OPEN, offset, 0.0, 0.0), sigma, rng))
    return sequence


def make_half_circle(rng: random.Random) -> list[list[Landmark]]:
    """半圆：累计转角约 175°，远不足 330°。"""
    frames = rng.choice([14, 16, 18])
    radius = rng.uniform(0.15, 0.20)
    start = rng.uniform(0.0, 2.0 * math.pi)
    step = math.pi * radius / frames
    sigma = _step_sigma(step)
    sequence: list[list[Landmark]] = []
    for index in range(frames):
        angle = start + math.radians(175.0) * (index / (frames - 1))
        tx = 0.5 + radius * math.cos(angle)
        ty = 0.52 + radius * math.sin(angle)
        sequence.append(jitter(aim_tip(BASE_OPEN, tx, ty), sigma, rng))
    return sequence


# --------------------------------------------------------------------------- #
# 静止 / 噪声
# --------------------------------------------------------------------------- #
def make_static_jitter(rng: random.Random, frames: int = 500) -> list[list[Landmark]]:
    """静止微抖：手保持张开不动，只有传感器级抖动（σ=0.0018）。"""
    return [jitter(BASE_OPEN, STATIC_JITTER_SIGMA, rng) for _ in range(frames)]


def make_random_jitter(rng: random.Random, frames: int = 500) -> list[list[Landmark]]:
    """强随机抖动：**保留手部结构**，关键点独立抖动（σ=0.012）+ 10% 离群点。

    为什么不生成"21 个互不相关的随机点"：那不是一只手，MediaPipe 永远不会输出
    这种结构（它只输出检测到的手）。用点云做负例会误伤所有基于手部几何的判据，
    得到的"误触发率"没有工程意义。这里的噪声更接近真实：检测抖动 + 偶发跳变。
    """
    sequence: list[list[Landmark]] = []
    for _ in range(frames):
        frame: list[Landmark] = []
        for point in BASE_OPEN:
            x = point.x + rng.gauss(0.0, 0.012)
            y = point.y + rng.gauss(0.0, 0.012)
            z = point.z + rng.gauss(0.0, 0.006)
            if rng.random() < 0.10:  # 偶发离群点（遮挡/跟踪跳变）
                x += rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.05)
                y += rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.05)
            frame.append(Landmark(x=min(1.0, max(0.0, x)), y=min(1.0, max(0.0, y)), z=z))
        sequence.append(frame)
    return sequence


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    id: str
    label: str
    kind: str
    frames: list[list[Landmark]]

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "noise_ratio": NOISE_RATIO,
            "fps": FPS,
            "frame_count": len(self.frames),
            "frames": [
                {
                    "ts": round(index * FRAME_INTERVAL, 6),
                    "landmarks": [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in lms],
                }
                for index, lms in enumerate(self.frames)
            ],
        }


def _write_jsonl(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def generate(out_dir: Path | None = None, *, seed: int = 20261101, force: bool = True) -> dict[str, Any]:
    """生成三类数据集，返回 manifest。"""
    target = out_dir or DATASET_DIR
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # ① 正例：五类各 100 组
    standard: list[Record] = []
    for index in range(100):
        standard.append(Record(f"G-02-{index:03d}", "G-02", "positive", make_fist_sequence(rng)))
    for index in range(100):
        standard.append(Record(f"G-10-{index:03d}", "G-10", "positive", make_push_sequence(rng)))
    for index in range(100):
        sequence, code = make_swipe_sequence(rng, direction="left", fast=False)
        standard.append(Record(f"G-11-{index:03d}", code.value, "positive", sequence))
    for index in range(100):
        sequence, code = make_swipe_sequence(rng, direction="right", fast=False)
        standard.append(Record(f"G-12-{index:03d}", code.value, "positive", sequence))
    for index in range(100):
        sequence, _ = make_circle_sequence(rng, clockwise=(index % 2 == 0))
        standard.append(Record(f"G-13-{index:03d}", "G-13", "positive", sequence))

    # ② 硬负例：四类各 100 组
    negatives: list[Record] = []
    for name, generator in (
        ("figure_eight", make_figure_eight),
        ("s_shape", make_s_shape),
        ("reciprocal", make_reciprocal),
        ("half_circle", make_half_circle),
    ):
        for index in range(100):
            negatives.append(
                Record(f"NEG-{name}-{index:03d}", name, "hard_negative", generator(rng))
            )

    # ③ 静止 / 噪声：合计 1000 帧
    noise: list[Record] = [
        Record("NOISE-static-000", "static_jitter", "noise", make_static_jitter(rng, 500)),
        Record("NOISE-random-000", "random_jitter", "noise", make_random_jitter(rng, 500)),
    ]

    paths = {
        "standard": target / "standard.jsonl",
        "hard_negative": target / "hard_negative.jsonl",
        "static_noise": target / "static_noise.jsonl",
    }
    counts = {
        "standard": _write_jsonl(paths["standard"], standard),
        "hard_negative": _write_jsonl(paths["hard_negative"], negatives),
        "static_noise": _write_jsonl(paths["static_noise"], noise),
    }

    manifest = {
        "version": DATASET_VERSION,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fps": FPS,
        "noise_ratio": NOISE_RATIO,
        "counts": counts,
        "bytes": {name: path.stat().st_size for name, path in paths.items()},
        "labels": {
            "positive": sorted({record.label for record in standard}),
            "hard_negative": sorted({record.label for record in negatives}),
            "noise": sorted({record.label for record in noise}),
        },
        "spec": {
            "standard": "G-02/G-10/G-11/G-12/G-13 各 100 组，噪声 = 单帧位移 5%",
            "hard_negative": "八字形/S 形/往复摆动/半圆不足 330°，各 100 组",
            "static_noise": "静止微抖 500 帧 + 强随机抖动 500 帧（保留手部结构）",
        },
    }
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def ensure_dataset(*, seed: int = 20261101, force: bool = False) -> dict[str, Any]:
    """数据集缺失或版本不符时自动重建。"""
    manifest_path = DATASET_DIR / MANIFEST_NAME
    if not force and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("version") == DATASET_VERSION and all(
            (DATASET_DIR / f"{name}.jsonl").exists()
            for name in ("standard", "hard_negative", "static_noise")
        ):
            return manifest
    return generate(force=True, seed=seed)


def load_records(kind: str) -> list[dict[str, Any]]:
    """读取指定类别数据集（自动确保已生成）。"""
    ensure_dataset()
    path = DATASET_DIR / f"{kind}.jsonl"
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_landmarks(frame: dict[str, Any]) -> list[Landmark]:
    """把数据集里的一帧转成 `Landmark` 列表。"""
    return [Landmark(x=float(x), y=float(y), z=float(z)) for x, y, z in frame["landmarks"]]


def record_landmarks(record: dict[str, Any]) -> list[list[Landmark]]:
    return [to_landmarks(frame) for frame in record["frames"]]


# --------------------------------------------------------------------------- #
# 评测报告（多个脚本共同写 logs/eval_report.md）
# --------------------------------------------------------------------------- #
_REPORT_HEADER = "# Gesture-Copilot 评测报告"


def update_report(section_key: str, title: str, body: str) -> Path:
    """写入/替换评测报告中的一个章节（幂等，多个脚本可各自写自己的章节）。"""
    sections: dict[str, tuple[str, str]] = {}
    if REPORT_PATH.exists():
        current: tuple[str, str] | None = None
        buffer: list[str] = []
        for line in REPORT_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if current is not None:
                    sections[current[0]] = (current[1], "\n".join(buffer).strip())
                heading = line[3:].strip()
                current = (heading, heading)
                buffer = []
            elif current is not None:
                buffer.append(line)
        if current is not None:
            sections[current[0]] = (current[1], "\n".join(buffer).strip())

    # 同名旧章节（历史无前缀键）以新内容替换，避免报告里累积重复段落
    sections = {key: value for key, value in sections.items() if value[0] != title}
    sections[f"{section_key}|{title}"] = (title, body)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = [sections[key] for key in sorted(sections)]
    content = [
        _REPORT_HEADER,
        "",
        f"> 数据集版本 v{DATASET_VERSION} · 更新于 {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]
    for heading, section_body in ordered:
        content.append(f"## {heading}")
        content.append("")
        content.append(section_body)
        content.append("")
    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")
    return REPORT_PATH


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 Gesture-Copilot 合成评测数据集")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认 logs/dataset）")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--force", action="store_true", help="强制重建")
    args = parser.parse_args(argv)

    if args.force or args.out is not None:
        manifest = generate(args.out, seed=args.seed, force=True)
    else:
        manifest = ensure_dataset(seed=args.seed)

    print("=" * 74)
    print("Gesture-Copilot · 合成数据集生成（T14）")
    print("=" * 74)
    print(
        f"版本 v{manifest['version']} · seed={manifest['seed']} · "
        f"噪声 = 单帧位移 {manifest['noise_ratio']:.0%}"
    )
    print(f"输出目录：{args.out or DATASET_DIR}")
    print("-" * 74)
    for name, count in manifest["counts"].items():
        size_mb = manifest["bytes"][name] / 1024 / 1024
        print(f"  {name:<14} {count:>4} 条  {size_mb:>6.2f} MB")
    print("-" * 74)
    print(f"正例标签：{', '.join(manifest['labels']['positive'])}")
    print(f"硬负例  ：{', '.join(manifest['labels']['hard_negative'])}")
    print(f"噪声序列：{', '.join(manifest['labels']['noise'])}")
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
