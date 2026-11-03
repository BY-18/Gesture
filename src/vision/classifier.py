"""静态姿态与瞬时动作分类（Plan V1.0 T8 / §3.2、§5.1）。

职责（只判"当下这一帧/这几帧"，不做防抖、不做仲裁）：
- `is_fist()`             → G-02 急停的前提：四指指尖低于其 PIP 关节；
- `palm_facing_camera()`  → G-10 前推的前提：掌心法线朝向镜头；
- `is_forward_push()`     → G-10 前推：掌心朝向 + 手腕 Z 轴连续减小 或 Y 轴上抬。

坐标系：MediaPipe 归一化坐标，y 向下、z 越小越靠近镜头。
因此"指尖低于 PIP"在数值上表现为 `tip.y > pip.y`。
"""

from __future__ import annotations

import math
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas import Landmark
from src.core.config import Thresholds, get_settings

__all__ = [
    "INDEX_TIP",
    "INDEX_PIP",
    "FINGER_TIP_PIP",
    "fist_hits",
    "fist_score",
    "is_fist",
    "hand_size",
    "finger_states",
    "extended_count",
    "is_thumb_up",
    "is_open_palm",
    "is_v_sign",
    "is_pinch",
    "palm_facing_camera",
    "PushResult",
    "push_score",
    "is_forward_push",
    "PushWindow",
]

#: 食指指尖 / PIP 索引（轨迹追踪点与握拳判据共用）
INDEX_TIP = 8
INDEX_PIP = 6

#: 四指 (tip, pip) 索引对：食指、中指、无名指、小指
FINGER_TIP_PIP: tuple[tuple[int, int], ...] = ((8, 6), (12, 10), (16, 14), (20, 18))

#: 掌面定义所需关键点：腕、食指 MCP、小指 MCP
_PALM_TRIPLE = (0, 5, 17)

#: 关键点数量下限
_MIN_LANDMARKS = 21


def _thresholds(thresholds: Thresholds | None) -> Thresholds:
    return thresholds if thresholds is not None else get_settings().thresholds


def _valid(landmarks: Sequence[Landmark] | None) -> bool:
    return bool(landmarks) and len(landmarks) >= _MIN_LANDMARKS  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# G-02：握拳
# --------------------------------------------------------------------------- #
def fist_hits(landmarks: Sequence[Landmark]) -> int:
    """四指中指"尖低于第二关节"的命中数（0~4）。"""
    if not _valid(landmarks):
        return 0
    hits = 0
    for tip, pip in FINGER_TIP_PIP:
        if landmarks[tip].y > landmarks[pip].y:
            hits += 1
    return hits


def fist_score(landmarks: Sequence[Landmark]) -> float:
    """握拳置信度（0~1），供 P2 的置信度加权表决使用。"""
    return fist_hits(landmarks) / len(FINGER_TIP_PIP)


def is_fist(
    landmarks: Sequence[Landmark] | None,
    thresholds: Thresholds | None = None,
) -> bool:
    """G-02 急停前提：四指中至少 ``fist_min_fingers``（默认 4）个指尖低于 PIP。

    2026-11 收紧：默认从 3 指改为 4 指，并要求手掌在画面中足够大。
    此前"随手一放"就会被判成握拳（实测 53 次识别里 45 次是误判急停）。
    """
    if not _valid(landmarks):
        return False
    th = _thresholds(thresholds)
    if fist_hits(landmarks) < th.fist_min_fingers:  # type: ignore[arg-type]
        return False
    if hand_size(landmarks) < th.min_hand_size:  # type: ignore[arg-type]
        return False
    # 排除"竖拇指"：手型上与握拳只差一个拇指，但语义是前进。
    # 判据差异很明确（拇指竖起且远离食指 MCP），普通握拳的拇指是横搭的。
    return not is_thumb_up(landmarks, thresholds)


# --------------------------------------------------------------------------- #
# 静态手型包的地基：手部尺寸与各手指伸展状态
# --------------------------------------------------------------------------- #
def hand_size(landmarks: Sequence[Landmark] | None) -> float:
    """手部包围盒对角线长度（归一化单位）。

    用于剔除"手离镜头太远、关键点几乎重合"导致的误判——那种情况下
    任何指尖/关节的相对比较都不可靠。
    """
    if not _valid(landmarks):
        return 0.0
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def finger_states(landmarks: Sequence[Landmark] | None) -> dict[str, bool]:
    """四指伸展状态：``tip.y < pip.y``（图像 y 向下）视为伸直。"""
    names = ("index", "middle", "ring", "pinky")
    if not _valid(landmarks):
        return {name: False for name in names}
    return {
        name: landmarks[tip].y < landmarks[pip].y
        for name, (tip, pip) in zip(names, FINGER_TIP_PIP)
    }


def thumb_extended(landmarks: Sequence[Landmark] | None) -> bool:
    """拇指伸展：指尖离腕的距离明显大于 IP 关节离腕的距离。"""
    if not _valid(landmarks):
        return False
    wrist, ip, tip = landmarks[0], landmarks[3], landmarks[4]
    d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_ip = math.hypot(ip.x - wrist.x, ip.y - wrist.y)
    return d_tip > d_ip * 1.15


def extended_count(landmarks: Sequence[Landmark] | None) -> int:
    """四指中伸直的根数（0~4），用于"手指计数调速"。"""
    return sum(1 for v in finger_states(landmarks).values() if v)


def is_thumb_up(
    landmarks: Sequence[Landmark] | None,
    thresholds: Thresholds | None = None,
) -> bool:
    """G-03 竖拇指 = 前进：四指蜷起，但拇指明显竖起并远离食指根部。

    ⚠️ 与握拳的分界在最关键处：普通握拳拇指是**横搭**在食指上的，
    指尖不会高出 IP 关节，也不会远离食指 MCP。因此这里要求同时满足
    "竖起高度" 与 "远离食指 MCP" 两个条件，避免把"翘着拇指的握拳"
    误判成前进（那会让急停失灵）。
    """
    th = _thresholds(thresholds)
    if not _valid(landmarks):
        return False
    if hand_size(landmarks) < th.min_hand_size:  # type: ignore[arg-type]
        return False
    if fist_hits(landmarks) < th.thumb_up_fist_fingers:  # type: ignore[arg-type]
        return False
    tip, ip, index_mcp = landmarks[4], landmarks[3], landmarks[5]
    rise = ip.y - tip.y
    if rise < th.thumb_up_min_rise:  # type: ignore[arg-type]
        return False
    away = math.hypot(tip.x - index_mcp.x, tip.y - index_mcp.y)
    return away >= th.thumb_up_min_away  # type: ignore[arg-type]


def is_open_palm(
    landmarks: Sequence[Landmark] | None,
    thresholds: Thresholds | None = None,
) -> bool:
    """G-04 张开五指 = 怠速：四指全伸 + 拇指伸展（捏合不算张开）。"""
    th = _thresholds(thresholds)
    if not _valid(landmarks):
        return False
    if hand_size(landmarks) < th.min_hand_size:  # type: ignore[arg-type]
        return False
    if not (all(finger_states(landmarks).values()) and thumb_extended(landmarks)):
        return False
    return not is_pinch(landmarks, thresholds)


def is_v_sign(
    landmarks: Sequence[Landmark] | None,
    thresholds: Thresholds | None = None,
) -> bool:
    """G-05 剪刀手 = 转圈：食指+中指伸，无名指+小指蜷，拇指不伸。

    要求拇指不伸是为了和"三指/四指张开"区分开。
    """
    th = _thresholds(thresholds)
    if not _valid(landmarks):
        return False
    if hand_size(landmarks) < th.min_hand_size:  # type: ignore[arg-type]
        return False
    states = finger_states(landmarks)
    return (
        states["index"]
        and states["middle"]
        and not states["ring"]
        and not states["pinky"]
        and not thumb_extended(landmarks)
    )


def is_pinch(
    landmarks: Sequence[Landmark] | None,
    thresholds: Thresholds | None = None,
) -> bool:
    """G-06 拇指食指捏合 = 急停（备用急停手势，手小或握不紧拳时更好做）。

    判据是"两指尖距离 / 手部尺寸"，与镜头远近无关。
    """
    th = _thresholds(thresholds)
    if not _valid(landmarks):
        return False
    size = hand_size(landmarks)
    if size < th.min_hand_size:  # type: ignore[arg-type]
        return False
    thumb_tip, index_tip = landmarks[4], landmarks[8]
    gap = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return (gap / size) <= th.pinch_ratio  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# G-10 前提：掌心朝向镜头
# --------------------------------------------------------------------------- #
def palm_facing_camera(landmarks: Sequence[Landmark] | None) -> float:
    """掌心正对镜头的程度（0~1，1 表示完全正对）。

    由 腕(0) / 食指 MCP(5) / 小指 MCP(17) 三点构成掌面，
    取其法向量的 z 分量绝对值归一化：掌面平行于像平面 → 法线沿光轴 → 接近 1；
    手掌侧立（只看到手背刀面）→ 法线落在像平面内 → 接近 0。
    """
    if not _valid(landmarks):
        return 0.0
    wrist, index_mcp, pinky_mcp = (landmarks[i] for i in _PALM_TRIPLE)
    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return 0.0  # 三点共线，掌面不可定义
    return min(1.0, abs(nz) / norm)


# --------------------------------------------------------------------------- #
# G-10：前推
# --------------------------------------------------------------------------- #
class PushResult(BaseModel):
    detected: bool = False
    score: float = 0.0
    dz_total: float = 0.0
    dy_total: float = 0.0
    palm_facing: float = 0.0
    frames: int = 0
    reason: str | None = None


class PushWindow(BaseModel):
    """手腕/掌面历史窗口（容量 = 判定所需帧数 + 1）。"""

    model_config = ConfigDict(validate_assignment=False)

    capacity: int = Field(default=4, ge=2)
    frames: list[list[Landmark]] = Field(default_factory=list)

    def push(self, landmarks: Sequence[Landmark]) -> None:
        self.frames.append(list(landmarks))
        overflow = len(self.frames) - self.capacity
        if overflow > 0:
            del self.frames[:overflow]

    def clear(self) -> None:
        self.frames.clear()

    @property
    def size(self) -> int:
        return len(self.frames)


def push_score(
    history: Sequence[Sequence[Landmark]],
    thresholds: Thresholds | None = None,
) -> PushResult:
    """基于最近若干帧的手腕位移评估"前推"程度。

    判据（Plan §5.1）：
    - 掌心朝向镜头 ≥ ``palm_facing_min``；
    - 满足其一：手腕 z 每帧减小 ≥ ``push_delta_z_per_frame``（连续 3 帧），
      或窗口内手腕 y 上抬 ≥ ``push_min_delta_y``（y 减小表示上抬）。
    """
    th = _thresholds(thresholds)
    result = PushResult(frames=len(history))
    if len(history) < 2:
        result.reason = "样本不足"
        return result

    latest = history[-1]
    if not _valid(latest):
        result.reason = "关键点缺失"
        return result

    need = max(2, th.push_consecutive_frames)
    recent = list(history)[-need:]
    if len(recent) < 2:
        result.reason = "样本不足"
        return result

    facing = palm_facing_camera(latest)
    result.palm_facing = round(facing, 4)

    wrist_z = [frame[0].z for frame in recent]
    wrist_y = [frame[0].y for frame in recent]
    dz_total = wrist_z[-1] - wrist_z[0]
    dy_total = wrist_y[-1] - wrist_y[0]
    result.dz_total = round(dz_total, 4)
    result.dy_total = round(dy_total, 4)

    # z 方向：逐帧均满足"靠近镜头"的阈值
    deltas = [wrist_z[i + 1] - wrist_z[i] for i in range(len(wrist_z) - 1)]
    z_push = len(deltas) >= need - 1 and all(
        d <= -th.push_delta_z_per_frame for d in deltas
    )
    y_push = -dy_total >= th.push_min_delta_y

    if facing < th.palm_facing_min:
        result.reason = f"掌心未正对镜头（{facing:.2f} < {th.palm_facing_min:.2f}）"
        result.score = round(0.3 * facing, 4)
        return result

    if z_push or y_push:
        result.detected = True
        result.reason = "ok(z)" if z_push else "ok(y)"
        result.score = 1.0 if (z_push and facing >= 0.9) else 0.9
        return result

    result.reason = "手腕无前推位移"
    result.score = round(min(0.5, abs(dz_total) / max(1e-6, th.push_delta_z_per_frame * need)), 4)
    return result


def is_forward_push(
    history: Sequence[Sequence[Landmark]],
    thresholds: Thresholds | None = None,
) -> tuple[bool, float]:
    """便利封装：返回 (是否前推, 置信度)。"""
    result = push_score(history, thresholds)
    return result.detected, result.score
