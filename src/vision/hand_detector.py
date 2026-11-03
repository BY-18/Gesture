"""MediaPipe 手部关键点封装（Plan V1.0 T7 / §3.1、§2.6）。

对外只暴露"一帧一只手"的 `HandFrame`：
- 每帧最多取 1 只手，规则见 `select_hand()`（Plan §2.6）；
- 支持从**合成坐标**构造 `HandFrame`（离线评测 / 虚拟注入 / 无摄像头环境）；
- MediaPipe 不可用时 `available` 为 False，`detect()` 返回 None，绝不抛异常阻断链路。

环境说明（Plan §8.1 决策点 D1）：本机为 Python 3.12 + mediapipe 0.10.21，
使用 **legacy solutions API**（`mp.solutions.hands`）。
MediaPipe 的 `handedness` 基于"自拍镜像"假设：画面中左手 → 标签 "Right"，
故本模块的 `handedness` 字段直接透传原始标签，不做镜像翻转，由上层按需解释。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.api.schemas import Landmark
from src.core.config import get_settings

__all__ = [
    "HandFrame",
    "HandDetector",
    "select_hand",
    "landmarks_from_mapping",
    "HAND_LANDMARK_COUNT",
    "hand_center",
]

#: MediaPipe Hands 关键点数量
HAND_LANDMARK_COUNT = 21

#: 腕关节索引（用于"距画面中心最近"的选手指标）
WRIST = 0

#: 分数并列判定阈值（§2.6：置信度相同时按距画面中心取近）
_SCORE_TIE_EPS = 1e-6


@dataclass(frozen=True)
class HandFrame:
    """单帧单手快照。"""

    landmarks: tuple[Landmark, ...]
    handedness: str = "Unknown"
    score: float = 1.0
    ts: float = field(default_factory=time.time)
    frame_id: int | None = None
    source: str = "mediapipe"  # mediapipe | synthetic

    def as_list(self) -> list[Landmark]:
        return list(self.landmarks)

    @property
    def valid(self) -> bool:
        return len(self.landmarks) == HAND_LANDMARK_COUNT

    @property
    def is_right(self) -> bool:
        return self.handedness.lower() == "right"

    def wrist(self) -> Landmark:
        return self.landmarks[WRIST]

    def tip(self, index: int = 8) -> Landmark:
        return self.landmarks[index]


def hand_center(hand: HandFrame) -> tuple[float, float]:
    """手部几何中心（取全部关键点均值），用于"距画面中心最近"判定。"""
    if not hand.landmarks:
        return (0.5, 0.5)
    xs = sum(p.x for p in hand.landmarks) / len(hand.landmarks)
    ys = sum(p.y for p in hand.landmarks) / len(hand.landmarks)
    return xs, ys


def _center_distance(hand: HandFrame) -> float:
    if not hand.landmarks:
        return float("inf")
    wrist = hand.landmarks[WRIST]
    return math.hypot(wrist.x - 0.5, wrist.y - 0.5)


def select_hand(hands: Sequence[HandFrame]) -> HandFrame | None:
    """按 Plan §2.6 选出唯一一只手。

    规则：**置信度最高者优先**；置信度并列（差 < 1e-6）时取**距画面中心最近**者。
    """
    if not hands:
        return None
    if len(hands) == 1:
        return hands[0]
    ordered = sorted(hands, key=lambda h: (-round(h.score, 6), _center_distance(h)))
    best = ordered[0]
    tied = [
        h for h in ordered if abs(round(h.score, 6) - round(best.score, 6)) < _SCORE_TIE_EPS
    ]
    if len(tied) > 1:
        return min(tied, key=_center_distance)
    return best


# --------------------------------------------------------------------------- #
# 合成坐标 → Landmark
# --------------------------------------------------------------------------- #
def landmarks_from_mapping(
    payload: Mapping[int, Any] | Sequence[Any] | Iterable[Landmark],
) -> list[Landmark]:
    """把多种输入形态统一成 21 个 `Landmark`。

    支持的形态：
    - ``list[Landmark]``
    - ``list[dict]`` / ``list[tuple(x, y, z)]``（按顺序 21 个）
    - ``dict[int, dict]`` / ``dict[str, dict]``（索引 → 坐标）
    - ``dict`` 形如 ``{"landmarks": [...]}``（兼容注入报文）
    """
    if isinstance(payload, Mapping):
        if "landmarks" in payload and isinstance(payload["landmarks"], (list, tuple)):
            return landmarks_from_mapping(payload["landmarks"])  # type: ignore[arg-type]
        items: list[tuple[int, Any]] = []
        for key, value in payload.items():
            try:
                items.append((int(key), value))
            except (TypeError, ValueError):
                continue
        items.sort(key=lambda kv: kv[0])
        return [_to_landmark(value) for _, value in items]

    result = [_to_landmark(item) for item in payload]
    if len(result) != HAND_LANDMARK_COUNT:
        raise ValueError(f"关键点数量必须为 {HAND_LANDMARK_COUNT}，实际 {len(result)}")
    return result


def _to_landmark(value: Any) -> Landmark:
    if isinstance(value, Landmark):
        return value
    if isinstance(value, Mapping):
        return Landmark(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            z=float(value.get("z", 0.0)),
            visibility=float(value.get("visibility", 1.0)),
        )
    if isinstance(value, (list, tuple)):
        coords = list(value) + [0.0, 0.0, 1.0]
        return Landmark(
            x=float(coords[0]),
            y=float(coords[1]),
            z=float(coords[2]),
            visibility=float(coords[3]) if len(coords) > 3 else 1.0,
        )
    if hasattr(value, "x") and hasattr(value, "y"):
        return Landmark(
            x=float(value.x),
            y=float(value.y),
            z=float(getattr(value, "z", 0.0)),
            visibility=float(getattr(value, "visibility", 1.0)),
        )
    raise TypeError(f"无法解析的关键点：{type(value)!r}")


# --------------------------------------------------------------------------- #
# 检测器
# --------------------------------------------------------------------------- #
class HandDetector:
    """MediaPipe Hands 封装（legacy solutions API）。"""

    def __init__(
        self,
        *,
        max_hands: int | None = None,
        model_complexity: int | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        settings = get_settings()
        self.max_hands = max_hands or settings.mediapipe.max_hands
        self.model_complexity = (
            model_complexity if model_complexity is not None else settings.mediapipe.model_complexity
        )
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands: Any = None
        self._mp: Any = None
        self._cv2: Any = None
        self._error: str | None = None
        self._init()

    # -- 生命周期 ---------------------------------------------------------- #
    def _init(self) -> None:
        try:
            import cv2  # noqa: PLC0415 - 延迟导入以便无 cv2 环境仍可构造
            import mediapipe as mp  # noqa: PLC0415

            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "hands"):
                raise RuntimeError(
                    f"mediapipe {getattr(mp, '__version__', '?')} 不含 legacy solutions API"
                )
            self._cv2 = cv2
            self._mp = mp
            self._hands = solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max(1, self.max_hands),
                model_complexity=int(self.model_complexity),
                min_detection_confidence=float(self.min_detection_confidence),
                min_tracking_confidence=float(self.min_tracking_confidence),
            )
        except Exception as exc:  # pragma: no cover - 环境缺失时的降级路径
            self._error = f"{type(exc).__name__}: {exc}"
            self._hands = None

    @property
    def available(self) -> bool:
        return self._hands is not None

    @property
    def error(self) -> str | None:
        return self._error

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:  # pragma: no cover
                pass
            self._hands = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- 检测 -------------------------------------------------------------- #
    def detect_all(self, frame_bgr: Any, *, frame_id: int | None = None) -> list[HandFrame]:
        """返回本帧全部手（最多 ``max_hands`` 只）。失败/不可用时返回空列表。"""
        if self._hands is None or frame_bgr is None:
            return []
        try:
            rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
        except Exception:  # pragma: no cover - 单帧失败不应中断采集
            return []
        if not getattr(results, "multi_hand_landmarks", None):
            return []

        now = time.time()
        handedness_list = getattr(results, "multi_handedness", None) or []
        hands: list[HandFrame] = []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label, score = "Unknown", 0.0
            if index < len(handedness_list):
                classification = handedness_list[index].classification
                if classification:
                    label = classification[0].label
                    score = float(classification[0].score)
            marks = tuple(
                Landmark(x=float(lm.x), y=float(lm.y), z=float(lm.z), visibility=1.0)
                for lm in hand_landmarks.landmark
            )
            if len(marks) != HAND_LANDMARK_COUNT:
                continue
            hands.append(
                HandFrame(
                    landmarks=marks,
                    handedness=label,
                    score=score,
                    ts=now,
                    frame_id=frame_id,
                    source="mediapipe",
                )
            )
        return hands[: max(1, self.max_hands)]

    def detect(self, frame_bgr: Any, *, frame_id: int | None = None) -> HandFrame | None:
        """检测并按 §2.6 选出唯一一只手。"""
        return select_hand(self.detect_all(frame_bgr, frame_id=frame_id))

    # -- 合成输入 ---------------------------------------------------------- #
    @staticmethod
    def from_landmarks(
        landmarks: Mapping[int, Any] | Sequence[Any] | Iterable[Landmark],
        *,
        handedness: str = "Right",
        score: float = 1.0,
        frame_id: int | None = None,
        ts: float | None = None,
    ) -> HandFrame:
        """从合成坐标构造 `HandFrame`（离线评测 / 虚拟注入）。"""
        marks = landmarks_from_mapping(landmarks)
        return HandFrame(
            landmarks=tuple(marks),
            handedness=handedness,
            score=score,
            ts=time.time() if ts is None else ts,
            frame_id=frame_id,
            source="synthetic",
        )
