"""配置中心（Plan V1.0 T1 / §8.3）。

职责：
1. 加载 `.env` 并做类型校验，任何非法值立即报错而非静默降级；
2. 集中托管 Plan §5.1 的全部阈值常量块——算法文件内 **禁止** 出现裸数字阈值；
3. 对外只读（frozen dataclass），进程内单例。

环境变量清单见 `.env.example`；阈值常量不经环境变量覆盖，一律在代码内定稿，
以避免现场调参出现"环境不同、判定不同"的不可复现问题。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=False)


# --------------------------------------------------------------------------- #
# 环境变量读取（带类型校验）
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"环境变量 {name}={raw!r} 不是合法布尔值")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - 启动期快速失败
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法整数") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法浮点数") from exc


# --------------------------------------------------------------------------- #
# 分组配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path = BASE_DIR
    log_dir: Path = BASE_DIR / "logs"
    static_dir: Path = BASE_DIR / "src" / "static"
    docs_dir: Path = BASE_DIR / "docs"

    def ensure(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_console: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    #: 启动时是否在守护线程中探测摄像头（不阻塞服务启动）
    probe_enabled: bool = True
    #: 探测看门狗超时（毫秒）：超时即判 UNAVAILABLE，线程自行结束
    probe_timeout_ms: int = 2000
    #: 连续取帧失败多少次自动降级为 INJECT_ONLY
    max_consecutive_failures: int = 5
    #: 水平镜像（自拍语义）。摄像头看到的是"观察者视角"，人手向自己的右侧移动时
    #: 归一化 x 反而减小；开启后统一翻转为"用户视角"，使
    #:   ① HUD 骨架与人手同向移动；② 向右挥 → CMD_TURN_RIGHT。
    #: 关闭则沿用 MediaPipe 原始坐标（观察者视角）。
    mirror_x: bool = True


@dataclass(frozen=True)
class MediaPipeConfig:
    model_complexity: int = 0
    max_hands: int = 2


@dataclass(frozen=True)
class ModeConfig:
    """运行模式。P0 阶段固定 DRY_RUN + INJECT_ONLY。"""

    inject_only: bool = True
    dry_run: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    """并发与安全（Plan §2.3 / §5.3）。"""

    estop_cooldown_ms: int = 500
    watchdog_miss_ms: int = 500
    bus_queue_size: int = 32
    dedupe_ms: int = 200
    multi_hand_cooldown_ms: int = 300


@dataclass(frozen=True)
class PerformanceConfig:
    """性能门限与打点（Plan §4）。"""

    target_fps: int = 30
    frame_budget_ms: float = 33.3
    frame_p95_limit_ms: float = 20.0
    degrade_p95_ms: float = 25.0
    metrics_history: int = 2000
    frame_push_hz: int = 15


@dataclass(frozen=True)
class Thresholds:
    """Plan V1.0 §5.1 阈值常量块（Phase 3 调参后回填定稿值）。

    坐标一律为 MediaPipe 归一化坐标（0~1，图像 y 轴向下）。
    """

    # 轨迹环形缓冲容量（Plan §2.1：容量保留 30，判定不再等满）
    window_capacity: int = 30

    # 握拳 G-02：四指 tip.y > pip.y 的命中手指数
    # 2026-11 收紧：3 → 4，并要求手掌足够大，避免"随手一放"被判成急停
    fist_min_fingers: int = 4
    #: 手部包围盒对角线下限（归一化单位）：低于此值视为"手太远/关键点不可靠"
    min_hand_size: float = 0.18

    # ---- 静态手型包（G-03 竖拇指 / G-04 张掌 / G-05 剪刀手 / G-06 捏合）----
    #: 竖拇指要求：四指中蜷起的根数（与握拳同源判据）
    thumb_up_fist_fingers: int = 3
    #: 拇指尖高出 IP 关节的最小高度（归一化）
    thumb_up_min_rise: float = 0.06
    #: 拇指尖远离食指 MCP 的最小距离（区分"翘拇指的握拳"）
    thumb_up_min_away: float = 0.12
    #: 捏合判据：拇指尖-食指尖距离 / 手部尺寸 的上限
    pinch_ratio: float = 0.30
    #: "静止张掌"判定：最近若干采样点内食指尖位移上限，超过即认为手在移动，
    #: 此时不得产出 G-04（否则挥手/前推过程会被反复判成"怠速"）
    open_palm_still_eps: float = 0.012
    #: 判断静止所回看的采样点数
    open_palm_still_frames: int = 3

    # ---- 上下挥动包（G-20 / G-21）----
    #: 竖直方向的最小位移（归一化，占画面高度比例）
    swipe_vertical_min_displacement: float = 0.10
    #: 竖直挥动时允许的水平漂移占比上限
    swipe_horizontal_ratio: float = 0.40

    # ---- 双手组合包（G-30 / G-31）----
    #: 判定双手手势时，两只手都必须达到的最小尺寸
    two_hand_min_size: float = 0.15

    # 左/右挥动 G-11 / G-12
    swipe_min_displacement: float = 0.12
    swipe_max_frames: int = 8
    swipe_vertical_ratio: float = 0.40
    #: 挥动窗口内逐帧水平位移必须同向（容差），用于剔除"来回摆动"
    swipe_monotonic_eps: float = 2e-4

    # 前推 G-10
    push_delta_z_per_frame: float = 0.02
    push_consecutive_frames: int = 3
    push_min_delta_y: float = 0.10
    #: 掌心正对镜头的最小程度（0~1），G-10 的前提条件
    palm_facing_min: float = 0.60

    # 转圈 G-13（Plan §2.1 裁决：累计转角驱动 + 12~24 帧自适应窗口）
    circle_min_cum_angle: float = 330.0
    circle_closure_max: float = 0.18
    circle_max_radius_cv: float = 0.35
    circle_sign_consistency: float = 0.85
    circle_min_frames: int = 12
    circle_max_frames: int = 24
    #: 环形趋势闸门：短窗累计转角超过该值即视为"正在画圆"，
    #: 期间抑制平移类候选（前推/左右挥），避免转圈过程被抢先判成转向
    arc_gate_angle: float = 45.0
    arc_gate_min_points: int = 5
    #: 帧间超时：相邻两帧间隔超过该值视为"手势中断"，重置轨迹与滑窗，
    #: 防止上一段手势的残留样本污染新动作的判定（离手/停顿即断链）
    gap_reset_ms: int = 200


@dataclass(frozen=True)
class DebounceConfig:
    """分级防抖参数（Plan §2.2 / §2.3 裁决）。"""

    window_size: int = 5
    confirm_frames: int = 3
    min_confidence: float = 0.85
    estop_confirm_frames: int = 2


@dataclass(frozen=True)
class GraphConfig:
    """LangGraph 温路径熔断参数（Plan §2.5）。"""

    #: graph_ms 的 P95 超过该值即熔断为直连路径
    p95_limit_ms: float = 15.0
    #: 熔断判据所需的最小样本数
    min_samples: int = 20


@dataclass(frozen=True)
class WatchdogConfig:
    """看门狗参数（Plan §3.2）。

    `tick_hz=25`：滴答间隔 40 ms（且循环内做周期补偿，无漂移），使"丢失手势后下发
    CMD_IDLE"的最坏延迟为 500 + 40 + 提交耗时 ≈ 555 ms，稳落在 §10.2 要求的
    500±100 ms 窗口内（10 Hz 且无补偿时实测可达 606 ms，已越界）。
    """

    #: 后台滴答频率
    tick_hz: int = 25
    #: LIVE 模式下的 FPS 下限
    fps_floor: float = 15.0
    #: FPS 低于下限持续多久后下发 CMD_IDLE
    fps_low_ms: int = 300


@dataclass(frozen=True)
class DegradeConfig:
    """自适应降级参数（Plan §4.1 / T13）。"""

    enabled: bool = True
    #: 单帧耗时 P95 超过该值触发降级
    frame_p95_ms: float = 25.0
    #: 触发所需的样本数
    min_samples: int = 20
    #: 两次降级的冷却时间
    cooldown_s: float = 10.0
    #: 降级目标分辨率
    target_width: int = 480
    target_height: int = 360


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    server: ServerConfig
    camera: CameraConfig
    mediapipe: MediaPipeConfig
    mode: ModeConfig
    safety: SafetyConfig
    performance: PerformanceConfig
    thresholds: Thresholds
    debounce: DebounceConfig
    graph: GraphConfig
    watchdog: WatchdogConfig
    degrade: DegradeConfig

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def load(cls) -> "Settings":
        log_level = _env_str("LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"LOG_LEVEL={log_level!r} 非法")

        port = _env_int("WS_PORT", 8000)
        if not 1 <= port <= 65535:
            raise ValueError(f"WS_PORT={port} 越界")

        settings = cls(
            paths=PathsConfig(),
            server=ServerConfig(
                host=_env_str("APP_HOST", "127.0.0.1"),
                port=port,
                log_level=log_level,
                log_console=_env_bool("LOG_CONSOLE", True),
            ),
            camera=CameraConfig(
                index=_env_int("CAMERA_INDEX", 0),
                width=_env_int("CAMERA_WIDTH", 640),
                height=_env_int("CAMERA_HEIGHT", 480),
                target_fps=_env_int("TARGET_FPS", 30),
                probe_enabled=_env_bool("CAMERA_PROBE_ENABLED", True),
                probe_timeout_ms=_env_int("CAMERA_PROBE_TIMEOUT_MS", 2000),
                max_consecutive_failures=_env_int("CAMERA_MAX_FAILURES", 5),
                mirror_x=_env_bool("MIRROR_X", True),
            ),
            mediapipe=MediaPipeConfig(
                model_complexity=_env_int("MP_MODEL_COMPLEXITY", 0),
                max_hands=_env_int("MP_MAX_HANDS", 2),
            ),
            mode=ModeConfig(
                inject_only=_env_bool("INJECT_ONLY", True),
                dry_run=_env_bool("DRY_RUN", True),
            ),
            safety=SafetyConfig(
                estop_cooldown_ms=_env_int("ESTOP_COOLDOWN_MS", 500),
                watchdog_miss_ms=_env_int("WATCHDOG_MISS_MS", 500),
                bus_queue_size=_env_int("BUS_QUEUE_SIZE", 32),
                dedupe_ms=_env_int("DEDUPE_MS", 200),
                multi_hand_cooldown_ms=_env_int("MULTI_HAND_COOLDOWN_MS", 300),
            ),
            performance=PerformanceConfig(
                frame_p95_limit_ms=_env_float("FRAME_P95_LIMIT_MS", 20.0),
                degrade_p95_ms=_env_float("DEGRADE_P95_MS", 25.0),
                metrics_history=_env_int("METRICS_HISTORY", 2000),
                frame_push_hz=_env_int("FRAME_PUSH_HZ", 15),
            ),
            thresholds=Thresholds(),
            debounce=DebounceConfig(),
            graph=GraphConfig(
                p95_limit_ms=_env_float("GRAPH_P95_LIMIT_MS", 15.0),
                min_samples=_env_int("GRAPH_MIN_SAMPLES", 20),
            ),
            watchdog=WatchdogConfig(
                # 必须与 WatchdogConfig.tick_hz 的默认值一致：25Hz → 滴答 40ms，
                # 使"500ms 丢失"的判定落在 500~540ms；若取 10Hz（滴答 100ms），
                # 量化误差会把回落时机推到 600ms 之外，触发门禁 500±100ms 失败。
                tick_hz=_env_int("WATCHDOG_TICK_HZ", 25),
                fps_floor=_env_float("WATCHDOG_FPS_FLOOR", 15.0),
                fps_low_ms=_env_int("WATCHDOG_FPS_LOW_MS", 300),
            ),
            degrade=DegradeConfig(
                enabled=_env_bool("DEGRADE_ENABLED", True),
                frame_p95_ms=_env_float("DEGRADE_FRAME_P95_MS", 25.0),
                min_samples=_env_int("DEGRADE_MIN_SAMPLES", 20),
                cooldown_s=_env_float("DEGRADE_COOLDOWN_S", 10.0),
                target_width=_env_int("DEGRADE_TARGET_WIDTH", 480),
                target_height=_env_int("DEGRADE_TARGET_HEIGHT", 360),
            ),
        )
        settings.validate()
        return settings

    # -- 校验 -------------------------------------------------------------- #
    def validate(self) -> None:
        t = self.thresholds
        if t.circle_min_frames >= t.circle_max_frames:
            raise ValueError("circle_min_frames 必须小于 circle_max_frames")
        if t.circle_min_frames <= 0 or t.circle_max_frames > t.window_capacity:
            raise ValueError("转圈帧窗口必须落在 (0, window_capacity] 内")
        if not 0.0 < t.swipe_min_displacement < 1.0:
            raise ValueError("swipe_min_displacement 必须落在 (0, 1)")
        if not 0.0 <= t.circle_sign_consistency <= 1.0:
            raise ValueError("circle_sign_consistency 必须落在 [0, 1]")
        if not 0.0 <= t.palm_facing_min <= 1.0:
            raise ValueError("palm_facing_min 必须落在 [0, 1]")
        if self.debounce.confirm_frames > self.debounce.window_size:
            raise ValueError("confirm_frames 不得大于 window_size（Plan §2.2）")
        if self.debounce.estop_confirm_frames >= self.debounce.confirm_frames:
            raise ValueError("急停必须比常规动作更早触发（Plan §2.3）")
        if self.safety.bus_queue_size < 2:
            raise ValueError("bus_queue_size 至少为 2")
        if self.performance.frame_push_hz <= 0:
            raise ValueError("frame_push_hz 必须为正")
        if self.graph.p95_limit_ms <= 0 or self.graph.min_samples <= 0:
            raise ValueError("graph 熔断参数必须为正")
        if self.watchdog.tick_hz <= 0 or self.watchdog.fps_floor < 0:
            raise ValueError("watchdog 参数非法")
        if self.degrade.frame_p95_ms <= 0 or self.degrade.min_samples <= 0:
            raise ValueError("degrade 参数必须为正")
        if self.degrade.target_width <= 0 or self.degrade.target_height <= 0:
            raise ValueError("degrade 目标分辨率必须为正")
        self.paths.ensure()

    # -- 视图 -------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return _stringify_paths(data)


def _stringify_paths(node: Any) -> Any:
    if isinstance(node, Path):
        return str(node)
    if isinstance(node, dict):
        return {k: _stringify_paths(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_stringify_paths(v) for v in node]
    return node


# --------------------------------------------------------------------------- #
# 进程内单例
# --------------------------------------------------------------------------- #
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """测试用：重新读取环境变量。"""
    global _settings
    _settings = Settings.load()
    return _settings
