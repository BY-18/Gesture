# Gesture-Copilot —— 纯本地实时代控手势识别 Agent

在 CPU-only 的本地环境下，构建一条从「摄像头视觉捕获」到「机器人控制指令下发」的 **端到端 ≤ 900 ms** 手势遥控通道。通过时序防抖状态机在「零误触」与「低延迟」之间取得工程平衡，并以**虚拟手势注入器**保证在无摄像头、无实机条件下 100% 完成闭环演示与验收。

> 实施计划与验收口径详见 [docs/plan_gesture_copilot.md](docs/plan_gesture_copilot.md)（指标冻结表 §1.2，验收矩阵 §11.3）。

---

## 1. 核心特性

- **五类手势 → 六种指令**：握拳急停、前推、左/右挥动、转圈，映射到完整冻结指令域
- **分级防抖**：热路径纯函数表决（急停 2 帧 / 方向 3 帧 / 转圈角度累积），零框架依赖、零 I/O
- **急停最高优先权**：2 帧直出 `CMD_STOP`（旁路滑窗、旁路队列、旁路 LangGraph），触发后 500 ms 冷却期拒绝一切运动指令
- **看门狗兜底**：手势丢失 500±100 ms 自动回落 `CMD_IDLE`（零速度怠速）
- **无摄像头 100% 闭环**：注入链路从防抖层入口汇入，与真实视觉链路共用同一套状态机与控制总线
- **全链路可观测**：逐事件审计日志（`logs/session_*.jsonl`）+ `/api/metrics` 延迟直方图 + 自适应降级

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│ 采集层  src/vision/capture.py     OpenCV VideoCapture (640×480@30FPS) │
├──────────────────────────────────────────────────────────────────────┤
│ 感知层  src/vision/hand_detector.py  MediaPipe 21 点 + handedness     │
├──────────────────────────────────────────────────────────────────────┤
│ 几何层  src/vision/classifier.py  静态姿态（G-02 / G-10 前提）         │
│         src/vision/trajectory.py  位移方向（G-11/12）+ 圆拟合（G-13）  │
├──────────────────────────────────────────────────────────────────────┤
│ 防抖层  src/agents/debounce.py    滑窗表决 / 急停快通道 【热路径·纯函数】│
├──────────────────────────────────────────────────────────────────────┤
│ 编排层  src/agents/graph.py       LangGraph StateGraph【温路径·异步旁路】│
│                                   急停硬拦截 → 意图确认 → 看门狗 → 审计 │
├──────────────────────────────────────────────────────────────────────┤
│ 执行层  src/execution/control_bus.py  标准控制帧 / DRY_RUN / 发送队列   │
├──────────────────────────────────────────────────────────────────────┤
│ 服务层  src/api/routes.py + main.py   FastAPI REST + WebSocket 推流    │
├──────────────────────────────────────────────────────────────────────┤
│ 呈现层  src/static/index.html     单文件 HUD：视频+骨架/状态面板/注入器 │
└──────────────────────────────────────────────────────────────────────┘
                               │
                         logs/session_*.jsonl（逐事件审计 + 延迟打点）
```

**数据流**：帧 → 检测 → 几何候选 → 防抖确认 → LangGraph 编排 → 控制帧 → 总线（DRY_RUN 打日志）→ WebSocket 下行 → HUD 渲染。
**虚拟注入器旁路**：`POST /api/inject` 从防抖层入口注入，因此注入链路与真实摄像头链路**共用同一套状态机与控制总线**。

## 3. 手势与指令集

共 **13 个手势码**，全部映射到冻结的 6 条指令（不扩展 `Cmd` 域）。

| 手势码 | 动作 | 指令 | 触发机制 | 冷却/语义 |
|---|---|---|---|---|
| G-02 | 握拳 | `CMD_STOP` | 2 帧直出（第 3 帧锁存） | 急停最高优先权，500 ms 冷却 |
| G-10 | 前推 | `CMD_FORWARD` | 3 帧几何 + 3 帧表决 | 武装看门狗 |
| G-11 | 左挥 | `CMD_TURN_LEFT` | 水平位移 ≥0.12（≤8 帧） | 第 4 帧反向可回滚 |
| G-12 | 右挥 | `CMD_TURN_RIGHT` | 同上 | 同上 |
| G-13 | 转圈 | `CMD_SPIN` | 累计转角 ≥330° + 闭环拟合 | 12~24 帧窗口，单帧直通 |
| G-03 | 竖拇指 | `CMD_FORWARD` | 静态手型，3 帧表决 | 拇指须明显竖起并远离食指 MCP，否则仍判握拳 |
| G-04 | 张开五指（静止） | `CMD_IDLE` | 静态手型，3 帧表决 | 须"手不动"才成立，松手后可再触发 |
| G-05 | 剪刀手 | `CMD_SPIN` | 静态手型，3 帧表决 | 食指+中指伸，无名指/小指/拇指蜷 |
| G-06 | 拇指食指捏合 | `CMD_STOP` | 2 帧直出 | 备用急停，手小或握不紧拳时更好做 |
| G-20 | 上挥 | `CMD_FORWARD` | 竖直位移 ≥0.10（≤8 帧） | 掌心未正对镜头时的竖直路径（正对时归 G-10） |
| G-21 | 下挥 | `CMD_IDLE` | 同上 | 反向 |
| G-30 | 双掌前推 | `CMD_FORWARD` | 双手 + 3 帧表决 | 需两只手都在画面内 |
| G-31 | 双手握拳 | `CMD_STOP` | 2 帧直出 | 语义最强的急停，要求两只手同时握拳 |
| — | 看门狗 | `CMD_IDLE` | 输入中断 500±40 ms | 零速度怠速，优先级仅次于急停 |

**方向语义（`MIRROR_X=true`，默认开启）**：按**用户视角**判方向——手往自己的右边挥 = 右转，HUD 骨架与人手同向移动。若需要 MediaPipe 原始"观察者视角"，设 `MIRROR_X=false`。

**手指计数调速**：运动指令按"食指尖以外"伸出的手指数分挡——0 根→0.4、1 根→0.7、≥2 根→1.0（轨迹固定追食指尖，故食指伸不伸不影响识别）。

## 4. 延迟预算（30 FPS，帧周期 33.3 ms）

**单帧热路径预算**：

| 阶段 | 模块 | 预算 (ms) | 上限 (ms) |
|---|---|---|---|
| 采集取帧 | `capture.py` | 2.0 | 5.0 |
| MediaPipe 检测（complexity=0，单手） | `hand_detector.py` | 16.0 | 20.0 |
| 几何特征与候选判定 | `classifier.py` / `trajectory.py` | 1.0 | 3.0 |
| 防抖状态机逐帧推进 | `debounce.py` | 0.5 | 1.0 |
| 指令封包 + 总线 | `control_bus.py` | 1.0 | 3.0 |
| WS 下行至浏览器（含渲染） | 浏览器 | 4.0 | 10.0 |
| **合计** | | **24.5** | **42.0** |

> 帧周期内余量 8.8 ms；单帧 P95 > 25 ms 自动降级 640×480 → 480×360。

**触发时延预算与验收实测**（`scripts/bench_latency.py`，每类 50 次）：

| 动作 | 帧数依据 | 总预算 (ms) | 冻结上限 (ms) | 断言口径 |
|---|---|---|---|---|
| G-02 急停 | 2 帧 | 71.7 | 100 | 总时延 P99 ≤ 100 ms |
| G-10/11/12 方向 | 3 帧 | 105.0 | 120 | 表决时延 P99 ≤ 120 ms |
| G-13 转圈 | 18~22 帧典型 | ≤ 743 | 800 | 总时延 P99 ≤ 760 ms |
| 端到端（含 WS 渲染） | 转圈路径 | ≤ 900 | 1000 | 按 trace_id 串联复算 |
| 单帧耗时 | — | — | P95 ≤ 20 | 1000 帧热路径统计 |

## 5. 快速启动

环境要求：Windows / Python 3.12（项目自带 `.venv`），CPU-only，无需 GPU；无摄像头时自动进入 `INJECT_ONLY` 模式。

```powershell
# ① 安装依赖（首次）
.venv\Scripts\python.exe -m pip install -r requirements.txt

# ② 配置（可选，默认即注入模式 + 干跑）
Copy-Item .env.example .env

# ③ 启动服务
.venv\Scripts\python.exe main.py

# ④ 打开 HUD
# 浏览器访问 http://127.0.0.1:8000/
```

关键配置项（`.env`，全部有默认值）：`INJECT_ONLY`（注入模式）、`DRY_RUN`（干跑，绝不驱动物理设备）、`WATCHDOG_MISS_MS=500`（看门狗超时）、`ESTOP_COOLDOWN_MS=500`（急停冷却）、`GRAPH_P95_LIMIT_MS=15`（温路径熔断阈值）。

## 6. HUD 演示指南

打开 `http://127.0.0.1:8000/` 后：

1. **注入器面板**：点击底部任一按钮（`G-02`~`G-31`，快捷键 `1~9` / `A S D F`）即可模拟对应手势；指令流水与状态面板即时滚动显示生成的 `ControlFrame`（`seq` / `cmd` / `speed` / `source`）。
2. **急停演示**：先点 G-10（前进），随后立即点 G-02 —— 观察急停直出 `CMD_STOP`，且随后 500 ms 内的运动指令被标记为抑制（`suppressed=true`）。
3. **看门狗演示**：点 G-10 后停止一切操作，约 500 ms 后流水自动出现 `source=WATCHDOG` 的 `CMD_IDLE`。
4. **延迟面板**：右上角实时显示 FPS 与 dispatch/graph/frame 的 P50/P95/P99 直方图。
5. **摄像头模式**：只要探测到摄像头，视觉循环就会自动启动（`INJECT_ONLY` 仅是模式标签，不拦截视觉链路）。手放入画面后，HUD 的 21 点骨架会实时跟随；**下行的是骨架而非原始视频**（`has_video` 恒为 false）。左上角 `SRC` 会显示当前姿态来源：`摄像头` / `注入` / `无信号`。
6. **会话审计**：每次演示的完整事件流落盘于 `logs/session_*.jsonl`，可离线回放比对。

## 7. 一键验收（门禁 G3）

```powershell
.venv\Scripts\python.exe scripts\verify_all.py
```

脚本自动**连接或拉起**服务（日志见 `logs/verify_server.log`），执行 §11.3 的 **5 类核心断言**，随后调度三个离线子套件，最后打印验收矩阵；全部通过退出码 `0`：

| # | 断言类别 | 内容 |
|---|---|---|
| 1 | 服务可达 | `/api/health` 返回 `INJECT_ONLY`/`LIVE`；`/`（HUD）返回 200 |
| 2 | 协议正确 | `/api/inject` 覆盖全部 6 种 `cmd`；`ControlFrame` 字段完整；`seq` 单调递增 |
| 3 | 急停优先权 | FORWARD 后立即 G-02：≤100 ms 出 `CMD_STOP`，随后 500 ms 运动指令严格被拒 |
| 4 | 看门狗超时 | FORWARD 后停止输入：500±100 ms 自动 `CMD_IDLE`（source=WATCHDOG） |
| 5 | 时延与稳态 | 300 次混合注入 + 三个子套件；`/api/metrics` 各 P99 达标 |

离线子套件（可单独运行，均不依赖摄像头/网络）：

```powershell
.venv\Scripts\python.exe scripts\gen_synthetic.py     # T14 生成合成数据集（v3，seed 固定）
.venv\Scripts\python.exe scripts\test_trajectory.py   # T15 识别率 ≥90% + 硬负例不误判
.venv\Scripts\python.exe scripts\test_debounce.py     # T15 静态误触发 ≤2% + 防抖语义回归
.venv\Scripts\python.exe scripts\bench_latency.py     # T16 热路径 P95 ≤20ms + 触发时延预算
```

评测结果统一写入 [logs/eval_report.md](logs/eval_report.md)（Markdown 表格，多次运行幂等覆盖）。

## 8. 项目结构

```
Gesture/
├── main.py                      # FastAPI 启动入口（lifespan 收拢全部单例与后台任务）
├── src/
│   ├── vision/                  # capture / hand_detector / classifier / trajectory / pipeline
│   ├── agents/                  # debounce（热路径纯函数）/ nodes / graph（LangGraph）/ state
│   ├── execution/control_bus.py # 控制帧封包、DRY_RUN、发送队列、去抖
│   ├── api/                     # routes / schemas / injector / ws
│   ├── core/                    # config / logger / metrics（直方图 + 自适应降级）
│   └── static/index.html        # 单文件 HUD（零外链，内网离线可用）
├── scripts/
│   ├── gen_synthetic.py         # T14 合成数据集生成器
│   ├── test_trajectory.py       # T15 识别率评测（门禁 G1 口径）
│   ├── test_debounce.py         # T15 静态误触发 + 防抖语义回归
│   ├── bench_latency.py         # T16 延迟基准
│   ├── verify_all.py            # T17 一键全链路验收（门禁 G3）
│   ├── g0_smoke.py / g2_smoke.py# 历史门禁 G0 / G2 冒烟脚本
├── logs/
│   ├── dataset/                 # 合成数据集 JSONL + manifest（版本化）
│   ├── eval_report.md           # 评测汇总报告
│   └── session_*.jsonl          # 逐事件审计日志
└── docs/plan_gesture_copilot.md # 实施计划（指标冻结表 + 验收口径 + 实施记录）
```

## 9. 安全约束（不可绕过）

- 默认 `DRY_RUN=true`：指令只落日志与 HUD，绝不驱动物理设备
- 急停 **永不丢弃、永不延迟**：队列满丢弃最旧**运动**指令，急停旁路一切队列
- 冷却期内运动指令在总线侧硬拒绝；看门狗怠速优先级高于普通运动指令、低于急停
- 温路径熔断：LangGraph 单次执行 P95 > 15 ms 自动降级为直接函数调用并告警
