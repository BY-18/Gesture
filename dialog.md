# Gesture-Copilot 开发对话记录

> 工作目录：`C:\Users\BY\PycharmProjects\Gesture`
> 环境：Windows / Python 3.12.15 / CPU-only / `.venv` 自带依赖
> 记录范围：从"跑起来看看成果"到"扩展手势词表并修复门禁失败"的完整过程
> 说明：用户消息逐字保留；助手回复按轮次整理，保留结论、数据与命令，省略重复的排版性内容。

---

## 第 1 轮 · 让项目跑起来

**用户：**

> 给我指令让我运行一下 ,看看项目成果

**助手：**

先确认环境已就绪：

- `.venv` 中 Python 3.12.15，依赖齐全（`mediapipe 0.10.21`、`numpy 1.26.4`、`fastapi`、`uvicorn`、`langgraph`、`opencv 4.11.0`）
- 8000 端口空闲，无残留 python 进程
- 无 `.env`，走默认 `INJECT_ONLY=true + DRY_RUN=true`（纯注入 + 干跑，不会驱动真实设备）

**方案 A：看 HUD（最直观）**

```powershell
cd C:\Users\BY\PycharmProjects\Gesture
.\.venv\Scripts\python.exe main.py
```

浏览器打开 `http://127.0.0.1:8000/`，按演示动线点一遍：

1. 点 **G-10 前推** → 指令流水出现 `CMD_FORWARD`
2. 紧接着点 **G-02 握拳** → 立刻直出 `CMD_STOP`，之后 500 ms 内的运动指令被标 `suppressed=true`
3. 只点一次 **G-10** 后停手 → 约 500 ms 后自动冒出 `source=WATCHDOG` 的 `CMD_IDLE`
4. 依次点 **G-11 / G-12 / G-13** → 看左右转、转圈与右上角 P50/P95/P99 直方图

**方案 B：一键全链路验收**

```powershell
.\.venv\Scripts\python.exe scripts\verify_all.py
```

跑 5 类核心断言 + 3 个离线子套件，最后打印验收矩阵；服务没起时脚本会自己拉起（日志 `logs/verify_server.log`），跑完自动关闭。

**方案 C：分项评测**

```powershell
.\.venv\Scripts\python.exe scripts\test_trajectory.py   # 五类手势识别率（门槛 ≥90%）
.\.venv\Scripts\python.exe scripts\test_debounce.py     # 静态误触发 ≤2% + 防抖语义
.\.venv\Scripts\python.exe scripts\bench_latency.py     # 热路径 P95 ≤20ms + 触发时延预算
```

**成果去哪看**：HUD（实时交互与延迟面板）、终端验收矩阵、`logs/eval_report.md`（评测汇总）、`logs/session_*.jsonl`（逐事件审计流）。

---

## 第 2 轮 · "没调用摄像头"与"手势到底怎么动"

**用户：**

> cd C:\Users\BY\PycharmProjects\Gesture
> .\.venv\Scripts\python.exe scripts\verify_all.py 运行这个指令并没有调用摄像头执行 ,此外这写手势是怎么动的

**助手：**

### 一、摄像头其实"调用了"，只是没有任何东西给你看

用户的 `logs/verify_server.log` 里明确记录：

```
camera_probe   available=True state=READY index=0 width=640 height=480 fps=17.29
note=已探测到摄像头…
vision_loop_started fps_target=30
```

摄像头**被成功打开、视觉闭环真的跑起来了**。之所以"看不到调用"，有三个原因：

1. **`verify_all.py` 本来就不走摄像头**：5 类断言全靠 `POST /api/inject` 注入 + 离线合成数据（结果里每条指令都是 `source=INJECT`，只有看门狗那条是 `source=WATCHDOG`）。它验的是协议、急停优先权、看门狗、时延。
2. **项目没有任何画面预览窗口**：全项目没有 `cv2.imshow`，HUD 也没有 `<video>` / `getUserMedia`；`ws.py` 里 `has_video` 被硬编码为 `False`——HUD 上的手部骨架是用注入的 21 点坐标画的合成骨架。
3. **脚本跑完会 `terminate()` 自己拉起的服务**，摄像头随进程一起释放。

> 同时还发现一个真实缺口：`main.py` 的视觉循环**不写** `app.state.latest_pose`（只有 `/api/inject` 和 `/api/frames` 写）。所以对着摄像头比手势时指令会正常产生，但 **HUD 骨架不会跟着手**动。

### 二、五个手势的动作要领

追踪点统一是**食指尖**（landmark 8）。坐标是画面归一化坐标：x 向右、y 向下、z 越小越靠近镜头。判定优先级：**急停 > 转圈 > 前推 > 左右挥**。

| 手势 | 手上怎么动 | 代码里真正在算的判据 | 需几帧 |
|---|---|---|---|
| **G-02 握拳急停** | 手掌正对镜头，四五指一起蜷进掌心握成拳 | 四指中 ≥3 个"指尖 y > PIP 第二关节 y" | **2 帧**（≈66ms）直出 |
| **G-10 前推** | 五指张开·掌心正对镜头，手腕朝镜头推出去；或明显往上抬手 | 掌心朝向 ≥0.60 且（连续 3 帧 z 靠近 ≥0.02 或 y 上抬 ≥0.10） | 3 帧几何 + 3 帧表决 |
| **G-11 左挥** | 伸出食指，向画面左侧划一道直的 | 8 帧内 \|dx\| ≥ 0.12、垂直占比 <40%、方向单向 | ≤8 帧 |
| **G-12 右挥** | 同上，向画面右侧划 | 同上，dx > 0 | ≤8 帧 |
| **G-13 转圈** | 食指伸出，空中画直径约 15~25% 屏宽的圆，一个方向画满近一整圈 | 12~24 帧内累计转角 ≥330°、闭合误差 ≤0.18、半径 CV ≤0.35、旋向一致 ≥0.85 | 12~24 帧 |
| **看门狗** | 停手不动 | 500±100ms 无输入 | — |

**实操要点**：握拳最容易先找感觉；前推最容易失败（靠单目估 z 不稳，优先"往上抬"）；挥动要单向干脆；画圈要快一点圆一点；手离镜头 40~70cm。

---

## 第 3 轮 · 补齐 HUD 实时骨架

**用户：**

> ok

（承接上一轮结尾的提议：是否补齐 HUD 实时骨架 / 加 cv2.imshow 调试窗口）

**助手：**

### 改了什么（4 个文件）

- **`main.py`**：新增 `_publish_pose()`，视觉循环每帧都发布 21 点姿态，**不等识别出手势**（之前只有产生指令那一帧才有数据，HUD 只能闪一下）。
- **`src/api/ws.py`**：下行帧新增 `source`（`GESTURE`/`INJECT`）和 `camera_available` 字段。
- **`src/api/routes.py`**：修两处不一致——注入链路写 `latest_pose` 时**漏了 `source` 字段**；`/api/frames` 回放链路同样改成"每帧都发布姿态"。
- **`src/static/index.html`**：左上角新增 `SRC` 行显示姿态来源；空态提示区分「摄像头就绪」与「等待手势注入」。

### 验证结果

离线断言（`latest_pose` → WS 下行）4/4 通过；端到端 3/3 通过：

```
PASS ① 无手势帧也发布姿态: source=GESTURE code=None landmarks=21
PASS ② 手势帧带标签:      code=G-02 label=停止/急停 source=GESTURE
PASS ③ 注入链路:          source=INJECT code=G-11 landmarks=21
```

G3 全链路门禁回归：**34/34 PASS**（急停 5.5ms、看门狗 557.5ms、`graph_ms` P95=7.60ms、离线子套件三项全过）。

### ⚠️ 摄像头打不开

当时复测发现 OpenCV 三个后端 × 索引 0/1 × 3 轮**全部 `isOpened=False`**，在源码之外就失败，与代码无关。给出排查顺序：Windows「相机」App 验证 → 关闭占用程序 → 设备管理器检查 → 物理隐私开关 → 隐私设置授权。

---

## 第 4 轮 · 运行指令 + 手势动作复述

**用户：**

> 给我运行指令 ,另外告诉我手势该怎么做

**助手：**

（此时摄像头仍不可用、8000 端口空闲）

**起服务**

```powershell
cd C:\Users\BY\PycharmProjects\Gesture
.\.venv\Scripts\python.exe main.py
```

浏览器打开 `http://127.0.0.1:8000/`。

**验收（可选，另开窗口）**

```powershell
.\.venv\Scripts\python.exe scripts\verify_all.py
```

**摄像头自检**（调好设备后先跑这条；服务开着时它会失败，因为摄像头被服务占用）

```powershell
.\.venv\Scripts\python.exe -c "import cv2;c=cv2.VideoCapture(0,cv2.CAP_DSHOW);print('可用' if c.isOpened() and c.read()[0] else '不可用');c.release()"
```

**手势动作**：统一前提是单手、40~70cm、手掌占画面够大、正面光、背景不花。按键 `1`~`5` 分别对应 G-02/G-10/G-11/G-12/G-13。动作与判据见表（同第 2 轮）。

**确认做对没有**：HUD 上 `SRC` 显示 `摄像头` 说明走真实视觉链路；`POSE` 会变成对应标签；右侧流水出现 `ControlFrame`。
- `SRC` 显示摄像头但 `POSE` 一直 NO SIGNAL → 手没被检测到（靠近、掌心对镜头、加光）
- `POSE` 有标签但没指令 → 帧数不够，动作做久一点

---

## 第 5 轮 · 两个门禁 FAIL + 镜像 + 手势太少

**用户：**

> （贴出 `verify_all.py` 完整输出，矩阵 32/34 FAIL）
>
> 这是什么原因 ,我不明白 ,另外识别的手势是镜像的 ,我不要镜像 ,另外 ,我需要多几种手势 ,就目前的手势太少

失败明细：

```
- [4. 看门狗超时] 回落时机落在 400~600 ms：624.4 ms
- [5. 时延与稳态] frame_ms P99 ≤ 25 ms：127.33 ms
```

**助手：**

### 一、两个 FAIL 的真正原因（都不是随机抖动）

**FAIL 1：看门狗 624.4ms** —— `WATCHDOG_TICK_HZ` 存在**三处不一致**：dataclass 默认 `25`、`.env.example` 写 `20`、`from_env` 实际默认 **`10`**。没有 `.env` 时跑的是 10Hz → 判定量化误差 100ms，`500+100` 已顶到 600 上限，任何调度抖动都会越界。

修复：统一为 25Hz（40ms 量化）。实测回落时机 **518.6 / 524.7 / 546.6 ms**。

**FAIL 2：frame_ms P99 = 127ms** —— 会话日志证明**摄像头当时是好的**（`available=true 640×480 fps=37.26`），视觉循环确实在跑。两个问题叠加：

1. `frame_ms` 把 **MediaPipe 启动预热帧**也算进去了（降级在循环启动 2.6 秒后就触发，样本 `P95=112.2ms`）。
   日志证据：
   ```
   event=degrade  reason="frame_ms P95=112.2ms > 25.0ms"  samples=22  applied={640x480 → 480x360}
   event=degrade_error  error="AttributeError: 'int' object has no attribute 'upper'"
   ```
   修复：加 1.5s 预热、预热帧不入直方图（与 `graph_warmup` 同口径）。
2. 预热触发的降级**当场崩了**：`logger.emit(..., level=action.level)` —— `level` 是控制台日志级别参数，传进去的却是 int 型降级级别，`level.upper()` 抛 `AttributeError`，导致 `hub.broadcast` 被跳过。
   修复：字段改名 `degrade_level`，并把 `_emit_console` 改成 `str(level).upper()` 兜底。

### 二、镜像的根因

MediaPipe 的 x 是**图像坐标**。用户面对摄像头时，手往自己右边移动，图像里 x 反而**减小** → 被判成左转；HUD 骨架也和手反向移动。

修复：新增 `MIRROR_X`（默认 `true`），在摄像头入口统一翻转 x（`x → 1-x`，y/z 不变），**决策与渲染同时修正**。自测复现并验证：

```
[PASS] 不开镜像：右挥手被判为左转（复现你遇到的问题） → G-11
[PASS] 开镜像：右挥手判为右转 → G-12
[PASS] 开镜像：左挥手判为左转
[PASS] 镜像只翻转 x，y/z 保持不变
```

### 三、手势从 5 个扩到 13 个

询问后用户的选择（多选全选 + 收紧急停）：

- 静态手型包 ✔
- 上下挥动包 ✔
- 双手组合包 ✔
- 手指计数调速 ✔
- 急停误触收紧 ✔（日志里 53 次识别有 45 次是握拳急停，多是手随手一放被误判）

### 四、实现要点

**契约层**：`GestureCode` 扩到 13 个码，全部映射到**冻结的 6 条指令域**（不扩展 `Cmd`）；新增 `ESTOP_CODES`，让 G-02/G-06/G-31 统一走 2 帧急停快通道。

**几何层**（`classifier.py`）：
- 新增 `hand_size`、`finger_states`、`thumb_extended`、`extended_count`
- 新增 `is_thumb_up`、`is_open_palm`、`is_v_sign`、`is_pinch`
- `is_fist` 收紧：`fist_min_fingers` 3→4 + 手掌尺寸门限 + **排除竖拇指姿态**
- `is_open_palm` 排除捏合；`dominant_swipe_vertical` 补齐上下挥动

**流水线优先级**（安全优先、静态手型垫底）：

```
TWO_FIST > 握拳 > 捏合 > 转圈 > (弧线进行中) > 双掌前推
> 竖拇指 > 剪刀手 > 前推 > 上/下挥 > 左右挥 > 静止张掌
```

**踩到的两个坑（均已修复）**：

1. **上挥与前推几何同源**：数据集里 G-10 本身就是"抬腕"动作（竖直位移中位数 0.191、最大 0.439），无法用阈值区分。让判据更严格的 push 优先；掌心正对镜头时按 G-10 走，未正对时才是 G-20。两者都是 `CMD_FORWARD`，用户无感。
2. **张掌候选污染表决窗口**：手在移动过程中每帧都产出 IDLE 候选，把"3 帧表决"拖成 4 帧，`G-11/G-12` 表决时延从 66.9ms 涨到 133.5ms，超出 120ms 预算。修复：G-04 只对**静止的手**成立（`open_palm_still_eps`）。

### 五、验证结果

```
词表自测 15/15     13 个手势码单帧判出自身 + 静态手型互斥 + 双手组合
注入链路 13/13     每个码都落到期望指令
G1 识别率          G-02 100% / G-10 100% / G-11 97% / G-12 96% / G-13 99%
延迟基准           PASS（G-10 表决 100.3ms / G-11 66.8 / G-12 67.1 / G-13 735.2）
门禁 G3            34/34 PASS
HUD JS 语法        OK
```

### 六、必须说清的遗留风险

这台机器的摄像头**时好时坏**（该轮测了 4 次，只有更早那 1 次成功）。因此新增的 9 个手势只用**合成手型**验证了"判据 → 候选 → 指令"链路正确且互不误判，**真实手部姿势下的阈值尚未标定**（`thumb_up_min_rise`、`pinch_ratio`、`min_hand_size` 均为可调参数）。建议摄像头稳定后先在 HUD 上加实时几何读数，用于标定。

---

## 第 6 轮 · 归档对话记录

**用户：**

> 把对话记录整理起来存放在项目中 ,命名为对话.txt 或者dialog.md

**助手：** 即本文件（`dialog.md`）。

---

## 附录 A · 最终手势词表（13 个码 → 6 条指令）

| 手势码 | 动作 | 指令 | 触发机制 | 语义要点 |
|---|---|---|---|---|
| G-02 | 握拳 | `CMD_STOP` | 2 帧直出（第 3 帧锁存） | 急停最高优先权，500ms 冷却 |
| G-03 | 竖拇指 | `CMD_FORWARD` | 静态手型，3 帧表决 | 拇指须明显竖起并远离食指 MCP，否则仍判握拳 |
| G-04 | 张开五指（静止） | `CMD_IDLE` | 静态手型，3 帧表决 | 须"手不动"才成立，松手后可再触发 |
| G-05 | 剪刀手 | `CMD_SPIN` | 静态手型，3 帧表决 | 食指+中指伸，无名指/小指/拇指蜷 |
| G-06 | 拇指食指捏合 | `CMD_STOP` | 2 帧直出 | 备用急停，手小或握不紧拳时更好做 |
| G-10 | 前推 | `CMD_FORWARD` | 3 帧几何 + 3 帧表决 | 掌心朝向 + z 靠近 / y 上抬 |
| G-11 | 左挥 | `CMD_TURN_LEFT` | 水平位移 ≥0.12（≤8 帧） | 第 4 帧反向可回滚 |
| G-12 | 右挥 | `CMD_TURN_RIGHT` | 同上 | 同上 |
| G-13 | 转圈 | `CMD_SPIN` | 累计转角 ≥330° + 闭环拟合 | 12~24 帧窗口，单帧直通 |
| G-20 | 上挥 | `CMD_FORWARD` | 竖直位移 ≥0.10（≤8 帧） | 掌心未正对镜头时的竖直路径 |
| G-21 | 下挥 | `CMD_IDLE` | 同上 | 反向 |
| G-30 | 双掌前推 | `CMD_FORWARD` | 双手 + 3 帧表决 | 需两只手都在画面内 |
| G-31 | 双手握拳 | `CMD_STOP` | 2 帧直出 | 语义最强的急停 |
| — | 看门狗 | `CMD_IDLE` | 输入中断 500±40ms | 优先级仅次于急停 |

**调速**：运动指令按"食指尖以外"伸出的手指数分挡 —— 0 根→0.4、1 根→0.7、≥2 根→1.0。

**方向语义**：`MIRROR_X=true`（默认）按用户视角判方向；`MIRROR_X=false` 为 MediaPipe 原始观察者视角。

---

## 附录 B · 本轮改动的文件清单

| 文件 | 改动要点 |
|---|---|
| `main.py` | HUD 姿态发布 `_publish_pose`、镜像 `_mirror_landmarks`、双手 `detect_all`、视觉预热、调速 `_speed_from_fingers`、降级日志字段修正 |
| `src/api/schemas.py` | 手势词表扩到 13 个、`GESTURE_TO_CMD`/`GESTURE_LABELS` 扩充、新增 `ESTOP_CODES` |
| `src/api/ws.py` | 下行帧新增 `source` / `camera_available` / `hands` |
| `src/api/routes.py` | `latest_pose` 每帧发布、补 `source` 字段、注入链路一致性 |
| `src/api/injector.py` | 新增 8 个合成手型（竖拇指/剪刀手/捏合/上下挥/双手） |
| `src/agents/debounce.py` | `ESTOP_CODES` 快通道、静态保持去刷屏、`MOTION_CODES`/`DIRECTION_CODES` 扩充 |
| `src/core/config.py` | `WATCHDOG_TICK_HZ` 默认 10→25、`CameraConfig.mirror_x`、新增几何阈值块 |
| `src/core/logger.py` | `_emit_console` 容忍任意类型 level |
| `src/vision/classifier.py` | 手部尺寸/指态/新静态手势判据、握拳收紧 |
| `src/vision/trajectory.py` | `dominant_swipe_vertical` 上下挥动 |
| `src/vision/pipeline.py` | 新候选优先级、双手组合、静止门槛 |
| `src/static/index.html` | 13 个注入按钮与快捷键、双手骨架渲染、姿态来源标注 |
| `README.md` | 手势词表、镜像语义、调速说明、HUD 操作更新 |
| `.env.example` | `MIRROR_X`、`WATCHDOG_TICK_HZ` |

---

## 附录 C · 常用命令速查

```powershell
# 起服务（HUD 在 http://127.0.0.1:8000/）
cd C:\Users\BY\PycharmProjects\Gesture
.\.venv\Scripts\python.exe main.py

# 一键验收（门禁 G3）
.\.venv\Scripts\python.exe scripts\verify_all.py

# 分项评测
.\.venv\Scripts\python.exe scripts\test_trajectory.py
.\.venv\Scripts\python.exe scripts\test_debounce.py
.\.venv\Scripts\python.exe scripts\bench_latency.py

# 摄像头自检（服务需先停）
.\.venv\Scripts\python.exe -c "import cv2;c=cv2.VideoCapture(0,cv2.CAP_DSHOW);print('可用' if c.isOpened() and c.read()[0] else '不可用');c.release()"

# 确认走的是真实视觉链路
Select-String -Path logs\session_*.jsonl -Pattern '"source": "GESTURE"' | Select-Object -Last 5
```

HUD 快捷键：`1`~`9` → G-02/G-10/G-11/G-12/G-13/G-03/G-04/G-05/G-06；`A S D F` → G-20/G-21/G-30/G-31。
