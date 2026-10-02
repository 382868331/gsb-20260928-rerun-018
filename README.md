# 空间声音引擎

纯 Python + NumPy/SciPy 实现的连续时间双声道空间声引擎。支持直达声
（方位声像、距离衰减、运动多普勒、平滑开关、多声源混合）以及房间/门洞
参与的传播：遮挡衰减与低通、按门开度连续变化的音量与频谱、每房间混响尾音。

## 模块

- spatial_audio/resources.py — SoundResource：不可变单声道采样数据，可被任意
  发声对象共享、独立替换；SoundResource.sine_tone(...) 生成确定性人工音调。
- spatial_audio/trajectory.py — Trajectory：按时间点的 3D Catmull-Rom 轨迹，
  任意连续时刻都可采样出位置和速度。
- spatial_audio/scene.py — Listener（位置/朝向，可独立更新）与
  Emitter（声音资源 + 轨迹 + 音量/开关），三者状态互不耦合。
- spatial_audio/engine.py — SpatialEngine：核心渲染。
- spatial_audio/recording.py — MotionRecording：把运动轨迹、音调参数、开关事件
  存为 JSON，可重建场景并重放，导出结果可复现。
- spatial_audio/rooms.py — Room（轴对齐房间盒 + 混响参数）、Portal（连接两个
  房间的门，DoorMotion 关键帧描述开度随时间变化，smoothstep 插值保证连续）、
  RoomScene（回答直达/混响透射查询）、RoomReverb（反馈梳状 + 全通滤波器组，
  状态跨块延续，尾音与分块大小无关）。
  AbsorptionZone（轴对齐吸声区域）按频带衰减几何路径穿过该区域的传播，
  材料可在重放途中按关键帧更换。

## 关键保证

- 连续时间推进：每个输出采样都在其真实时刻采样轨迹；播放头为浮点值并跨块延续，
  增益包络/开关包络状态同样逐采样延续。因此改 block_size 只改变计算分块，
  不改变声音（test_block_size_independence 用 1/37/256/1024/10007 对比）。
- 平滑收尾：声源停用走 3 ms 起音 / 15 ms 释音的包络，无咔哒声；
  开关事件按采样时刻精确调度，不依赖块边界。
- 混合不溢出：各声源独立渲染后相加；超过阈值时进入 tanh 软限制器，
  输出峰值不超过阈值，且 RenderStats.limited 与衰减量会明确报告，不会静默削波。
- 空间效应：等功率左右声像（基于听者右轴）、轻微前后差别、
  1/(1+d/d0) 距离衰减、沿听者-声源连线投影速度的多普勒变调。
- 遮挡与门：声源与听者不在同一房间时，直达声按门开度连续衰减并经固定低通
  交叉淡化（频谱随门开度连续变化）；门关闭后直达声明显受阻，但源房间的
  混响尾音仍按较低的透射增益传入，可清晰分辨。
- 多房间与绕行路径：RoomScene 在门户图上枚举两房间之间的所有简单路径
  （直达 + 绕行），每条路径增益为沿途各门透射的乘积，各路径能量相加
  （上限 1）。走廊–房间1–房间2 串联时，关闭直达门后绕行路径仍然可闻，
  与"路径完全消失"（NO_PATH 级别）明显可区分；远端房间的混响尾音也按
  最强门户路径继续传入并随时间衰减。
- 频带化吸声：AbsorptionZone 按命名频带（默认 low/mid/high，交叉频率
  400/2500 Hz，RoomScene(bands=..., band_edges=...) 可改）给出各自的
  线性增益；信号经互补频带分解后逐带着色，各带增益都为 1 时严格还原
  原信号，旧的宽频 gain/muffle 写法继续可用。只有听者↔门户↔声源折线
  穿过区域的传播才受影响，未改造区域频率组成逐采样不变。
- 中途换装：zone.set_schedule(key, [(time, band_gains, muffle), ...]) 可
  在重放途中安装/更换材料（带 blend 秒交叉淡化，无咔哒声）。换装只对
  换后新传播的声音生效：动态材料在声源进入混响发送时按当前频带增益
  着色，此前已经进入房间的衰减尾音保留它实际经过的频谱，不会被整体
  重新染色。静态材料则继续沿收听侧路径作用于可闻尾音。
- 多材料与路径追踪：连续穿过多块吸声区域时各带增益依次相乘，与只
  穿过一块明显不同；RoomScene.path_report(listener, source, t) 返回某
  收听时刻所有传播路径（直达/绕行、各门开度与增益、沿途材料及每带
  总增益），按强度排序，可据此追到主要传播路径与所用材料安排。
- 一次记录、多种重放：MotionRecording 保存轨迹/音调/开关事件与房间方案；
  render_recording(..., scene=...) 或 recording.with_scene(scene) 可用
  同一行走记录重放不同的门安排或加吸声区域的方案。
- 中途保存与续听：engine.render_chunk 推进会话时钟；save_checkpoint 把
  记录与全部延续状态（播放头、包络、滤波器与混响尾音状态）存入 JSON，
  load_checkpoint 恢复后继续 render_chunk，与不间断渲染逐采样一致
  （当时仍在衰减的余音被完整保留）。
- 区间截取一致：engine.render_interval(t0, t1) 先快进到 t0 再渲染，
  结果与整段渲染的 [t0, t1] 切片逐采样一致。
- 混响：每个房间一条立体声混响总线（梳状 + 全通，lfilter zi 跨块延续），
  声源按所在房间发送；听者听到的混响增益同样随门开度连续变化。
- 状态独立：每个声源的播放头、包络、滤波器状态各自独立，启动/停止某个
  声源不会改变其他声源的播放位置；声源瞬移、听者穿越房间边界都经
  平滑处理，无爆音。

## 运行

    .venv/bin/python demo.py

生成：

- outputs/demo_spatial.wav — 6 秒 48 kHz 立体声可听结果（移动 440 Hz 音调
  从左到右；右侧 660 Hz 声源在 t=2 s 被平滑关闭）。
- outputs/channel_check.png — 左右声道 20 ms RMS 与 L-R 平衡曲线，用于核对声像。
- outputs/motion_recording.json — 运动过程记录，可用不同分块重放。
- outputs/demo_spatial_replay.wav — 从 JSON 重放的结果（与原始差异约 1e-12）。
- outputs/room_recording.json / demo_rooms.wav / demo_rooms_replay.wav /
  room_check.png — 双房间 + 门场景：门在 2 s 关闭、4.5 s  reopen，
  5 s 声源停止后仍可听到隔壁房间的混响尾音。
- outputs/corridor_recording.json / corridor_walk.wav — 走廊 + 两个相连
  房间：声源沿走廊行进，door2 关闭期间声音经 room1 绕行进入 room2。
- outputs/corridor_alt_doors.wav — 同一行走记录换一套门安排重放。
- outputs/absorb_*_plain.wav / absorb_*_treated.wav / corridor_check.png —
  吸声改造对比：同一记录在 room2 与走廊两个收听位置、原墙面与新增
  吸声区域两种方案下的声音与停止后的衰减。
- outputs/corridor_checkpoint.json / corridor_resumed.wav — 中途保存的
  检查点及续听结果（与整段渲染对应区间差异约 1e-12）。

测试：

    .venv/bin/python -m pytest tests/ -q
