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
  render_interval(...) 只渲染选定区间，结果与整段渲染后截取完全一致；
  save_session/load_session 在中途保存/恢复整个引擎状态（含仍在衰减的
  混响尾音），重开后从断点继续听。
- spatial_audio/rooms.py — Room（轴对齐房间盒 + 混响参数）、Portal（连接两个
  房间的门，DoorMotion 关键帧描述开度随时间变化，smoothstep 插值保证连续）、
  RoomScene（房间-门构成图，直达声沿所有简单门路径求和：近路被关门切断时
  绕行路径仍在传声；混响透射同样多跳传播）、AbsorptionZone（吸声区域，
  只衰减几何上穿过该区域的传播路径，纯增益、不改变频谱，与关门造成的
  低通闷化可区分）、RoomReverb（反馈梳状 + 全通滤波器组，状态跨块延续，
  尾音与分块大小无关，可整体快照/恢复）。

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
- 多房间路径：走廊与多个相连房间构成门图，直达声为各条门路径增益之和
  （每条路径增益为沿途各门透射的乘积），门逐渐开合时直达与绕行声各自
  连续变化；已产生的余音继续在各自房间衰减并沿可用路径传到听者。
- 吸声改造：AbsorptionZone 只影响穿过它的传播（直达线段、听者到房间中心
  的混响路径），不经过的路径逐采样不变，不会让全场统一变小声。
- 断点续听：engine.snapshot()/restore_state() 捕获播放头、包络、滤波器与
  每个房间混响的滤波器状态；save_session 把房间方案、轨迹与状态一起存成
  JSON，load_session 恢复后 render(..., start_time=t) 与一次性整段渲染
  逐采样一致（误差约 1e-12）。
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
- outputs/corridor_recording.json / demo_corridor.wav — 走廊 + 三房间
  （含绕行环路）行走记录：door_b 在 2.5-3.5 s 逐渐关闭，直达路径消失后
  绕行路径（经 room3）继续传声，8 s 声源停止后余音继续衰减。
- outputs/corridor_replay_alt_doors.wav — 同一条行走记录换一套门安排
  （door_b 常开、door_c 常闭）重放的结果。
- outputs/corridor_session.json / corridor_resumed.wav — 5 s 处保存会话，
  重开后接着渲染，与整段渲染一致。
- outputs/corridor_treated.wav / absorption_compare.png — 同一条记录分别
  采用原墙面与新增吸声区域，在两个收听位置的电平对比及声源停止后的
  衰减曲线。

测试：

    .venv/bin/python -m pytest tests/ -q
