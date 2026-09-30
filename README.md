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
  RoomScene（在房间-门图上枚举所有简单路径，回答多路径直达/混响透射查询）、
  RoomReverb（反馈梳状 + 全通滤波器组，zi 跨块延续且可导出/恢复，尾音与
  分块大小无关）。

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
- 多房间图传播：房间是节点、门是边；声源房间到听者房间之间的每条简单路径
  并行贡献并求和，经直达门的声音与绕经其他房间的声音可同时到达。链条上任一
  门移动都沿整条链连续改变音量与频谱（多跳累积轻微低通）；房间不连通时声音
  完全阻断而非人为泄漏。
- 遮挡与门：每扇门按开度把信号分到“干”与“低通闷声”两路，直达声增益与频谱
  随门开度连续变化；门关闭后直达声明显受阻，但源房间的混响尾音仍按较低的
  透射增益沿房间图传入，可清晰分辨。
- 混响：每个房间一条立体声混响总线（梳状 + 全通，lfilter zi 跨块延续，可用
  get_state/set_state 导出恢复）。声源只发送到自己所在房间的总线，滤波器状态
  始终挂在源房间上：听者跨界或路径切换不会抹掉尚未衰减的尾音；听到的各房间
  混响增益沿房间图随门开度连续变化。
- 中途保存续渲：全部跨块状态（播放头、包络、增益/低通/遮挡平滑器、各房间混响
  zi 与增益、整数采样时钟）可由 SpatialEngine.suspend_state/restore_state（或
  save_state/load_state）整体快照并 JSON 序列化。每个采样时刻由整数索引
  (k+0.5)/sr 给出，不按分块累加浮点时长，所以“渲染→保存→重建→恢复→继续”
  与一次连续渲染等长且逐采样相符，两段用不同 block_size 也不引入额外爆音。
- 换门状态重放：RoomScene.with_door_motions / recording_with_doors 只替换门的
  DoorMotion，声源轨迹、资源、音量与启停事件原样保留，可用同一条行走记录反复
  试听不同门状态。
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
- outputs/multipath_recording.json / demo_multipath.wav — 三房间 A-B-C 链：
  听者连续穿过两个房间边界，两扇门在各自片段内独立渐开渐合，声源停止后仍有
  房间 C 混响尾音；demo_multipath_doors_shut.wav 是同一记录仅把门换成全关的
  重放（启停安排不变）；multipath_state.json + demo_multipath_resumed.wav 是
  2.31 s 处保存状态、用不同 block_size 续渲的结果（与连续渲染差异约 1e-12）。

测试：

    .venv/bin/python -m pytest tests/ -q
