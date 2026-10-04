# 空间声音引擎

纯 Python + NumPy/SciPy 实现的连续时间双声道空间声引擎。支持直达声
（方位声像、距离衰减、运动多普勒、平滑开关、多声源混合）以及房间/门洞
参与的传播：遮挡衰减与低通、按门开度连续变化的音量与频谱、每房间混响尾音；
Panel 提供有限大小、有朝向、可移动旋转、可按区间修订的频带透过面板。

## 模块

- spatial_audio/resources.py — SoundResource：不可变单声道采样数据，可被任意
  发声对象共享、独立替换；SoundResource.sine_tone(...) 生成确定性人工音调。
- spatial_audio/trajectory.py — Trajectory：按时间点的 3D Catmull-Rom 轨迹，
  任意连续时刻都可采样出位置和速度。
- spatial_audio/scene.py — Listener（位置/朝向，可独立更新）与
  Emitter（声音资源 + 轨迹 + 音量/开关），三者状态互不耦合。
- spatial_audio/engine.py — SpatialEngine：核心渲染。
- spatial_audio/recording.py — MotionRecording：把运动轨迹、音调参数、开关事件
  存为 JSON，可重建场景并重放，导出结果可复现；kind="mix_tones" 可在同一
  声源中放低/中/高多个正弦成分，用于逐频带核对。
- spatial_audio/rooms.py — Room（轴对齐房间盒 + 混响参数）、Portal（连接两个
  房间的门，DoorMotion 关键帧描述开度随时间变化，smoothstep 插值保证连续）、
  RoomScene（回答直达/混响透射查询）、RoomReverb（反馈梳状 + 全通滤波器组，
  状态跨块延续，尾音与分块大小无关）。
   AbsorptionZone（轴对齐吸声区域）只衰减几何路径穿过该区域的传播；
   可用 bands=[...] 按频带给出作用程度（每频带一个增益，或 [(t, v), ...]
   关键帧随时间 smoothstep 变化，即可在重放途中更换材料），频带划分由
   RoomScene(band_edges=...) 决定（默认 1 kHz 单分频，低/高两个频带）。

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
- 吸声改造对比：AbsorptionZone 只作用于几何路径（听者↔门户↔声源折线）
  穿过该区域的传播——增益乘以 zone.gain 并增加额外低通混合；不穿过的
  路径逐采样不变，因此不会让所有房间统一变小声。门关闭（路径消失 +
  混响透射同时被压）与吸声增强（仅过区路径被衰减、室内尾音保留）在
  输出上可明确区分。
- 频带化吸声：给区域指定 bands 后，直达路径与混响发送都按频带分别
  加权（引擎用互补一阶分频把信号拆成各频带，状态逐采样延续，分块
  大小不改变结果）。同一材料可以明显削弱高频而保留低频；连续穿过
  多块区域的路径按各区域逐频带增益连乘，与只穿过一块结果不同。
  材料关键帧在重放途中改变时，正在衰减的混响尾音保持其产生时的
  频谱（频带加权施加在进入混响总线的发送端），只有新产生的声音
  使用新材料。RoomScene.explain_path(listener, source, t) 返回某收听
  时刻的最强传播路径：经过的门及开度、穿过的吸声区域及其当时的
  逐频带增益。未指定 bands 的区域仍按原宽频 gain/muffle 工作。
- 可移动可修订面板：Panel 是有限矩形频带面板，位置与法向朝向按时间
  smoothstep 插值；RoomScene(band_edges=(500, 2000)) 时可直接给低/中/高
  三频透过量。面板移动到一条路径外时，只是不再给新声音加滤波，进入
  混响总线的旧尾音继续按其产生时的面板版本衰减；另一条从不相交的路径
  逐采样不变。PanelRevision(start, end, ..., bands=[...]) 仅在指定区间
  替换位置/朝向/频带透过量，频带填 None 表示沿用基础版本。
  SpatialEngine.render_recomputed_from(boundary, duration) 先快进到边界、
  再清空混响尾音总线，供比较“保留历史”与“从边界重算”的听感差别。
  panel_version_manifest(recording) 导出每段实际使用的面板版本、位置、
  朝向和逐频带增益；explain_path 也会列出当时穿过的面板。
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

- outputs/band_room2_plain.wav / band_room2_treated.wav /
  band_room2_swap.wav / band_check.png — 频带化吸声对比：同一行走记录，
  原墙面、低通透/高频吸收面板、以及 t=5 s 中途更换材料三种方案，
  图中按 1 kHz 分频分别对比低/高频带能量。
- outputs/panel_history.wav / panel_recomputed_boundary.wav /
  panel_route_crossing.wav / panel_route_other.wav / panel_check.png —
  低/中/高三频声音与两条路径上的移动可修订面板：面板滑离相交路径后旧
  尾音保留，另一条路径全程不受影响；history 与 boundary recompute 两个
  WAV 可直接 A/B 试听。
- outputs/panel_versions.json / panel_recording.json /
  panel_checkpoint.json — 每段实际使用的面板版本（位置、朝向、三频透过量）、
  可复放记录和边界暂停检查点。

测试：

    .venv/bin/python -m pytest tests/ -q
