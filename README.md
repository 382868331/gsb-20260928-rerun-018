# 空间声音引擎（基础版）

Python 实现的连续时间双声道空间声音引擎。声音资源（`SignalSource`）、
发声对象（`SoundSource`）和听者（`Listener`）状态彼此独立，可分别更新；
当前实现移动声源的双声道渲染，接口为后续房间反射预留扩展点。

## 模块

- `spatial_audio/signals.py` — 可复用声音资源，如相位连续的人工音 `SineTone`。
- `spatial_audio/source.py` — 发声对象：位置/轨迹、音量、启用状态、停用时刻。
- `spatial_audio/listener.py` — 听者位置与朝向，独立于声源更新。
- `spatial_audio/trajectory.py` — 分段线性轨迹（位置连续、速度逐段恒定）与 JSON 录制。
- `spatial_audio/engine.py` — 连续时间渲染器：
  - 每耳传播延迟（距离/声速），可变延迟同时产生多普勒；
  - 距离衰减 `r0/(r0+d)`；
  - 头影 ILD（最大 6 dB）实现左右定位；
  - 启用 5 ms 起音、停用 30 ms 线性收尾，消除突发咔哒声；
  - 多声源求和后 `tanh` 软限制，保证永不超 [-1,1]，并在报告中记录越限样本数。
- `spatial_audio/scenario.py` — 场景（含轨迹/停用事件）保存为带 SHA-256 的 JSON 并可重放。
- `spatial_audio/export.py` — 导出 16-bit 立体声 WAV。

## 连续时间与分块无关

渲染内部以固定 512 样本 DSP 子块推进，所有几何量按音频采样逐点计算，
相位、延迟线、包络状态都建立在连续时间轴上。对外可用任意 `process(n)`
分块拉取，结果逐位一致（测试中块大小 1/37/256/4096 的最大差为 0）。

## 运行

```bash
pip install -r requirements.txt
python demo_moving_source.py
```

产物位于 `outputs/`：

- `moving_source.wav` — 7 秒可听结果：440 Hz 音从听者左侧沿 y=4 m 移动到右侧，
  另有 660 Hz 声源在 t=2.5 s 突然停用并平滑收尾；
- `channel_levels.csv` — 250 ms 窗口的左右 RMS 与主导声道；
- `scenario.json` — 可重放的运动/事件描述（含校验和）；
- `verification.json` — 声道、分块无关性、重放一致性、收尾与电平报告。

## 核对结果

- 起始窗口左声道占优（RMS 0.199 vs 0.103），末段右声道占优（0.099 vs 0.056）；
- 块大小 1/256/4096 渲染最大绝对差为 0；保存后重放最大差为 0；
- 声源突然停用后进入 30 ms 收尾，尾部最大样本跳变约 0.04（硬切断约 1.6）；
- 20 个同放声源的越限场景被软限制吸收，输出不削波，且报告越限样本数。

运行测试：`python -m pytest tests -q`。
