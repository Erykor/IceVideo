# IceVideo

一套用来**辅助 AI 剪视频**的小工具集。

设计哲学：**AI 是编辑，工具是眼睛和手**。每个工具独立可用，不互相耦合，没有"一键出片"的 black box。AI 自己读帧、读时间线、做编辑决策，然后用工具去切、去拼。

适合：让 AI agent（Claude Code、自定义 agent 等）在终端里读视频内容并做出剪辑决定。**不适合**：希望按一个按钮就出片的非 AI 用户。

## 文档地图

| 文件 | 给谁看 |
|---|---|
| **README.md** | 想用工具的人（人或 AI）— 工具清单、典型工作流 |
| [AGENTS.md](./AGENTS.md) | **代用户剪片子的 AI** — 怎么思考、怎么读、怎么决定、什么时候用哪个工具 |
| [config.toml](./config.toml) | 调参的人 — 分析器默认参数（权重、阈值、prompts）|

## 安装

```bash
git clone <repo> ~/IceVideo
cd ~/IceVideo
uv sync                       # 基础：CPU torch + Whisper + CLIP
uv sync --extra faces         # 加 InsightFace（人脸识别）
uv sync --extra critique      # 加 Anthropic SDK（VLM 帧描述）
uv sync --extra music         # 加 librosa（音乐节拍分析）
uv sync --extra diarize       # 加 pyannote（说话人识别，需 HF_TOKEN）
```

切到 GPU torch：编辑 `pyproject.toml` 的 `[tool.uv.sources]`，把 `pytorch-cpu` 换成 `pytorch-cu124` 然后 `uv sync`。

依赖外部：
- `ffmpeg 6+`（必需）
- `exiftool`（可选，做 GoPro GPMF 解析）
- `ANTHROPIC_API_KEY`（可选，用 VLM 描述帧）
- `HF_TOKEN`（可选，用 pyannote）

## 工具分类

### Sensors —— 从视频抽数据

| 命令 | 输出 | 用途 |
|---|---|---|
| `icevideo probe <video>...` | stdout 表 | 时长 / 分辨率 / fps / 编码 / 音轨 |
| `icevideo extract` | `work/audio/*.wav` + `work/clip_frames/<v>/*.jpg` | 16kHz 单声道 WAV + 0.5fps CLIP 帧 |
| `icevideo transcribe` | `work/transcripts/*.{json,srt}` | stable-whisper 带 word probability |
| `icevideo signals` | `work/signals/*.json` | 每秒 scene/flow/RMS/voiceness/laughter |
| `icevideo clip-score` | `work/clip_scores/*.json` + `clip_embs/*.npy` | CLIP 与 prompt 的相似度 + 帧嵌入 |
| `icevideo boundaries` | `work/boundaries/*.json` | silence + scene cut 时间戳（snap 用）|
| `icevideo telemetry` | `work/telemetry/*.json` | EXIF / GPMF（黄金时刻 + GPS）|
| `icevideo faces` | `work/faces/*.json` | per-second 主角脸面积 + identity 聚类 |
| `icevideo diarize` | `work/diarize/*.json` | 说话人 turn |
| `icevideo music <file>` | `work/music/onsets.json` | librosa 节拍点 + BPM |
| `icevideo frames <video> --at T...` | `<out>/<base>_t<T>.jpg` | 抽指定时刻帧给 AI 看 |
| `icevideo frames <video> --around T --window N --step S` | 同上 | 某时刻 ±N 秒密集抽帧（找精确切点）|
| `icevideo audio-clip <src> --start T --end T --out F.mp3 [--spectrogram]` | mp3 + png | 抽音频段（可顺带出 spectrogram）|

### Analyzers —— 把数据变成 AI 能读的东西

| 命令 | 输出 | 用途 |
|---|---|---|
| `icevideo timeline <video>` | stdout markdown 表 | 把所有信号合并成 per-second 可读时间线 |
| `icevideo describe <img>...` | stdout 文本 | Claude/Gemini VLM 描述帧（批量扫帧用） |

### Effectors —— 真正的剪辑动作

| 命令 | 输出 | 用途 |
|---|---|---|
| `icevideo cut <src> --start T --end T --out F [--slow X --interp mci]` | 单段 mp4 | NVENC + loudnorm + 慢放（含 motion-compensated 插帧）+ J cut + 亮度调整 |
| `icevideo cut --from-csv FILE [--out-dir DIR]` | N×mp4 | 批量模式（CSV 列：src,start,end,out,slow,interp,brightness,saturation,audio_offset,snap_start,snap_end,name）|
| `icevideo cut ... --snap-start --snap-end` | mp4 | 自动 snap 到最近 silence/scene（需 `boundaries` 先跑）|
| `icevideo concat <clips...> --xfade 0.3 [--audio-xfades 0.5,0.8]` | 拼好的 mp4 | xfade 视频 + per-seam 自定义 acrossfade（实现跨段 J/L cut）|
| `icevideo subtitle <video> <srt> --out F [--softsub]` | 带字幕 mp4 | 烧字幕（默认）或软字幕轨道 |
| `icevideo music-mix <video> <music> --out F` | 带配乐 mp4 | sidechain 压缩 ducking，音乐自动循环 |

### Advisors —— 给 AI 提建议（不替 AI 决定）

| 命令 | 输出 | 用途 |
|---|---|---|
| `icevideo select` | `work/plan.json` | 跑综合打分给出一份"候选计划"。**AI 通常会编辑这份计划**，不是直接用 |
| `icevideo lint` | stderr 报告 | 检查 plan.json：太短、太长、切到 word 中间、相邻太像 |
| `icevideo critique` | `work/critique.json` | VLM/启发式给成片打分（0-10）+ 改进建议 |
| `icevideo render` | `output/highlight_*.mp4` | 便捷包装：吃 plan.json 直接出片（等价于 N 次 cut + concat）|

## 典型的 AI 工作流

下面是一个 AI agent 应该怎么用这些工具的示例。**没有任何一步是必须的**，AI 按需求挑：

```bash
# 0. 看看素材是啥
icevideo probe *.MP4

# 1. 抽通用数据（一次性，缓存到 work/）
icevideo extract --input-glob '*.MP4'
icevideo transcribe
icevideo signals
icevideo clip-score
icevideo boundaries
icevideo faces            # 可选
icevideo telemetry        # 可选

# 2. 让 AI 读这一段视频的时间线（每秒一行）
icevideo timeline GX010078.MP4 --only-active > /tmp/078.md
# AI 用 Read 工具看 /tmp/078.md，识别有意思的时刻

# 3. AI 想看某些时刻长什么样
icevideo frames GX010078.MP4 --at 93 99 105 --out /tmp/check/
# AI 用 Read 直接看 jpeg

# 4. （可选）让 VLM 批量描述一堆帧
icevideo describe /tmp/check/*.jpg

# 5. AI 决定要剪哪几段（可以一条一条切，也可以批量）
icevideo cut GX010054.MP4 --start 7 --end 14 --out /tmp/clips/01.mp4
icevideo cut GX010078.MP4 --start 93 --end 106 --slow 0.7 --interp mci --out /tmp/clips/02.mp4
icevideo cut GX010079.MP4 --start 32 --end 45 --snap-start --snap-end --out /tmp/clips/03.mp4

# 或一次 CSV 批量
cat <<'EOF' | icevideo cut --from-csv - --out-dir /tmp/clips
src,start,end,slow,interp,snap_start,snap_end,name
GX010054.MP4,7,14,,,,,01
GX010078.MP4,93,106,0.7,mci,,,02
GX010079.MP4,32,45,,,true,true,03
EOF

# 6. 拼起来（普通 xfade）
icevideo concat /tmp/clips/*.mp4 --xfade 0.3 --out /tmp/final.mp4

# 7. 跨段 J cut（第 2 个seam 让下一段音频提前 0.5s 进来）
icevideo concat /tmp/clips/*.mp4 --xfade 0.3 --audio-xfades 0.3,0.8 --out /tmp/final_jl.mp4

# 8. 加配乐 + 字幕（可选）
icevideo music-mix /tmp/final.mp4 ./song.mp3 --out /tmp/final_music.mp4
icevideo subtitle /tmp/final_music.mp4 ./captions.srt --out /tmp/final_done.mp4

# 9. 自己看成果
icevideo frames /tmp/final.mp4 --every 5s --out /tmp/final_check/
icevideo critique         # VLM/启发式评分
```

如果 AI 想要一个"起点"参考，可以跑 `icevideo select` 拿到一份候选 plan.json，**自己编辑这份 plan**（移除/替换段、调时间），然后跑 `icevideo render` 出片。但这只是一个起点，不是强制路径。

## 关键设计选择

1. **没有 `auto` / `run-all` 命令**。AI 是编辑，编辑不按按钮。
2. **每个工具读自己需要的东西，不假设别的工具跑过没**。timeline 在没有 clip_scores 时还是能输出（只是 clip 列空），cut 在没有信号时也能跑。
3. **缓存友好**。每个 sensor 跑完写 `work/<step>/<video>.json`，下次跳过。改了 config 也只需要重跑相关步骤。
4. **AI 直接读 timeline + jpeg**，不需要解析 JSON。markdown 表格 + 抽出来的帧是 AI 最自然的输入。
5. **cut 和 concat 是原语**。所有剪辑动作都可以表示成"切 N 个段 + 拼一次"。慢放、亮度、loudnorm、J cut 都是 cut 的参数，不是单独的命令。

## 配置

`config.toml` 影响的是**分析器**（signals 的权重、clip 的 prompts、render 的默认参数）。每个工具也接受 CLI 参数覆盖。复制一份 `config.toml` 到你的项目目录会自动覆盖默认。

## 目录结构

```
~/IceVideo/
├── pyproject.toml          # uv + 可选 extras (faces / diarize / music / critique)
├── config.toml             # 分析器默认参数
├── README.md / AGENTS.md
├── tests/                  # pytest 烟雾测试（45 个）
└── icevideo/
    ├── cli.py              # 总入口，每个 subcommand 转发到对应模块
    ├── config.py           # 加载 config.toml + Paths（支持多 input_dir）
    ├── utils.py            # zscore/smooth/cosine_sim/ffmpeg/ffprobe
    │
    ├── # SENSORS (data extractors)
    ├── probe.py            # 元数据
    ├── extract.py          # WAV + CLIP frames
    ├── transcribe.py       # stable-whisper
    ├── signals.py          # 每秒运动+音频
    ├── clip_score.py       # CLIP 相似度
    ├── boundaries.py       # silence + scene
    ├── telemetry.py        # ffprobe + GPMF
    ├── faces.py            # InsightFace + Haar fallback
    ├── diarize.py          # pyannote (graceful skip)
    ├── music.py            # librosa onset
    ├── frames.py           # 抽指定时刻帧（含 --around / --around-peaks）
    ├── audio_clip.py       # 音频段 + spectrogram
    │
    ├── # ANALYZERS (data → readable)
    ├── timeline.py         # 可读时间线
    ├── describe.py         # VLM 帧描述
    │
    ├── # EFFECTORS (precise execution)
    ├── cut.py              # 单段切片（含批量 / snap / mci 插帧）
    ├── concat.py           # xfade + 跨段 J/L cut
    ├── subtitle.py         # 烧字幕 / 软字幕
    ├── music_mix.py        # 配乐 bed + sidechain ducking
    │
    └── # ADVISORS (suggestion, not decision)
        ├── select.py       # 候选 plan 推荐
        ├── lint.py         # plan 体检
        ├── critique.py     # 成片自评 (VLM / 启发式)
        └── render.py       # 便捷：吃 plan 直接出片
```

## 已知限制

- **慢放有三档**：`--interp linear`（设 PTS，默认）/ `--interp blend`（帧混合）/ `--interp mci`（运动补偿）。`mci` 比 linear 丝滑，但仍非 RIFE/FILM 级别。
- **scene 边界对齐效果有限**：GoPro 单镜头连拍，硬切点很少。silence snap 是主力。
- **CLIP 是英文模型**，prompts 用英文最稳。
- **describe / critique 需 ANTHROPIC_API_KEY**，否则 critique 退化到规则评分，describe 不可用。
- **diarize 需 HF_TOKEN**，否则跳过。

## 测试

```bash
uv run pytest tests/        # 45 个烟雾测试，约 8 秒跑完
```

测试用 `ffmpeg lavfi` 现合成微型测试视频，所以不依赖任何真实素材。
