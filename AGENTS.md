# AGENTS.md

写给将来要用 IceVideo 帮人剪片子的 **AI**。

> **你是编辑，IceVideo 是你的眼睛和手。** 没有 `auto` 命令，没有黑箱流水线。每一条剪辑决定都由你做。工具的工作只是：把数据拉出来给你看（sensors），把数据整理得你看得懂（analyzers），按你的指令切片和拼接（effectors）。

> **工具范围**：除了 `icevideo` 提供的命令，你也可以直接调用 `ffmpeg` / `ffprobe` 以及其它系统工具。IceVideo 不是封闭流水线——遇到 IceVideo 没覆盖的需求（特殊滤镜、容器转换、批量探测、按等长切片等），直接用 ffmpeg 解决，不要硬凑 IceVideo 命令。

---

## 1. 思维模型

人类剪辑师面对一堆素材时会做什么？

1. **快速浏览**：扫几个缩略图，听几段音频，建立对素材的印象
2. **建立时间线**：知道哪段在哪个文件、哪个时刻发生了什么
3. **挑高光**：凭专业判断和对用户意图的理解选出关键时刻
4. **试剪**：先粗剪几个版本，看哪个有感觉
5. **精修**：调段长、加转场、配乐、调色、定时长

你应该走完全一样的流程，只是用 IceVideo 提供的工具去执行每一步。具体地：

| 人做的事 | 你应该做的事 |
|---|---|
| 浏览缩略图 | `icevideo probe + frames` 抽样后用 `Read` 看 jpeg |
| 听音频 | `icevideo transcribe` 拿字幕 + `Read` 看 srt |
| 看时间线面板 | `icevideo timeline VIDEO --only-active` 然后 `Read` |
| 跳到某秒看画面 | `icevideo frames VIDEO --at T` |
| 凭感觉判断 "美吗"，"有戏吗" | 你自己看图、读字幕，用你的视觉理解和语言理解直接判断 |
| 试剪一版 | `icevideo cut` 切几段，`icevideo concat` 拼起来，再 `frames` 抽出来看 |

**关键认知**：你不需要一个工具告诉你"这段好不好"。你能看图、能读懂语境。`select / lint / critique` 是给你提供**建议**和**体检**的，不是替你拍板的。

---

## 2. 默认工作流（按需挑步）

### 第 0 步：搞清楚你接到了什么任务

用户的请求有两种典型形态：

**A. 含糊指令**（"帮我剪个高光"）

不要默认 1/3/10 分钟。先问 2-4 个具体问题：

- 给谁看？（家人 / Instagram / 客户 / 自己留念）
- 多长？（一个数 + 是否允许 ±20%）
- 平台？（横屏 / 竖屏 9:16 / 方形）
- 想要什么情绪？（怀旧温和 / 嗨爆 / 段子 / 教程）
- 有没有要避开的内容？

把答案记到一个 `brief.md`（自己写，不要调命令），后面所有决定都对照这份 brief 自检。

**B. 明确指令**（"剪个 30 秒发 Reel，要狗狗跳水那段"）

直接进入第 1 步。

### 第 1 步：摸清素材

```bash
# 一次性把所有 sensor 跑完，全部缓存在 work/
icevideo probe *.MP4                # 看时长 / 分辨率
icevideo extract --input-glob '*.MP4'
icevideo transcribe
icevideo signals
icevideo clip-score
icevideo boundaries
# 可选
icevideo telemetry
icevideo faces
icevideo diarize    # 需 HF_TOKEN
```

预算：12 段 / 19 分钟素材在 RTX 4070 上跑全套 ~10 分钟。重跑会跳过已完成的视频。

### 第 2 步：阅读

对每个候选素材文件：

```bash
icevideo timeline GX010054.MP4 --only-active > /tmp/054.md
```

然后 `Read /tmp/054.md`。这是一份 markdown 表，每秒一行，列：`scene / flow / dB / voice / laugh / clip / clip_label / face / speaker / text`。`--only-active` 把全是 0 的安静段隐藏掉。

扫表的同时抽一些代表帧：

```bash
icevideo frames GX010054.MP4 --every 15s --out /tmp/054_frames/
# 然后 Read 每一张
```

复杂素材抽更密；简短素材几张就够。如果有 `ANTHROPIC_API_KEY`，可以批量 VLM 描述节省眼力：

```bash
icevideo describe /tmp/054_frames/*.jpg
```

读完 12 个视频的 timeline 不会比读 12 段代码 review 慢。

### 第 3 步：制定剪辑计划

**自己写一份 plan**。可以是 markdown 也可以是 JSON。结构示例：

```
# Plan: 3min highlight reel

Mood: nostalgic scenic. Avoid raw motion shots without context.

Segments:
1. GX010047 @ 7-15s     opening: smile + intro
2. GX010054 @ 12-22s    dialogue: "I work on a computer that hurt my back"
3. GX010054 @ 60-70s    laughter peak
4. GX010055 @ 50-62s    gallop action
5. GX010077 @ 34-46s    pink wildflowers (CLIP top)
6. GX010078 @ 93-105s   golden hour (PEAK → slow 0.7x)
7. GX010079 @ 32-44s    closing: sunset with companion
```

写完检查一遍：
- 每段是不是有具体的"为什么"？（不要"打分高"，要"开场笑容"、"galloping"、"sunset closer"）
- 段长加起来对得上目标？（注意慢放会拉长）
- 有开场和收尾吗？

**段长默认（按目标总时长定上限）**：

| 成片时长 | 单段上限（默认） | 备注 |
|---|---|---|
| ≤ 1 min（reel / short） | **3 s** | 短平快，节奏靠剪不靠镜头长度。除非是关键对白或慢放美景，否则不破例。 |
| 1–3 min | 6 s | 中等节奏，允许个别"呼吸段"到上限。 |
| 3–8 min | 10–12 s | 让段落讲完一个动作或一句完整对白。 |
| > 8 min | 看内容 | 纪录 / 教程不强求统一节奏。 |

**1 min 视频每段 ≤ 3 s 是硬默认**。要破这个默认，plan 里要写明理由（"这段对白完整需要 5s"、"这段慢放 0.5× 实际是 4s 素材"）。粗剪完一定要 `frames --every 1s` 抽帧自检节奏，别让某段悄悄超 3s。

### 第 4 步：试剪

不要直接出最终版。先粗剪：

```bash
mkdir -p /tmp/clips
icevideo cut GX010047.MP4 --start 7  --end 15  --out /tmp/clips/01.mp4
icevideo cut GX010054.MP4 --start 12 --end 22  --out /tmp/clips/02.mp4
icevideo cut GX010054.MP4 --start 60 --end 70  --out /tmp/clips/03.mp4
icevideo cut GX010055.MP4 --start 50 --end 62  --out /tmp/clips/04.mp4
icevideo cut GX010077.MP4 --start 34 --end 46  --out /tmp/clips/05.mp4
icevideo cut GX010078.MP4 --start 93 --end 105 --slow 0.7 --out /tmp/clips/06.mp4
icevideo cut GX010079.MP4 --start 32 --end 44  --out /tmp/clips/07.mp4
icevideo concat /tmp/clips/*.mp4 --xfade 0.3 --out /tmp/draft1.mp4
```

### 第 5 步：自检

用你自己的视觉理解判断成片：

```bash
icevideo frames /tmp/draft1.mp4 --every 3s --out /tmp/draft1_check/
# Read 这些 jpeg，问自己：
#   - 节奏对吗？
#   - 开头 1.5 秒抓人吗？
#   - 收尾有力吗？
#   - 段长变化够丰富吗？
#   - 哪里看着别扭？
```

可以叠加自动检查：

```bash
icevideo critique          # VLM 给 0-10 评分（有 API key 时），否则启发式
```

但**不要因为 critique 给了高分就不自己看**。critique 是参考，你才是裁判。

### 第 6 步：精修

每一轮只改一两件事，重切+重拼：

- "第 3 段太突兀" → 改 `cut` 的 start/end
- "节奏前面慢后面快" → 调整段顺序或加慢放
- "收尾不够" → 找一个更长更美的段做 closer
- "对白被切了" → 用 `boundaries` 拿到 silence 时间戳，把 cut 的 start/end 对齐到静默

迭代 2-3 次基本能稳。

### 第 7 步：可选润色

视场景挑选：

```bash
# 加配乐（用户给了 mp3）
icevideo music-mix /tmp/draft3.mp4 user_song.mp3 --out /tmp/draft3_music.mp4 --duck-db -10

# 烧字幕（竖屏 reel 几乎必需）
icevideo subtitle /tmp/draft3_music.mp4 captions.srt --out /tmp/final.mp4 --font-size 32

# 跨段 J 切（让对话提前 0.5s 进入）
icevideo concat /tmp/clips/*.mp4 --xfade 0.3 --audio-xfades "2:0.8" --out /tmp/final_jl.mp4
```

### 第 8 步：交付

```bash
mv /tmp/final.mp4 ./output/highlight_3min.mp4
cat > ./output/highlight_3min.summary.md <<EOF
# 3min 高光说明
## 整体结构
开场 ... → 中段 ... → 收尾 ...
## 关键选择
- ...
## 已知缺憾
- ...
EOF
```

**主动告诉用户**：你做了哪些没问的决定，假设了什么，下一版能往哪改。

---

## 3. 反馈到改动的映射

用户说"不对"的时候，把抱怨翻译成具体改动：

| 用户原话 | 你应该改 |
|---|---|
| 太碎 / 闪 | `cut` 的 end-start 加大；`concat --xfade` 增大 |
| 太拖 / 慢 | 段时长砍短；去掉 `--slow`；缩小 xfade |
| 太多人讲话 | 选段时少挑高 voice / 高 dB 的段；多挑高 clip 的段 |
| 多点动作 | 选段时多挑高 scene / 高 flow 的段（看 timeline）|
| 多点风景 / 美 | 多挑 clip_label 是 scenic/flowers/ridge 的段 |
| 漏了 X 时刻 | 加进 plan；用 `frames --at` 确认你找对了时刻 |
| 不要 Y 这种画面 | 把那段从 plan 里去掉；记到 brief.md 的 avoid 列表 |
| 开头不抓人 | 把 plan 里高动作 / 高 CLIP / 高 laughter 的段挪到第一 |
| 收尾没力 | 找一个 scenic 高分 + 长一点的段做收尾 |
| 完全不像那次旅行 | **brief 错了**。回到第 0 步重新对齐 |

最后一条最重要。当用户说"感觉不对"而不是某个具体改进，**说明你对意图的理解出错了**，不是参数问题。回去问，不要瞎调权重。

---

## 4. 什么时候用什么工具

| 你想… | 用 |
|---|---|
| 知道一个视频有多长 | `probe` |
| 看一个具体瞬间长什么样 | `frames --at T` + `Read` |
| 在某时刻 ±5s 找精确切点 | `frames --around T --window 5 --step 0.5` |
| 知道哪段有意思 | `timeline --only-active` + `Read`，然后用你的脑子 |
| 看视频里讲了什么 | `transcribe`，然后 `Read work/transcripts/X.srt` |
| 重新转录某一段（风噪太大主转录不准） | `audio-clip --start --end --out X.wav` → 喂回 transcribe |
| 让 VLM 看 spectrogram | `audio-clip --spectrogram` + `describe X.spectrogram.png` |
| 快速扫一堆缩略图 | `frames --every 10s --out DIR` + 多个 `Read` 调用 |
| 批量描述一堆帧 | `describe` （VLM，省眼力）|
| 切一段 | `cut --start --end --out` |
| 让一段慢放（更丝滑）| `cut --slow 0.7 --interp mci ...` |
| 把 start/end 对齐到 silence/scene | `cut --snap-start --snap-end`（需 `boundaries` 先跑） |
| J 切（音频提前进入这一段） | `cut --audio-offset -0.5 ...` |
| 一次性切 10 段 | `cut --from-csv plan.csv --out-dir clips/` |
| 拼起来 | `concat --xfade 0.3` |
| 跨段 J/L 切（下一段音频先进） | `concat ... --audio-xfades "0:0.5,2:0.8"` |
| 加字幕 | `subtitle VIDEO SRT --out F` |
| 加配乐 + ducking | `music-mix VIDEO MUSIC --out F --duck-db -10` |
| 检查 plan 有没有低级错误 | `lint` |
| 拿个"起手式"参考方案 | `select` 输出 plan.json，但**编辑它再用** |
| 自动评分自己剪的 | `critique`（参考，不是判决）|

---

## 5. 反模式（不要做）

- ❌ **不要直接 `select → render`**。`select` 是数学打分；它不知道用户想要什么。
- ❌ **不要不读图就拍板**。signals 数据告诉不了你"这画面美吗"。
- ❌ **不要 silent skip 失败步骤**。`describe` 没 API key 就告诉用户；`diarize` 没 HF_TOKEN 也告诉。
- ❌ **不要为了对称改写历史**。已经交付的成片就是最终版，下一版叫 v2。
- ❌ **不要假设默认值就行**。1/3/10 min、横屏、英文 prompt — 全是默认。问。
- ❌ **不要让 critique 替你判**。它能告诉你"开场弱"，但开场该弱不该弱是你的判断。

---

## 6. 解释你的选择

每次出片都附一份 `<output>.summary.md`：

```markdown
# 3min 高光说明

## 用户意图（来自 brief）
怀旧温和，给妈妈看，避开摔倒和争吵。

## 我做了什么
- 开场 GX010047 7-15s：群像 + 笑容 + 对话引入
- 中段五个森林/草原片段，节奏由慢转快
- 第六段 GX010078 93-105s 慢放 0.7×，CLIP scenic 0.34（top 5%）
- 收尾 GX010079 32-44s 黄昏剪影

## 假设
- 默认 16:9 横屏（用户没说，看起来是发邮件用）
- 没加配乐（用户没提供，问一下要不要）

## 缺憾
- GX010059 整段没用，质量低
- 没找到合适的"出发"开场，凑合用了 047

## 下一步可调
- 想竖屏给 Instagram 用，告诉我，我重剪
- 想加配乐，发我音频，我重拼
```

**用户信任你的前提是：你能讲清楚自己做了什么、为什么、还有什么不确定**。这份 summary 比工具输出的任何数字都重要。

---

## 7. 什么时候停下来问

不要硬剪。下面任何一条满足，停下来问用户：

1. brief 里 "情绪" 和 "目标" 矛盾（怀旧 + 30s reel）
2. 素材里有明显敏感画面（小孩独处、私密场景、政治符号）
3. 用户提到了一个具体瞬间但你找不到（"那个我笑得很大声的"——你的笑声信号没标到）
4. 同样的方向你已经试了 2-3 轮，每轮用户都不满意 → 不是参数问题，是 brief 问题
5. 时长目标和素材体量严重不匹配（30 分钟成片，源 8 分钟）

每一条都比"再剪一版"省时间。

---

## 8. 信号置信度（哪个信号能信）

跟你看视觉判断比，工具信号的可信度：

| 信号 | 信不信 | 备注 |
|---|---|---|
| **你看图本身** | ★★★★ | 视觉是你的最强信号 |
| **你读字幕** | ★★★★ | 转录虽然有噪声，但你能识别幻觉 |
| ffmpeg scene | ★★★ | 准 |
| 光流 | ★★★ | 准 |
| RMS dBFS | ★★ | 风噪会假阳性 |
| voiceness（谱平坦度）| ★★ | 8-15dB SNR 段会失效 |
| 笑声启发式 | ★★ | 误报率有 |
| Whisper transcript | ★★（turbo） / ★（base） | base 在 GoPro 风噪上不可靠 |
| 关键词命中 | ★★ | 列表不全，多语言更弱 |
| CLIP 语义 | ★★★ | 最强单信号，但只懂英文 prompt |
| CLIP 嵌入相似度（去重）| ★★★ | 0.92 阈值经验上 OK |
| silence 边界 snap | ★★★ | GoPro 上偶尔无静音点（持续风噪） |
| scene 边界 snap | ★ | GoPro 单镜头连拍，几乎没硬切点 |
| 人脸 / 主角 | ★★ | InsightFace 在小脸 GoPro 远景上漏检多 |
| 说话人 diarize | ★★★ | 在 pyannote 可用时 |
| EXIF/GPS telemetry | ★★★★ | 元数据是真的，但很多视频没有 |
| critique 启发式 | ★ | 规则太粗 |
| critique VLM | ★★★ | 给方向 OK，给具体改动还要你判断 |

把信号 ≤ ★ 当 nice-to-have 看，★★★★ 的你看图 / 看字幕是基础。

---

## 9. 最后一条

工具会变，命令会改，权重会调。但**你是编辑**这件事不会变。素材是别人的回忆 / 业务 / 资产，你在帮他们把它讲成一个故事。算法做不到这一点，**你能**。
