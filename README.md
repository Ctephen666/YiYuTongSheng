# 异语同声：跨语言个性化歌声转换系统

YiyuTongsheng 是一个课程项目级的 Python 工程骨架，用于展示跨语言个性化歌声转换的完整数据流。当前版本不下载模型、不调用真实推理，只提供可运行的 mock 流水线，方便后续逐步替换为真实模块。

## 技术路线

原始歌曲 -> 人声/伴奏分离 -> F0 提取与旋律乐谱重建 -> 中文歌词切句 -> 目标语言歌词翻译 -> 可唱性歌词适配 -> 目标语言音素化 -> 音节-音符对齐 -> 目标语言歌声合成 SVS -> RVC/SVC 个性化音色迁移 -> 人声伴奏混音 -> 最终跨语言歌曲输出

## 项目结构

- `data/`: 流水线中间数据与最终音频输出。
- `src/common/`: 路径、JSON、音频 placeholder、日志等公共工具。
- `src/preprocess/`: 人声/伴奏分离、清理、归一化。
- `src/melody/`: F0 提取、音符转换、MIDI/MusicXML 导出。
- `src/lyrics/`: 中文分句、翻译、可唱性改写与评分。
- `src/phoneme/`: 英文/日文目标语言音素化。
- `src/alignment/`: 音节-音符对齐。
- `src/svs/`: OpenUtau/DiffSinger 项目导出与歌声渲染占位。
- `src/svc/`: RVC/SVC 音色转换占位。
- `src/mix/`: 人声对齐、响度规划、最终混音。
- `src/evaluate/`: F0、节奏、歌词、说话人相似度评价占位与报告生成。
- `configs/`: 项目、歌词适配、SVS、RVC 配置。
- `examples/`: 示例歌词、旋律、音素和对齐 JSON。
- `outputs/`: 日志、报告和后续图表输出。

## 环境安装

```bash
conda create -n yiyu_svc python=3.10 -y
conda activate yiyu_svc
pip install -r requirements.txt
```

当前 `app.py` 对 PyYAML 有轻量 fallback，因此即使暂未安装依赖，也可以先跑通基础 mock 流程。

## 运行方式

```bash
python app.py --step all --target-language en
python app.py --step lyrics
python app.py --step alignment
python app.py --target-language ja
```

支持参数：

- `--config configs/project.yaml`
- `--step all / preprocess / melody / lyrics / phoneme / alignment / svs / svc / mix / evaluate`
- `--target-language en / ja`

## 模块说明

- `preprocess`: 检查原始音频，生成 mock 人声与伴奏 stem。后续接入 Demucs 或 UVR。
- `melody`: 生成 mock `f0.csv`、`melody_notes.json`、`melody.mid`、`melody.musicxml`。后续接入 RMVPE、librosa.pyin、MuseScore。
- `lyrics`: 生成中文 phrase map、literal translation 和 singable lyric。后续接入 LLM 翻译与可唱性改写。
- `phoneme`: 英文使用小型 mock 发音词典，日文使用字符级占位。后续接入 phonemizer、CMUdict 或 pyopenjtalk。
- `alignment`: 实现简单音节-音符规则对齐。后续替换为动态规划、强制对齐或 SVS 时长模型。
- `svs`: 生成 OpenUtau 导出计划和目标语言歌声 placeholder。后续生成 `.ustx` 并调用 OpenUtau 或 DiffSinger。
- `svc`: 生成 RVC converted voice placeholder。后续接入 RVC WebUI 或命令行推理。
- `mix`: 生成 final mix placeholder。后续用 pydub、librosa 或 ffmpeg 完成真实混音。
- `evaluate`: 生成 `outputs/reports/pipeline_report.md`，后续加入 F0 相似度、节奏对齐、歌词适配和音色相似度指标。

## 后续真实实现计划

1. 接入 Demucs/UVR，替换 `VocalSeparator` 的 placeholder 输出。
2. 接入 RMVPE 或 `librosa.pyin`，替换 mock F0。
3. 使用 MuseScore、pretty_midi 或 music21 校正并导出 MIDI/MusicXML。
4. 接入 LLM，完成歌词翻译、可唱性改写、押韵和语义约束。
5. 接入 phonemizer、CMUdict、pyopenjtalk，生成可用于 SVS 的音素和音节边界。
6. 实现动态规划音节-音符对齐，支持多音符拖腔和一音多字。
7. 接入 OpenUtau 或 DiffSinger，渲染目标语言歌声。
8. 接入 RVC WebUI 或命令行推理，完成目标歌手音色迁移。
9. 使用 pydub/librosa/ffmpeg 完成真实人声与伴奏混音。
10. 输出 F0 相似度、歌词适配评分、RVC 参数对比表和实验报告图表。

## 当前验收点

运行 `python app.py --step all --target-language en` 后，会自动创建 `data/` 子目录、示例输入、中间 JSON、placeholder 音频文件、日志和 `outputs/reports/pipeline_report.md`。所有 mock 文件都会明确标识为 placeholder，便于后续逐步替换为真实模型输出。
