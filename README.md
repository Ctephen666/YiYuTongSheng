# YiYuTongSheng

YiYuTongSheng（异语同声）是一个面向中文歌声合成与音色转换的端到端实验系统。当前主流程已经切换为 OpenCpop 数据解析、DiffSinger cascade 中文 SVS、F0 自然化、RVC/SVC 音色转换，以及 Web Dashboard 展示与操作。

当前系统重点不是早期 MeloTTS baseline，而是：

```text
OpenCpop 歌曲选择
  -> MIDI / TextGrid / WAV 解析
  -> OpenCpop SVS score 生成
  -> DiffSinger word-level input 导出
  -> DiffSinger cascade 中文 SVS
  -> notes fallback F0 naturalization
  -> target_language_vocal.wav
  -> RVC / SVC 音色转换
  -> Web 前端展示与操作
```

## 目录结构

```text
YiYuTongSheng/
  app.py                         # 主入口，推荐用于 SVS 主流程
  configs/
    project.yaml                 # 主项目配置
    web_demo.yaml                # Web Dashboard 配置
  src/
    common/                      # 通用路径、IO、日志工具
    dataset/                     # OpenCpop 数据读取
    melody/                      # MIDI 导入
    lyrics/                      # 中文歌词 phrase 构建
    phoneme/                     # 中文音素处理
    alignment/                   # note / lyric 对齐
    svs/                         # DiffSinger SVS、RVC 封装、报告生成
    web/                         # FastAPI + Jinja2 Web Dashboard
  tools/
    run_web_demo.py              # Web 启动入口
    run_web_task.py              # Web 后台任务入口
    run_diffsinger_infer.py      # DiffSinger phrase-level 推理 runner
  data/
    dataset/opencpop/            # OpenCpop 数据集
    svs/                         # SVS 输出、报告、segments
    svc/                         # SVC/RVC 输出
    web_logs/                    # Web 后台任务日志
  external/
    DiffSinger/                  # DiffSinger 源码与运行目录
    rvc/                         # RVC 源码、模型、index
  checkpoints/                   # DiffSinger 预训练/导出权重
```

模型权重、数据集、`.ckpt`、`.pth`、`.index` 等大文件通常不随 Git 仓库提交，需要本地准备。

## 环境准备

推荐使用已经配置好的 RVC / DiffSinger Python 环境，例如：

```cmd
D:\Anaconda_envs\envs\rvc\python.exe
```

需要本地存在以下关键资源：

```text
data/dataset/opencpop/
external/DiffSinger/
external/rvc/
checkpoints/diffsinger/
external/rvc/assets/hubert/hubert_base.pt
external/rvc/assets/rmvpe/rmvpe.pt
external/rvc/assets/weights/*.pth
external/rvc/assets/indices/*.index
```

当前默认 RVC F0 模型配置在 `configs/project.yaml`：

```yaml
svs:
  rvc:
    enabled: true
    model_name: "bofan_voice_f0.pth"
    index_path: "external/rvc/assets/indices/bofan_voice_f0_IVF591_Flat_nprobe_1_bofan_voice_f0_v2.index"
    f0method: "rmvpe"
```

## 启动 Web Dashboard

```cmd
D:\Anaconda_envs\envs\rvc\python.exe tools\run_web_demo.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

Web 页面支持：

- 扫描并选择 OpenCpop 歌曲
- 扫描并选择 RVC/SVC 人声模型
- 设置 SVS 参数：`start_phrase`、`max_phrases`、`assembly_mode`、`normalize`、`f0_naturalize`
- 设置 SVC 参数：`f0_method`、`f0_up_key`、`index_rate`、`protect`、`filter_radius`、`resample_sr`
- 真实执行 Score / SVS / SVC / 完整流程
- 查看任务状态、后台日志、报告摘要、phrase 表格
- 播放原始参考音频和完整版生成结果

## Web 按钮与真实执行

Web 后端使用 `subprocess.Popen` 异步执行任务，按钮不是 dry run。

主要按钮：

```text
生成 Score
生成基础歌声
执行音色转换
一键完整流程
刷新
刷新报告
打开输出目录
```

任务日志输出到：

```text
data/web_logs/
```

每个任务会记录：

```text
job_id
name
status
command
start_time
end_time
returncode
log_path
```

## 命令行运行主流程

推荐主入口：

```cmd
D:\Anaconda_envs\envs\rvc\python.exe app.py --step svs --target-language zh --opencpop-id 2001
```

说明：

- `--step svs` 会执行当前中文 OpenCpop SVS 主链路。
- `--opencpop-id` 用于选择 OpenCpop 歌曲，例如 `2001`。
- `configs/project.yaml` 控制 DiffSinger、RVC、输出路径和推理参数。

Web 中“生成基础歌声”和“一键完整流程”也会生成运行时配置，并调用：

```cmd
python app.py --config data/web_configs/<runtime>.yaml --step svs --target-language zh --opencpop-id <song_id>
```

## 选择歌曲

Web 会扫描：

```text
data/dataset/opencpop/midis/
data/dataset/opencpop/wavs/
data/dataset/opencpop/TextGrid/
data/dataset/opencpop/textgrids/
```

识别出的 `song_id` 会显示在歌曲下拉框中。选择歌曲后，页面会显示：

```text
MIDI 是否存在
WAV 是否存在
TextGrid 是否存在
文件路径
```

## 选择人声模型

Web 会扫描：

```text
external/rvc/assets/weights/
external/rvc/weights/
external/rvc/logs/
external/RVC/assets/weights/
external/RVC/weights/
```

并扫描 index：

```text
external/rvc/assets/indices/
external/rvc/indices/
external/rvc/logs/
external/RVC/assets/indices/
external/RVC/indices/
```

支持识别：

```text
*.pth
*.index
```

带 `f0` 的模型会优先显示。若模型疑似不是 F0 模型，页面会提示音准可能不稳定。

## 输出文件

主要输出：

```text
data/svs/target_language_vocal_diffsinger.wav   # DiffSinger 基础 SVS 备份
data/svs/target_language_vocal.wav              # 当前最终歌声
data/svc/converted_target_voice.wav             # SVC/RVC 转换结果
data/svc/final_mix.wav                          # 若存在，则作为最终混音优先展示
```

主要报告：

```text
data/svs/checkpoint_status.json
data/svs/diffsinger_opencpop_export_plan.json
data/svs/diffsinger_infer_report.json
data/svs/neural_svs_render_plan.json
data/svs/opencpop_svs_score.json
data/svs/rvc_voice_conversion_report.json
```

Web 音频区默认只展示：

```text
原始 OpenCpop 参考音频
完整版生成结果
```

完整版生成结果按优先级选择：

```text
data/svc/final_mix.wav
data/svc/converted_target_voice.wav
data/svs/target_language_vocal.wav
data/svs/target_language_vocal_diffsinger.wav
```

## 当前核心模块

```text
src/svs/diffsinger_opencpop_exporter.py       # OpenCpop -> DiffSinger word-level input
src/svs/opencpop_neural_svs_backend.py        # DiffSinger staging 与推理子进程封装
tools/run_diffsinger_infer.py                 # phrase-level DiffSinger runner
src/svs/rvc_voice_conversion.py               # RVC/SVC 调用封装
src/svs/run_opencpop_svs.py                   # OpenCpop 中文 SVS 主流程
src/web/app.py                                # FastAPI API 与页面路由
src/web/services.py                           # Web 扫描、命令生成、任务管理
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'src'`

请从项目根目录运行：

```cmd
cd /d D:\YiYuTongSheng
D:\Anaconda_envs\envs\rvc\python.exe tools\run_web_demo.py
```

当前 `tools/run_web_demo.py` 已经会自动把项目根目录加入 `sys.path`。

### 2. Web 页面 500，Jinja2 TemplateResponse 报错

已使用新版兼容写法：

```python
templates.TemplateResponse(request=request, name="index.html", context={...})
```

如果仍有问题，先重启 Web 服务。

### 3. Windows 出现 `ConnectionResetError: [WinError 10054]`

这通常是浏览器刷新、关闭标签页、轮询请求中断导致的 socket 关闭日志。只要 Uvicorn 仍显示：

```text
Uvicorn running on http://127.0.0.1:7860
```

服务一般仍可继续使用。

### 4. RVC 输出不是目标音色

检查：

```text
configs/project.yaml -> svs.rvc.model_name
configs/project.yaml -> svs.rvc.index_path
Web 页面选择的人声模型
Web 页面选择的 index
```

单独执行 SVC 时，系统优先使用：

```text
data/svs/target_language_vocal_diffsinger.wav
```

这样可以避免对已经 RVC 过的音频二次转换。

### 5. DiffSinger CUDA OOM

当前 runner 使用 phrase-level inference。可在 Web 或配置中先设置：

```yaml
svs:
  max_phrases: 1
  assembly_mode: concat
```

确认单句能跑后，再改回整首：

```yaml
svs:
  max_phrases: null
  assembly_mode: timeline
```

## Git 与大文件说明

不要把以下本地大文件提交到 Git：

```text
*.wav
*.mp3
*.flac
*.npy
*.npz
*.ckpt
*.pth
*.index
data/svs/segments/
data/web_logs/
outputs/
external/DiffSinger/checkpoints/
external/rvc/assets/weights/
external/rvc/assets/indices/
external/rvc/logs/
```

模型权重、OpenCpop 数据集、DiffSinger checkpoints、RVC 模型和 index 需要用户在本地自行准备。

## 推荐入口

```text
Web Dashboard:
  tools/run_web_demo.py

总入口:
  app.py

SVS 内部 runner:
  tools/run_diffsinger_infer.py

Web 后台任务:
  tools/run_web_task.py

DiffSinger 配置:
  configs/project.yaml

Web 配置:
  configs/web_demo.yaml
```
