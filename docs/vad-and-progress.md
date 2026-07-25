# VAD 与进度回调

transcribe.cpp 可选地集成 **VAD(语音活动检测)** 和 **进度回调**。两者都
**默认关闭**,不影响现有行为。

- VAD 把输入音频切成语音段,逐段解码,提升长音频/稀疏音频的速度并避免单次
  解码超长输入导致的截断或显存溢出。
- 进度回调在解码的 chunk 边界触发,语义与 audio.cpp 的 `audiocpp_progress_fn`
  一致 —— 客户端可在两套 API 上复用同一份回调逻辑。

---

## VAD(通过 audiocpp.dll)

### 算法

| `transcribe_vad_mode` | 算法 | 精度 | 需 model | 说明 |
|---|---|---|---|---|
| `TRANSCRIBE_VAD_OFF`(默认) | 无 | — | — | 全量解码,行为与未集成 VAD 时完全一致 |
| `TRANSCRIBE_VAD_SILERO` | 神经(Silero v4/v5) | 高 | 是 | audiocpp.dll 内嵌权重,无需外部模型文件 |
| `TRANSCRIBE_VAD_ENERGY` | 能量/RMS | 中(噪声敏感) | 否 | 纯 DSP,最轻量 |

### 启用 VAD 的两步

1. **构建时开编译开关**:

   ```bash
   cmake -B build -S . -DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON
   cmake --build build --target transcribe-cli
   ```

   开关关时(默认),VAD 实现代码完全不编译,`transcribe_vad`/`transcribe_free_vad`
   符号不存在;`run_params.vad` 字段仍存在但被解析为 OFF。

2. **运行时提供 audiocpp.dll**:把内嵌了 Silero 权重的 `audiocpp.dll`(从
   audio.cpp 项目获取)放到以下任一位置(按顺序查找):

   - `transcribe_vad_params.dll_path`(显式指定)
   - 环境变量 `TRANSCRIBE_VAD_DLL`
   - 可执行文件同目录
   - 当前工作目录
   - 系统 PATH

   找不到 dll 时,VAD 自动降级为全量解码(记一条 WARN 日志),**不报错** —— VAD
   是增强,不是必需。

### 在 transcribe_run 里用 VAD

```c
struct transcribe_run_params rp;
transcribe_run_params_init(&rp);
rp.language     = "en";
rp.vad.mode     = TRANSCRIBE_VAD_SILERO;  // 或 TRANSCRIBE_VAD_ENERGY
rp.vad.backend  = 1;   // 0=CPU; audiocpp dll 内置多后端,1=CUDA,2=Vulkan...
rp.vad.device_id = 0;
// 其他字段 <=0 / 0 表示用默认值:
//   max_chunk_ms   = family effective_max_audio_ms(或 30000)
//   merge_gap_ms   = 500
//   padding_ms     = 250
transcribe_run(session, pcm, n_samples, &rp);
```

### 独立 VAD(不跑 ASR)

只想做语音分段,不需要转录:

```c
struct transcribe_vad_params vp;
memset(&vp, 0, sizeof(vp));
vp.struct_size = sizeof(vp);
vp.mode = TRANSCRIBE_VAD_SILERO;

transcribe_vad_segment * segs = nullptr;
int64_t                  n    = 0;
transcribe_status st = transcribe_vad(pcm, n_samples, 16000, &vp, &segs, &n);
if (st == TRANSCRIBE_OK) {
    for (int64_t i = 0; i < n; ++i) {
        printf("[%lld-%lld ms] conf %.3f\n",
               (long long)segs[i].start_ms, (long long)segs[i].end_ms, segs[i].confidence);
    }
}
transcribe_free_vad(segs);  // NULL 安全
```

`transcribe_vad` / `transcribe_free_vad` 仅在 `-DTRANSCRIBE_VAD_VIA_AUDIOCPP=ON`
编译时存在;否则是链接错误。

### CLI

```bash
# Qwen3-ASR + Silero VAD(神经),VAD 走 CUDA
TRANSCRIBE_VAD_DLL=/path/to/audiocpp.dll \
./transcribe-cli -m model.gguf --backend cuda \
    --vad-mode silero --vad-backend 1 meeting.wav
```

`--vad-mode {off,silero,energy}` / `--vad-dll PATH` / `--vad-backend N` /
`--vad-device N`。`--vad-backend` 是 audiocpp.dll 内部的后端选择(0=CPU,
1=CUDA,2=Vulkan,...),与 ASR 的 `--backend` 独立。

### 降级保证

VAD 路径的任何失败(dll 缺失、符号不匹配、VAD 调用异常、空切分计划)都降级
为全量 `arch->run` 解码并记一条 WARN。**VAD 永远不会让 ASR 比不开 VAD 时更
坏。** 这是 `run_one_inner` 里 VAD 分支的核心合同。

### 流式路径

`transcribe_stream_*` **不接 VAD**,也不触发进度回调。流式进度用 pull-based
的 `transcribe_stream_update`(`input_received_ms` / `audio_committed_ms` /
`buffered_ms`)。这是设计决策,非遗漏。

---

## 进度回调

```c
int on_progress(float progress, const char * stage,
                int64_t completed, int64_t total, void * ud) {
    fprintf(stderr, "[%s] %.0f%% (%lld/%lld)\n",
            stage, progress * 100.0, (long long)completed, (long long)total);
    return 0;  // 非 0 = 请求取消
}

transcribe_set_progress_callback(session, on_progress, nullptr);
transcribe_run(session, pcm, n_samples, &rp);
```

**语义与 `audiocpp_progress_fn` 完全一致**:

- `progress ∈ [0.0, 1.0]`,`stage` 是短标签(如 `"asr+qwen3_asr"`,回调期内有效,
  不要保留指针),`completed` / `total` 是 chunk 计数。
- 返回 0 继续;**返回非 0 请求取消** —— 正在进行的 run 在下一个 chunk 边界停止,
  已完成段作为部分结果保留,`transcribe_run` 返回 `TRANSCRIBE_ERR_ABORTED`。
- 回调在调用 `transcribe_run` 的线程上**同步**触发。回调内不要调 `transcribe_*`
  API,不要抛异常(异常会被当成取消请求并在 C ABI 边界吞掉)。

取消也可经原有的 `transcribe_set_abort_callback` 触发,两者并存、终态等价 ——
提供进度回调兼任取消是为了让客户端在 transcribe.cpp 与 audio.cpp 间复用同一份
回调逻辑。

### 触发点

- VAD 分块循环:每个 chunk 边界(初始 0/N + 每段完成后的 i/N)。
- 非 VAD 路径:当前阶段不触发(各家族 `run()` 内部全量挂点在后续阶段)。
- 流式路径:**不触发**(见上)。
