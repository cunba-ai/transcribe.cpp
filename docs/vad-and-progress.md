# VAD 与进度回调

transcribe.cpp 提供原生的 **VAD(语音活动检测)** 和 **进度回调**。

- VAD 把输入音频切成语音段,逐段处理,提升长音频/稀疏音频的速度。详见
  `docs/vad.md` 与 `include/transcribe/vad.h`。
- 进度回调在离线 run/batch 的 utterance 边界触发,语义与 audio.cpp 的
  `audiocpp_progress_fn` 一致 —— 客户端可在两套 API 上复用同一份回调逻辑。

---

## VAD(原生)

VAD 是库内置的原生能力:一个 `TRANSCRIBE_ROLE_VAD` 角色的模型(Silero v6.2)
对 16 kHz 音频逐帧打分,聚合成语音段。没有外部 DLL、没有运行时加载。

- 模型从磁盘加载(`transcribe_model_load_file`,接受 Silero v5.1.2 / v6.2 的
  ggml 权重),或直接用构建期内嵌的权重
  `transcribe_model_load_embedded_vad(NULL, &m)` —— 无需任何模型文件。
- 会话 API:`transcribe_vad_session_init` / `transcribe_vad_run` /
  `transcribe_vad_get_result`,完整参数与示例见 `docs/vad.md`。

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
  不要保留指针),`completed` / `total` 是 utterance 计数。
- 返回 0 继续;**返回非 0 请求取消** —— 正在进行的 run 在下一个 utterance 边界
  停止,已完成段作为部分结果保留,`transcribe_run` 返回 `TRANSCRIBE_ERR_ABORTED`。
- 回调在调用 `transcribe_run` 的线程上**同步**触发。回调内不要调 `transcribe_*`
  API。回调不得抛异常 —— 抛出的异常会在触发点被吞掉并视为取消请求,绝不会
  逃出 C ABI(与 `transcribe_log_invoke` 的日志回调同一纪律)。

取消也可经原有的 `transcribe_set_abort_callback` 触发,两者并存、终态等价 ——
提供进度回调兼任取消是为了让客户端在 transcribe.cpp 与 audio.cpp 间复用同一份
回调逻辑。

### 触发点

- 单次 `transcribe_run`:成功解码结束后一次 `1/1`。
- `transcribe_run_batch`:batch 分发器发出 `0/n` 与收尾的 `n/n` bookend;
  中间每条 utterance 完成后发 `(i+1, n)` —— 由共享循环发出(串行回退
  `run_batch_serial` 与切片解码 `decode_batch_slices`,均在
  `src/transcribe-batch-util.cpp`)。最后一条 utterance 不在循环里发,由
  分发器的 `n/n` 收尾握手覆盖。
- 流式路径:**不触发**;流式进度用 pull-based 的 `transcribe_stream_update`
  (`input_received_ms` / `audio_committed_ms` / `buffered_ms`)。这是设计决策,
  非遗漏。
