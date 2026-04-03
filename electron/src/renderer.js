window.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  const state = { videoPath: '', outputPath: '' };

  let isRunning = false;
  let progressMode = 'whisper';

  const runBtn = $('run');
  const whisperBar = $('whisperProgressBar');
  const whisperText = $('whisperProgressText');
  const translationBar = $('translationProgressBar');
  const translationText = $('translationProgressText');
  const ocrModeBtn = $('ocrModeBtn');
  const whisperLabel = $('whisperProgressLabel');
  const whisperStep = $('whisperStep');
  const whisperModelInput = $('whisperModel');
  const pickWhisperBtn = $('pickWhisper');

  // max stage progress
  const stageMax = { whisper: 0, translation: 0 };

  const syncWhisperModelState = () => {
    const disableWhisper = progressMode === 'ocr';

    if (whisperModelInput) {
      whisperModelInput.disabled = disableWhisper;
      whisperModelInput.placeholder = disableWhisper
        ? 'OCR模式下无需Whisper模型'
        : '输入whisper语音模型路径（.bin）';
    }

    if (pickWhisperBtn) {
      pickWhisperBtn.disabled = disableWhisper;
    }
  };


  const syncWhisperLabel = () => {
    if (whisperLabel) {
      whisperLabel.textContent =
      progressMode === 'ocr' ? 'OCR 识别进度' : 'Whisper 识别进度';
    }
    if (ocrModeBtn) {
      ocrModeBtn.textContent = progressMode === 'ocr' ? 'OCR：开' : 'OCR';
    } 
    ocrModeBtn?.classList.toggle('is-on', progressMode === 'ocr');
    if (whisperStep) {
      whisperStep.textContent = progressMode === 'ocr' ? 'OCR 识别' : '语音识别';
    }
  };
  syncWhisperLabel();
  syncWhisperModelState();

  const setRunButtonState = (running) => {
    if (!runBtn) return;
    runBtn.disabled = running;
    runBtn.textContent = running ? '处理中...' : '开始处理';
  };

  const setStageProgress = (stage, p, { allowDecrease = false } = {}) => {
    // Number(p) || 0 avoid invalid number
    const v = Math.max(0, Math.min(100, Number(p) || 0));
    if (!allowDecrease && (stage === 'whisper' || stage === 'translation')) {
      stageMax[stage] = Math.max(stageMax[stage] || 0, v);
    }
    const finalV = (stage === 'whisper' || stage === 'translation') ? stageMax[stage] : v;

    if (stage === 'whisper') {
      if (whisperBar) whisperBar.style.width = `${finalV}%`;
      if (whisperText) whisperText.textContent = `${finalV}%`;
    } else if (stage === 'translation') {
      if (translationBar) translationBar.style.width = `${finalV}%`;
      if (translationText) translationText.textContent = `${finalV}%`;
    }
  };

  // running / done / error
  const setStep = (step, status = 'running') => {
    const all = document.querySelectorAll('#stepper .step');
    const order = ['prepare', 'video', 'whisper', 'translation', 'done'];
    const idx = order.indexOf(step);

    all.forEach((el) => {
      el.classList.remove('is-running', 'is-done', 'is-error');
      const i = order.indexOf(el.dataset.step);

      if (idx >= 0 && i < idx) el.classList.add('is-done');

      if (i === idx) {
        if (status === 'done') el.classList.add('is-done');
        else if (status === 'error') el.classList.add('is-error');
        else el.classList.add('is-running');
      }
    });

    const map = {
      prepare: '准备',
      video: '视频处理',
      whisper: '语音识别',
      translation: 'AI翻译',
      done: '完成',
    };
    // const t = $('stepCurrentText');
    // if (t) t.textContent = status === 'error' ? `失败（${map[step] || step}）` : (map[step] || '处理中');
  };

  $('pickWhisper')?.addEventListener('click', async () => {
    const p = await window.api.pickFile({ filters: [{ name: 'Model', extensions: ['bin', 'gguf'] }] });
    if (p) $('whisperModel').value = p;
  });

  $('pickTranslation')?.addEventListener('click', async () => {
    const p = await window.api.pickFile({ filters: [{ name: 'Model', extensions: ['bin', 'gguf'] }] });
    if (p) $('translationModel').value = p;
  });

  $('pickVideo')?.addEventListener('click', async () => {
    const p = await window.api.pickVideo();
    if (!p) return;
    state.videoPath = p;
    $('videoPath').textContent = p;
  });

  $('pickOutput')?.addEventListener('click', async () => {
    const p = await window.api.pickDir();
    if (!p) return;
    state.outputPath = p;
    $('outputPath').value = p;
  });

  ocrModeBtn?.addEventListener('click', () => {
    if (isRunning) return;
    progressMode = progressMode === 'whisper' ? 'ocr' : 'whisper';
    syncWhisperLabel();
    syncWhisperModelState();
  });

  // drag and drop video file
  const dz = $('dropZone');
  dz?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dz.classList.add('dragover');
  });
  dz?.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz?.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    state.videoPath = f.path;
    $('videoPath').textContent = f.path;
  });

  // start processing
  $('run')?.addEventListener('click', async () => {
    if (isRunning) return;
    if (!state.videoPath) return;

    isRunning = true;
    setRunButtonState(true);

    // progress reset
    stageMax.whisper = 0;
    stageMax.translation = 0;
    setStageProgress('whisper', 0, { allowDecrease: true });
    setStageProgress('translation', 0, { allowDecrease: true });
    setStep('prepare', 'running');

    try {
      const ret = await window.api.runCppPipeline({
        videoPath: state.videoPath,
        outputPath: $('outputPath')?.value?.trim() || state.outputPath || '',
        whisperModel: $('whisperModel')?.value?.trim() || '',
        translationModel: $('translationModel')?.value?.trim() || '',
        threads: Number($('threads')?.value || 4),
        ocrEnabled: progressMode === 'ocr',
      });

      if (ret?.code === 0) {
        setStageProgress('whisper', 100);
        setStageProgress('translation', 100);
        setStep('done', 'done');
      } else {
        setStep('done', 'error');
      }
    } catch (e) {
      setStep('done', 'error');
    } finally {
      isRunning = false;
      setRunButtonState(false);
    }
  });

  window.api.onCppProgress?.((m) => {
    try {
      if (!m) return;

      const kind = m.kind || 'progress';
      const stage = m.stage;
      const p = Math.max(0, Math.min(100, Number(m.progress || 0)));

      if (kind === 'output') {
        const el = $('resultPath');
        if (el) el.textContent = m.path || '-';
        return;
      }

      if (kind === 'stage' && stage) {
        setStep(stage, m.status || 'running');
        return;
      }

      if (kind === 'progress' && stage) {
        if (stage === 'whisper') {
          setStep('whisper', 'running');
          setStageProgress('whisper', p);
        } else if (stage === 'translation') {
          setStep('translation', 'running');
          setStageProgress('translation', p);
        } else if (stage === 'ocr') {
          setStep('whisper', 'running');
          setStageProgress('whisper', p);
        } else if (stage === 'done') {
          setStageProgress('whisper', 100);
          setStageProgress('translation', 100);
          setStep('done', 'done');
        }
      }
    } catch (e) {
      console.error('onCppProgress error:', e);
    }
  });

  // window.api.onCppLog?.((m) => {
  //   log(`[${m.type}] ${String(m.text || '').trimEnd()}`);
  // });

  // log('[ready] renderer loaded');
});