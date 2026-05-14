window.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  const state = { videoPath: '', outputPath: '' };

  let isRunning = false;
  let progressMode = 'whisper';
  let translationOnline = false;

  const previewVideo = $('previewVideo');
  const videoPathText = $('videoPath');
  const overlay = $('videoProgressOverlay');
  const overlayBar = $('videoProgressBar');
  const overlayText = $('videoProgressText');
  const videoWrap = document.querySelector('.video-preview-wrap');

  let objectUrlToRevoke = '';

  const setOverlayVisible = (visible) => {
    overlay?.classList.toggle('is-visible', !!visible);
    videoWrap?.classList.toggle('is-progressing', !!visible);
  };

  const setOverlayProgress = (p) => {
    const v = Math.max(0, Math.min(100, Number(p) || 0));
    if (overlayBar) overlayBar.style.width = `${v}%`;
    if (overlayText) overlayText.textContent = `${v}%`;
  };

  // 合成进度：Whisper 50% + Translation 50%
  const calcOverall = () => {
    const w = stageMax.whisper || 0;
    const t = stageMax.translation || 0;
    return Math.round(w * 0.5 + t * 0.5);
  };

  const cleanupObjectUrl = () => {
    if (objectUrlToRevoke) {
      URL.revokeObjectURL(objectUrlToRevoke);
      objectUrlToRevoke = '';
    }
  };

  const setVideoPreview = (filePath, src) => {
    state.videoPath = filePath || '';
    if (videoPathText) videoPathText.textContent = filePath || '未选择视频';
    if (!previewVideo) return;

    cleanupObjectUrl();

    previewVideo.pause();
    previewVideo.currentTime = 0;
    previewVideo.removeAttribute('src');

    if (!src) {
      previewVideo.load();
      return;
    }

    previewVideo.src = src;
    previewVideo.load(); // 不自动播放
  };

  if (previewVideo) {
    previewVideo.addEventListener('loadedmetadata', () => {
      console.log('[preview] loadedmetadata ok:', previewVideo.duration);
    });

    previewVideo.addEventListener('error', () => {
      const err = previewVideo.error;
      console.error('[preview] error', {
        code: err?.code,
        message: err?.message,
        currentSrc: previewVideo.currentSrc,
      });
      alert('视频无法播放，可能是编码不支持。请先用 mp4(h264+aac) 测试。');
    });
  }

  const safeToFileUrl = (filePath) => {
    const normalized = String(filePath || '').replace(/\\/g, '/');
    try {
      if (window.api && typeof window.api.toFileUrl === 'function') {
        const url = window.api.toFileUrl(filePath);
        if (typeof url === 'string' && url) return url;
      }
    } catch (e) {
      console.warn('window.api.toFileUrl failed:', e);
    }
    return `file://${encodeURI(normalized)}`;
  };

  const syncPreviewVideo = (filePath) => {
    if (!previewVideo) return;

    if (!filePath) {
      previewVideo.pause();
      previewVideo.removeAttribute('src');
      previewVideo.load();
      return;
    }

    const url = safeToFileUrl(filePath);
    console.log('[preview] set src =', url);

    previewVideo.pause();
    previewVideo.currentTime = 0;
    previewVideo.src = url;
    previewVideo.load(); // 不自动播放，用户手动点 controls
  };

  const setVideoPath = (filePath) => {
    state.videoPath = filePath || '';
    if (videoPathText) videoPathText.textContent = filePath || '未选择视频';
    syncPreviewVideo(filePath);
  };

  const validateBeforeRun = () => {
    const videoPath = state.videoPath?.trim() || '';
    const outputPath = $('outputPath')?.value?.trim() || state.outputPath || '';
    const whisperModel = $('whisperModel')?.value?.trim() || '';
    const translationModel = $('translationModel')?.value?.trim() || '';

    if (!videoPath) return '请先选择视频文件';
    if (!outputPath) return '请先选择输出路径';

    if (progressMode !== 'ocr' && !whisperModel) {
      return '请先选择 Whisper 模型';
    }

    if (!translationOnline && !translationModel) {
      return '请先选择翻译模型，或开启在线翻译';
    }

    return '';
  };

  const runBtn = $('run');
  const whisperBar = $('whisperProgressBar');
  const whisperText = $('whisperProgressText');
  const translationBar = $('translationProgressBar');
  const translationText = $('translationProgressText');
  const ocrModeBtn = $('ocrModeBtn');
  const onlineModeBtn = $('onlineModeBtn');
  const whisperLabel = $('whisperProgressLabel');
  const whisperStep = $('whisperStep');
  const whisperModelInput = $('whisperModel');
  const pickWhisperBtn = $('pickWhisper');

  const stageMax = { whisper: 0, translation: 0 };

  let ocrEntity = [];
  let ocrFilePath = '';

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

  const syncOnlineModeState = () => {
    if (!onlineModeBtn) return;
    onlineModeBtn.textContent = translationOnline ? '在线翻译：开' : '在线翻译';
    onlineModeBtn.classList.toggle('is-on', translationOnline);
  };

  syncWhisperLabel();
  syncWhisperModelState();
  syncOnlineModeState();

  const setRunButtonState = (running) => {
    if (!runBtn) return;
    runBtn.disabled = running;
    runBtn.textContent = running ? '处理中...' : '开始处理';
  };

  const setStageProgress = (stage, p, { allowDecrease = false } = {}) => {
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
    const picked = await window.api.pickVideo();
    if (!picked) return;
    console.log('ocr results loaded from picked.ocrResults path:', picked.ocrResults);
    // picked: { path, url }
    // 判断一下这个视频是否已经有ocr结果了，如果有的话直接加载预览和字幕
    if (picked.ocrResults) {
      const text = await window.api.readTextFile(picked.ocrResults);
      const parsed = JSON.parse(text || '{}');
      ocrFilePath = picked.ocrResults;
      ocrEntity = Array.isArray(parsed.entries) ? parsed.entries : [];
    } else {
      ocrEntity = [];
    }
    setVideoPreview(picked.path, picked.url);
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

  onlineModeBtn?.addEventListener('click', () => {
    if (isRunning) return;
    translationOnline = !translationOnline;
    syncOnlineModeState();
  });

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

    const objectUrl = URL.createObjectURL(f);
    objectUrlToRevoke = objectUrl;

    // f.path 用于后续 pipeline，objectUrl 只用于前端预览
    setVideoPreview(f.path || '', objectUrl);
  });

  // start processing
  $('run')?.addEventListener('click', async () => {
    if (isRunning) return;
    const err = validateBeforeRun();
    if (err) {
      alert(err);
      return;
    }

    isRunning = true;
    setRunButtonState(true);
    setOverlayVisible(true);
    setOverlayProgress(0);
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
        threads: Number($('threads')?.value || 2),
        ocrEnabled: progressMode === 'ocr',
        onlineTranslation: translationOnline,
      });

      if (ret?.code === 0) {
        setStageProgress('whisper', 100);
        setStageProgress('translation', 100);
        setStep('done', 'done');
        setOverlayProgress(100); // 成功时可选
        setOverlayVisible(false);
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

  // 渲染字幕行（使用 .subtitle-list 容器）
  const renderSubtitleList = (list) => {
    const container = document.querySelector('.subtitle-list');
    if (!container) return;

    container.innerHTML = '';
    list.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'subtitle-row';
      row.dataset.index = String(i);

      row.innerHTML = `
        <div class="subtitle-col">
          <textarea class="subtitle-textarea" placeholder="原字幕">${ocrEntity[item]?.text || ''}</textarea>
        </div>
        <div class="subtitle-col">
          <textarea class="subtitle-textarea subtitle-textarea--trans" placeholder="翻译字幕">${ocrEntity[item]?.trans_text || ''}</textarea>
        </div>
      `;
      row.querySelector('.subtitle-textarea')?.addEventListener('input', (e) => {
        ocrEntity[item].text = e.target.value;
      });

      row.querySelector('.subtitle-textarea--trans')?.addEventListener('input', (e) => {
        ocrEntity[item].trans_text = e.target.value;
      });
      container.appendChild(row);
    });
  };

  const saveBtn = $('saveSubtitles');
  saveBtn?.addEventListener('click', async () => {
    if (!ocrFilePath) {
      alert('没有 OCR 文件路径，无法保存');
      return;
    }
    try {
      const payload = JSON.stringify({ entries: ocrEntity }, null, 2);

      await window.api.writeTextFile(ocrFilePath, payload);
      console.log('保存成功');
    } catch (e) {
      console.error('save failed:', e);
      alert('保存失败');
    }
  });

  window.api.onCppProgress?.(async (m) => {
    try {
      if (!m) return;

      const kind = m.kind || 'progress';
      const stage = m.stage;
      const p = Math.max(0, Math.min(100, Number(m.progress || 0)));

      if (kind === 'output') {
        const el = $('resultPath');
        if (el) el.textContent = m.path || '-';
        console.log('[onCppProgress] output path:', m.path);
        if (m.path) {
          cleanupObjectUrl(); // 如果之前用过 object URL，清理掉
          console.log('[onCppProgress] preview file URL ->', safeToFileUrl(m.path));
          setVideoPath(m.path);
        }
        return;
      }

      if (kind === 'ocr_path') {
        try {
          const text = await window.api.readTextFile(m.path);
          const parsed = JSON.parse(text || '{}');
          ocrEntity = Array.isArray(parsed.entries) ? parsed.entries : [];
          console.log('ocr nums:', ocrEntity.length)
        } catch (e) {
          console.error('load ocr json failed:', e);
        }
        return;
      }

      if (kind === 'stage' && stage) {
        setStep(stage, m.status || 'running');
        return;
      }

      if (kind === 'progress' && stage) {
        setOverlayProgress(calcOverall());
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

  const highlightSubtitleRows = (indices) => {
    const rows = document.querySelectorAll('.subtitle-row');
    rows.forEach((r) => r.classList.remove('is-active'));

    indices.forEach((i, n) => {
      const target = rows[i];
      if (target) {
        target.classList.add('is-active');
        if (n === 0) target.scrollIntoView({ block: 'nearest' });
      }
    });
  };

  const findSubtitleIndicesByRange = (tCs) => {
    const result = [];
    ocrEntity.forEach((item, i) => {
      const t0 = Number(item.t0_cs ?? 0);
      const t1 = Number(item.t1_cs ?? t0);
      // console.log(`[findSubtitleIndicesByRange] check idx=${i}, t0=${t0}, t1=${t1}, tCs=${tCs}`);
      if (tCs >= t0 && tCs <= t1) result.push(i);
    });
    return result;
  };


  previewVideo?.addEventListener('pause', () => {
    console.log('[preview] video paused at', previewVideo.currentTime);
    const tCs = Math.round(previewVideo.currentTime * 100);
    const indices = findSubtitleIndicesByRange(tCs);
    console.log('[preview] subtitle indices text ans trans_text for current time:', indices.map((i) => ({
      index: i,
      text: ocrEntity[i]?.text,
      trans_text: ocrEntity[i]?.trans_text
    })));
    renderSubtitleList(indices);
    highlightSubtitleRows(indices);
  });
});