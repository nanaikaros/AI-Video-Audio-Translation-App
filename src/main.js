const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const log = require('electron-log');
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');
const net = require('net');

function initLogger() {
  const logDir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDir, { recursive: true });

  log.transports.file.format = '[{y}-{m}-{d} {h}:{i}:{s}.{ms}] [{level}] {text}';
  log.transports.file.level = 'info';
  log.transports.file.resolvePathFn = () => path.join(logDir, `${new Date().toISOString().slice(0,10)}.log`);
  log.info(`[main] log file: ${log.transports.file.resolvePathFn()}`);
}

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 780,
    minWidth: 1000,
    minHeight: 680,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  initLogger();
  createWindow();
});

// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') {
//     app.quit();
//   }
// });

ipcMain.on('frontend-log', (_e, payload) => {
  const level = payload?.level || 'info';
  const msg = `[frontend] ${payload?.msg || ''}`;
  if (level === 'error') log.error(msg);
  else if (level === 'warn') log.warn(msg);
  else log.info(msg);
});

ipcMain.handle('pick-video', async () => {
  const ret = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'avi', 'flv'] }]
  });
  if (ret.canceled || ret.filePaths.length === 0) return null;
  return ret.filePaths[0];
});

ipcMain.handle('pick-file', async (_e, options = {}) => {
  const ret = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: options.filters || [{ name: 'All', extensions: ['*'] }]
  });
  if (ret.canceled || ret.filePaths.length === 0) return null;
  return ret.filePaths[0];
});

ipcMain.handle('pick-dir', async () => {
  const ret = await dialog.showOpenDialog({
    properties: ['openDirectory', 'createDirectory']
  });
  if (ret.canceled || ret.filePaths.length === 0) return null;
  return ret.filePaths[0];
});

function getBackendBin() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend', 'my_app');
  }
  return path.join(app.getAppPath(), 'backend', 'my_app');
}

ipcMain.handle('run-cpp-pipeline', async (event, payload) => {
  const bin = getBackendBin();
  const runBase = app.getPath('userData');

  log.info(`[run] isPackaged=${app.isPackaged}`);
  log.info(`[run] appPath=${app.getAppPath()}`);
  log.info(`[run] resourcesPath=${process.resourcesPath}`);
  log.info(`[run] backendBin=${bin}`);

  const exists = fs.existsSync(bin);
  let canExec = false;
  try {
    fs.accessSync(bin, fs.constants.X_OK);
    canExec = true;
  } catch {}

  log.info(`[run] binCheck exists=${exists} canExec=${canExec}`);

  if (!exists || !canExec) {
    log.error(`[run] backend missing or not executable: ${bin}`);
    event.sender.send('cpp-log', {
      type: 'error',
      text: `[backend missing or not executable] ${bin}\n`
    });
    return { code: -2 };
  }

  const isWin = process.platform === 'win32';
  const sockPath = isWin
    ? `\\\\.\\pipe\\cpp-progress-${process.pid}-${Date.now()}`
    : path.join(os.tmpdir(), `cpp-progress-${process.pid}-${Date.now()}.sock`);

  if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);

  let gotDoneStage = false;
  let gotAnyProgress = false;
  let currentStage = 'prepare';
  let lastWhisper = -1;
  let lastTranslation = -1;
  let resolved = false;

  const emitStage = (stage, status = 'running') => {
    currentStage = stage;
    event.sender.send('cpp-progress', { kind: 'stage', stage, status });
  };
  const emitProgress = (stage, progress) => {
    event.sender.send('cpp-progress', { kind: 'progress', stage, progress: Number(progress) || 0 });
  };
  const emitOutput = (p) => {
    event.sender.send('cpp-progress', { kind: 'output', path: p });
  };

  const safeResolve = (ret) => {
    if (resolved) return;
    resolved = true;
    try { server.close(); } catch {}
    try { if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath); } catch {}
    return ret;
  };

  emitStage('prepare', 'running');

  const server = net.createServer((conn) => {
    log.info('[run] progress socket connected');
    let acc = '';
    conn.on('data', (buf) => {
      acc += buf.toString('utf8');
      const lines = acc.split('\n');
      acc = lines.pop() || '';

      for (const line of lines) {
        const s = line.trim();
        if (!s) continue;

        try {
          const msg = JSON.parse(s);
          gotAnyProgress = true;
          if (msg.path) emitOutput(msg.path);

          if (msg.stage === 'video') {
            emitStage('video', 'running');
          } else if (msg.stage === 'whisper') {
            emitStage('whisper', 'running');
            const p = Number(msg.progress) || 0;
            if (p !== lastWhisper) {
              lastWhisper = p;
              emitProgress('whisper', p);
            }
            if (p >= 100) emitStage('whisper', 'done');
            mainWindow.setProgressBar(lastWhisper / 200);
          } else if (msg.stage === 'translation') {
            if (currentStage === 'whisper') emitStage('whisper', 'done');
            emitStage('translation', 'running');
            const p = Number(msg.progress) || 0;
            if (p !== lastTranslation) {
              lastTranslation = p;
              emitProgress('translation', p);
            }
            if (p >= 100) emitStage('translation', 'done');
            mainWindow.setProgressBar((lastTranslation + 100) / 200);
          } else if (msg.stage === 'done') {
            mainWindow.setProgressBar(-1);
            gotDoneStage = true;
            emitStage('done', 'done');
          }
          log.info(`[progress] ${s}`);
        } catch (e) {
          log.error(`[progress] parse/handle error: ${e.message}; raw=${s}`);
        }
      }
    });
  });

  await new Promise((r) => server.listen(sockPath, r));
  log.info(`[run] progress socket listening: ${sockPath}`);

  return await new Promise((resolve) => {
    const args = [
      '--video', payload.videoPath || '',
      '--whisper-model', payload.whisperModel || '',
      '--translation-model', payload.translationModel || '',
      '--threads', String(payload.threads || 4),
      '--output', payload.outputPath || '',
      '--progress-sock', sockPath,
    ];

    log.info(`[run] cmd=${bin} ${args.join(' ')}`);
    const child = spawn(bin, args, { cwd: runBase, stdio: ['ignore', 'pipe', 'pipe'] });

    child.on('spawn', () => log.info(`[run] child spawned pid=${child.pid}`));

    child.on('error', (err) => {
      log.error(`[run] spawn error code=${err.code} msg=${err.message}`);
      emitStage(currentStage, 'error');
      
      resolve(safeResolve({ code: -1, error: err.message }));
    });

    child.stdout.on('data', (buf) => {
      const t = buf.toString();
      event.sender.send('cpp-log', { type: 'stdout', text: t });
      log.info(`[cpp stdout] ${t.trimEnd()}`);
    });

    child.stderr.on('data', (buf) => {
      const t = buf.toString();
      event.sender.send('cpp-log', { type: 'stderr', text: t });
      try {
        if(t.toLowerCase().includes('could not update timestamps for skipped samples')) {}
        else {
          dialog.showMessageBox(mainWindow, {
            type: 'error',
            title: 'Error from backend',
            message: t,
          });
        }
      } catch (e) {}
      log.error(`[cpp stderr] ${t.trimEnd()}`);
    });

    child.on('close', (code) => {
      log.info(`[run] close code=${code} gotAnyProgress=${gotAnyProgress} gotDoneStage=${gotDoneStage}`);
      if (code === 0 && gotDoneStage) {
        emitStage('done', 'done');
      } else {
        emitStage(currentStage, 'error');
        event.sender.send('cpp-log', {
          type: 'error',
          text: `[close] code=${code}, gotAnyProgress=${gotAnyProgress}, gotDoneStage=${gotDoneStage}\n`
        });
      }
      resolve(safeResolve({ code }));
    });
  });
});