const { contextBridge, ipcRenderer } = require('electron');
const { pathToFileURL } = require('url');

// Context Isolation
contextBridge.exposeInMainWorld('api', {
  pickVideo: () => ipcRenderer.invoke('pick-video'),
  pickFile: (options) => ipcRenderer.invoke('pick-file', options),
  pickDir: () => ipcRenderer.invoke('pick-dir'),
  onCppLog: (cb) => ipcRenderer.on('cpp-log', (_e, m) => cb(m)),
  onCppProgress: (cb) => ipcRenderer.on('cpp-progress', (_e, msg) => cb?.(msg)),
  runCppPipeline: (payload) => ipcRenderer.invoke('run-cpp-pipeline', payload),
  logFront: (level, msg) => ipcRenderer.send('frontend-log', { level, msg }),
  toFileUrl: (filePath) => (filePath ? pathToFileURL(filePath).href : ''),
  readTextFile: (filePath) => ipcRenderer.invoke('read-text-file', filePath),
  writeTextFile: (filePath, text) => ipcRenderer.invoke('write-text-file', filePath, text),
});