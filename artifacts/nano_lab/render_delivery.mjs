// Render the local listening page in an isolated, bounded headless process.
import { spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const pagePath = path.resolve(process.argv[2] ?? 'artifacts/nano_lab/delivery/index.html');
const output = path.resolve(process.argv[3] ?? 'artifacts/nano_lab/delivery/preview.png');
const child = spawn('/home/terryd/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--disable-background-networking',
   '--single-process', '--no-zygote', '--remote-debugging-port=0', 'about:blank'],
  { detached: true, stdio: ['ignore', 'ignore', 'pipe'] });
let ws;
let nextId = 1;
const pending = new Map();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function call(method, params = {}, sessionId) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 10000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
}
try {
  const endpoint = await new Promise((resolve, reject) => {
    let buffer = '';
    const timer = setTimeout(() => reject(new Error('Browser startup timeout')), 10000);
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.stderr.on('data', data => {
      buffer = (buffer + data.toString()).slice(-10000);
      const match = buffer.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    const entry = pending.get(message.id);
    if (!entry) return;
    clearTimeout(entry.timer);
    pending.delete(message.id);
    if (message.error) entry.reject(new Error(JSON.stringify(message.error)));
    else entry.resolve(message.result);
  };
  const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
  await call('Page.enable', {}, sessionId);
  await call('Emulation.setDeviceMetricsOverride', { width: 1100, height: 1500, deviceScaleFactor: 1, mobile: false }, sessionId);
  const nav = await call('Page.navigate', { url: pathToFileURL(pagePath).href }, sessionId);
  if (nav.errorText) throw new Error(nav.errorText);
  let ready = false;
  for (let i = 0; i < 100; i++) {
    const result = await call('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true }, sessionId);
    if (result.result.value === 'complete') { ready = true; break; }
    await pause(100);
  }
  if (!ready) throw new Error('Local page did not finish loading');
  const media = await call('Runtime.evaluate', { awaitPromise: true, returnByValue: true,
    expression: `Promise.all([...document.querySelectorAll('audio')].map(a => new Promise((resolve, reject) => {
      const finish = () => { if (Number.isFinite(a.duration) && a.duration > 0) resolve({src: a.getAttribute('src'), duration: a.duration}); };
      a.addEventListener('loadedmetadata', finish, {once: true});
      a.addEventListener('error', () => reject(new Error('Audio failed: ' + a.src)), {once: true});
      setTimeout(() => reject(new Error('Audio metadata timeout: ' + a.src)), 5000);
      if (a.readyState >= 1) finish(); else a.load();
    })))` }, sessionId);
  if (media.exceptionDetails || !Array.isArray(media.result.value)) throw new Error('Audio metadata verification failed: ' + JSON.stringify(media));
  const playback = await call('Runtime.evaluate', { awaitPromise: true, returnByValue: true, userGesture: true,
    expression: `(async () => {
      const results = [];
      for (const a of document.querySelectorAll('audio')) {
        a.muted = true;
        a.currentTime = 0;
        await a.play();
        await new Promise(resolve => setTimeout(resolve, 120));
        a.pause();
        if (!(a.currentTime > 0)) throw new Error('Playback did not advance: ' + a.src);
        results.push({src: a.getAttribute('src'), advancedSeconds: a.currentTime});
        a.currentTime = 0;
        a.muted = false;
      }
      return results;
    })()` }, sessionId);
  if (playback.exceptionDetails || !Array.isArray(playback.result.value)) throw new Error('Muted playback verification failed: ' + JSON.stringify(playback));
  const metrics = await call('Page.getLayoutMetrics', {}, sessionId);
  const { data } = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: 1100, height: Math.min(10000, Math.ceil(metrics.cssContentSize.height)), scale: 1 } }, sessionId);
  await writeFile(output, Buffer.from(data, 'base64'));
  console.log(JSON.stringify({ screenshot: output, contentHeight: metrics.cssContentSize.height, media: media.result.value, mutedPlayback: playback.result.value }));
  await call('Browser.close');
} finally {
  if (ws) ws.close();
  for (const entry of pending.values()) clearTimeout(entry.timer);
  await pause(500);
  if (child.exitCode === null && child.signalCode === null) {
    try { process.kill(-child.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
}
