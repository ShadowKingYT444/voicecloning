import { workerData } from 'node:worker_threads';
import { readFileSync, writeSync } from 'node:fs';
const { fd, high } = workerData;
setInterval(() => {
  try {
    const peak = process.resourceUsage().maxRSS / 1024;
    const available = Number(readFileSync('/proc/meminfo', 'utf8').match(/^MemAvailable:\s+(\d+)/m)?.[1]) / 1024;
    const noSwap = readFileSync('/proc/swaps', 'utf8').trim().split('\n').length === 1;
    const reason = available < 4096 ? 'headroom_below_4096_mib' : !noSwap ? 'swap_enabled'
      : peak > high ? 'rss_exceeded_high_threshold' : null;
    writeSync(fd, JSON.stringify({ peak_rss_mib: peak, stopped_reason: reason }) + '\n');
    if (reason) process.kill(process.pid, 'SIGKILL');
  } catch { process.kill(process.pid, 'SIGKILL'); }
}, 100);
