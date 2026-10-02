// Restricted CPU component entry point. RLIMIT_AS, seccomp, CPU affinity,
// no swap, inherited flock, and outer reserve watchdog are already active.
import { Worker } from 'node:worker_threads';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { writeSync, readFileSync } from 'node:fs';

const fd = Number(process.env.NANO_GUARD_REPORT_FD);
const high = Number(process.env.NANO_GUARD_HIGH_MIB);
if (!Number.isInteger(fd) || !Number.isInteger(high)) throw Error('Missing resource supervisor.');
if (typeof WebAssembly !== 'undefined') throw Error('Restricted Node checks must disable WASM.');
const args = process.argv.slice(2);
if (args[0] !== '--test' || args.length < 2 || args.slice(1).some((arg) => arg.startsWith('-'))) {
  throw Error('Restricted Node entry supports --test followed by explicit component test files only.');
}
function sample() {
  const peak = process.resourceUsage().maxRSS / 1024;
  const available = Number(readFileSync('/proc/meminfo', 'utf8').match(/^MemAvailable:\s+(\d+)/m)?.[1]) / 1024;
  const noSwap = readFileSync('/proc/swaps', 'utf8').trim().split('\n').length === 1;
  const reason = available < 4096 ? 'headroom_below_4096_mib' : !noSwap ? 'swap_enabled'
    : peak > high ? 'rss_exceeded_high_threshold' : null;
  writeSync(fd, JSON.stringify({ peak_rss_mib: peak, stopped_reason: reason }) + '\n');
  if (reason) process.kill(process.pid, 'SIGKILL');
}
sample();
const watchdog = new Worker(new URL('./guarded_node_watchdog.mjs', import.meta.url), {
  workerData: { fd, high }, resourceLimits: { maxOldGenerationSizeMb: 16, maxYoungGenerationSizeMb: 2, stackSizeMb: 1 },
});
watchdog.on('error', () => process.kill(process.pid, 'SIGKILL'));
try {
  // node:test also runs when imported directly; no child process isolation.
  for (const file of args.slice(1)) await import(pathToFileURL(resolve(file)).href);
} finally {
  // Tests run after module loading. Keep the watchdog alive through TAP output.
  watchdog.unref();
  process.on('beforeExit', sample);
  process.on('exit', sample);
}
