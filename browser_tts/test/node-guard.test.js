import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

test('restricted Node guard retains kernel cap and denies children', { skip: !process.env.NANO_GUARD_REPORT_FD }, () => {
  assert.equal(typeof WebAssembly, 'undefined');
  const limits = readFileSync('/proc/self/limits', 'utf8');
  assert.match(limits, /Max address space\s+671088640\s+671088640/);
  const child = spawnSync('/bin/true');
  assert.equal(child.error?.code, 'EPERM');
  const status = readFileSync('/proc/self/status', 'utf8');
  assert.match(status, /^Seccomp:\s+2$/m);
  const cpuList = status.match(/^Cpus_allowed_list:\s+(.*)$/m)[1];
  const cpus = cpuList.split(',').reduce((count, span) => {
    const [first, last = first] = span.split('-').map(Number); return count + last - first + 1;
  }, 0);
  assert.ok(cpus <= 2);
});
