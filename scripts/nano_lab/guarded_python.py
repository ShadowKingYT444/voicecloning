"""Python workload entry point with an in-process memory/headroom watchdog.

Some container PID namespaces expose the parent's bootstrap rather than the
executed workload through /proc/<pid>. Sample self and rusage inside the actual
workload, emit measurements to its supervisor, and enforce the high threshold.
The seccomp/RLIMIT_AS bootstrap has already run before this entry point.
"""
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import threading
import time


def main():
    fd = int(os.environ.pop('NANO_GUARD_REPORT_FD'))
    high = int(os.environ.pop('NANO_GUARD_HIGH_MIB'))
    done = threading.Event()
    def sample():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(info['MemAvailable'].split()[0])/1024
        reason = None
        if available < 4096:
            reason = 'headroom_below_4096_mib'
        elif len(Path('/proc/swaps').read_text().splitlines()) != 1:
            reason = 'swap_enabled'
        elif peak > high:
            reason = 'rss_exceeded_high_threshold'
        try:
            os.write(fd, (json.dumps({'peak_rss_mib': peak, 'stopped_reason': reason})+'\n').encode())
        except BrokenPipeError:
            os._exit(125)  # Do not continue after losing the supervisor.
        if reason:
            os._exit(125)
    def watch():
        while not done.wait(.1):
            sample()
    arguments = sys.argv[1:]
    if not arguments or arguments[0].startswith('-') and arguments[0] not in ('-c', '-m'):
        raise SystemExit('Single-process Python guard supports a script, -c, or -m')
    thread = threading.Thread(target=watch, name='nano-resource-watchdog', daemon=True)
    sample()
    thread.start()
    try:
        if arguments[0] == '-c':
            sys.argv = ['-c', *arguments[2:]]
            exec(compile(arguments[1], '<string>', 'exec'), {'__name__': '__main__'})
        elif arguments[0] == '-m':
            sys.argv = arguments[1:]
            runpy.run_module(arguments[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = arguments
            sys.path.insert(0, str(Path(arguments[0]).resolve().parent))
            runpy.run_path(arguments[0], run_name='__main__')
    finally:
        done.set()
        thread.join()
        sample()
        os.close(fd)


if __name__ == '__main__':
    main()
