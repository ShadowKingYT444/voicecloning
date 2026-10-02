"""Run one lab job with a hard 3 GiB limit and desktop headroom.

Default: the existing systemd cgroup. Explicit single-process backend: a
stricter Linux address-space cap and in-process RSS stop for containers.
"""
import argparse, fcntl, os, signal, subprocess, time, sys, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def available_mib():
    values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])/1024

def job_rss_mib(unit):
    """Sample service RSS in addition to the kernel-enforced cgroup cap."""
    result=subprocess.run(['systemctl','--user','show',unit,'--property=ControlGroup','--value'],capture_output=True,text=True)
    group=result.stdout.strip()
    if not group:return 0.
    try:pids=(Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.procs').read_text().split()
    except FileNotFoundError:return 0.
    total=0
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])
        except FileNotFoundError:pass
    return total/1024

def single_process_job(command, memory_mib, high_mib, lock, report_path=None):
    """Stricter Linux backend: one process, capped virtual memory, no swap."""
    from single_process_guard import no_swap
    if not no_swap():
        raise SystemExit('Single-process guard refuses hosts with enabled swap')
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'OPENBLAS_NUM_THREADS':'2','NUMEXPR_MAX_THREADS':'2',
         'TOKENIZERS_PARALLELISM':'false','MALLOC_ARENA_MAX':'2'}
    executable=Path(command[0]).name
    if not (executable.startswith('python') or executable=='node'):
        raise SystemExit('Single-process backend requires Python or restricted Node component checks')
    read_fd,write_fd=os.pipe()
    os.set_blocking(read_fd,False)
    env.update(NANO_GUARD_REPORT_FD=str(write_fd),NANO_GUARD_HIGH_MIB=str(high_mib))
    if executable=='node':
        # No JIT/WASM reservation or child test runners: these are CPU-only
        # source/component checks, never ONNX, browser, or Vite model jobs.
        entry=Path(__file__).with_name('guarded_node.mjs')
        guarded_command=[command[0],'--jitless','--max-old-space-size=128',
                         '--v8-pool-size=1',str(entry),*command[1:]]
    else:
        entry=Path(__file__).with_name('guarded_python.py')
        guarded_command=[command[0],str(entry),*command[1:]]
    child=Path(__file__).with_name('single_process_guard.py')
    process=subprocess.Popen([sys.executable,str(child),'--memory-mib',str(memory_mib),'--',*guarded_command],
                             cwd=ROOT,env=env,start_new_session=True,pass_fds=(lock.fileno(),write_fd))
    os.close(write_fd)
    started=time.monotonic(); peak_rss=0.; stopped=False; reason=None
    pending=b''
    def read_measurements():
        nonlocal pending,peak_rss,reason,stopped
        while True:
            try:data=os.read(read_fd,65536)
            except BlockingIOError:break
            if not data:break
            pending+=data
            while b'\n' in pending:
                line,pending=pending.split(b'\n',1)
                row=json.loads(line)
                peak_rss=max(peak_rss,float(row['peak_rss_mib']))
                if row['stopped_reason']:
                    reason=row['stopped_reason']; stopped=True
    def stop():
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while process.poll() is None:
            read_measurements()
            if available_mib()<4096:reason='headroom_below_4096_mib'
            elif not no_swap():reason='swap_enabled'
            elif peak_rss>high_mib:reason='rss_exceeded_high_threshold'
            if reason:
                stopped=True; print('Stopping single-process job: '+reason,flush=True); stop(); break
            time.sleep(.1)
        code=process.wait()
        read_measurements()
        return 125 if stopped else code
    finally:
        if process.poll() is None:stop(); process.wait()
        os.close(read_fd)
        for s,handler in previous.items():signal.signal(s,handler)
        if report_path:
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(dict(backend='single-process',hard_address_space_mib=memory_mib,
                rss_stop_mib=high_mib,desktop_reserve_mib=4096,child_processes='seccomp_denied',
                threads='allowed_on_two_cpus',swap='host_disabled_required',sampled_peak_rss_mib=peak_rss,
                measurement_source='in_process_getrusage',
                stopped_reason=reason,returncode=process.returncode,elapsed_s=time.monotonic()-started),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('systemd','single-process'),default='systemd',
                        help='single-process is stricter: hard address-space cap, seccomp denies child processes, no host swap')
    parser.add_argument('--guard-report',type=Path,help='Write single-process watchdog measurements')
    budget=parser.add_mutually_exclusive_group()
    budget.add_argument('--small-job',action='store_true',help='Hard 1280 MiB cap; require that budget plus 4 GiB desktop reserve')
    budget.add_argument('--max-memory-mib',type=int,choices=(640,768,1024),help='Stricter cap for tiny component checks; always reserve another 4 GiB at launch')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command[:1]==['--']:args.command=args.command[1:]
    if not args.command:parser.error('A command is required')
    lock_path=ROOT/'artifacts/nano_lab/model_job.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another bounded lab job is running')
        memory_mib=args.max_memory_mib or (1280 if args.small_job else 3072)
        start_mib=4096+memory_mib if memory_mib<3072 else 6144
        high_mib=int(memory_mib*.8) if memory_mib<3072 else 2500
        if available_mib()<start_mib:
            raise SystemExit(f'Not starting: less than {start_mib/1024:g} GiB RAM is available')
        if args.backend=='single-process':
            print(f'Hard address-space cap={memory_mib}MiB, RSS stop={high_mib}MiB, seccomp single process, no swap, 2 CPUs; reserve=4GiB',flush=True)
            raise SystemExit(single_process_job(args.command,memory_mib,high_mib,lock,args.guard_report))
        # A fixed service name also prevents overlap if a prior wrapper was
        # interrupted but its service is still being stopped.
        unit='nano-lab-model'
        command=['systemd-run','--user','--wait','--pipe','--collect',f'--unit={unit}',
                 '-p',f'MemoryMax={memory_mib}M','-p',f'MemoryHigh={high_mib}M','-p','MemorySwapMax=0',
                 '-p','CPUQuota=200%','-p','Nice=10','-p','OOMPolicy=kill',
                 f'--working-directory={ROOT}',
                 '--setenv=OMP_NUM_THREADS=2','--setenv=MKL_NUM_THREADS=2',
                 '--setenv=OPENBLAS_NUM_THREADS=2','--setenv=NUMEXPR_MAX_THREADS=2',
                 '--setenv=TOKENIZERS_PARALLELISM=false',*args.command]
        print(f'MemoryMax={memory_mib}MiB, MemoryHigh={high_mib}MiB, no swap, 2 CPU cores, one job; desktop reserve=4GiB',flush=True)
        process=subprocess.Popen(command)
        def interrupted(signum,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        stopped_for_resources=False
        try:
            while process.poll() is None:
                if available_mib()<4096:
                    print('Stopping job: desktop headroom fell below 4 GiB',flush=True)
                    stopped_for_resources=True
                    subprocess.run(['systemctl','--user','stop',unit],check=False)
                    break
                if job_rss_mib(unit)>memory_mib:
                    print(f'Stopping job: aggregate process RSS exceeded {memory_mib} MiB',flush=True)
                    stopped_for_resources=True
                    subprocess.run(['systemctl','--user','stop',unit],check=False)
                    break
                time.sleep(1)
            code=process.wait()
            raise SystemExit(125 if stopped_for_resources else code)
        finally:
            if process.poll() is None:
                subprocess.run(['systemctl','--user','stop',unit],check=False)
                process.wait()

if __name__=='__main__':main()
