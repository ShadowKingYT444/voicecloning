"""Run one lab job with a hard 3 GiB cgroup limit and desktop headroom."""
import argparse, fcntl, os, signal, subprocess, time
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    budget=parser.add_mutually_exclusive_group()
    budget.add_argument('--small-job',action='store_true',help='Hard 1280 MiB cap; require that budget plus 4 GiB desktop reserve')
    budget.add_argument('--max-memory-mib',type=int,choices=(640,768,1024),help='Stricter cap for tiny component checks; always reserve another 4 GiB at launch')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
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
