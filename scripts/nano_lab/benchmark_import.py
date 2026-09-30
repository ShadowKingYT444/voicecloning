"""Measure one package import in a fresh process, without loading weights."""
import argparse,importlib,json,resource,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
start=time.perf_counter();importlib.import_module(a.module);elapsed=time.perf_counter()-start
status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
r={'module':a.module,'import_seconds':elapsed,'rss_mib':int(status['VmRSS'].split()[0])/1024,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
