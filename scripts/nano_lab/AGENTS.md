# Resource constraints

The user reported desktop crashes from this task's memory use. This constraint
applies to all subsequent training, synthesis, export, and evaluation work.

- Run at most one model process at a time. Do not run model workloads in agents
  concurrently with the main process.
- Launch model work through `bounded_job.py`. It uses a systemd cgroup with a
  hard 3 GiB memory limit, 2500 MiB high threshold, zero swap, two CPU cores,
  and reduced scheduling priority. It requires 6 GiB available before launch
  and stops the job if system headroom drops below 4 GiB.
- Do not increase these limits without a new user instruction.
- Lightweight checks may use `bounded_job.py --small-job`: a stricter 1280 MiB
  hard memory/RSS cap, 1024 MiB high threshold, no swap, and 5376 MiB required
  at launch. That launch threshold reserves the entire job budget plus 4 GiB
  for the desktop. The 4 GiB runtime headroom stop remains mandatory. Do not
  use this mode to bypass a failed large job; it must fit the smaller cap.
- Tiny component tests can select a still stricter 640, 768, or 1024 MiB cap
  with `--max-memory-mib`. Their start threshold is the full cap plus 4096 MiB;
  the same lock, no-swap policy, two-core quota, and 4 GiB runtime stop apply.
  A lower cap is for a smaller workload, not a retry of an oversized workload.
- Use the optimized loader. Preserve checkpoints and completed artifacts.
- Do not kill unrelated desktop processes or flush system caches.
- Report a memory-limit failure accurately. Reduce the job's working set before
  retrying. Do not bypass the resource wrapper to make a failed job pass.
