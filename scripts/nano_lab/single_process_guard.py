"""Linux child bootstrap for the stricter, non-cgroup lab backend.

The address-space cap bounds RSS too. Seccomp forbids child processes (threads
remain allowed), so that cap also bounds aggregate job RSS. CPU affinity limits
all inherited threads to two logical CPUs. Refuse hosts with any enabled swap.
This is intentionally less permissive than the existing cgroup backend.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import resource
from pathlib import Path

ALLOW = 0x7FFF0000
ERRNO = 0x00050000
CLONE_THREAD = 0x00010000


class ArgCompare(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]


def no_swap():
    return len(Path('/proc/swaps').read_text().splitlines()) == 1


def install_process_filter():
    if platform.machine() not in {'x86_64', 'aarch64'}:
        raise RuntimeError('Single-process guard supports x86_64/aarch64 Linux only')
    # Loading the stable SONAME directly avoids find_library spawning ldconfig
    # before the process filter is installed.
    try:
        lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    except OSError as exc:
        raise RuntimeError('libseccomp is required; no unfiltered fallback') from exc
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                         ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ArgCompare)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(ALLOW)
    if not context:
        raise RuntimeError('Cannot initialize seccomp')
    try:
        def deny(name, error=errno.EPERM, comparison=None):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number == -1:
                return  # This architecture has no such syscall.
            count = 1 if comparison is not None else 0
            pointer = ctypes.pointer(comparison) if comparison is not None else None
            code = lib.seccomp_rule_add_array(context, ERRNO | error, number, count, pointer)
            if code != 0:
                raise RuntimeError(f'Cannot install seccomp rule for {name}: {code}')
        deny('fork')
        deny('vfork')
        # Return ENOSYS so libc falls back to clone for pthread creation; clone
        # is then allowed only when CLONE_THREAD is present. clone3's flags
        # live behind a pointer and cannot be safely inspected by classic BPF.
        deny('clone3', errno.ENOSYS)
        deny('clone', comparison=ArgCompare(0, 7, CLONE_THREAD, 0))  # MASKED_EQ
        deny('sched_setaffinity')
        deny('setrlimit')
        deny('prlimit64', comparison=ArgCompare(2, 1, 0, 0))  # NE: new_limit != NULL
        deny('swapon')
        if lib.seccomp_load(context) != 0:
            raise RuntimeError('Cannot load seccomp; refusing to execute workload')
    finally:
        lib.seccomp_release(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-mib', type=int, choices=(640, 768, 1024, 1280, 3072), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A workload is required')
    if not no_swap():
        raise SystemExit('Single-process guard refuses hosts with enabled swap')
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if not cpus:
        raise SystemExit('No CPUs available')
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    limit = args.memory_mib * 2**20
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    install_process_filter()
    # pass_fds from the parent preserves its flock across this exec. The lock
    # therefore remains held even if the parent watchdog exits unexpectedly.
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
