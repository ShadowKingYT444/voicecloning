#!/usr/bin/env python3
"""Measure the browser TTS reader in an isolated, bounded Chrome profile.

The caller must launch this script through scripts/nano_lab/bounded_job.py.
The Vite server must already listen on 127.0.0.1:4187. This runner does not
start or stop the app server.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = "Take a slow breath in, and let your shoulders relax."
GPU_INIT_SCRIPT = r"""(() => {
  const state = {
    installed: false,
    installationPending: true,
    instrumentationStatus: 'installationPending',
    requestAdapterHookInstalled: false,
    requestDeviceHookInstalled: false,
    deviceCreateBufferHookInstalled: false,
    destroyHookedBufferCount: 0,
    destroyHookFailureCount: 0,
    adapterCallCount: 0,
    adapterCount: 0,
    deviceCallCount: 0,
    deviceCount: 0,
    adapterDiagnostics: [],
    deviceFeatures: [],
    hookFailures: [],
    createCalls: 0,
    destroyCalls: 0,
    createdRequestedBytes: 0,
    destroyedRequestedBytes: 0,
    notDestroyedRequestedBytes: 0,
    peakNotDestroyedRequestedBytes: 0,
    note: 'Requested GPUBuffer descriptor sizes; not a physical VRAM measurement.'
  };
  globalThis.__voiceStudyGpuBufferStats = state;
  const sizes = new WeakMap();
  const destroyed = new WeakSet();
  const wrappedTargets = new WeakMap();
  const adapterDetails = (adapter) => {
    let info = {};
    try { info = adapter.info || {}; } catch (_) {}
    const features = [];
    try { for (const feature of adapter.features || []) features.push(String(feature)); } catch (_) {}
    const diagnostics = {
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
    };
    const fallbackStatus = typeof info.isFallbackAdapter === 'boolean' ? info.isFallbackAdapter
      : (typeof adapter.isFallbackAdapter === 'boolean' ? adapter.isFallbackAdapter : null);
    const record = {
      isFallbackAdapter: fallbackStatus,
      fallbackStatusSource: typeof info.isFallbackAdapter === 'boolean' ? 'GPUAdapterInfo.isFallbackAdapter'
        : (typeof adapter.isFallbackAdapter === 'boolean' ? 'GPUAdapter.isFallbackAdapter' : 'unknown'),
      info: diagnostics,
      features,
      infoError: null,
    };
    const refreshAssertions = () => {
      const label = Object.values(record.info).filter(Boolean).join(' ').toLowerCase();
      const softwareMarkers = ['swiftshader', 'llvmpipe', 'lavapipe', 'softpipe', 'software renderer', 'software adapter'];
      const hasSoftwareLabel = softwareMarkers.some((marker) => label.includes(marker));
      record.hardwareAssertions = {
        hasAdapterIdentity: Object.values(record.info).some(Boolean),
        notFallback: record.isFallbackAdapter === false,
        noKnownSoftwareLabel: !hasSoftwareLabel,
        noGoogleSwiftShader: !(String(record.info.vendor || '').toLowerCase() === 'google' && label.includes('swiftshader')),
        shaderF16: features.includes('shader-f16'),
      };
      record.knownSoftwareLabel = hasSoftwareLabel;
      record.hardwareAssertionsPassed = Object.values(record.hardwareAssertions).every(Boolean);
      state.hardwareAssertionsPassed = record.hardwareAssertionsPassed;
    };
    refreshAssertions();
    if (!Object.values(record.info).some(Boolean) && typeof adapter.requestAdapterInfo === 'function') {
      try {
        Promise.resolve(adapter.requestAdapterInfo()).then((laterInfo) => {
          record.info = {
            vendor: laterInfo && laterInfo.vendor || null,
            architecture: laterInfo && laterInfo.architecture || null,
            device: laterInfo && laterInfo.device || null,
            description: laterInfo && laterInfo.description || null,
          };
          if (typeof (laterInfo && laterInfo.isFallbackAdapter) === 'boolean') {
            record.isFallbackAdapter = laterInfo.isFallbackAdapter;
            record.fallbackStatusSource = 'GPUAdapterInfo.isFallbackAdapter';
          }
          refreshAssertions();
        }, (error) => { record.infoError = String(error && error.message || error).slice(0, 500); });
      } catch (error) { record.infoError = String(error && error.message || error).slice(0, 500); }
    }
    return record;
  };
  const pushFailure = (hook, error) => {
    state.hookFailures.push({ hook, error: String(error && error.message || error).slice(0, 500) });
    state.installed = false;
    state.instrumentationStatus = state.deviceCreateBufferHookInstalled ? 'partial' : 'hookFailed';
    state.installationPending = false;
  };
  const markActive = () => {
    state.installed = state.deviceCreateBufferHookInstalled && state.destroyHookedBufferCount > 0 &&
      state.destroyHookFailureCount === 0 && state.hookFailures.length === 0;
    if (state.installed) {
      state.instrumentationStatus = 'active';
      state.installationPending = false;
    } else if (state.deviceCreateBufferHookInstalled &&
      (state.destroyHookFailureCount > 0 || state.hookFailures.length > 0)) {
      state.instrumentationStatus = 'partial';
      state.installationPending = false;
    }
  };
  const wasWrapped = (target, name) => wrappedTargets.get(target)?.has(name) || false;
  const rememberWrapped = (target, name) => {
    let names = wrappedTargets.get(target);
    if (!names) { names = new Set(); wrappedTargets.set(target, names); }
    names.add(name);
  };
  const patchMethod = (target, name, makeWrapper, hook) => {
    if (!target) { pushFailure(hook, 'API object is unavailable.'); return false; }
    if (wasWrapped(target, name)) return true;
    const original = target[name];
    if (typeof original !== 'function') { pushFailure(hook, `${name} is not callable.`); return false; }
    const wrapped = makeWrapper(original);
    try {
      Object.defineProperty(target, name, { configurable: true, writable: true, value: wrapped });
      if (target[name] === wrapped) { rememberWrapped(target, name); return true; }
    } catch (error) {}
    const ownDescriptor = Object.getOwnPropertyDescriptor(target, name);
    if (ownDescriptor) {
      pushFailure(hook, `Could not replace non-configurable own ${name} method.`);
      return false;
    }
    const prototype = Object.getPrototypeOf(target);
    if (prototype && typeof prototype[name] === 'function' && !wasWrapped(prototype, name)) {
      const protoOriginal = prototype[name];
      try {
        Object.defineProperty(prototype, name, { configurable: true, writable: true, value: makeWrapper(protoOriginal) });
        if (prototype[name] !== protoOriginal) { rememberWrapped(prototype, name); return true; }
      } catch (error) {
        pushFailure(hook, error);
        return false;
      }
    } else if (prototype && wasWrapped(prototype, name)) {
      return true;
    }
    pushFailure(hook, `Could not wrap ${name} on the API object or its prototype.`);
    return false;
  };
  const installBufferDestroy = (buffer) => {
    const installed = patchMethod(buffer, 'destroy', (original) => function(...args) {
      const result = Reflect.apply(original, this, args);
      if (!destroyed.has(this) && sizes.has(this)) {
        destroyed.add(this);
        const size = sizes.get(this) || 0;
        state.destroyCalls += 1;
        state.destroyedRequestedBytes += size;
        state.notDestroyedRequestedBytes = Math.max(0, state.notDestroyedRequestedBytes - size);
      }
      return result;
    }, 'GPUBuffer.destroy');
    if (installed) state.destroyHookedBufferCount += 1;
    else state.destroyHookFailureCount += 1;
    markActive();
    return installed;
  };
  const installDevice = (device) => {
    state.deviceCount += 1;
    let features = [];
    try { features = [...device.features].map(String); } catch (_) {}
    state.deviceFeatures.push(features);
    const installed = patchMethod(device, 'createBuffer', (original) => function(descriptor, ...args) {
      const buffer = Reflect.apply(original, this, [descriptor, ...args]);
      const size = Number(descriptor && descriptor.size);
      if (Number.isFinite(size) && size >= 0) {
        sizes.set(buffer, size);
        state.createCalls += 1;
        state.createdRequestedBytes += size;
        state.notDestroyedRequestedBytes += size;
        state.peakNotDestroyedRequestedBytes = Math.max(
          state.peakNotDestroyedRequestedBytes, state.notDestroyedRequestedBytes);
        installBufferDestroy(buffer);
      }
      return buffer;
    }, 'GPUDevice.createBuffer');
    state.deviceCreateBufferHookInstalled = state.deviceCreateBufferHookInstalled || installed;
    if (installed && state.instrumentationStatus !== 'hookFailed') {
      state.instrumentationStatus = 'deviceCreateHookInstalled';
    }
    markActive();
  };
  const installAdapter = (adapter) => {
    if (!adapter) return;
    state.adapterCount += 1;
    const diagnostics = adapterDetails(adapter);
    state.adapterDiagnostics.push(diagnostics);
    state.hardwareAssertionsPassed = Object.values(diagnostics.hardwareAssertions).every(Boolean);
    const installed = patchMethod(adapter, 'requestDevice', (original) => function(...args) {
      state.deviceCallCount += 1;
      let result;
      try { result = Reflect.apply(original, this, args); }
      catch (error) { pushFailure('GPUAdapter.requestDevice-call', error); throw error; }
      return Promise.resolve(result).then((device) => {
        if (device) installDevice(device);
        else pushFailure('GPUAdapter.requestDevice-result', 'requestDevice returned no device.');
        return device;
      }, (error) => { pushFailure('GPUAdapter.requestDevice-rejection', error); throw error; });
    }, 'GPUAdapter.requestDevice');
    state.requestDeviceHookInstalled = state.requestDeviceHookInstalled || installed;
    if (installed && state.instrumentationStatus !== 'hookFailed') {
      state.instrumentationStatus = 'requestDeviceHookInstalled';
    }
  };
  if (!globalThis.navigator || !navigator.gpu) {
    state.instrumentationStatus = 'unavailable';
    state.installationPending = false;
    state.reason = 'navigator.gpu is unavailable in this worker.';
    return { installed: false, installationPending: false, instrumentationStatus: state.instrumentationStatus, reason: state.reason };
  }
  const adapterHook = patchMethod(navigator.gpu, 'requestAdapter', (original) => function(...args) {
    state.adapterCallCount += 1;
    let result;
    try { result = Reflect.apply(original, this, args); }
    catch (error) { pushFailure('GPU.requestAdapter-call', error); throw error; }
    return Promise.resolve(result).then((adapter) => {
      if (adapter) installAdapter(adapter);
      else pushFailure('GPU.requestAdapter-result', 'requestAdapter returned no adapter.');
      return adapter;
    }, (error) => { pushFailure('GPU.requestAdapter-rejection', error); throw error; });
  }, 'GPU.requestAdapter');
  state.requestAdapterHookInstalled = adapterHook;
  if (adapterHook && state.instrumentationStatus !== 'hookFailed') state.instrumentationStatus = 'adapterHookInstalled';
  return {
    installed: false,
    installationPending: state.installationPending,
    instrumentationStatus: state.instrumentationStatus,
    requestAdapterHookInstalled: state.requestAdapterHookInstalled,
    hookFailures: state.hookFailures,
  };
})()"""
class RunnerError(RuntimeError):
    pass


class SignalStop(RunnerError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_local_model_base(model_base: str | None, app_url: str) -> str | None:
    if model_base is None:
        return None
    page = urllib.parse.urlsplit(app_url)
    resolved = urllib.parse.urlsplit(urllib.parse.urljoin(app_url, model_base))
    app_origin = (page.scheme.lower(), page.netloc.lower())
    model_origin = (resolved.scheme.lower(), resolved.netloc.lower())
    if model_origin != app_origin or resolved.scheme.lower() not in ("http", "https"):
        raise ValueError("--model-base must use the same local HTTP(S) origin as --url.")
    if resolved.username or resolved.password:
        raise ValueError("--model-base cannot contain URL credentials.")
    if resolved.query or resolved.fragment:
        raise ValueError("--model-base cannot contain a query or fragment.")
    path = resolved.path if resolved.path.endswith("/") else resolved.path + "/"
    return urllib.parse.urlunsplit((resolved.scheme, resolved.netloc, path, "", ""))


def mib(kib: float) -> float:
    return round(kib / 1024.0, 3)


def bounded_timeout(deadline: float, maximum: float = 15.0) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunnerError("Overall hard timeout expired.")
    return min(maximum, remaining)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_path(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return ROOT / "artifacts" / "nano_lab" / "browser_measurements" / f"run-{stamp}"


def guard_context() -> dict[str, Any]:
    lines: list[str] = []
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        pass
    expected = any("nano-lab-model" in line for line in lines)
    tiny_guard = bool(os.environ.get("NANO_GUARD_REPORT_FD"))
    return {
        "cgroup": lines,
        "bounded_job_unit_detected": expected,
        "single_process_guard_detected": tiny_guard,
        "accepted": expected,
    }


def check_local_server(url: str, timeout: float = 4.0) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != 4187:
        raise RunnerError("Use the existing local app URL with explicit port 4187, for example http://127.0.0.1:4187/.")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read(128)
            status = response.status
    except (OSError, urllib.error.URLError) as error:
        raise RunnerError(f"The existing app server is not reachable at {url}: {error}") from error
    return {"url": url, "status": status, "serverStartedByRunner": False}


def scan_processes() -> dict[int, tuple[int, int]]:
    """Return pid -> (parent pid, start ticks) from procfs."""
    output: dict[int, tuple[int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return output
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            if len(fields) < 20:
                continue
            ppid = int(fields[1])
            start_ticks = int(fields[19])
            output[int(entry.name)] = (ppid, start_ticks)
        except (OSError, ValueError, IndexError):
            continue
    return output


def process_tree(root_pid: int, process_map: dict[int, tuple[int, int]]) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in process_map.items():
        children.setdefault(ppid, []).append(pid)
    found = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return sorted(pid for pid in found if pid in process_map)


def process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    except OSError:
        return []


def process_executable(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


class BrowserOwnership:
    def __init__(self, pid: int, chrome_path: Path, profile: Path):
        self.pid = pid
        self.chrome_path = str(chrome_path.resolve())
        self.profile = str(profile.resolve())
        self.identities: dict[int, dict[str, Any]] = {}

    def current_pids(self) -> list[int]:
        process_map = scan_processes()
        pids = process_tree(self.pid, process_map)
        for pid in pids:
            stat = process_map[pid]
            cmdline = process_cmdline(pid)
            exe = process_executable(pid)
            self.identities[pid] = {
                "startTicks": stat[1],
                "executable": exe,
                "cmdline": cmdline,
                "observedDescendantOfChromePid": True,
            }
        return pids

    def matches_identity(self, pid: int, identity: dict[str, Any]) -> bool:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[19]) != identity["startTicks"]:
                return False
            cmdline = process_cmdline(pid)
            exe = process_executable(pid) or ""
            observed_executable = identity.get("executable")
            return (exe == self.chrome_path or
                    (observed_executable is not None and exe == observed_executable) or
                    any(self.profile in item for item in cmdline))
        except (OSError, ValueError, IndexError):
            return False

    def terminate(self, process: subprocess.Popen[bytes], timeout: float = 5.0) -> list[int]:
        self.current_pids()
        owned = sorted(self.identities, reverse=True)
        for pid in owned:
            if self.matches_identity(pid, self.identities[pid]):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            self.current_pids()
            time.sleep(0.1)
        self.current_pids()
        killed: list[int] = []
        for pid in sorted(self.identities, reverse=True):
            if self.matches_identity(pid, self.identities[pid]):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        return killed


class DevTools:
    """Small RFC 6455 WebSocket client for local Chrome DevTools Protocol."""

    def __init__(self, url: str, timeout: float = 30.0):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "ws" or parsed.port is None:
            raise RunnerError(f"Unexpected DevTools WebSocket URL: {url}")
        host = parsed.hostname or "127.0.0.1"
        if host == "localhost":
            host = "127.0.0.1"
        self.sock = socket.create_connection((host, parsed.port), timeout=timeout)
        self.sock.settimeout(min(5.0, timeout))
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        host_header = f"{host}:{parsed.port}"
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        self.sock.sendall(handshake)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RunnerError("Chrome closed the DevTools handshake.")
            response.extend(chunk)
            if len(response) > 16384:
                raise RunnerError("DevTools handshake response is too large.")
        header, extra = bytes(response).split(b"\r\n\r\n", 1)
        if extra:
            self._prefetched = bytearray(extra)
        else:
            self._prefetched = bytearray()
        lines = header.decode("latin1").split("\r\n")
        if not lines[0].startswith("HTTP/1.1 101"):
            raise RunnerError(f"DevTools WebSocket handshake failed: {lines[0]}")
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(
            __import__("hashlib").sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raise RunnerError("DevTools returned an invalid WebSocket accept key.")
        self.next_id = 0
        self.responses: dict[int, dict[str, Any]] = {}
        self.event_handler = None
        self.closed = False

    def _send_frame(self, payload: bytes, opcode: int = 1) -> None:
        mask = os.urandom(4)
        size = len(payload)
        header = bytearray([0x80 | opcode])
        if size < 126:
            header.append(0x80 | size)
        elif size < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", size))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", size))
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_exact(self, size: int) -> bytes:
        chunks = bytearray()
        if self._prefetched:
            count = min(size, len(self._prefetched))
            chunks.extend(self._prefetched[:count])
            del self._prefetched[:count]
        while len(chunks) < size:
            chunk = self.sock.recv(size - len(chunks))
            if not chunk:
                raise RunnerError("DevTools WebSocket closed during a frame.")
            chunks.extend(chunk)
        return bytes(chunks)

    def _recv_frame(self) -> tuple[int, bool, bytes]:
        first, second = self._recv_exact(2)
        opcode = first & 0x0F
        fin = bool(first & 0x80)
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, fin, payload

    def _recv_message(self) -> str | None:
        fragments = bytearray()
        fragment_opcode = None
        while True:
            opcode, fin, payload = self._recv_frame()
            if opcode == 8:
                raise RunnerError("Chrome closed the DevTools WebSocket.")
            if opcode == 9:
                self._send_frame(payload, opcode=10)
                continue
            if opcode == 10:
                continue
            if opcode in (1, 2):
                fragment_opcode = opcode
                fragments = bytearray(payload)
            elif opcode == 0:
                fragments.extend(payload)
            else:
                continue
            if fin:
                if fragment_opcode != 1:
                    return None
                return fragments.decode("utf-8")

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message:
            self.responses[int(message["id"])] = message
        elif self.event_handler and message.get("method"):
            self.event_handler(message)

    def call(self, method: str, params: dict[str, Any] | None = None, session_id: str | None = None,
             timeout: float = 15.0) -> dict[str, Any]:
        self.next_id += 1
        call_id = self.next_id
        message: dict[str, Any] = {"id": call_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        self._send_frame(json.dumps(message, separators=(",", ":")).encode("utf-8"))
        deadline = time.monotonic() + timeout
        while call_id not in self.responses:
            if time.monotonic() >= deadline:
                raise RunnerError(f"DevTools timed out during {method}.")
            self.sock.settimeout(max(0.1, min(5.0, deadline - time.monotonic())))
            try:
                payload = self._recv_message()
            except socket.timeout:
                continue
            if payload is None:
                continue
            self._dispatch(json.loads(payload))
        response = self.responses.pop(call_id)
        if "error" in response:
            raise RunnerError(f"DevTools {method} failed: {response['error']}")
        return response.get("result", {})

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._send_frame(b"", opcode=8)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class BrowserMeasurement:
    def __init__(self, args: argparse.Namespace, report: dict[str, Any], browser: subprocess.Popen[bytes],
                 chrome_path: Path, profile: Path, devtools: DevTools, page_session: str,
                 sample_log_path: Path, report_path: Path, started: float, deadline: float):
        self.args = args
        self.report = report
        self.browser = browser
        self.ownership = BrowserOwnership(browser.pid, chrome_path, profile)
        self.devtools = devtools
        self.page_session = page_session
        self.started = started
        self.deadline = deadline
        self.worker_sessions: dict[str, dict[str, Any]] = {}
        self.gpu_detached: list[dict[str, Any]] = []
        self.stage = None
        self.stage_name: str | None = None
        self.sample_log_path = sample_log_path
        self.report_path = report_path
        self.baseline_pss_mib: float | None = None
        self.last_nvidia_sample = 0.0
        self.report.setdefault("stages", {})
        self.report.setdefault("workers", [])
        self.report.setdefault("networkRequests", [])
        self.devtools.event_handler = self.on_event

    def protocol(self, method: str, params: dict[str, Any] | None = None, session: str | None = None,
                 timeout: float = 15.0) -> dict[str, Any]:
        return self.devtools.call(method, params, session, timeout=bounded_timeout(self.deadline, timeout))

    def on_event(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params", {})
        if method == "Target.attachedToTarget":
            target = params.get("targetInfo", {})
            session = params.get("sessionId")
            if not session:
                return
            is_worker = target.get("type") in ("worker", "service_worker", "shared_worker")
            record = {
                "sessionId": session,
                "targetId": target.get("targetId"),
                "type": target.get("type"),
                "url": target.get("url"),
                "instrumentation": "pending",
            }
            self.worker_sessions[session] = record
            self.report["workers"].append(record)
            try:
                self.protocol("Runtime.enable", {}, session)
                self.protocol("Network.enable", {}, session)
                if is_worker and not self.args.ui_only:
                    result = self.protocol("Runtime.evaluate", {
                        "expression": GPU_INIT_SCRIPT,
                        "returnByValue": True,
                        "awaitPromise": False,
                    }, session)
                    injected = result.get("result", {}).get("value", {})
                    record["instrumentation"] = injected.get("instrumentationStatus", "unavailable")
                    record["installationPending"] = bool(injected.get("installationPending"))
                    record["concreteDeviceInstrumentation"] = bool(injected.get("deviceCreateBufferHookInstalled"))
                    record["bufferInstrumentationActive"] = bool(injected.get("installed"))
                    record["instrumentationResult"] = injected
            except Exception as error:
                record["instrumentation"] = "failed"
                record["instrumentationError"] = str(error)
            finally:
                try:
                    self.protocol("Runtime.runIfWaitingForDebugger", {}, session, timeout=10)
                except Exception as error:
                    record["resumeError"] = str(error)
        elif method == "Target.detachedFromTarget":
            session = params.get("sessionId")
            record = self.worker_sessions.pop(session, None)
            if record:
                record["detachedAtUtc"] = utc_now()
                self.gpu_detached.append(record)
        elif method == "Network.requestWillBeSent":
            request = params.get("request", {})
            url = request.get("url", "")
            # Keep this focused on asset requests. The full URL is useful for
            # checking whether UI-only mode reached the model host.
            if ("huggingface.co" in url or "/voice/" in url or "/models/" in url
                    or "federalist-no-10.txt" in url):
                self.report["networkRequests"].append({
                    "timestamp": params.get("timestamp"),
                    "wallTime": params.get("wallTime"),
                    "url": url,
                    "method": request.get("method"),
                    "type": params.get("type"),
                })

    def check_timeout(self, stage: str, stage_deadline: float) -> None:
        if self.browser.poll() is not None:
            raise RunnerError(f"Chrome exited during stage {stage} with code {self.browser.returncode}.")
        now = time.monotonic()
        if now >= self.deadline:
            raise RunnerError(f"Overall hard timeout expired during stage {stage}.")
        if now >= stage_deadline:
            raise RunnerError(f"Stage timeout expired during {stage}.")

    def start_stage(self, name: str) -> None:
        stage = {
            "startedAtUtc": utc_now(),
            "startedElapsedSeconds": round(time.monotonic() - self.started, 3),
            "samples": [],
            "status": "running",
        }
        self.report["stages"][name] = stage
        self.stage = stage
        self.stage_name = name
        self.sample(name)

    def finish_stage(self, name: str, status: str = "complete", error: str | None = None) -> None:
        stage = self.report["stages"][name]
        self.sample(name)
        stage["endedAtUtc"] = utc_now()
        stage["endedElapsedSeconds"] = round(time.monotonic() - self.started, 3)
        stage["durationSeconds"] = round(stage["endedElapsedSeconds"] - stage["startedElapsedSeconds"], 3)
        stage["status"] = status
        if error:
            stage["error"] = error
        samples = stage["samples"]
        if samples:
            stage["sampledPeakRssMiB"] = round(max(item["hostProcessTree"]["rssMiB"] for item in samples), 3)
            stage["sampledPeakPssMiB"] = round(max(item["hostProcessTree"]["pssMiB"] for item in samples), 3)
            stage["sampledPeakIncrementalPssMiB"] = (None if name == "clean_baseline" else round(
                stage["sampledPeakPssMiB"] - (self.baseline_pss_mib or 0.0), 3))
            stage["sampledPeakProcessCount"] = max(item["hostProcessTree"]["processCount"] for item in samples)
            heap_used = [item['javascriptHeap'].get('observedTargetUsedBytes') for item in samples]
            stage['sampledPeakObservedJsHeapUsedMiB'] = max(
                (value / 1048576 for value in heap_used if value is not None), default=None)
            gpu_rows = [sample.get("webgpuRequestedBuffers", {}) for sample in samples]
            current_rows = [row.get("currentNotDestroyedRequestedBytes", 0) for row in gpu_rows]
            highwater_rows = [row.get("largestWorkerLifetimePeakRequestedBytes", 0) for row in gpu_rows]
            stage["sampledPeakCurrentNotDestroyedRequestedMiB"] = round(max(current_rows, default=0) / 1048576, 3)
            stage["largestWorkerLifetimePeakRequestedMiB"] = round(max(highwater_rows, default=0) / 1048576, 3)
            stage["workerLifetimePeakIncreaseDuringStageMiB"] = round(
                max(0, max(highwater_rows, default=0) - highwater_rows[0]) / 1048576, 3)
            elapsed = [float(item["elapsedSeconds"]) for item in samples]
            intervals = [round(right - left, 3) for left, right in zip(elapsed, elapsed[1:])]
            stage["sampleIntervalSeconds"] = {
                "configured": self.args.sample_seconds,
                "meanObserved": round(sum(intervals) / len(intervals), 3) if intervals else None,
                "maxObserved": max(intervals, default=None),
            }
            stage["createdRequestedBytesAtEnd"] = gpu_rows[-1].get("createdRequestedBytes", 0)
            stage["destroyedRequestedBytesAtEnd"] = gpu_rows[-1].get("destroyedRequestedBytes", 0)
            stage["createCallsAtEnd"] = gpu_rows[-1].get("createCalls", 0)
            stage["destroyCallsAtEnd"] = gpu_rows[-1].get("destroyCalls", 0)
            stage["concreteDeviceInstrumentationWorkerCountAtEnd"] = gpu_rows[-1].get(
                "concreteDeviceInstrumentationWorkerCount", 0)
            stage["instrumentedWorkerCountAtEnd"] = gpu_rows[-1].get("instrumentedWorkerCount", 0)
            stage["installationPendingWorkerCountAtEnd"] = gpu_rows[-1].get(
                "installationPendingWorkerCount", 0)
            stage["gpuInstrumentationStatusesAtEnd"] = gpu_rows[-1].get("instrumentationStatuses", [])
        self.stage = None
        self.stage_name = None
        write_json(self.report_path, self.report)

    def wait_sample(self, interval: float, name: str) -> None:
        time.sleep(interval)
        self.sample(name)

    def sample(self, name: str) -> None:
        process_map = scan_processes()
        pids = process_tree(self.browser.pid, process_map)
        for pid in pids:
            if pid in process_map:
                self.ownership.identities[pid] = {
                    "startTicks": process_map[pid][1],
                    "executable": process_executable(pid),
                    "cmdline": process_cmdline(pid),
                    "observedDescendantOfChromePid": True,
                }
        rss_kib = 0.0
        pss_kib = 0.0
        missing = []
        for pid in pids:
            try:
                lines = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="ascii").splitlines()
                values = {}
                for line in lines:
                    if line.startswith("Rss:"):
                        values["rss"] = float(line.split()[1])
                    elif line.startswith("Pss:"):
                        values["pss"] = float(line.split()[1])
                if "rss" not in values or "pss" not in values:
                    missing.append(pid)
                else:
                    rss_kib += values["rss"]
                    pss_kib += values["pss"]
            except (OSError, ValueError, IndexError):
                missing.append(pid)
        gpu = self.gpu_snapshot()
        rss_mib = mib(rss_kib)
        pss_mib = mib(pss_kib)
        row: dict[str, Any] = {
            "timestampUtc": utc_now(),
            "elapsedSeconds": round(time.monotonic() - self.started, 3),
            "hostProcessTree": {
                "rootPid": self.browser.pid,
                "pids": pids,
                "processCount": len(pids),
                "rssMiB": rss_mib,
                "pssMiB": pss_mib,
                "incrementalPssMiB": (None if name == "clean_baseline" or self.baseline_pss_mib is None
                                       else round(pss_mib - self.baseline_pss_mib, 3)),
                "smapsRollupMissingPids": missing,
                "scope": "Chrome root PID and its current /proc parent-child descendants only.",
            },
            "webgpuRequestedBuffers": gpu,
            "javascriptHeap": self.js_heap_snapshot(),
        }
        if self.args.nvidia_smi:
            row["nvidiaSmi"] = self.nvidia_memory(pids)
        self.report["stages"][name]["samples"].append(row)
        with self.sample_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "sample": row}, separators=(",", ":")) + "\n")
            stream.flush()
        if name == "clean_baseline" and self.baseline_pss_mib is None:
            self.baseline_pss_mib = row["hostProcessTree"]["pssMiB"]

    def gpu_snapshot(self) -> dict[str, Any]:
        rows = []
        for session, record in list(self.worker_sessions.items()):
            try:
                result = self.protocol("Runtime.evaluate", {
                    "expression": "globalThis.__voiceStudyGpuBufferStats || {installed:false, installationPending:false, instrumentationStatus:'unavailable', reason:'Instrumentation state not found.'}",
                    "returnByValue": True,
                    "awaitPromise": False,
                }, session, timeout=4)
                value = result.get("result", {}).get("value", {})
                record["lastGpuSnapshot"] = value
                record["instrumentation"] = value.get("instrumentationStatus", record.get("instrumentation"))
                record["installationPending"] = bool(value.get("installationPending"))
                record["concreteDeviceInstrumentation"] = bool(value.get("deviceCreateBufferHookInstalled"))
                record["bufferInstrumentationActive"] = bool(value.get("installed"))
                rows.append({"targetId": record.get("targetId"), "type": record.get("type"), **value})
            except Exception as error:
                rows.append({"targetId": record.get("targetId"), "type": record.get("type"),
                             "installed": False, "installationPending": False,
                             "instrumentationStatus": "unavailable", "readError": str(error)})
        for record in self.gpu_detached:
            value = record.get("lastGpuSnapshot") or {
                "installed": False,
                "reason": "Worker detached before a final instrumentation snapshot was available.",
            }
            rows.append({"targetId": record.get("targetId"), "type": record.get("type"),
                         "detached": True, **value})
        current = sum(int(row.get("notDestroyedRequestedBytes", 0) or 0) for row in rows)
        created = sum(int(row.get("createdRequestedBytes", 0) or 0) for row in rows)
        destroyed = sum(int(row.get("destroyedRequestedBytes", 0) or 0) for row in rows)
        peak = max((int(row.get("peakNotDestroyedRequestedBytes", 0) or 0) for row in rows), default=0)
        creates = sum(int(row.get("createCalls", 0) or 0) for row in rows)
        destroys = sum(int(row.get("destroyCalls", 0) or 0) for row in rows)
        return {
            "instrumentedWorkerCount": sum(bool(row.get("installed")) for row in rows),
            "concreteDeviceInstrumentationWorkerCount": sum(bool(row.get("deviceCreateBufferHookInstalled")) for row in rows),
            "installationPendingWorkerCount": sum(bool(row.get("installationPending")) for row in rows),
            "workerCount": len(rows),
            "createCalls": creates,
            "destroyCalls": destroys,
            "currentNotDestroyedRequestedBytes": current,
            "currentNotDestroyedRequestedMiB": round(current / 1048576, 3),
            "createdRequestedBytes": created,
            "destroyedRequestedBytes": destroyed,
            "largestWorkerLifetimePeakRequestedBytes": peak,
            "instrumentationStatuses": [row.get("instrumentationStatus", "unavailable") for row in rows],
            "workerAdapterDiagnostics": [
                {"targetId": row.get("targetId"), "type": row.get("type"),
                 "adapters": row.get("adapterDiagnostics", []),
                 "deviceFeatures": row.get("deviceFeatures", []),
                 "hookFailures": row.get("hookFailures", [])}
                for row in rows if row.get("adapterDiagnostics") or row.get("deviceFeatures") or row.get("hookFailures")
            ],
            "workers": rows,
            "measurement": "Requested JavaScript GPUBuffer descriptor sizes. Not physical VRAM or total device residency.",
        }

    def js_heap_snapshot(self) -> dict[str, Any]:
        targets = [{'session': self.page_session, 'type': 'page'}]
        targets.extend({'session': session, 'type': record.get('type'), 'targetId': record.get('targetId')}
                       for session, record in list(self.worker_sessions.items()))
        rows = []
        for target in targets:
            row = {key: value for key, value in target.items() if key != 'session'}
            try:
                usage = self.protocol('Runtime.getHeapUsage', {}, target['session'], timeout=4)
                if not isinstance(usage.get('usedSize'), (float, int)) or usage['usedSize'] < 0:
                    raise RunnerError('Runtime.getHeapUsage did not return a valid usedSize.')
                row.update(usage); row['available'] = True
            except Exception as error:
                row.update(available=False, error=str(error))
            rows.append(row)
        complete = all(row['available'] for row in rows)
        return {
            'targets': rows, 'allObservedTargetsAvailable': complete,
            'observedTargetUsedBytes': sum(row['usedSize'] for row in rows) if complete else None,
            'scope': 'CDP V8 heap for the page and currently attached worker targets only. Missing targets remain unknown.',
            'limitations': 'Not browser RSS/PSS, WASM linear memory, physical GPU memory, or total ArrayBuffer residency. Optional backingStorageSize/embedderHeapUsedSize are reported separately when returned; do not sum them into RSS.',
        }

    def nvidia_memory(self, owned_pids: list[int]) -> dict[str, Any]:
        now = time.monotonic()
        if now - self.last_nvidia_sample < max(1.0, self.args.sample_seconds):
            return {"sampled": False, "reason": "sample interval"}
        self.last_nvidia_sample = now
        binary = shutil.which("nvidia-smi")
        if not binary:
            return {"available": False, "reason": "nvidia-smi not found"}
        try:
            result = subprocess.run(
                [binary, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=False,
            )
            if result.returncode != 0:
                return {"available": True, "querySucceeded": False, "error": result.stderr.strip()[-1000:]}
            rows = []
            owned = set(owned_pids)
            for line in result.stdout.splitlines():
                fields = [part.strip() for part in line.split(",", 1)]
                if len(fields) == 2:
                    try:
                        pid, memory = int(fields[0]), int(fields[1])
                    except ValueError:
                        continue
                    if pid in owned:
                        rows.append({"pid": pid, "usedMiB": memory})
            return {
                "available": True,
                "querySucceeded": True,
                "ownedComputeProcessRows": rows,
                "coverage": "nvidia-smi compute-app query only; graphics/Vulkan allocations may not appear.",
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"available": True, "querySucceeded": False, "error": str(error)}

    def evaluate(self, expression: str, *, await_promise: bool = False, user_gesture: bool = False,
                 session: str | None = None, timeout: float = 15.0) -> Any:
        result = self.protocol("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
            "userGesture": user_gesture,
        }, session or self.page_session, timeout=timeout)
        details = result.get("exceptionDetails")
        if details:
            exception = details.get("exception", {}).get("description") or details.get("text")
            raise RunnerError(f"Page JavaScript failed: {exception}")
        remote = result.get("result", {})
        if remote.get("type") == "undefined":
            return None
        return remote.get("value")

    def wait_js(self, expression: str, *, timeout: float, interval: float | None = None,
                stage: str, success: Any = None) -> Any:
        end = min(time.monotonic() + timeout, self.deadline)
        pace = interval or self.args.sample_seconds
        while True:
            self.check_timeout(stage, end)
            value = self.evaluate(expression)
            if success is None:
                if value:
                    return value
            else:
                result = success(value)
                if result:
                    return value
            if time.monotonic() + pace >= end:
                self.check_timeout(stage, end)
            self.wait_sample(pace, stage)

    def run_api_call(self, name: str, stage: str, args_js: str, timeout: float) -> dict[str, Any]:
        expression = f"""(() => {{
          const api = window.voiceStudy;
          if (!api || typeof api[{json.dumps(name)}] !== 'function') throw new Error('window.voiceStudy.{name} is unavailable.');
          const state = {{ done: false, error: null, value: null }};
          window.__browserMeasurementRun = state;
          try {{
            const pending = api[{json.dumps(name)}]({args_js});
            Promise.resolve(pending).then(value => {{ state.value = value; state.done = true; }}).catch(error => {{
              state.error = error && error.message ? error.message : String(error); state.done = true;
            }});
          }} catch (error) {{ state.error = error && error.message ? error.message : String(error); state.done = true; }}
          return true;
        }})()"""
        self.evaluate(expression, user_gesture=True)
        poll = """(() => ({
          run: window.__browserMeasurementRun || null,
          snapshot: window.voiceStudy && typeof window.voiceStudy.snapshot === 'function' ? window.voiceStudy.snapshot() : null
        }))()"""
        value = self.wait_js(
            poll,
            timeout=timeout,
            stage=stage,
            success=lambda row: isinstance(row, dict) and isinstance(row.get("run"), dict) and row["run"].get("done"),
        )
        if value["run"].get("error"):
            raise RunnerError(f"voiceStudy.{name} failed: {value['run']['error']}")
        return {"result": value["run"].get("value"), "snapshot": value.get("snapshot")}

    def begin_recording_stage(self, name: str, start: Any, complete: Any, timeout: float) -> dict[str, Any]:
        self.start_stage(name)
        stage_deadline = min(time.monotonic() + timeout, self.deadline)
        try:
            start()
            poll = """(() => ({
              run: window.__browserMeasurementRun || null,
              snapshot: window.voiceStudy && typeof window.voiceStudy.snapshot === 'function' ? window.voiceStudy.snapshot() : null
            }))()"""
            interval = self.args.sample_seconds
            while True:
                self.check_timeout(name, stage_deadline)
                value = self.evaluate(poll)
                if complete(value):
                    self.finish_stage(name)
                    return value
                if time.monotonic() + interval >= stage_deadline:
                    self.check_timeout(name, stage_deadline)
                self.wait_sample(interval, name)
        except Exception as error:
            self.finish_stage(name, status="failed", error=str(error))
            raise

    def start_read_text(self, text: str, seed: int) -> None:
        text_js = json.dumps(text, ensure_ascii=True)
        options_js = json.dumps({"seed": seed, "chunkWords": self.args.chunk_words}, separators=(",", ":"))
        expression = f"""(() => {{
          const api = window.voiceStudy;
          if (!api || typeof api.readText !== 'function') throw new Error('window.voiceStudy.readText is unavailable.');
          const state = {{ done:false, error:null, value:null }};
          window.__browserMeasurementRun = state;
          try {{ Promise.resolve(api.readText({text_js}, {options_js})).then(value => {{ state.value=value; state.done=true; }}).catch(error => {{
            state.error=error && error.message ? error.message : String(error); state.done=true;
          }}); }} catch(error) {{ state.error=error && error.message ? error.message : String(error); state.done=true; }}
          return true;
        }})()"""
        self.evaluate(expression, user_gesture=True)

    def read_text(self, name: str, text: str, seed: int, timeout: float) -> dict[str, Any]:

        def start() -> None:
            self.start_read_text(text, seed)

        def complete(value: Any) -> bool:
            run = value.get("run") if isinstance(value, dict) else None
            snap = value.get("snapshot") if isinstance(value, dict) else None
            if not isinstance(run, dict) or not run.get("done"):
                return False
            if run.get("error"):
                raise RunnerError(f"voiceStudy.readText ({name}) failed: {run['error']}")
            if not isinstance(snap, dict) or snap.get("busy"):
                raise RunnerError("readText resolved while the reader was still busy.")
            if snap.get('readingOutcome') != 'complete':
                raise RunnerError('The reader did not report complete playback.')
            rows = snap.get("chunks")
            if not isinstance(rows, list) or not rows:
                raise RunnerError("readText returned without any completed audio chunks.")
            token_counts = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                tokens = row.get("speechTokens", 0)
                token_counts.append(len(tokens) if isinstance(tokens, list) else int(tokens or 0))
            if not any(count > 0 for count in token_counts):
                raise RunnerError("No generated speech token count was reported.")
            assert_complete_reading(snap, text)
            return True

        result = self.begin_recording_stage(name, start, complete, timeout)
        run_result = {"result": result.get("run", {}).get("value"), "snapshot": result.get("snapshot")}
        stage = self.report["stages"][name]
        metrics = (run_result.get("snapshot") or {}).get("metrics")
        stage["readerMetrics"] = metrics
        stage["readerMetricsScope"] = "Synthesis metrics exclude audio playback. Stage wall time includes synthesis, storage, and scheduled playback completion."
        write_json(self.report_path, self.report)
        return run_result

    def stop_restart_probe(self, text: str, seed: int, timeout: float) -> dict[str, Any]:
        probe_text = make_stop_probe_text(text, self.args.chunk_words)
        passage_poll = """(() => ({
          run: window.__browserMeasurementRun || null,
          snapshot: window.voiceStudy && typeof window.voiceStudy.snapshot === 'function' ? window.voiceStudy.snapshot() : null,
          passageLabel: document.getElementById('reading-progress-label')?.textContent || ''
        }))()"""
        click_stop = """(() => {
          const api = window.voiceStudy;
          const button = document.getElementById('stop-button');
          const snapshot = api && typeof api.snapshot === 'function' ? api.snapshot() : null;
          if (!snapshot?.busy || !Array.isArray(snapshot.chunks) || snapshot.chunks.length < 1 || !button || button.disabled) {
            return {clicked:false, snapshot, passageLabel:document.getElementById('reading-progress-label')?.textContent || ''};
          }
          const passageLabel = document.getElementById('reading-progress-label')?.textContent || '';
          button.click();
          return {clicked:true, snapshot, passageLabel, buttonDisabledAfterClick:button.disabled};
        })()"""
        self.start_stage('stop_restart')
        stage_deadline = min(time.monotonic() + timeout, self.deadline)
        stage_started = time.monotonic()
        stop_clicked_at = None
        click_snapshot = None
        passage_label = ''
        try:
            self.check_timeout('stop_restart', stage_deadline)
            self.start_read_text(probe_text, seed)
            interval = self.args.sample_seconds
            while stop_clicked_at is None:
                self.check_timeout('stop_restart', stage_deadline)
                current = self.evaluate(passage_poll)
                run = current.get('run') if isinstance(current, dict) else None
                if isinstance(run, dict) and run.get('done'):
                    if run.get('error'):
                        raise RunnerError(f"Stop probe read failed before Stop was clicked: {run['error']}")
                    raise RunnerError('The stop probe completed before the runner could click Stop.')
                snap = current.get('snapshot') if isinstance(current, dict) else None
                if isinstance(snap, dict) and snap.get('busy') and snap.get('chunks'):
                    result = self.evaluate(click_stop, user_gesture=True)
                    if isinstance(result, dict) and result.get('clicked'):
                        stop_clicked_at = time.monotonic()
                        click_snapshot = result.get('snapshot')
                        passage_label = result.get('passageLabel') or ''
                        match = re.search(r'\bof\s+(\d+)\b', passage_label)
                        observed_passages = int(match.group(1)) if match else None
                        stage = self.report['stages']['stop_restart']
                        stage['stopRestartProbe'] = {
                            'probeText': probe_text,
                            'probeInputWords': len(probe_text.split()),
                            'minimumPassages': 4,
                            'observedPassageCount': observed_passages,
                            'chunksBeforeStop': len((click_snapshot or {}).get('chunks', [])),
                            'stopButtonClicked': True,
                            'stopButtonDisabledAfterClick': result.get('buttonDisabledAfterClick'),
                            'stopRequestElapsedSeconds': round(stop_clicked_at - stage_started, 3),
                            'stopRequestedSnapshot': click_snapshot,
                        }
                        write_json(self.report_path, self.report)
                        break
                if time.monotonic() + interval >= stage_deadline:
                    self.check_timeout('stop_restart', stage_deadline)
                self.wait_sample(interval, 'stop_restart')

            passage_match = re.search(r'\bof\s+(\d+)\b', passage_label)
            passage_count = int(passage_match.group(1)) if passage_match else None
            stop_wait = """(() => ({
              run: window.__browserMeasurementRun || null,
              snapshot: window.voiceStudy && typeof window.voiceStudy.snapshot === 'function' ? window.voiceStudy.snapshot() : null
            }))()"""
            interval = self.args.sample_seconds
            stopped_row = None
            while stopped_row is None:
                self.check_timeout('stop_restart', stage_deadline)
                current = self.evaluate(stop_wait)
                run = current.get('run') if isinstance(current, dict) else None
                snap = current.get('snapshot') if isinstance(current, dict) else None
                if isinstance(run, dict) and run.get('done'):
                    if run.get('error'):
                        raise RunnerError(f"Stop probe read failed after Stop was clicked: {run['error']}")
                    stopped_row = run.get('value')
                    if not isinstance(stopped_row, dict):
                        raise RunnerError('The stopped read did not return a snapshot.')
                    assert_stopped_reading(stopped_row)
                    break
                if time.monotonic() + interval >= stage_deadline:
                    self.check_timeout('stop_restart', stage_deadline)
                self.wait_sample(interval, 'stop_restart')

            if passage_count is None or passage_count < 4:
                raise RunnerError(f'The stop probe started {passage_count!r} passages; it must start at least four.')
            if len((click_snapshot or {}).get('chunks', [])) < 1:
                raise RunnerError('The Stop control was clicked before the first completed audio chunk.')
            click_elapsed = round(stop_clicked_at - stage_started, 3)
            stop_ack_seconds = round(time.monotonic() - stop_clicked_at, 3)
            button_disabled_after_click = (result.get('buttonDisabledAfterClick') is True)
            if not button_disabled_after_click:
                raise RunnerError('The Stop control did not enter its disabled stopping state.')
            detail = {
                'probeText': probe_text,
                'probeInputWords': len(probe_text.split()),
                'minimumPassages': 4,
                'observedPassageCount': passage_count,
                'chunksBeforeStop': len((click_snapshot or {}).get('chunks', [])),
                'stopButtonClicked': True,
                'stopButtonDisabledAfterClick': button_disabled_after_click,
                'stopRequestElapsedSeconds': click_elapsed,
                'stopAcknowledgementSeconds': stop_ack_seconds,
                'stopRequestedSnapshot': click_snapshot,
                'stoppedSnapshot': stopped_row,
                'assertions': {
                    'stoppedOutcome': stopped_row.get('readingOutcome') == 'stopped',
                    'notBusy': stopped_row.get('busy') is False,
                    'modelRemainsReady': stopped_row.get('modelReady') is True,
                    'scheduledQueueEmpty': stopped_row.get('metrics', {}).get('scheduledQueueSize') == 0,
                },
            }
            stage = self.report['stages']['stop_restart']
            stage['stopRestartProbe'] = detail
            stage['readerMetrics'] = stopped_row.get('metrics')
            stage['readerMetricsScope'] = 'A real model read was stopped after its first completed audio chunk. Stop acknowledgement includes the active model call and scheduled-playback cancellation.'
            self.finish_stage('stop_restart')
            write_json(self.report_path, self.report)
            return detail
        except Exception as error:
            if self.stage_name == 'stop_restart':
                self.finish_stage('stop_restart', status='failed', error=str(error))
            raise

    def export_wav(self, stage_name: str, artifact_name: str, path: Path,
                   deadline: float | None = None) -> dict[str, Any]:
        self.start_stage(stage_name)
        stage_deadline = min(deadline, self.deadline) if deadline is not None else self.deadline
        opened = False
        try:
            info = self.evaluate('window.voiceStudy.openWavExport()')
            opened = True
            if not isinstance(info, dict) or not isinstance(info.get('bytes'), int) or not 44 < info['bytes'] <= 0xffffffff + 8:
                raise RunnerError('Invalid streamed WAV export metadata.')
            digest = hashlib.sha256(); written = 0; max_chunk = 0
            with path.open('wb') as output:
                while True:
                    self.check_timeout(stage_name, stage_deadline)
                    read_timeout = max(0.05, min(30.0, stage_deadline - time.monotonic()))
                    row = self.evaluate('window.voiceStudy.readWavExport()', await_promise=True, timeout=read_timeout)
                    if not isinstance(row, dict) or row.get('offsetBytes') != written:
                        raise RunnerError('Streamed WAV offset is missing, duplicated, or out of order.')
                    if row.get('done') is True:
                        opened = False; break
                    contents = base64.b64decode(row.get('base64', ''), validate=True)
                    if not 0 < len(contents) <= 65536 or row.get('byteLength') != len(contents) or written + len(contents) > info['bytes']:
                        raise RunnerError('Invalid or oversized streamed WAV chunk.')
                    output.write(contents); digest.update(contents); written += len(contents)
                    max_chunk = max(max_chunk, len(contents))
                    self.sample(stage_name)
            if written != info['bytes']:
                raise RunnerError('Streamed WAV ended before all declared bytes were exported.')
            metadata = inspect_wav_file(path)
            metadata.update({
                "path": str(path),
                "sha256": digest.hexdigest(),
                "hashAlgorithm": "sha256",
                'exportMode': 'bounded-base64-chunks-to-disk',
                'maximumTransferChunkBytes': max_chunk,
                'exportedPassages': info.get('passages'),
            })
            self.report["artifacts"][artifact_name] = metadata
            self.finish_stage(stage_name)
            return metadata
        except Exception as error:
            if self.stage_name == stage_name:
                self.finish_stage(stage_name, status="failed", error=str(error))
            raise
        finally:
            if opened:
                try:
                    cancel_timeout = max(0.05, min(5.0, stage_deadline - time.monotonic()))
                    self.evaluate('window.voiceStudy.cancelWavExport()', await_promise=True, timeout=cancel_timeout)
                except Exception: pass

    def capture_screenshot(self, path: Path, viewport: tuple[int, int] | None = None) -> dict[str, Any]:
        self.evaluate("document.fonts && document.fonts.ready ? document.fonts.ready.then(() => true) : Promise.resolve(true)", await_promise=True, timeout=20)
        self.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve(true))))", await_promise=True, timeout=10)
        if viewport:
            width, height = viewport
        else:
            metrics = self.protocol("Page.getLayoutMetrics", {}, self.page_session)
            content = metrics.get("cssContentSize", {})
            width = max(320, min(1800, int(content.get("width", 1200))))
            height = max(700, min(12000, int(content.get("height", 1200))))
        result = self.protocol("Page.captureScreenshot", {
            "format": "png",
            "captureBeyondViewport": True,
            "fromSurface": True,
            "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": 1},
        }, self.page_session, timeout=30)
        image_bytes = base64.b64decode(result["data"], validate=True)
        path.write_bytes(image_bytes)
        return {"path": str(path), "bytes": len(image_bytes), "width": width, "height": height}


def read_devtools_version(profile: Path, process: subprocess.Popen[bytes], timeout: float) -> tuple[int, str, dict[str, Any]]:
    active_file = profile / "DevToolsActivePort"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"Chrome exited during startup with code {process.returncode}.")
        try:
            lines = active_file.read_text(encoding="utf-8").splitlines()
            port = int(lines[0])
            path = lines[1]
            request_timeout = max(0.1, min(3.0, deadline - time.monotonic()))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=request_timeout) as response:
                version = json.load(response)
            return port, path, version
        except (OSError, ValueError, IndexError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.1)
    raise RunnerError("Chrome DevTools did not become ready before startup timeout.")


def chrome_arguments(chrome: Path, profile: Path, headless: bool, ui_only: bool = False,
                     hardware_webgpu: bool = False, adapter_only: bool = False) -> list[str]:
    headless_shell = chrome.name == "chrome-headless-shell"
    args = [
        str(chrome),
        f"--user-data-dir={profile}",
        "--remote-debugging-port=0",
        "--remote-debugging-address=127.0.0.1",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-component-update",
    ]
    if headless:
        args.append("--headless" if headless_shell else "--headless=new")
    if ui_only:
        # Reduce process count for the guarded static UI check. Never use this
        # layout for inference or for reported model-performance measurements.
        args.extend(("--no-zygote", "--no-sandbox", "--renderer-process-limit=1",
                     "--disable-gpu", "--disable-software-rasterizer"))
    elif adapter_only:
        args.extend(("--no-zygote", "--no-sandbox", "--renderer-process-limit=1"))
    if hardware_webgpu:
        # Chrome's Linux headless WebGPU recipe uses Vulkan. Keep these flags
        # opt-in so the ordinary browser measurement retains its normal setup.
        args.extend(("--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface", "--enable-unsafe-webgpu"))
    args.append("about:blank")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4187/", help="Existing Vite server URL; port 4187 is required.")
    parser.add_argument("--chrome", default="google-chrome", help="Chrome executable name or path.")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON, WAV, screenshot, and browser log files.")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="Hard wall-clock limit for the entire browser run.")
    parser.add_argument("--startup-timeout-seconds", type=float, default=20)
    parser.add_argument("--load-timeout-seconds", type=float, default=900)
    parser.add_argument("--inference-timeout-seconds", type=float, default=300)
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Short text used for both matched synthesis passes.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--embedding-manifest", help="Optional explicit Q4 embedding manifest URL for this full measurement.")
    parser.add_argument('--lossless-embedding-manifest', help='Optional independently hash-pinned FP16 shard manifest; no embedding ONNX session.')
    parser.add_argument('--chunk-words', type=int, choices=range(15, 25), default=18)
    parser.add_argument('--full-paper', action='store_true', help='After matched short first/warm readings, read the entire checked-in Federalist No. 10 and stream its WAV to disk.')
    parser.add_argument('--full-reading-timeout-seconds', type=float, default=1800)
    parser.add_argument('--check-stop-restart', action='store_true', help='Run a real-model multi-passage stop followed by a completed restart and WAV export.')
    parser.add_argument('--trace-inference', action='store_true', help='Record bounded per-passage tokenizer IDs, ONNX input metadata, and raw logit samples for diagnostic analysis.')
    parser.add_argument("--voice-state-manifest", help="Explicit same-origin candidate voice-state manifest; default selected voice stays unchanged.")
    parser.add_argument("--model-base", help="Optional same-origin model asset base URL, for example /models/chatterbox-nano-browser/.")
    parser.add_argument("--nvidia-smi", action="store_true", help="Sample nvidia-smi compute-app memory separately when available.")
    parser.add_argument("--headed", action="store_true", help="Use a visible Chrome window instead of headless Chrome.")
    parser.add_argument("--hardware-webgpu", action="store_true", help="Use Chrome's Linux Vulkan WebGPU flags and require a non-fallback adapter with shader-f16.")
    parser.add_argument('--require-hardware-webgpu', action='store_true', help='Require an identified non-fallback FP16 adapter without changing GPU backend or unsafe-WebGPU flags.')
    parser.add_argument('--power-preference', choices=('low-power', 'high-performance'), default='low-power', help='WebGPU adapter preference. Reported identity decides which GPU was actually selected.')
    parser.add_argument("--ui-only", action="store_true", help="Capture rendered UI and check the missing-state retry path. Do not load models or synthesize speech.")
    parser.add_argument("--adapter-only", action="store_true", help="Check the WebGPU adapter without preparing a model or synthesizing speech. This component check is not an inference benchmark.")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.startup_timeout_seconds <= 0 or args.load_timeout_seconds <= 0 or args.inference_timeout_seconds <= 0:
        parser.error("Timeout values must be greater than zero.")
    if args.idle_seconds < 0 or args.sample_seconds <= 0:
        parser.error("Idle time must be non-negative and sample interval must be greater than zero.")
    if not args.text.strip() and not args.ui_only:
        parser.error("Synthesis text must contain at least one non-space character.")
    if args.ui_only and args.embedding_manifest:
        parser.error("--embedding-manifest applies only to full model measurements, not --ui-only.")
    if args.ui_only and args.model_base:
        parser.error("--model-base applies only to full model measurements, not --ui-only.")
    if args.ui_only and args.hardware_webgpu:
        parser.error("--hardware-webgpu applies only to full model measurements, not --ui-only.")
    if args.ui_only and args.require_hardware_webgpu:
        parser.error('--require-hardware-webgpu does not apply to UI-only checks.')
    if args.check_stop_restart and (args.ui_only or args.adapter_only):
        parser.error('--check-stop-restart requires a full model measurement.')
    if args.voice_state_manifest and (args.ui_only or args.adapter_only):
        parser.error('--voice-state-manifest requires a full model measurement.')
    if args.trace_inference and (args.ui_only or args.adapter_only):
        parser.error('--trace-inference requires a full model measurement.')
    if args.trace_inference and args.full_paper:
        parser.error('--trace-inference is limited to short diagnostic runs, not --full-paper.')
    if args.adapter_only and (args.ui_only or args.embedding_manifest or args.model_base):
        parser.error('--adapter-only cannot be combined with UI, embedding, or model-loading options.')
    if args.embedding_manifest and args.lossless_embedding_manifest:
        parser.error('Select either Q4 or lossless streamed embeddings.')
    if (args.ui_only or args.adapter_only) and (args.lossless_embedding_manifest or args.full_paper):
        parser.error('Full-paper and lossless embeddings require a full measurement.')
    if args.full_reading_timeout_seconds <= 0:
        parser.error('Full reading timeout must be greater than zero.')
    try:
        args.resolved_model_base = normalize_local_model_base(args.model_base, args.url)
    except ValueError as error:
        parser.error(str(error))
    return args


def app_snapshot_js() -> str:
    return "window.voiceStudy && typeof window.voiceStudy.snapshot === 'function' ? window.voiceStudy.snapshot() : null"


def main() -> int:
    args = parse_args()
    run_started = time.monotonic()
    global_deadline = run_started + args.timeout_seconds
    out = output_path(args)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "startedAtUtc": utc_now(),
        "status": "starting",
        "mode": "ui-only" if args.ui_only else "adapter-only" if args.adapter_only else "browser-tts-measurement",
        "limitations": [
            "Host RSS and PSS sum /proc/smaps_rollup for only Chrome's owned profile process tree.",
            "The WebGPU counters record requested JavaScript GPUBuffer descriptor sizes. They do not measure physical VRAM residency or every backend allocation.",
            "Host PSS and GPU metrics are separate measures. Do not add them as an exact resident-memory total.",
            "This runner does not verify spoken words or perceptual voice quality.",
        ],
        "configuration": {
            "url": args.url,
            "outputDir": str(out),
            "timeoutSeconds": args.timeout_seconds,
            "startupTimeoutSeconds": args.startup_timeout_seconds,
            "loadTimeoutSeconds": args.load_timeout_seconds,
            "inferenceTimeoutSeconds": args.inference_timeout_seconds,
            "idleSeconds": args.idle_seconds,
            "sampleSeconds": args.sample_seconds,
            "text": args.text,
            "seed": args.seed,
            "embeddingManifestUrl": args.embedding_manifest,
            'losslessEmbeddingManifestUrl': args.lossless_embedding_manifest,
            'chunkWords': args.chunk_words,
            'fullPaper': args.full_paper,
            'checkStopRestart': args.check_stop_restart,
            'traceInference': args.trace_inference,
            "modelBaseUrl": args.model_base,
            "resolvedModelBaseUrl": args.resolved_model_base,
            "hardwareWebgpuDiagnostic": args.hardware_webgpu,
            'requireHardwareWebgpu': args.require_hardware_webgpu,
            "nvidiaSmiEnabled": args.nvidia_smi,
            "uiOnly": args.ui_only,
        },
        "guard": guard_context(),
        "server": None,
        "browser": None,
        "app": {},
        "stages": {},
        "workers": [],
        "networkRequests": [],
        "artifacts": {},
        "error": None,
    }
    browser: subprocess.Popen[bytes] | None = None
    ownership: BrowserOwnership | None = None
    devtools: DevTools | None = None
    measurement: BrowserMeasurement | None = None
    temp_profile: tempfile.TemporaryDirectory[str] | None = None
    chrome_log = out / "chrome.stderr.log"
    log_handle = None
    signal_handlers = {}
    cleaning_up = False
    sample_log_path = out / "measurement-samples.jsonl"
    report_path = out / "measurement.json"

    def interrupted(signum: int, frame: Any) -> None:
        if not cleaning_up:
            raise SignalStop(f"Received signal {signum}.")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal_handlers[signum] = signal.signal(signum, interrupted)

    try:
        sample_log_path.write_text("", encoding="utf-8")
        report["reportPath"] = str(report_path)
        report["artifacts"]["sampleLog"] = str(sample_log_path)
        write_json(report_path, report)
        if not report["guard"]["accepted"]:
            raise RunnerError("Refusing to launch Chrome outside bounded_job.py. Use the documented guarded command.")
        report["server"] = check_local_server(args.url, bounded_timeout(global_deadline, 4.0))
        chrome_name = shutil.which(args.chrome)
        if not chrome_name:
            raise RunnerError(f"Chrome executable not found: {args.chrome}")
        chrome_path = Path(chrome_name).resolve()
        headless_shell = chrome_path.name == "chrome-headless-shell"
        if headless_shell and not (args.ui_only or args.adapter_only):
            raise RunnerError("chrome-headless-shell supports only UI or adapter component checks. Full measurements require normal Chrome.")
        if headless_shell and args.headed:
            raise RunnerError("chrome-headless-shell cannot run a headed UI check.")
        temp_profile = tempfile.TemporaryDirectory(prefix="voice-study-chrome-", dir=out)
        profile = Path(temp_profile.name)
        report["browser"] = {
            "executable": str(chrome_path),
            "implementation": "chrome-headless-shell" if headless_shell else "Chrome",
            "headlessImplementation": "old headless shell" if headless_shell else ("Chrome headless=new" if not args.headed else "headed Chrome"),
            "cleanTemporaryProfile": True,
            "headless": not args.headed,
            "flags": chrome_arguments(chrome_path, profile, not args.headed, args.ui_only,
                                       args.hardware_webgpu, args.adapter_only)[1:-1],
            "hardwareWebgpuDiagnostic": args.hardware_webgpu,
            "hardwareWebgpuRequestedFlags": (["--use-angle=vulkan", "--enable-features=Vulkan",
                                                "--disable-vulkan-surface", "--enable-unsafe-webgpu"]
                                               if args.hardware_webgpu else []),
            "processLayout": ("one-renderer adapter-only component diagnostic; no model inference" if args.adapter_only
                              else "headless shell, one-renderer UI-only diagnostic with GPU disabled" if headless_shell
                              else ("one-renderer UI-only diagnostic with GPU disabled" if args.ui_only
                                    else "normal Chrome process layout")),
            "profilePath": str(profile),
            "processTreeScope": "Chrome root process and current /proc descendants; never system-wide Chrome processes.",
        }
        log_handle = chrome_log.open("wb")
        command = chrome_arguments(chrome_path, profile, not args.headed, args.ui_only,
                                   args.hardware_webgpu, args.adapter_only)
        browser = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=log_handle, start_new_session=True)
        ownership = BrowserOwnership(browser.pid, chrome_path, profile)
        startup_timeout = bounded_timeout(global_deadline, args.startup_timeout_seconds)
        port, devtools_path, version = read_devtools_version(profile, browser, startup_timeout)
        report["browser"].update({
            "pid": browser.pid,
            "devtoolsPort": port,
            "version": version,
            "devtoolsPath": devtools_path,
        })
        devtools = DevTools(version["webSocketDebuggerUrl"], timeout=bounded_timeout(global_deadline, startup_timeout))
        def startup_protocol(method: str, params: dict[str, Any] | None = None,
                             session: str | None = None, timeout: float = 15.0) -> dict[str, Any]:
            return devtools.call(method, params, session, timeout=bounded_timeout(global_deadline, timeout))

        targets = startup_protocol("Target.getTargets").get("targetInfos", [])
        target = next((item for item in targets if item.get("type") == "page" and item.get("url") == "about:blank"), None)
        if not target:
            raise RunnerError("Chrome did not expose its startup about:blank page.")
        report["browser"]["reusedStartupPage"] = True
        attached = startup_protocol("Target.attachToTarget", {"targetId": target["targetId"], "flatten": True})
        page_session = attached["sessionId"]
        startup_protocol("Page.enable", {}, page_session)
        startup_protocol("Runtime.enable", {}, page_session)
        startup_protocol("Network.enable", {}, page_session)
        startup_protocol("Emulation.setDeviceMetricsOverride", {
            "width": 1280, "height": 1000, "deviceScaleFactor": 1, "mobile": False,
        }, page_session)

        measurement = BrowserMeasurement(args, report, browser, chrome_path, profile, devtools, page_session,
                                         sample_log_path, report_path, run_started, global_deadline)
        measurement.start_stage("clean_baseline")
        baseline_end = time.monotonic() + max(2.0, min(5.0, args.sample_seconds * 3))
        while time.monotonic() < baseline_end:
            measurement.check_timeout("clean_baseline", baseline_end)
            measurement.wait_sample(min(args.sample_seconds, max(0.05, baseline_end - time.monotonic())), "clean_baseline")
        measurement.finish_stage("clean_baseline")
        baseline_samples = report["stages"]["clean_baseline"]["samples"]
        report["memoryBaseline"] = {
            "cleanBrowserPssMiB": round(sum(row["hostProcessTree"]["pssMiB"] for row in baseline_samples) / len(baseline_samples), 3),
            "cleanBrowserRssMiB": round(sum(row["hostProcessTree"]["rssMiB"] for row in baseline_samples) / len(baseline_samples), 3),
            "definition": "Mean measured Chrome process-tree PSS/RSS during a clean about:blank profile before app navigation.",
        }
        measurement.baseline_pss_mib = report["memoryBaseline"]["cleanBrowserPssMiB"]
        write_json(report_path, report)

        measurement.start_stage("page")
        nav = measurement.protocol("Page.navigate", {"url": args.url}, page_session)
        if nav.get("errorText"):
            raise RunnerError(f"App navigation failed: {nav['errorText']}")
        page_wait = min(args.startup_timeout_seconds, max(10.0, args.timeout_seconds / 4))
        page_value = measurement.wait_js(
            """(() => ({ ready: document.readyState, hasApi: !!window.voiceStudy,
              text: document.getElementById('passage')?.value || '', status: window.voiceStudy?.snapshot?.().status || null }))()""",
            timeout=page_wait,
            stage="page",
            success=lambda row: isinstance(row, dict) and row.get("ready") == "complete" and row.get("hasApi") and bool(row.get("text", "").strip()),
        )
        measurement.finish_stage("page")
        report["app"].update({
            "readyState": page_value.get("ready"),
            "voiceStudyApiPresent": page_value.get("hasApi"),
            "textLoaded": bool(page_value.get("text", "").strip()),
            "loadedTextCharacters": len(page_value.get("text", "")),
            "initialStatus": page_value.get("status"),
        })

        if args.ui_only:
            before = measurement.evaluate("""(() => {
              const visible = (id) => { const item=document.getElementById(id); const style=item && getComputedStyle(item);
                return !!item && style.visibility!=='hidden' && !!(item.offsetWidth||item.offsetHeight||item.getClientRects().length); };
              return { snapshot:window.voiceStudy.snapshot(),
                readDisabled:document.getElementById('read-button').disabled,
                prepareDisabled:document.getElementById('load-model').disabled,
                saveAudioVisible:visible('download-reading'), nowReadingVisible:visible('now-reading') };
            })()""")
            if not isinstance(before, dict) or before.get("snapshot", {}).get("modelReady"):
                raise RunnerError("UI-only check expected a not-ready voice before prepare.")
            if not before.get("readDisabled"):
                raise RunnerError("Read control was enabled before modelReady.")
            report["app"]["uiBeforePrepare"] = before

            # Observe worker fetches without pausing or injecting worker code.
            measurement.protocol("Target.setAutoAttach", {
                "autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True,
            }, page_session)

            missing_manifest_url = "/voice/__measurement_missing_state__.json"
            manifest_response = measurement.evaluate(f"""(async () => {{
              const response = await fetch({json.dumps(missing_manifest_url)}, {{cache:'no-store'}});
              const body = await response.text();
              const contentType = response.headers.get('content-type') || '';
              const trimmed = body.trimStart();
              let bodyKind = 'text';
              if (!trimmed) bodyKind = 'empty';
              else {{
                try {{ JSON.parse(body); bodyKind = 'json'; }}
                catch (_) {{ if (/html/i.test(contentType) || /^<!doctype\\s+html|^<html/i.test(trimmed)) bodyKind = 'html'; }}
              }}
              return {{requestedUrl:{json.dumps(missing_manifest_url)}, responseUrl:response.url,
                status:response.status, ok:response.ok, contentType, bodyKind,
                bodyBytes:new TextEncoder().encode(body).length, bodyPreview:body.slice(0,240)}};
            }})()""", await_promise=True, timeout=15)
            report["app"]["simulatedMissingStateProbe"] = {
                "setup": "The runner requests a unique same-origin manifest path and records the actual HTTP response. It then passes that path to prepareVoice. The app validates the manifest before model assets. This is a controlled invalid-path probe, not a claim that the installed voice-state asset is missing.",
                "requestedManifestUrl": missing_manifest_url,
                "manifestResponse": manifest_response,
            }

            def start_ui_probe() -> None:
                expression = """(() => {
                  const state = {done:false,error:null,value:null};
                  window.__browserMeasurementRun = state;
                  try { Promise.resolve(window.voiceStudy.prepareVoice({voiceStateManifestUrl:'/voice/__measurement_missing_state__.json'})).then(value => { state.value=value; state.done=true; }).catch(error => {
                    state.error=error && error.message ? error.message : String(error); state.done=true;
                  }); } catch(error) { state.error=error && error.message ? error.message : String(error); state.done=true; }
                  return true;
                })()"""
                measurement.evaluate(expression, user_gesture=True)

            def probe_complete(row: Any) -> bool:
                run = row.get("run") if isinstance(row, dict) else None
                return isinstance(run, dict) and bool(run.get("done"))

            probe = measurement.begin_recording_stage("missing_state_probe", start_ui_probe, probe_complete,
                                                      min(args.load_timeout_seconds, 30.0))
            error = probe.get("run", {}).get("error")
            snap = probe.get("snapshot") or measurement.evaluate(app_snapshot_js())
            status_text = measurement.evaluate("document.getElementById('model-status-text')?.textContent || ''")
            button = measurement.evaluate("""(() => { const item=document.getElementById('load-model'); return {
              disabled:item.disabled, text:item.innerText.trim(), visible:!!(item.offsetWidth||item.offsetHeight||item.getClientRects().length)
            }; })()""")
            controls_after = measurement.evaluate("""(() => {
              const visible = (id) => { const item=document.getElementById(id); const style=item && getComputedStyle(item);
                return !!item && style.visibility!=='hidden' && !!(item.offsetWidth||item.offsetHeight||item.getClientRects().length); };
              return { saveAudioVisible:visible('download-reading'), nowReadingVisible:visible('now-reading'),
                loadProgressVisible:visible('load-progress-wrap') };
            })()""")
            model_network_requests = [row for row in report["networkRequests"] if "huggingface.co" in row.get("url", "")]
            runtime_asset_requests = [row for row in report["networkRequests"]
                                      if "/onnx/" in row.get("url", "") or "/tokenizer" in row.get("url", "")]
            checks = {
                "prepareRejected": bool(error),
                "errorMentionsFixedVoiceState": bool(re.search(r"fixed voice state|voice-state|asmr-state|state manifest", str(error or ""), re.I)),
                "snapshotNotReady": isinstance(snap, dict) and not snap.get("modelReady"),
                "snapshotErrorSet": isinstance(snap, dict) and snap.get("status") == "error" and bool(snap.get("error")),
                "statusTextMentionsFailure": bool(re.search(r"state|fail|could not|missing|voice", str(status_text), re.I)),
                "retryControlEnabled": isinstance(button, dict) and not button.get("disabled") and bool(re.search(r"retry|prepare", button.get("text", ""), re.I)),
                "readControlDisabled": bool(measurement.evaluate("document.getElementById('read-button').disabled")),
                "saveAudioHiddenBeforePrepare": not before.get("saveAudioVisible"),
                "nowReadingHiddenBeforePrepare": not before.get("nowReadingVisible"),
                "saveAudioHiddenAfterFailure": not controls_after.get("saveAudioVisible"),
                "nowReadingHiddenAfterFailure": not controls_after.get("nowReadingVisible"),
                "loadProgressHiddenAfterFailure": not controls_after.get("loadProgressVisible"),
                "missingManifestResponseRecorded": isinstance(manifest_response, dict)
                    and manifest_response.get("requestedUrl") == missing_manifest_url
                    and isinstance(manifest_response.get("status"), int)
                    and bool(manifest_response.get("bodyKind")),
                "noHuggingFaceRequestsObserved": not model_network_requests,
                "noRuntimeAssetRequestsObserved": not runtime_asset_requests,
            }
            report["app"].update({
                "simulatedMissingStateProbe": {**report["app"]["simulatedMissingStateProbe"],
                                                "error": error, "snapshot": snap, "visibleStatusText": status_text,
                                                "prepareButton": button, "controlsAfterFailure": controls_after,
                                                "checks": checks},
                "uiOnlyNoHuggingFaceRequestsObserved": checks["noHuggingFaceRequestsObserved"],
            })
            measurement.start_stage("rendered_ui")
            screenshot = measurement.capture_screenshot(out / "browser-ui.png")
            measurement.finish_stage("rendered_ui")
            report["artifacts"]["screenshot"] = screenshot
            measurement.start_stage("rendered_ui_mobile")
            measurement.protocol("Emulation.setDeviceMetricsOverride", {
                "width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True,
                "screenWidth": 390, "screenHeight": 844,
            }, page_session)
            mobile_width = measurement.evaluate("document.documentElement.scrollWidth")
            mobile_screenshot = measurement.capture_screenshot(out / "browser-ui-mobile-390x844.png")
            mobile_screenshot.update({"viewportWidth": 390, "viewportHeight": 844,
                                      "documentScrollWidth": mobile_width})
            measurement.finish_stage("rendered_ui_mobile")
            report["artifacts"]["mobileScreenshot"] = mobile_screenshot
            checks["mobileContentFits390px"] = isinstance(mobile_width, (int, float)) and mobile_width <= 390
            report["app"]["simulatedMissingStateProbe"]["checks"] = checks
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                raise RunnerError("UI-only acceptance checks failed: " + ", ".join(failed_checks))
            report["status"] = "complete"
        else:
            measurement.protocol("Target.setAutoAttach", {
                "autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True,
            }, page_session)
            adapter = measurement.evaluate("""(async () => {
              if (!navigator.gpu) return { available:false, reason:'navigator.gpu is absent' };
              const item = await navigator.gpu.requestAdapter({powerPreference:POWER_PREFERENCE});
              if (!item) return { available:false, reason:'No WebGPU adapter was returned' };
              let info = item.info || {};
              if ((!info.vendor && !info.architecture) && item.requestAdapterInfo) {
                try { info = await item.requestAdapterInfo(); } catch (_) {}
              }
              const infoFields = { vendor:info.vendor || null, architecture:info.architecture || null,
                device:info.device || null, description:info.description || null };
              const label = Object.values(infoFields).filter(Boolean).join(' ').toLowerCase();
              const fallbackStatus = typeof info.isFallbackAdapter === 'boolean' ? info.isFallbackAdapter
                : (typeof item.isFallbackAdapter === 'boolean' ? item.isFallbackAdapter : null);
              const fallbackStatusSource = typeof info.isFallbackAdapter === 'boolean' ? 'GPUAdapterInfo.isFallbackAdapter'
                : (typeof item.isFallbackAdapter === 'boolean' ? 'GPUAdapter.isFallbackAdapter' : 'unknown');
              const softwareMarkers = ['swiftshader', 'llvmpipe', 'lavapipe', 'softpipe', 'software renderer', 'software adapter'];
              const features = [...item.features].map(String);
              const hardwareAssertions = {
                hasAdapterIdentity:Object.values(infoFields).some(Boolean),
                notFallback: fallbackStatus === false,
                noKnownSoftwareLabel: !softwareMarkers.some(marker => label.includes(marker)),
                noGoogleSwiftShader: !(String(infoFields.vendor || '').toLowerCase() === 'google' && label.includes('swiftshader')),
                shaderF16: features.includes('shader-f16'),
              };
              return { available:true, isFallbackAdapter:fallbackStatus, fallbackStatusSource,
                info:infoFields, features, hardwareAssertions,
                hardwareAssertionsPassed:Object.values(hardwareAssertions).every(Boolean),
                limits:{maxBufferSize:item.limits.maxBufferSize,
                  maxStorageBufferBindingSize:item.limits.maxStorageBufferBindingSize} };
            })()""".replace('POWER_PREFERENCE', json.dumps(args.power_preference)), await_promise=True, timeout=20)
            if not isinstance(adapter, dict) or not adapter.get("available"):
                raise RunnerError(f"WebGPU adapter check failed: {adapter}")
            report["app"]["webgpuAdapter"] = adapter
            if args.hardware_webgpu or args.require_hardware_webgpu:
                report["app"]["hardwareWebgpuPreflight"] = {
                    "adapterAssertions": adapter.get("hardwareAssertions", {}),
                    "passed": bool(adapter.get("hardwareAssertionsPassed")),
                    "requirements": ["adapter identity must be available",
                                     "isFallbackAdapter must be false",
                                     "adapter info must not identify a known software renderer",
                                     "adapter must expose shader-f16"],
                }
                write_json(report_path, report)
                if not adapter.get("hardwareAssertionsPassed"):
                    raise RunnerError("Hardware verification requires a non-fallback adapter without a known software-renderer label and with shader-f16.")

            if args.adapter_only:
                report['app']['adapterOnly'] = True
                report['app']['modelPreparationCalled'] = False
                report['status'] = 'complete'
                print(json.dumps({'status':'complete','mode':'adapter-only','report':str(report_path),
                                  'adapter':adapter}, indent=2), flush=True)
                return 0

            expected_words = len(re.findall(r"\S+", args.text.strip()))
            report["synthesis"] = {"text": args.text, "expectedInputWords": expected_words, "seed": args.seed,
                                   "first": None, "warm": None, "matchedTextAndSeed": True}
            prepare_call = "window.voiceStudy.prepareVoice()"
            prepare_options = {'powerPreference': args.power_preference}
            if args.voice_state_manifest:
                prepare_options['voiceStateManifestUrl'] = args.voice_state_manifest
                report['app']['voiceStateManifestUrl'] = args.voice_state_manifest
            if args.trace_inference:
                prepare_options['traceInference'] = True
            if args.embedding_manifest:
                prepare_options["embeddingManifestUrl"] = args.embedding_manifest
                report["app"]["embeddingManifestUrl"] = args.embedding_manifest
            if args.lossless_embedding_manifest:
                prepare_options['losslessEmbeddingManifestUrl'] = args.lossless_embedding_manifest
            if args.model_base:
                prepare_options["modelBaseUrl"] = args.model_base
                report["app"]["modelBaseUrl"] = args.model_base
                report["app"]["resolvedModelBaseUrl"] = args.resolved_model_base
            if prepare_options:
                options = json.dumps(prepare_options, ensure_ascii=True)
                prepare_call = f"window.voiceStudy.prepareVoice({options})"
            report["app"]["embeddingSelection"] = ('lossless FP16 shards' if args.lossless_embedding_manifest
                else 'explicit Q4 manifest' if args.embedding_manifest else 'pinned default embed_tokens_fp16')
            report["app"]["modelAssetSelection"] = "explicit same-origin model base" if args.model_base else "pinned default model base"
            load_start = """(() => {
              const state = {done:false,error:null,value:null}; window.__browserMeasurementRun=state;
              try { Promise.resolve(PREPARE_CALL).then(value => { state.value=value; state.done=true; }).catch(error => {
                state.error=error && error.message ? error.message : String(error); state.done=true;
              }); } catch(error) { state.error=error && error.message ? error.message : String(error); state.done=true; }
              return true;
            })()""".replace("PREPARE_CALL", prepare_call)

            # The saved voice path and an optional Q4 embedding candidate are
            # selected before loading. Keep the model-load stage explicit.
            measurement.start_stage("load")
            stage_deadline = min(time.monotonic() + args.load_timeout_seconds, measurement.deadline)
            measurement.evaluate(load_start, user_gesture=True)
            load_value = None
            while True:
                measurement.check_timeout("load", stage_deadline)
                load_value = measurement.evaluate("""(() => ({run:window.__browserMeasurementRun,
                  snapshot:window.voiceStudy.snapshot()}))()""")
                run = load_value.get("run", {}) if isinstance(load_value, dict) else {}
                snap = load_value.get("snapshot", {}) if isinstance(load_value, dict) else {}
                if run.get("done"):
                    if run.get("error"):
                        raise RunnerError(f"voiceStudy.prepareVoice failed: {run['error']}")
                    if not snap.get("modelReady"):
                        raise RunnerError("prepareVoice resolved before modelReady became true.")
                    break
                measurement.wait_sample(args.sample_seconds, "load")
            measurement.finish_stage("load")
            report["app"]["loadedSnapshot"] = load_value.get("snapshot")
            if args.hardware_webgpu or args.require_hardware_webgpu:
                gpu_load_snapshot = measurement.gpu_snapshot()
                report["app"]["gpuBufferInstrumentationAfterLoad"] = gpu_load_snapshot
                worker_adapters = [adapter_row
                                   for worker_row in gpu_load_snapshot.get("workers", [])
                                   if worker_row.get("type") == "worker"
                                   for adapter_row in worker_row.get("adapterDiagnostics", [])]
                worker_hardware_passed = bool(worker_adapters) and all(
                    adapter_row.get("hardwareAssertionsPassed")
                    and all(adapter_row.get("hardwareAssertions", {}).values())
                    for adapter_row in worker_adapters)
                report["app"]["hardwareWebgpuWorkerCheck"] = {
                    "passed": worker_hardware_passed,
                    "workerAdapterCount": len(worker_adapters),
                    "workerAdapters": worker_adapters,
                    "instrumentationStatuses": gpu_load_snapshot.get("instrumentationStatuses", []),
                    "requirements": ["the model worker must expose adapter identity",
                                     "every observed worker adapter must be non-fallback and not a known software renderer",
                                     "every observed worker adapter must expose shader-f16"],
                }
                write_json(report_path, report)
                if not worker_hardware_passed:
                    raise RunnerError("--hardware-webgpu could not verify a worker adapter with hardware identity and shader-f16.")
            report['artifacts']['readyScreenshot'] = measurement.capture_screenshot(out / 'voice-ready.png')
            write_json(report_path, report)

            measurement.start_stage("idle")
            idle_end = min(time.monotonic() + args.idle_seconds, measurement.deadline)
            while time.monotonic() < idle_end:
                measurement.check_timeout("idle", idle_end)
                measurement.wait_sample(min(args.sample_seconds, max(0.05, idle_end - time.monotonic())), "idle")
            measurement.finish_stage("idle")

            first = measurement.read_text("first", args.text, args.seed, args.inference_timeout_seconds)
            report["synthesis"]["first"] = first
            report["artifacts"]["firstWav"] = measurement.export_wav(
                "first_artifact_export", "firstWav", out / "first-reading.wav")
            warm = measurement.read_text("warm", args.text, args.seed, args.inference_timeout_seconds)
            report["synthesis"]["warm"] = warm
            report["app"]["finalSnapshot"] = warm.get("snapshot")
            report["artifacts"]["warmWav"] = measurement.export_wav(
                "warm_artifact_export", "warmWav", out / "warm-reading.wav")
            report['artifacts']['completedScreenshot'] = measurement.capture_screenshot(out / 'reading-completed.png')
            first_hash = report["artifacts"]["firstWav"]["sha256"]
            warm_hash = report["artifacts"]["warmWav"]["sha256"]
            report["audioComparison"] = {
                "sameText": True,
                "sameSeed": True,
                "firstWarmWavByteIdentical": first_hash == warm_hash,
                "firstSha256": first_hash,
                "warmSha256": warm_hash,
                "interpretation": "Hash equality establishes byte identity only. A difference can result from runtime nondeterminism and requires listening and waveform checks.",
            }
            report['latencyScopes'] = {
                'coldLoadSeconds': report['stages']['load'].get('durationSeconds'),
                'firstRequest': first.get('snapshot', {}).get('metrics'),
                'warmRequest': warm.get('snapshot', {}).get('metrics'),
                'note': 'Cold means a fresh Chrome profile; OS/HTTP-server caches are not flushed. First/warm TTFA excludes model preparation. Playback and export are separate stages.',
            }
            if args.check_stop_restart:
                stop_restart_deadline = min(time.monotonic() + args.inference_timeout_seconds,
                                            measurement.deadline)
                report['synthesis']['stopRestart'] = measurement.stop_restart_probe(
                    args.text, args.seed, max(0.05, stop_restart_deadline - time.monotonic()))
                restart_timeout = min(stop_restart_deadline - time.monotonic(),
                                      measurement.deadline - time.monotonic())
                if restart_timeout <= 0:
                    raise RunnerError('The combined stop/restart inference timeout expired before the restart read.')
                restarted = measurement.read_text('restart', args.text, args.seed, restart_timeout)
                report['synthesis']['restart'] = restarted
                report['app']['finalSnapshot'] = restarted.get('snapshot')
                report['artifacts']['restartWav'] = measurement.export_wav(
                    'restart_artifact_export', 'restartWav', out / 'restart-reading.wav',
                    deadline=stop_restart_deadline)
                report['audioComparison']['restartWavByteIdenticalToFirst'] = (
                    report['artifacts']['restartWav']['sha256'] == first_hash)
                write_json(report_path, report)
            if args.full_paper:
                full_path = ROOT / 'browser_tts/public/federalist-no-10.txt'
                full_text = full_path.read_text(encoding='utf-8').strip()
                report['fullReading'] = {
                    'inputPath': str(full_path), 'inputSha256': hashlib.sha256(full_path.read_bytes()).hexdigest(),
                    'inputWords': len(full_text.split()),
                    'reading': measurement.read_text('full_reading', full_text, args.seed, args.full_reading_timeout_seconds),
                }
                report['app']['finalSnapshot'] = report['fullReading']['reading'].get('snapshot')
                report['artifacts']['fullWav'] = measurement.export_wav('full_artifact_export', 'fullWav', out / 'full-reading.wav')
            write_json(report_path, report)
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "atUtc": utc_now()}
        if isinstance(error, KeyboardInterrupt):
            report["error"]["message"] = "Interrupted by user."
        if isinstance(error, SignalStop):
            report["status"] = "interrupted"
    finally:
        cleaning_up = True
        if measurement and measurement.stage_name:
            name = measurement.stage_name
            detail = (report.get("error") or {}).get("message", "Runner stopped during this stage.")
            try:
                measurement.finish_stage(name, status="failed", error=detail)
            except Exception as stage_error:
                stage = report.get("stages", {}).get(name, {})
                stage["status"] = "failed"
                stage["error"] = detail
                stage["endedAtUtc"] = utc_now()
                stage["summaryError"] = str(stage_error)
        if devtools:
            try:
                devtools.call("Browser.close", {}, timeout=3)
            except Exception:
                pass
            devtools.close()
        terminated_pids = []
        if browser and ownership:
            try:
                terminated_pids = ownership.terminate(browser)
            except Exception as error:
                report.setdefault("cleanup", {})["error"] = str(error)
        report.setdefault("cleanup", {}).update({
            "terminatedOwnedPids": terminated_pids,
            "browserExitCode": browser.returncode if browser else None,
        })
        if log_handle:
            try:
                log_handle.close()
            except OSError:
                pass
        profile_removed = temp_profile is None
        if temp_profile:
            try:
                temp_profile.cleanup()
            except OSError as error:
                report.setdefault("cleanup", {})["profileRemovalError"] = str(error)
            profile_removed = not Path(temp_profile.name).exists()
        if report.get("browser"):
            report["browser"]["profileRemovedAfterRun"] = profile_removed
        report.setdefault("cleanup", {})["profileRemoved"] = profile_removed
        report["endedAtUtc"] = utc_now()
        report["elapsedSeconds"] = round(time.monotonic() - run_started, 3)
        report["reportPath"] = str(out / "measurement.json")
        try:
            write_json(out / "measurement.json", report)
        except Exception as error:
            sys.stderr.write(f"Could not write measurement JSON: {error}\n")
        for signum, handler in signal_handlers.items():
            signal.signal(signum, handler)

    print(json.dumps({"status": report.get("status"), "report": str(out / "measurement.json"),
                      "error": report.get("error"), "artifacts": report.get("artifacts", {})}, indent=2))
    return 0 if report.get("status") == "complete" else 1


def inspect_wav(data: bytes) -> dict[str, Any]:
    return inspect_wav_header(data[:44], len(data))


def inspect_wav_file(path: Path) -> dict[str, Any]:
    with path.open('rb') as source:
        header = source.read(44)
    return inspect_wav_header(header, path.stat().st_size)


def inspect_wav_header(data: bytes, file_size: int) -> dict[str, Any]:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RunnerError("The exported audio is not a RIFF/WAVE file.")
    channels = struct.unpack_from("<H", data, 22)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    pcm_bytes = struct.unpack_from("<I", data, 40)[0]
    frame_size = channels * bits // 8
    if channels != 1 or sample_rate != 24000 or bits != 16 or pcm_bytes <= 0 or file_size != 44 + pcm_bytes or pcm_bytes % 2:
        raise RunnerError("The exported WAV header has invalid or incomplete PCM data.")
    if data[12:16] != b'fmt ' or data[36:40] != b'data' or struct.unpack_from('<H', data, 20)[0] != 1 or struct.unpack_from('<I', data, 4)[0] != file_size - 8:
        raise RunnerError('The exported WAV format/RIFF length is invalid.')
    return {
        "bytes": file_size,
        "channels": channels,
        "sampleRateHz": sample_rate,
        "bitsPerSample": bits,
        "audioSeconds": round(pcm_bytes / frame_size / sample_rate, 3),
        "validPcmPayload": True,
    }


def make_stop_probe_text(text: str, chunk_words: int, minimum_passages: int = 4) -> str:
    if not isinstance(chunk_words, int) or chunk_words <= 0 or minimum_passages <= 0:
        raise RunnerError('The stop/restart passage limits must be positive integers.')
    words = text.strip().split()[:chunk_words]
    if not words:
        raise RunnerError('The stop/restart probe requires non-empty synthesis text.')
    unit = ' '.join(words)
    target_words = chunk_words * minimum_passages
    copies = max(1, (target_words + len(words) - 1) // len(words))
    return ' '.join([unit] * copies)


def assert_stopped_reading(snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict):
        raise RunnerError('The stop probe did not return a reader snapshot.')
    if snapshot.get('readingOutcome') != 'stopped':
        raise RunnerError('The real-model read did not report a stopped outcome.')
    if snapshot.get('busy') is not False:
        raise RunnerError('The reader remained busy after Stop completed.')
    if snapshot.get('modelReady') is not True:
        raise RunnerError('The model was not ready after Stop completed.')
    if snapshot.get('status') != 'ready':
        raise RunnerError('The reader did not return to ready status after Stop.')
    metrics = snapshot.get('metrics')
    if not isinstance(metrics, dict) or metrics.get('scheduledQueueSize') != 0:
        raise RunnerError('Scheduled audio remained after Stop completed.')
    chunks = snapshot.get('chunks')
    if not isinstance(chunks, list) or not chunks:
        raise RunnerError('The stop probe stopped before any audio chunk was recorded.')
    for index, row in enumerate(chunks):
        tokens = row.get('speechTokens') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get('index') != index or row.get('truncated') is not False
                or not tokens or row.get('audioSeconds', 0) <= 0):
            raise RunnerError('The stopped read contains a missing, truncated, or invalid passage.')


def assert_complete_reading(snapshot: dict[str, Any], text: str) -> None:
    rows = snapshot.get('chunks', [])
    joined = ' '.join(row.get('text', '') for row in rows)
    if joined.split() != text.split():
        raise RunnerError('Reading input passages omit, repeat, or reorder text. This checks submitted text, not spoken words.')
    for index, row in enumerate(rows):
        if row.get('index') != index or row.get('truncated') is not False or not row.get('speechTokens') or row.get('audioSeconds', 0) <= 0:
            raise RunnerError('The reading contains a missing, truncated, or invalid passage.')
    if snapshot.get('metrics', {}).get('maximumScheduledQueueSize', 3) > 2:
        raise RunnerError('The playback queue exceeded two passages.')


if __name__ == "__main__":
    sys.exit(main())
