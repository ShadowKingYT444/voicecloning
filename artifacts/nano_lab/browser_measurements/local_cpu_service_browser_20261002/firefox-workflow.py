import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/nano_lab/browser_measurements/local_cpu_service_browser_20261002'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('runner', ROOT / 'browser_tts/scripts/measure-browser.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
server_log = (OUT / 'server.log').open('w')
server = subprocess.Popen([str(ROOT / '.venv-nano-cpu/bin/python'), str(ROOT / 'browser_tts/scripts/local-reader-server.py')], env=env, stdout=server_log, stderr=server_log)
firefox = None
protocol = None
try:
    deadline = time.monotonic() + 20
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:4187/api/reader/status', timeout=1) as response:
                status = json.load(response)
            if status.get('schema') == 'voice-study.local-reader/v1': break
        except OSError: pass
        if server.poll() is not None or time.monotonic() > deadline: raise RuntimeError('CPU service did not start')
        time.sleep(0.2)
    # Firefox uses an isolated profile and its documented WebDriver BiDi endpoint.
    with tempfile.TemporaryDirectory(prefix='voice-cpu-firefox-', dir=OUT) as profile:
        firefox_log = (OUT / 'firefox.log').open('w')
        firefox = subprocess.Popen(['/snap/firefox/current/usr/lib/firefox/firefox', '--headless', '--no-remote', '--profile', profile, '--remote-debugging-port', '0', 'about:blank'], env=env, stdout=firefox_log, stderr=firefox_log)
        deadline = time.monotonic() + 30
        while True:
            match = re.search(r'WebDriver BiDi listening on (ws://127\.0\.0\.1:\d+)', (OUT / 'firefox.log').read_text())
            if match: break
            if firefox.poll() is not None or time.monotonic() > deadline: raise RuntimeError('Firefox did not start')
            time.sleep(0.2)
        protocol = runner.DevTools(match.group(1) + '/session')
        session = protocol.call('session.new', {'capabilities': {}})
        context = protocol.call('browsingContext.create', {'type': 'tab'})['context']
        protocol.call('browsingContext.navigate', {'context': context, 'url': 'http://127.0.0.1:4187/', 'wait': 'complete'}, timeout=30)
        def evaluate(expression, timeout=20):
            result = protocol.call('script.evaluate', {'expression': expression, 'target': {'context': context}, 'awaitPromise': True, 'resultOwnership': 'none', 'userActivation': True}, timeout=timeout)
            if result.get('type') != 'success': raise RuntimeError(str(result))
            return json.loads(result['result']['value'])
        deadline = time.monotonic() + 15
        while not evaluate('JSON.stringify(!!window.voiceStudy)'):
            if time.monotonic() > deadline: raise RuntimeError('Firefox app did not load')
            time.sleep(0.2)
        prepared = evaluate('window.voiceStudy.prepareVoice().then(v=>JSON.stringify(v))')
        if prepared['modelIdentity']['provider'] != 'CPUExecutionProvider': raise RuntimeError('Firefox did not select CPU')
        first = evaluate('window.voiceStudy.readText("Take a slow breath in, and let your shoulders relax.",{seed:1337}).then(v=>JSON.stringify(v))', 200)
        wav = evaluate('window.voiceStudy.exportWavBase64().then(v=>JSON.stringify(v))')
        (OUT / 'firefox-reading.wav').write_bytes(base64.b64decode(wav))
        screenshot = protocol.call('browsingContext.captureScreenshot', {'context': context, 'origin': 'document', 'format': {'type': 'image/png'}}, timeout=20)
        (OUT / 'firefox-completed.png').write_bytes(base64.b64decode(screenshot['data']))
        facts = evaluate('JSON.stringify({userAgent:navigator.userAgent,gpu:!!navigator.gpu,jspi:typeof WebAssembly.Suspending=== "function",execution:document.getElementById("runtime-execution").textContent,ram:document.getElementById("metric-cpu-rss").textContent})')
        with urllib.request.urlopen('http://127.0.0.1:4187/api/reader/status') as response: service_status = json.load(response)
        (OUT / 'firefox-result.json').write_text(json.dumps({'session': session, 'browser': facts, 'prepared': prepared, 'first': first, 'serviceStatus': service_status}, indent=2))
        protocol.call('session.end'); protocol.close(); protocol = None
        firefox.terminate(); firefox.wait(timeout=15); firefox = None
        firefox_log.close()
    print('Firefox CPU reader check completed.', flush=True)
finally:
    if protocol:
        try: protocol.call('browser.close', timeout=5)
        except Exception: pass
        protocol.close()
    if firefox and firefox.poll() is None:
        firefox.terminate()
        try: firefox.wait(timeout=15)
        except subprocess.TimeoutExpired: firefox.kill(); firefox.wait()
    server.terminate()
    try: server.wait(timeout=30)
    except subprocess.TimeoutExpired: server.kill(); server.wait()
    server_log.close()
