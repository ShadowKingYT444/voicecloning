"""Guarded static listening-page checks. Does not load a speech model."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

spec = importlib.util.spec_from_file_location('measurement', Path(__file__).with_name('measure-browser.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    args = runner.parse_args()
    if not runner.guard_context()['accepted']:
        raise RuntimeError('Run this check through bounded_job.py.')
    out = args.output_dir or Path('artifacts/nano_lab/browser_measurements/local_listening_ui_20261002')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'purpose': 'static listening-page verification', 'model_inference': False, 'audio': []}
    chrome = Path(shutil.which(args.chrome)).resolve()
    with tempfile.TemporaryDirectory(prefix='voice-listening-', dir=out) as temporary:
        profile = Path(temporary)
        with (out / 'chrome.stderr.log').open('wb') as log:
            process = subprocess.Popen(runner.chrome_arguments(chrome, profile, True, ui_only=True),
                                       stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
            ownership = runner.BrowserOwnership(process.pid, chrome, profile)
            devtools = None
            try:
                _, _, version = runner.read_devtools_version(profile, process, 20)
                devtools = runner.DevTools(version['webSocketDebuggerUrl'])
                target = next(t for t in devtools.call('Target.getTargets')['targetInfos'] if t['type'] == 'page')
                session = devtools.call('Target.attachToTarget', {'targetId': target['targetId'], 'flatten': True})['sessionId']
                measurement = runner.BrowserMeasurement(args, report, process, chrome, profile, devtools, session,
                    out / 'samples.jsonl', out / 'report.json', started, started + 90)
                measurement.protocol('Page.enable', {}, session)
                measurement.protocol('Emulation.setDeviceMetricsOverride',
                    {'width': 1280, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False}, session)
                measurement.protocol('Page.navigate',
                    {'url': 'http://127.0.0.1:4187/experiments/native-donor-listening/'}, session)
                measurement.start_stage('page')
                measurement.wait_js("document.readyState === 'complete' && document.querySelectorAll('audio').length === 4",
                                    timeout=20, stage='page', success=lambda value: value is True)
                measurement.finish_stage('page')
                report['desktop'] = measurement.capture_screenshot(out / 'desktop.png')
                for index in range(4):
                    measurement.evaluate(f"new Promise((resolve, reject) => {{ const a = document.querySelectorAll('audio')[{index}]; a.onloadedmetadata = () => resolve(a.duration); a.onerror = () => reject(new Error('Audio load failed')); a.load(); }})",
                                         await_promise=True, timeout=10)
                    result = measurement.evaluate(f"new Promise((resolve, reject) => {{ const a = document.querySelectorAll('audio')[{index}]; a.onended = () => resolve({{src:a.currentSrc, duration:a.duration, ended:a.ended}}); a.onerror = () => reject(new Error('Audio playback failed')); a.play().catch(reject); }})",
                                                  await_promise=True, user_gesture=True, timeout=10)
                    if not result['ended'] or result['duration'] <= 0:
                        raise RuntimeError('Audio did not finish playback.')
                    report['audio'].append(result)
                measurement.protocol('Emulation.setDeviceMetricsOverride',
                    {'width': 390, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True}, session)
                report['mobile'] = measurement.capture_screenshot(out / 'mobile.png')
                report['mobileNoOverflow'] = measurement.evaluate('document.documentElement.scrollWidth <= innerWidth')
                if not report['mobileNoOverflow']:
                    raise RuntimeError('Mobile page has horizontal overflow.')
                report['status'] = 'passed'
                report['limitations'] = ['Playback API verification only. No human listening or physical speaker assessment.']
            finally:
                ownership.terminate(process)
                if devtools:
                    devtools.close()
    runner.write_json(out / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
