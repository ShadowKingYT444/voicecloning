// Model-free diagnostic shared by the visible page and CPU component tests.
// No runtime import, fetch, device request, or GPU buffer allocation.
export const LIMIT_NAMES = Object.freeze([
  'maxBufferSize', 'maxStorageBufferBindingSize', 'maxUniformBufferBindingSize',
  'maxStorageBuffersPerShaderStage', 'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension', 'maxBindGroups', 'maxBindingsPerBindGroup',
]);

export async function checkCapabilities({
  scope = globalThis, powerPreference = 'low-power',
} = {}) {
  if (!['low-power', 'high-performance'].includes(powerPreference)) {
    throw new TypeError('Select low-power or high-performance.');
  }
  const nav = scope.navigator || {};
  const wasm = scope.WebAssembly || {};
  const report = {
    schema: 'voice-study.browser-capabilities/v1',
    checkedAtUtc: new Date().toISOString(),
    userAgent: nav.userAgent || null,
    powerPreference,
    secureContext: scope.isSecureContext === true,
    crossOriginIsolated: scope.crossOriginIsolated === true,
    runtime: {
      target: 'onnxruntime-web@1.30.0 / JSPI / WebGPU / FP16',
      webgpu: typeof nav.gpu?.requestAdapter === 'function',
      jspiSuspending: typeof wasm.Suspending === 'function',
      jspiPromising: typeof wasm.promising === 'function',
      audioContext: typeof (scope.AudioContext || scope.webkitAudioContext) === 'function',
      indexedDB: !!scope.indexedDB,
      filePicker: typeof scope.showSaveFilePicker === 'function',
      userAgentMemoryApi: typeof scope.performance?.measureUserAgentSpecificMemory === 'function',
    },
    adapter: null, capabilityPrerequisitesPassed: false, hardwareAdapterVerified: false,
    modelsDownloaded: false, inferenceRun: false, gpuDeviceRequested: false,
    limitations: [
      'Capability checks do not establish ONNX operator support, inference parity, memory, speed, or voice quality.',
      'Unknown fallback status or redacted identity does not establish a hardware adapter.',
      'Adapter limits are capabilities, not measurements of allocated or physical GPU memory.',
    ],
  };
  if (!report.runtime.webgpu) return report;
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference });
    if (!adapter) { report.adapterError = 'No adapter was returned.'; return report; }
    let info = adapter.info || {};
    if (!Object.values(info).some(Boolean) && typeof adapter.requestAdapterInfo === 'function') {
      info = await adapter.requestAdapterInfo();
    }
    const identity = Object.fromEntries(['vendor', 'architecture', 'device', 'description']
      .map((name) => [name, typeof info[name] === 'string' ? info[name] : null]));
    const fallback = typeof info.isFallbackAdapter === 'boolean' ? info.isFallbackAdapter
      : typeof adapter.isFallbackAdapter === 'boolean' ? adapter.isFallbackAdapter : null;
    const features = [...adapter.features].map(String).sort();
    const limits = Object.fromEntries(LIMIT_NAMES.map((name) => [name,
      Number.isFinite(adapter.limits?.[name]) ? adapter.limits[name] : null]));
    const label = Object.values(identity).filter(Boolean).join(' ').toLowerCase();
    const software = ['swiftshader', 'llvmpipe', 'lavapipe', 'softpipe', 'software renderer', 'software adapter']
      .some((marker) => label.includes(marker));
    report.adapter = { identity, isFallbackAdapter: fallback,
      fallbackStatusSource: typeof info.isFallbackAdapter === 'boolean' ? 'GPUAdapterInfo.isFallbackAdapter'
        : typeof adapter.isFallbackAdapter === 'boolean' ? 'GPUAdapter.isFallbackAdapter' : 'unknown',
      knownSoftwareLabel: software, shaderF16: features.includes('shader-f16'), features, limits };
    report.capabilityPrerequisitesPassed = report.secureContext && report.runtime.jspiSuspending
      && report.runtime.jspiPromising && report.runtime.audioContext && report.runtime.indexedDB
      && report.adapter.shaderF16;
    report.hardwareAdapterVerified = fallback === false && !software && Object.values(identity).some(Boolean);
  } catch (error) { report.adapterError = error?.message || String(error); }
  return report;
}
