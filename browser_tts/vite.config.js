import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 4187,
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4187,
    strictPort: true,
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        wasmComponent: fileURLToPath(new URL('./wasm-component.html', import.meta.url)),
      },
    },
  },
});
