import { defineConfig } from 'vite';

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
    target: 'esnext',
  },
});
