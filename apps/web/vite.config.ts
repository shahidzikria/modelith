import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
  preview: { port: 4173 },
  // Pre-bundle lazily imported deps so the dev server doesn't reload the page when they first load.
  optimizeDeps: { include: ['@monaco-editor/react', 'monaco-editor/esm/vs/editor/editor.api', 'elkjs/lib/elk.bundled.js', 'jszip'] },
  build: { chunkSizeWarningLimit: 4000 },
});
