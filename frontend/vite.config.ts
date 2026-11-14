import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the app on :5173 and forwards /api to FastAPI on :8000,
// so the browser sees one origin and the API needs no extra CORS setup.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
