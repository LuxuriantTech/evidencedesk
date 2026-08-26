import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000", changeOrigin: true }, "/health": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000" }, "/ready": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000" } } },
  test: { environment: "jsdom", globals: true, exclude: ["e2e/**", "node_modules/**"], setupFiles: ["./src/test/setup.ts"] },
});
