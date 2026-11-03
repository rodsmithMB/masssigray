import { defineConfig, type Plugin } from "vitest/config";

/**
 * Load the bundle as a classic deferred script instead of an ES module.
 * Browsers refuse module scripts (and crossorigin assets) on file:// pages,
 * so this lets dist/index.html open straight from disk as well as from a web server.
 */
function classicScript(): Plugin {
  return {
    name: "classic-script",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler: (html) =>
        html.replace(/<script type="module" crossorigin/g, "<script defer").replace(/ crossorigin(?=[ >])/g, ""),
    },
  };
}

export default defineConfig({
  // Relative asset paths, so the built site works from any folder (e.g. rodsmithlab.ca/masssigray/)
  base: "./",
  plugins: [classicScript()],
  build: {
    outDir: "dist",
    target: "es2022",
    chunkSizeWarningLimit: 5000,
    modulePreload: false,
    rolldownOptions: { output: { format: "iife" } },
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
