import { ProxyOptions, defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, process.cwd());
  const backendUrl = env.VITE_QUEUE_API_URL || "http://127.0.0.1:8080";
  const proxyConf:Record<string, string | ProxyOptions> = {
    "/api": {
      target: backendUrl,
      changeOrigin: true,
      ws: true,
    },
    "/mcp": {
      target: backendUrl,
      changeOrigin: true,
    },
  };
  return {
    server: {
      host: "0.0.0.0",
      https: {
        cert: "./cert.pem",
        key: "./key.pem",
      },
      proxy:{
        ...proxyConf,
      }
    },
    plugins: [
      topLevelAwait({
        // The export name of top-level await promise for each chunk module
        promiseExportName: "__tla",
        // The function to generate import names of top-level await promise in each chunk module
        promiseImportName: i => `__tla_${i}`,
      }),
    ],
  };
});
