import { existsSync } from "fs";
import { watch } from "fs";
import { join, extname } from "path";
import { build } from "./build.ts";

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".map": "application/json",
};

const LIVERELOAD_SCRIPT = `<script>new EventSource("/__reload").onmessage = () => location.reload();</script>`;

const PORT = 3000;

// Initial build
await build();

// SSE clients
const clients = new Set<ReadableStreamDefaultController>();

// Watch slides.md with debounce
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
watch("slides.md", () => {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    await build();
    for (const controller of clients) {
      try {
        controller.enqueue("data: reload\n\n");
      } catch {
        clients.delete(controller);
      }
    }
  }, 100);
});

Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);

    // SSE endpoint for livereload
    if (url.pathname === "/__reload") {
      const stream = new ReadableStream({
        start(controller) {
          clients.add(controller);
        },
        cancel(controller) {
          clients.delete(controller);
        },
      });
      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    const distPath = url.pathname === "/" ? "/index.html" : url.pathname;
    let path = `/dist${distPath}`;
    const filePath = join(import.meta.dir, path);

    if (!existsSync(filePath)) {
      return new Response("Not Found", { status: 404 });
    }

    const ext = extname(filePath);
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    // Inject livereload script into HTML responses
    if (ext === ".html") {
      let html = Bun.file(filePath).text();
      return html.then((content) => {
        content = content.replace("</body>", `${LIVERELOAD_SCRIPT}\n</body>`);
        return new Response(content, {
          headers: { "Content-Type": contentType },
        });
      });
    }

    return new Response(Bun.file(filePath), {
      headers: { "Content-Type": contentType },
    });
  },
});

console.log(`Dev server running at http://localhost:${PORT} (watching slides.md)`);
