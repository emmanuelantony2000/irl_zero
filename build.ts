import { marked } from "marked";
import { mkdirSync, cpSync } from "fs";

export async function build(): Promise<number> {
  const slidesRaw = await Bun.file("slides.md").text();
  const slideChunks = slidesRaw.split("\n---\n");

  const sections = slideChunks.map((chunk) => {
    let attrs = "";
    let content = chunk.trim();

    // Parse <!-- .slide: class="..." --> comments
    const slideAttrMatch = content.match(/<!--\s*\.slide:\s*(.*?)\s*-->/);
    if (slideAttrMatch) {
      attrs = ` ${slideAttrMatch[1]}`;
      content = content.replace(slideAttrMatch[0], "").trim();
    }

    const html = marked.parse(content, { async: false }) as string;
    return `        <section${attrs}>\n${html}\n        </section>`;
  });

  const fullHtml = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI on Chip and Diffusion Models</title>
    <link rel="stylesheet" href="reveal.css" />
    <link rel="stylesheet" href="theme/black.css" />
    <link rel="stylesheet" href="plugin/highlight/monokai.css" />
    <style>
      :root {
        --accent: #00d4ff;
        --accent2: #ff6ec7;
        --bg-dark: #0a0a1a;
      }

      .reveal {
        font-family: "Inter", "SF Pro Display", system-ui, sans-serif;
      }

      .reveal h1,
      .reveal h2,
      .reveal h3 {
        text-transform: none;
        font-weight: 700;
        letter-spacing: -0.02em;
      }

      .reveal h1 {
        font-size: 2.2em;
      }

      .reveal h2 {
        font-size: 1.6em;
        color: var(--accent);
      }

      .reveal h3 {
        font-size: 1.2em;
        color: var(--accent2);
      }

      .reveal .slides section {
        text-align: left;
        padding: 20px 40px;
        box-sizing: border-box;
        width: calc(100%) !important;
      }

      .reveal .title-slide {
        text-align: center;
      }

      .reveal .section-divider {
        text-align: center;
      }

      .reveal .section-divider h2 {
        font-size: 2.2em;
        color: #fff;
      }

      .reveal .section-divider .subtitle {
        font-size: 1.1em;
        color: #888;
        margin-top: 0.5em;
      }

      .reveal ul {
        font-size: 0.85em;
        line-height: 1.7;
      }

      .reveal li {
        margin-bottom: 0.4em;
      }

      .reveal .highlight {
        color: var(--accent);
        font-weight: 600;
      }

      .reveal .highlight2 {
        color: var(--accent2);
        font-weight: 600;
      }

      .reveal .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
      }

      .reveal .stat-box {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
      }

      .reveal .stat-box .number {
        font-size: 2em;
        font-weight: 800;
        color: var(--accent);
        display: block;
      }

      .reveal .stat-box .label {
        font-size: 0.75em;
        color: #aaa;
        margin-top: 4px;
      }

      .reveal .comparison-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75em;
        margin-top: 15px;
      }

      .reveal .comparison-table th {
        background: rgba(0, 212, 255, 0.15);
        color: var(--accent);
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
      }

      .reveal .comparison-table td {
        padding: 8px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .reveal .comparison-table tr:hover td {
        background: rgba(255, 255, 255, 0.03);
      }

      .reveal .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        margin-top: 15px;
      }

      .reveal .col-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 20px;
      }

      .reveal .col-box.pros {
        border-left: 3px solid #00ff88;
      }

      .reveal .col-box.cons {
        border-left: 3px solid #ff4466;
      }

      .reveal .col-box h3 {
        margin-top: 0;
        font-size: 1em;
      }

      .reveal .col-box.pros h3 {
        color: #00ff88;
      }

      .reveal .col-box.cons h3 {
        color: #ff4466;
      }

      .reveal .col-box ul {
        font-size: 0.8em;
        margin: 0;
        padding-left: 1.2em;
      }

      .reveal .flow-diagram {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin: 25px 0;
        font-size: 0.8em;
        flex-wrap: wrap;
      }

      .reveal .flow-step {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        min-width: 90px;
      }

      .reveal .flow-arrow {
        color: var(--accent);
        font-size: 1.4em;
      }

      .reveal .tag {
        display: inline-block;
        background: rgba(255, 110, 199, 0.15);
        color: var(--accent2);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.7em;
        margin: 2px;
      }

      .reveal .small {
        font-size: 0.65em;
        color: #666;
      }

      .reveal .center-text {
        text-align: center;
      }

      .reveal blockquote {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid var(--accent);
        padding: 15px 20px;
        font-style: italic;
        font-size: 0.85em;
        margin: 15px 0;
      }

      .reveal .diffusion-visual {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        margin: 20px 0;
        font-size: 0.7em;
      }

      .reveal .noise-block {
        width: 60px;
        height: 60px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9em;
      }

      .reveal .token-parallel {
        display: flex;
        gap: 6px;
        justify-content: center;
        margin: 15px 0;
      }

      .reveal .token-box {
        background: rgba(0, 212, 255, 0.12);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.7em;
        text-align: center;
      }

      .reveal .token-box.active {
        background: rgba(0, 212, 255, 0.25);
        border-color: var(--accent);
        box-shadow: 0 0 12px rgba(0, 212, 255, 0.2);
      }
    </style>
  </head>
  <body>
    <div class="reveal">
      <div class="slides">
${sections.join("\n\n")}
      </div>
    </div>

    <script src="reveal.js"></script>
    <script src="plugin/highlight/highlight.js"></script>
    <script>
      Reveal.initialize({
        hash: true,
        slideNumber: true,
        transition: "slide",
        backgroundTransition: "fade",
        plugins: [RevealHighlight],
        width: 1280,
        height: 720,
      });
    </script>
  </body>
</html>`;

  mkdirSync("dist", { recursive: true });
  mkdirSync("dist/theme", { recursive: true });
  mkdirSync("dist/plugin/highlight", { recursive: true });
  cpSync("node_modules/reveal.js/dist/reveal.js", "dist/reveal.js");
  cpSync("node_modules/reveal.js/dist/reveal.css", "dist/reveal.css");
  cpSync("node_modules/reveal.js/dist/theme/black.css", "dist/theme/black.css");
  cpSync("node_modules/reveal.js/plugin/highlight/highlight.js", "dist/plugin/highlight/highlight.js");
  cpSync("node_modules/reveal.js/plugin/highlight/monokai.css", "dist/plugin/highlight/monokai.css");
  await Bun.write("dist/index.html", fullHtml);
  console.log(`Rebuilt ${sections.length} slides → dist/index.html`);
  return sections.length;
}

// Run directly when executed as a script
await build();
