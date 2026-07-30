// Bundles src/ into a single self-contained dist/index.html.
//
// The deliverable is one file you can open from disk with no server: the CSS and the
// bundled JS are inlined into the markup, and nothing is fetched at runtime. Sources
// stay split for editing; only the build output is a single file.
//
//   node build.mjs                 one-shot production build (minified)
//   node build.mjs --watch --serve rebuild on change, served over HTTP
//
// Use --serve rather than opening dist/index.html directly when testing anything that
// touches localStorage or reload behaviour: a file:// page does not reload cleanly in a
// preview pane, which has already produced one false "persistence works" result here.

import * as esbuild from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const SRC = path.join(ROOT, 'src');
const OUT_DIR = path.join(ROOT, 'dist');
const OUT_FILE = path.join(OUT_DIR, 'index.html');

const watch = process.argv.includes('--watch');
const serve = process.argv.includes('--serve');
const minify = !watch;
const PORT = 8750;

// Substitution is done with split/join, never String.replace: both the CSS and the
// bundled JS contain `$` sequences that replace would read as capture-group references
// and silently mangle.
function inject(template, marker, payload) {
  if (!template.includes(marker)) throw new Error(`template is missing ${marker}`);
  return template.split(marker).join(payload);
}

async function build() {
  const result = await esbuild.build({
    entryPoints: [path.join(SRC, 'main.ts')],
    bundle: true,
    format: 'iife',
    target: 'es2020',
    minify,
    legalComments: 'none',
    write: false
  });

  const script = result.outputFiles[0].text;

  // An inline <script> ends at the first literal `</script>` regardless of context, so a
  // bundle containing one would silently truncate the document. Fail loudly instead.
  if (script.includes('</script>')) {
    throw new Error('bundle contains a literal </script>; it would truncate the inline script');
  }

  const template = await readFile(path.join(SRC, 'index.html'), 'utf8');
  const styles = await readFile(path.join(SRC, 'styles.css'), 'utf8');

  let html = inject(template, '__STYLES__', styles);
  html = inject(html, '__SCRIPT__', script);

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(OUT_FILE, html);

  const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
  console.log(`dist/index.html  ${kb} kB${minify ? '' : '  (unminified)'}`);
}

if (watch) {
  const ctx = await esbuild.context({
    entryPoints: [path.join(SRC, 'main.ts')],
    bundle: true,
    write: false,
    plugins: [{
      name: 'rebuild-html',
      setup(b) {
        b.onEnd(async (r) => {
          if (r.errors.length) return;
          try { await build(); } catch (e) { console.error(e.message); }
        });
      }
    }]
  });
  await ctx.watch();
  console.log('watching src/');
} else {
  await build();
}

if (serve) {
  createServer(async (req, res) => {
    try {
      const html = await readFile(OUT_FILE);
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(html);
    } catch {
      res.writeHead(503).end('not built yet');
    }
  }).listen(PORT, '127.0.0.1', () => console.log(`http://127.0.0.1:${PORT}/`));
}
