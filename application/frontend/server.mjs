import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const host = "127.0.0.1";
const port = Number(process.env.PORT || 5173);
const root = process.cwd();

const types = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
]);

function resolvePath(url) {
  const pathname = decodeURIComponent(new URL(url, `http://${host}`).pathname);
  const clean = normalize(pathname).replace(/^(\.\.[/\\])+/, "");
  return join(root, clean === "/" ? "index.html" : clean);
}

const server = createServer(async (req, res) => {
  try {
    const filePath = resolvePath(req.url || "/");
    const body = await readFile(filePath);
    res.writeHead(200, {
      "Content-Type": types.get(extname(filePath)) || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch {
    const body = await readFile(join(root, "index.html"));
    res.writeHead(200, {"Content-Type": "text/html; charset=utf-8"});
    res.end(body);
  }
});

server.listen(port, host, () => {
  console.log(`Frontend running at http://${host}:${port}`);
});
