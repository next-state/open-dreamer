import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Article } from "@/components/Article";

// Read at build time (static export prerenders this once): the ~700-line
// Distill article is kept as verbatim HTML rather than rewritten to JSX. The
// client component injects it, loads the Distill/D3/KaTeX runtime, and portals
// the live demo into the placeholder inside it.
export default function Page() {
  const html = readFileSync(
    join(process.cwd(), "article", "article.html"),
    "utf8"
  );
  return <Article html={html} />;
}
