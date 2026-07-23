"use client";

import { useEffect, useRef } from "react";
import { createRoot, type Root } from "react-dom/client";

import { asset } from "@/lib/site-config";
import { EmbeddedDemo } from "@/components/demo/EmbeddedDemo";

declare global {
  interface Window {
    renderMathInElement?: (
      el: HTMLElement,
      options: Record<string, unknown>
    ) => void;
  }
}

// Append a <script> once, resolving when it has loaded. Guarded by id so React
// StrictMode's double-mount (Next dev) and any remount can't register Distill's
// custom elements twice (which would throw "already defined").
function loadScript(
  id: string,
  src: string,
  attrs: { integrity?: string; crossOrigin?: string } = {}
): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.getElementById(id) as HTMLScriptElement | null;
    if (existing) {
      if (existing.dataset.loaded === "true") resolve();
      else {
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () =>
          reject(new Error(`failed to load ${src}`))
        );
      }
      return;
    }
    const script = document.createElement("script");
    script.id = id;
    script.src = src;
    if (attrs.integrity) script.integrity = attrs.integrity;
    if (attrs.crossOrigin) script.crossOrigin = attrs.crossOrigin;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    });
    script.addEventListener("error", () =>
      reject(new Error(`failed to load ${src}`))
    );
    document.body.appendChild(script);
  });
}

// Loads the article runtime exactly once, in the same order as the original
// index.html: Distill template (defines the <d-*> elements and, seeing the
// article already in the DOM, runs its post-DOMContentLoaded transform), then
// D3, then KaTeX + its auto-render extension.
let runtimePromise: Promise<void> | null = null;
function ensureArticleRuntime(): Promise<void> {
  if (runtimePromise) return runtimePromise;
  runtimePromise = (async () => {
    await loadScript(
      "distill-template",
      asset("/website/scripts/distill.pub_template.v2.js")
    );
    await loadScript("d3", "https://d3js.org/d3.v7.min.js");
    await loadScript(
      "katex",
      "https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.js",
      {
        integrity:
          "sha384-GxNFqL3r9uRJQhR+47eDxuPoNE7yLftQM8LcxzgS4HT73tp970WS/wV5p8UzCOmb",
        crossOrigin: "anonymous",
      }
    );
    await loadScript(
      "katex-auto-render",
      "https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/contrib/auto-render.min.js",
      {
        integrity:
          "sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl",
        crossOrigin: "anonymous",
      }
    );
  })();
  return runtimePromise;
}

/**
 * Renders the verbatim Distill article and mounts the live demo into it.
 *
 * The article is SSR'd as raw HTML (so the post reads with JS disabled and is
 * fully indexable). The demo is mounted as an independent React root into the
 * `#live-demo` placeholder — not a portal — so the Distill runtime, which
 * mutates the surrounding article DOM, and React never contend over the same
 * subtree. Scripts embedded in `innerHTML` never execute, so the Distill/D3/
 * KaTeX runtime is loaded explicitly once the markup is in the DOM; KaTeX math
 * and the Figure 7 caption bridge (wired on DOMContentLoaded in the original
 * page) are re-created here since that event has fired by mount time.
 */
export function Article({ html }: { html: string }) {
  const demoRootRef = useRef<Root | null>(null);

  useEffect(() => {
    const target = document.getElementById("live-demo");
    if (target && demoRootRef.current === null) {
      const root = createRoot(target);
      demoRootRef.current = root;
      root.render(<EmbeddedDemo />);
    }

    let cancelled = false;
    let cleanupFigure7: (() => void) | undefined;

    void ensureArticleRuntime().then(() => {
      if (cancelled) return;

      window.renderMathInElement?.(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });

      cleanupFigure7 = wireFigure7Caption();
    });

    return () => {
      cancelled = true;
      cleanupFigure7?.();
      // Defer the unmount out of the render/commit phase (React forbids
      // unmounting a root synchronously while another root is rendering, which
      // is exactly what StrictMode's mount→cleanup→mount cycle would trigger).
      const root = demoRootRef.current;
      demoRootRef.current = null;
      if (root) setTimeout(() => root.unmount(), 0);
    };
  }, []);

  return (
    // The injected article is parsed/normalized by the browser (it contains
    // markup like <ul> nested in <p>) and then mutated by the Distill runtime,
    // so its DOM never byte-matches the raw string at hydration. This subtree is
    // managed outside React — skip hydration diffing.
    <div suppressHydrationWarning dangerouslySetInnerHTML={{ __html: html }} />
  );
}

// The interactive Figure 7 iframe posts caption updates back to the page; keep
// the caption in sync and restore the default on mouseleave (verbatim behavior
// from the original index.html).
function wireFigure7Caption(): (() => void) | undefined {
  const frame = document.querySelector<HTMLIFrameElement>(
    'iframe[src$="space-layer-mask.html"]'
  );
  const caption = document.getElementById("figure-7-caption");
  if (!frame || !caption) return undefined;

  const defaultCaption = caption.textContent?.trim() ?? "";
  const restore = () => {
    caption.textContent = defaultCaption;
  };
  const onMessage = (event: MessageEvent) => {
    if (event.origin !== window.location.origin) return;
    if (event.source !== frame.contentWindow) return;
    const data = event.data as { type?: string; description?: unknown };
    if (data?.type !== "figure-7-caption") return;
    if (typeof data.description === "string") {
      caption.textContent = `Figure 7: ${data.description}`;
    } else {
      restore();
    }
  };

  frame.addEventListener("mouseleave", restore);
  window.addEventListener("message", onMessage);
  return () => {
    frame.removeEventListener("mouseleave", restore);
    window.removeEventListener("message", onMessage);
  };
}
