// The site ships as a static export served from a sub-path on GitHub Pages, so
// asset URLs the browser requests at runtime (the Distill stylesheet, template
// script, and figures referenced by the injected article HTML) must carry the
// same prefix Next uses as `basePath`. Kept here so the two can't drift.
export const BASE_PATH = "/open-dreamer";

/** Prefix a public-asset path with the deployment base path. */
export function asset(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PATH}${suffix}`;
}
