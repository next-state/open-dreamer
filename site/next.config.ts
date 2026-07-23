import type { NextConfig } from "next";

// The site is published to GitHub Pages at next-state.github.io/open-dreamer/,
// so it is served from a sub-path and must be a fully static export (no SSR /
// API routes). The demo is entirely client-side — the queue, JWT minting, and
// session CRUD all live on the PartyKit worker — so nothing here needs a server.
const nextConfig: NextConfig = {
  output: "export",
  basePath: "/open-dreamer",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
