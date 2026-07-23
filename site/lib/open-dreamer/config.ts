// Connection config for the Open Dreamer demo.
//
// A PartyKit queue worker fronts the demo: it owns the session lifecycle (mints
// JWTs, creates the session on claim, deletes it when the member leaves or time
// runs out) and the browser only adopts the session via
// `connect({ sessionId, connectionId })`.

// The browser attaches to the session the queue pre-mints, so this MUST match
// the queue worker's `coordinatorUrl` (the model name is baked into the
// generated typed client in open-dreamer.gen.ts). Kept in code (not env) so the
// two ends can't drift.
export const COORDINATOR_URL = "https://api.reactor.inc";

// The deployed open-dreamer queue Worker host, baked in at build time by the
// Pages workflow via NEXT_PUBLIC_OPEN_DREAMER_PARTYKIT_HOST.
export const PARTYKIT_HOST =
  process.env.NEXT_PUBLIC_OPEN_DREAMER_PARTYKIT_HOST || "127.0.0.1:1999";

// The 127.0.0.1 fallback is for local dev only. A production build shipped
// without the env var would retry a localhost WebSocket forever, which looks
// like an infinite queue to every visitor — surface a clear message instead.
export const PARTYKIT_HOST_MISSING =
  process.env.NODE_ENV === "production" &&
  !process.env.NEXT_PUBLIC_OPEN_DREAMER_PARTYKIT_HOST;
