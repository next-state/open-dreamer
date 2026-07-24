"use client";

import { useOpenDreamer } from "@/lib/open-dreamer/open-dreamer.gen.react";

const LABELS: Record<string, string> = {
  ready: "Connected",
  waiting: "Waiting for GPU…",
  connecting: "Connecting…",
  disconnected: "Offline",
  error: "Error",
};

/** Small glass connection indicator (dot + label) for the top bar. */
export function StatusPill() {
  const status = useOpenDreamer().status;

  const dotColor =
    status === "ready"
      ? "#34d399"
      : status === "connecting" || status === "waiting"
        ? "#facc15"
        : "rgba(255,255,255,0.3)";

  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/[0.12] bg-[rgba(12,12,12,0.5)] px-3 py-[7px] backdrop-blur-xl">
      <span
        className="h-[7px] w-[7px] rounded-full"
        style={{
          backgroundColor: dotColor,
          boxShadow:
            status === "ready" ? "0 0 8px rgba(52,211,153,0.7)" : "none",
          animation:
            status === "connecting" || status === "waiting"
              ? "dreamxPulse 1.5s infinite"
              : "none",
        }}
      />
      <span className="font-mono text-xs text-white/75">
        {LABELS[status] ?? status}
      </span>
    </div>
  );
}
