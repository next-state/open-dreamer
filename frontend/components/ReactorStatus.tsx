"use client";

import { useReactor, useStats } from "@reactor-team/js-sdk";

interface ReactorStatusProps {
  className?: string;
}

export function ReactorStatus({ className }: ReactorStatusProps) {
  const { status } = useReactor((state) => ({ status: state.status }));
  const stats = useStats();
  const fps = stats?.framesPerSecond;
  const rtt = stats?.rtt;

  const dotClass =
    status === "disconnected"
      ? "bg-red-500"
      : status === "ready"
      ? "bg-emerald-400 animate-ambient"
      : "bg-amber-400 animate-ambient";

  const label =
    status === "disconnected"
      ? "Offline"
      : status === "waiting"
      ? "Waiting"
      : status === "connecting"
      ? "Connecting"
      : "Live";

  return (
    <div className={`glass flex items-center gap-2.5 px-3 py-1.5 rounded-full ${className || ""}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
      <span className="text-xs font-medium text-white/90 tracking-wide">{label}</span>
      {status === "ready" && (
        <span className="hidden sm:flex items-center gap-3 pl-2.5 ml-1 border-l border-white/10 font-mono text-[11px] text-white/55">
          <span>{fps !== undefined ? `${fps.toFixed(0)} fps` : "— fps"}</span>
          {rtt !== undefined && <span>{rtt.toFixed(0)} ms</span>}
        </span>
      )}
    </div>
  );
}
