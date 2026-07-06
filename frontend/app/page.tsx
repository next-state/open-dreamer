"use client";

import { useEffect, useRef, useState } from "react";
import { ReactorProvider, useReactor } from "@reactor-team/js-sdk";
import { ReactorStatus } from "@/components/ReactorStatus";
import { KeyboardController } from "@/components/KeyboardController";
import { AgentToggle } from "@/components/AgentToggle";
import { NewSceneButton } from "@/components/NewSceneButton";

function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      {/* Voxel-block monogram — a subtle Minecraft nod without leaning into kitsch. */}
      <div
        className="relative w-7 h-7 rounded-md"
        style={{
          background:
            "linear-gradient(135deg, #34d399 0%, #10b981 50%, #047857 100%)",
          boxShadow:
            "inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -1px 0 rgba(0,0,0,0.25), 0 0 20px rgba(52,211,153,0.35)",
        }}
      >
        <div
          className="absolute inset-1 rounded-sm opacity-60"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.15) 1px, transparent 1px)",
            backgroundSize: "5px 5px",
          }}
        />
      </div>
      <div className="text-lg font-semibold tracking-tight leading-none">
        <span className="text-white/55 font-light">Open</span>
        <span className="text-white">Dreamer</span>
      </div>
    </div>
  );
}

function ControlsHint() {
  // Grouped by intent so each row reads as one coherent cluster.
  const rows: [React.ReactNode, string][][] = [
    // Move & look
    [
      ["WASD", "move"],
      [<MouseMoveIcon />, "camera"],
      ["Space", "jump"],
      ["Ctrl", "sprint"],
      ["Shift", "sneak"],
    ],
    // Act on the world + select what's in hand
    [
      [<MouseIcon side="left" />, "break"],
      [<MouseIcon side="right" />, "place"],
      [<MouseIcon side="middle" />, "pick"],
      ["1–9", "hotbar"],
      ["Scroll", "cycle"],
    ],
    // Inventory, items & HUD
    [
      ["E", "inventory"],
      ["Q", "drop"],
      ["F", "swap"],
      ["G", "debug"],
    ],
  ];
  return (
    <div className="hidden md:flex flex-col items-center gap-1.5 text-[11px] text-white/60 font-mono">
      {rows.map((items, r) => (
        <div key={r} className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
          {items.map(([key, label], idx) => (
            <span key={idx} className="flex items-center gap-1.5">
              <Kbd>{key}</Kbd>
              <span>{label}</span>
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center px-1.5 py-0.5 rounded bg-white/8 border border-white/10 text-white/80 text-[10px] font-mono">
      {children}
    </kbd>
  );
}

/** Small mouse glyph with the active button filled. */
function MouseIcon({ side }: { side: "left" | "right" | "middle" }) {
  const clipId = `mouse-clip-${side}`;
  return (
    <svg width="11" height="16" viewBox="0 0 14 20" aria-hidden="true">
      <defs>
        <clipPath id={clipId}>
          <rect x="1" y="1" width="12" height="18" rx="6" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        {side === "left" && <rect x="0" y="0" width="7" height="8.5" fill="currentColor" />}
        {side === "right" && <rect x="7" y="0" width="7" height="8.5" fill="currentColor" />}
        {side === "middle" && <rect x="5.5" y="2.5" width="3" height="5" rx="1.5" fill="currentColor" />}
      </g>
      <rect x="1" y="1" width="12" height="18" rx="6" fill="none" stroke="currentColor" strokeWidth="1.1" />
      <path d="M7 1.5 V8.5 M1 8.5 H13" stroke="currentColor" strokeWidth="0.9" />
    </svg>
  );
}

/** Mouse with motion arrows — "move the mouse", distinct from the click glyphs. */
function MouseMoveIcon() {
  return (
    <svg width="20" height="15" viewBox="0 0 24 20" aria-hidden="true">
      <rect x="6" y="1" width="12" height="18" rx="6" fill="none" stroke="currentColor" strokeWidth="1.1" />
      <path d="M12 1.5 V8.5 M6 8.5 H18" stroke="currentColor" strokeWidth="0.9" />
      <path
        d="M3.5 7 L1 10 L3.5 13 M20.5 7 L23 10 L20.5 13"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function GameInterface() {
  const { status, connect } = useReactor((state) => ({
    status: state.status,
    connect: state.connect,
  }));
  const [isLocked, setIsLocked] = useState(false);
  const retriesRef = useRef(0);

  // Auto-connect on mount and auto-reconnect if the session drops — but with
  // exponential backoff so a failing connection doesn't hammer start_session.
  // First attempt is immediate; failures back off 2s, 4s, ... capped at 15s.
  // A successful connection ("ready") resets the backoff.
  useEffect(() => {
    if (status === "ready") {
      retriesRef.current = 0;
      return;
    }
    if (status !== "disconnected" && status !== "error") return;

    const attempt = retriesRef.current;
    const delay = attempt === 0 ? 0 : Math.min(2000 * 2 ** (attempt - 1), 15000);
    const timer = setTimeout(() => {
      retriesRef.current += 1;
      connect();
    }, delay);
    return () => clearTimeout(timer);
  }, [status, connect]);

  const playable = status === "ready";
  const chromeFade = isLocked ? "opacity-0 pointer-events-none" : "opacity-100";

  return (
    <main className="absolute inset-0 flex items-center justify-center px-4 sm:px-8 py-10 sm:py-14">
      <div className="relative w-full max-w-[1600px] aspect-video">
        {/* Outer glow, sits behind the canvas */}
        <div className="absolute inset-0 game-glow rounded-2xl" />

        <div className="relative h-full w-full rounded-2xl overflow-hidden bg-black">
          {/* KeyboardController renders the video (GameView) and owns the
              click-to-pointer-lock target, so it must be the top layer. */}
          <KeyboardController enabled={playable} onLockChange={setIsLocked} />

          {/* Top chrome — overlays the top of the game so backdrop blur
              has actual frames to diffuse. */}
          <header
            className={`absolute top-0 left-0 right-0 z-20 flex items-center justify-between gap-4 px-4 sm:px-5 py-4 transition-opacity duration-500 ${chromeFade}`}
          >
            <Wordmark />
            <ReactorStatus />
          </header>

          {/* Bottom chrome — same idea, single glass deck. */}
          <footer
            className={`absolute bottom-4 left-1/2 -translate-x-1/2 z-20 transition-opacity duration-500 ${chromeFade}`}
          >
            <div className="glass-strong rounded-2xl px-2.5 py-1.5 flex flex-col items-center gap-1">
              <div className="flex items-center gap-1">
                <NewSceneButton />
                <span className="w-px h-5 bg-white/10 mx-1" />
                <AgentToggle />
              </div>
              <div className="h-px w-full bg-white/10 hidden md:block" />
              <div className="hidden md:block pb-1 pt-0.5">
                <ControlsHint />
              </div>
            </div>
          </footer>
        </div>
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <div className="app-backdrop relative h-screen w-screen overflow-hidden">
      <ReactorProvider
        modelName="world-model"
        local
        apiUrl={process.env.NEXT_PUBLIC_REACTOR_URL ?? "http://localhost:8096"}
      >
        <GameInterface />
      </ReactorProvider>
    </div>
  );
}
