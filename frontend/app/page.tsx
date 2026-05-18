"use client";

import { useEffect, useState } from "react";
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
  const items: [string, string][] = [
    ["WASD", "move"],
    ["Space", "jump"],
    ["Mouse", "look"],
    ["LMB", "attack"],
    ["RMB", "use"],
  ];
  return (
    <div className="hidden md:flex items-center gap-4 text-[11px] text-white/60 font-mono">
      {items.map(([key, label]) => (
        <span key={key} className="flex items-center gap-1.5">
          <Kbd>{key}</Kbd>
          <span>{label}</span>
        </span>
      ))}
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="px-1.5 py-0.5 rounded bg-white/8 border border-white/10 text-white/80 text-[10px] font-mono">
      {children}
    </kbd>
  );
}

function GameInterface() {
  const { status, connect } = useReactor((state) => ({
    status: state.status,
    connect: state.connect,
  }));
  const [isLocked, setIsLocked] = useState(false);

  // Auto-connect on mount and auto-reconnect if the session ever drops.
  useEffect(() => {
    if (status === "disconnected") connect();
  }, [status, connect]);

  const playable = status === "ready";
  const chromeFade = isLocked ? "opacity-0 pointer-events-none" : "opacity-100";

  return (
    <main className="absolute inset-0 flex items-center justify-center px-4 sm:px-8 py-10 sm:py-14">
      <div className="relative w-full max-w-[1600px] aspect-video">
        {/* Outer glow, sits behind the canvas */}
        <div className="absolute inset-0 game-glow rounded-2xl" />

        <div className="relative h-full w-full rounded-2xl overflow-hidden">
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
      <ReactorProvider modelName="world-model" local>
        <GameInterface />
      </ReactorProvider>
    </div>
  );
}
