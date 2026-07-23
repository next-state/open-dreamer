"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  OpenDreamerMainVideoView,
  useOpenDreamer,
  useOpenDreamerWorldStatus,
} from "@/lib/open-dreamer/open-dreamer.gen.react";
import { PointerLockControls } from "./PointerLockControls";
import { TouchControlBar, useTouchInput } from "./TouchControls";
import { ModeToggle } from "./ModeToggle";
import { StatusPill } from "./StatusPill";

interface GameRoomProps {
  /** Leave the session and return to the queue lobby. */
  onExit: () => void;
  /** Server-truth time left in the queue slot; null when unlimited. */
  remainingMs?: number | null;
}

function formatClock(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function useIsTouch(): boolean {
  const [isTouch, setIsTouch] = useState(false);
  useEffect(() => {
    setIsTouch(
      window.matchMedia?.("(pointer: coarse)").matches ||
        "ontouchstart" in window ||
        navigator.maxTouchPoints > 0
    );
  }, []);
  return isTouch;
}

/**
 * In-session chrome for the live Minecraft world. The video sits in a bare
 * screen; every visible control and hint lives in the chrome bars *around* it
 * (never over the frame) so the model output is never obscured. The session
 * opens in the real playable game and a Live⟷Dream toggle (or the Tab key on
 * desktop) hands the stream to the world model. Supports fullscreen (kept
 * letterboxed so nothing is cropped) and touch play.
 */
export function GameRoom({ onExit, remainingMs = null }: GameRoomProps) {
  const { status, switchToPolicy } = useOpenDreamer();
  const isTouch = useIsTouch();

  const [worldLoading, setWorldLoading] = useState(true);
  const [usePolicy, setUsePolicy] = useState(false);
  const [everDreamed, setEverDreamed] = useState(false);
  useOpenDreamerWorldStatus((msg) => {
    setWorldLoading(msg.loading);
    setUsePolicy(msg.use_policy);
    if (msg.use_policy) setEverDreamed(true);
  });

  const [activeKeys, setActiveKeys] = useState<string[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const shellRef = useRef<HTMLDivElement>(null);

  const connected = status === "ready";
  const connecting = status === "connecting" || status === "waiting";
  const playable = connected && !worldLoading;
  const preparing = connecting || (connected && worldLoading);
  const overlayText = connecting ? "Connecting…" : "Dreaming up a world…";
  const nudge = playable && !usePolicy && !everDreamed;

  const touch = useTouchInput(playable);

  const toggleDream = useCallback(() => {
    if (!connected) return;
    void switchToPolicy({ enable: !usePolicy }).catch(console.error);
  }, [connected, usePolicy, switchToPolicy]);

  // Tab flips Live⟷Dream from anywhere in the session (also while pointer
  // locked). Prevent the browser's focus-cycling default. Desktop-only — touch
  // devices toggle with the on-screen switch.
  useEffect(() => {
    if (isTouch) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      e.preventDefault();
      toggleDream();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isTouch, toggleDream]);

  // Keep the <video> painting — the SDK's autoplay can lose a race and leave a
  // black frame even with a live stream attached.
  useEffect(() => {
    if (!connected) return;
    const id = window.setInterval(() => {
      const v = shellRef.current?.querySelector("video");
      if (v && v.paused) {
        v.muted = true;
        void v.play().catch(() => {});
      }
    }, 1500);
    return () => window.clearInterval(id);
  }, [connected]);

  // Mirror the browser dropping native fullscreen (Esc / edge-swipe) back into
  // our state so the pseudo-fullscreen overlay collapses with it.
  useEffect(() => {
    const onChange = () => {
      if (document.fullscreenElement === null) setIsFullscreen(false);
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // A fixed-overlay "fullscreen" that works everywhere, plus a best-effort
  // native request for the immersive desktop experience. iOS Safari has no
  // Fullscreen API for non-<video> elements, so the overlay is what actually
  // fills the viewport on phones.
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((on) => {
      const next = !on;
      if (next) void shellRef.current?.requestFullscreen?.().catch(() => {});
      else if (document.fullscreenElement)
        void document.exitFullscreen().catch(() => {});
      return next;
    });
  }, []);

  // In fullscreen, letterbox (contain) inside the filled area so no part of the
  // view is cropped; windowed, fill the 16:9 box edge to edge (cover).
  const videoFit = isFullscreen ? "contain" : "cover";

  return (
    <div
      ref={shellRef}
      className={`flex w-full flex-col bg-[#0c0c0c] ${
        isFullscreen ? "fixed inset-0 z-[9999] h-full" : "h-full"
      }`}
    >
      {/* ── Top chrome (outside the frame) ── */}
      <div className="flex items-center justify-between gap-2 border-b border-white/[0.06] px-3 py-2">
        <StateBanner dreaming={usePolicy} live={playable} />
        <div className="flex items-center gap-2">
          {connected && remainingMs !== null && (
            <span
              className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1 font-mono text-xs tabular-nums"
              style={{
                color:
                  remainingMs <= 15_000 ? "#f87171" : "rgba(255,255,255,0.75)",
              }}
            >
              {formatClock(remainingMs)}
            </span>
          )}
          <StatusPill />
          {/* Native/pseudo fullscreen is a desktop affordance; on phones the
              demo is already sized to the viewport and iOS lacks the API. */}
          {!isTouch && (
            <button
              type="button"
              onClick={toggleFullscreen}
              aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
              className="flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-white/70 transition-colors hover:text-white"
            >
              {isFullscreen ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M8 3v4a1 1 0 0 1-1 1H3M21 8h-4a1 1 0 0 1-1-1V3M3 16h4a1 1 0 0 1 1 1v4M16 21v-4a1 1 0 0 1 1-1h4" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 8V5a2 2 0 0 1 2-2h3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M21 16v3a2 2 0 0 1-2 2h-3" />
                </svg>
              )}
            </button>
          )}
        </div>
      </div>

      {/* ── The screen (video only) ── */}
      <div
        className={`relative w-full overflow-hidden bg-black ${
          isFullscreen
            ? "flex min-h-0 flex-1 items-center justify-center"
            : "aspect-[4/3] sm:aspect-video"
        }`}
      >
        {isTouch ? (
          <div
            className="absolute inset-0 touch-none select-none"
            onPointerDown={touch.look.onPointerDown}
            onPointerMove={touch.look.onPointerMove}
            onPointerUp={touch.look.onPointerUp}
            onPointerCancel={touch.look.onPointerCancel}
          >
            <OpenDreamerMainVideoView
              videoObjectFit={videoFit}
              className="h-full w-full bg-black"
            />
            <div
              className="pointer-events-none absolute inset-0"
              style={{ boxShadow: "inset 0 0 140px rgba(0,0,0,0.55)" }}
            />
          </div>
        ) : (
          <PointerLockControls
            enabled={playable}
            onActiveKeysChange={setActiveKeys}
            videoObjectFit={videoFit}
          />
        )}

        {preparing && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-4 bg-black/70 backdrop-blur-sm">
            <span className="h-8 w-8 animate-spin rounded-full border-2 border-emerald-300 border-t-transparent" />
            <span className="font-mono text-sm tracking-wide text-white/70">
              {overlayText}
            </span>
          </div>
        )}
      </div>

      {/* ── Touch gamepad (its own bar, below the frame) ── */}
      {isTouch && <TouchControlBar input={touch} disabled={!playable} />}

      {/* ── Bottom chrome (outside the frame) ── */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-t border-white/[0.06] px-3 py-2.5">
        <div className="flex items-center gap-2.5">
          <ModeToggle
            dreaming={usePolicy}
            disabled={!connected}
            highlight={nudge}
          />
          <span className="hidden h-5 w-px bg-white/10 sm:block" />
          <button
            type="button"
            onClick={onExit}
            className="flex items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.04] px-3 py-1.5 font-mono text-xs text-white/70 transition-colors hover:bg-white/[0.08] hover:text-white"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M11 6l-6 6 6 6" />
            </svg>
            Leave
          </button>
        </div>

        {/* Hint / live key readout — text, never over the frame. */}
        <div className="flex min-h-[24px] items-center gap-1.5">
          {playable && !isTouch && activeKeys.length > 0 ? (
            activeKeys.map((label, i) => (
              <span
                key={i}
                className="rounded border border-emerald-400/30 bg-emerald-400/15 px-2 py-0.5 font-mono text-[10px] text-emerald-200"
              >
                {label}
              </span>
            ))
          ) : (
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/35">
              {isTouch
                ? nudge
                  ? "Tap the toggle to switch to Dream"
                  : "Drag to look · pad to move · Break / Place"
                : nudge
                  ? "Press Tab (or the toggle) to switch to Dream"
                  : "Click frame to play · WASD move · Tab to dream"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** Persistent, high-contrast label telling the visitor what they're looking at. */
function StateBanner({ dreaming, live }: { dreaming: boolean; live: boolean }) {
  const short = !live ? "Starting…" : dreaming ? "Dream" : "Game";
  const full = !live
    ? "Starting…"
    : dreaming
      ? "Dream — every frame is generated"
      : "Game — the real, playable game";
  return (
    <div
      className={`flex items-center gap-2 whitespace-nowrap rounded-md border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] ${
        !live
          ? "border-white/10 text-white/40"
          : dreaming
            ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-100"
            : "border-white/20 bg-white/[0.06] text-white/85"
      }`}
    >
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{
          backgroundColor: !live ? "#6b7280" : dreaming ? "#34d399" : "#e5e7eb",
          boxShadow: live && dreaming ? "0 0 8px rgba(52,211,153,0.8)" : "none",
        }}
      />
      {/* Compact on phones; the full phrase only where there's room. */}
      <span className="sm:hidden">{short}</span>
      <span className="hidden sm:inline">{full}</span>
    </div>
  );
}
