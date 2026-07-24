"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  OpenDreamerMainVideoView,
  useOpenDreamer,
} from "@/lib/open-dreamer/open-dreamer.gen.react";
import type { OpenDreamerSendKeyboardStateParams } from "@/lib/open-dreamer/open-dreamer.gen";

// KeyboardEvent.code → the backend's held-key names (VPT action space).
const KEY_CODE_TO_NAME: Record<
  string,
  keyof OpenDreamerSendKeyboardStateParams
> = {
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  Space: "space",
  ShiftLeft: "shift",
  ShiftRight: "shift",
  ControlLeft: "ctrl",
  ControlRight: "ctrl",
  KeyE: "e",
  KeyQ: "q",
  KeyF: "f",
  Digit1: "n1",
  Digit2: "n2",
  Digit3: "n3",
  Digit4: "n4",
  Digit5: "n5",
  Digit6: "n6",
  Digit7: "n7",
  Digit8: "n8",
  Digit9: "n9",
  // The browser reserves F3, so the debug overlay is bound to G.
  KeyG: "f3",
};

const KEY_DISPLAY: Partial<
  Record<keyof OpenDreamerSendKeyboardStateParams, string>
> = {
  w: "W",
  a: "A",
  s: "S",
  d: "D",
  space: "Jump",
  shift: "Sneak",
  ctrl: "Sprint",
  e: "Inv",
  q: "Drop",
  f: "Swap",
  f3: "Debug",
  n1: "1",
  n2: "2",
  n3: "3",
  n4: "4",
  n5: "5",
  n6: "6",
  n7: "7",
  n8: "8",
  n9: "9",
};

const ALL_KEY_NAMES = Array.from(
  new Set(Object.values(KEY_CODE_TO_NAME))
) as (keyof OpenDreamerSendKeyboardStateParams)[];

interface PointerLockControlsProps {
  /** Input is only captured while enabled (session ready and world live). */
  enabled: boolean;
  onLockChange?: (locked: boolean) => void;
  /** Report the currently-held control labels so the HUD can render them
   *  outside the video frame. */
  onActiveKeysChange?: (labels: string[]) => void;
  /** How the frame fills its box — `contain` letterboxes (used in fullscreen so
   *  no part of the view is cropped away). */
  videoObjectFit?: "cover" | "contain";
}

/**
 * Pointer-lock input layer over the model's video: once the pointer is
 * captured, streams held keys, mouse buttons, accumulated camera delta, and
 * scroll-wheel ticks to the model at the game tick rate.
 */
export function PointerLockControls({
  enabled,
  onLockChange,
  onActiveKeysChange,
  videoObjectFit = "cover",
}: PointerLockControlsProps) {
  const { sendKeyboardState, sendMouseState, sendMouseWheel } =
    useOpenDreamer();

  const containerRef = useRef<HTMLDivElement>(null);
  const pressed = useRef<Set<string>>(new Set());
  const buttons = useRef({ left: false, right: false, middle: false });
  const dx = useRef(0);
  const dy = useRef(0);

  const [isLocked, setIsLocked] = useState(false);

  const flushKeyboard = useCallback(() => {
    const state: OpenDreamerSendKeyboardStateParams = {};
    for (const name of ALL_KEY_NAMES) state[name] = false;
    for (const code of pressed.current) {
      const name = KEY_CODE_TO_NAME[code];
      if (name) state[name] = true;
    }
    void sendKeyboardState(state).catch(console.error);

    const active = ALL_KEY_NAMES.filter((n) => state[n]).map(
      (n) => KEY_DISPLAY[n] ?? n
    );
    if (buttons.current.left) active.push("Break");
    if (buttons.current.right) active.push("Place");
    if (buttons.current.middle) active.push("Pick");
    onActiveKeysChange?.(active);
  }, [sendKeyboardState, onActiveKeysChange]);

  const flushMouse = useCallback(() => {
    void sendMouseState({
      left: buttons.current.left,
      right: buttons.current.right,
      middle: buttons.current.middle,
      dx: dx.current,
      dy: dy.current,
    }).catch(console.error);
    dx.current = 0;
    dy.current = 0;
  }, [sendMouseState]);

  const enterLock = useCallback(() => {
    if (!enabled) return;
    const el = containerRef.current;
    if (el && document.pointerLockElement !== el) el.requestPointerLock();
  }, [enabled]);

  // Pointer-lock state — clears and zeroes input the moment the lock releases.
  useEffect(() => {
    const onChange = () => {
      const locked = document.pointerLockElement === containerRef.current;
      setIsLocked(locked);
      onLockChange?.(locked);
      if (!locked) {
        pressed.current.clear();
        buttons.current = { left: false, right: false, middle: false };
        dx.current = 0;
        dy.current = 0;
        onActiveKeysChange?.([]);
        flushKeyboard();
        flushMouse();
      }
    };
    document.addEventListener("pointerlockchange", onChange);
    return () => document.removeEventListener("pointerlockchange", onChange);
  }, [flushKeyboard, flushMouse, onLockChange, onActiveKeysChange]);

  // Release the lock if the world stops being playable underneath us.
  useEffect(() => {
    if (!enabled && document.pointerLockElement) document.exitPointerLock();
  }, [enabled]);

  // Keyboard / mouse / wheel — only bound while pointer-locked.
  useEffect(() => {
    if (!enabled || !isLocked) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.code in KEY_CODE_TO_NAME)) return;
      e.preventDefault();
      if (!pressed.current.has(e.code)) {
        pressed.current.add(e.code);
        flushKeyboard();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (!(e.code in KEY_CODE_TO_NAME)) return;
      e.preventDefault();
      pressed.current.delete(e.code);
      flushKeyboard();
    };
    const onMove = (e: MouseEvent) => {
      dx.current += e.movementX;
      dy.current += e.movementY;
    };
    const setButton = (button: number, down: boolean) => {
      if (button === 0) buttons.current.left = down;
      if (button === 1) buttons.current.middle = down;
      if (button === 2) buttons.current.right = down;
    };
    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      setButton(e.button, true);
      flushKeyboard();
      flushMouse();
    };
    const onMouseUp = (e: MouseEvent) => {
      e.preventDefault();
      setButton(e.button, false);
      flushKeyboard();
      flushMouse();
    };
    const onContext = (e: Event) => e.preventDefault();
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.deltaY === 0) return;
      void sendMouseWheel({ dwheel: e.deltaY < 0 ? 1 : -1 }).catch(
        console.error
      );
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("contextmenu", onContext);
    window.addEventListener("wheel", onWheel, { passive: false });

    // Push accumulated camera delta at ~20fps to match the game tick.
    const interval = window.setInterval(() => {
      if (dx.current !== 0 || dy.current !== 0) flushMouse();
    }, 50);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("contextmenu", onContext);
      window.removeEventListener("wheel", onWheel);
      window.clearInterval(interval);
    };
  }, [enabled, isLocked, flushKeyboard, flushMouse, sendMouseWheel]);

  return (
    <div
      ref={containerRef}
      onClick={enterLock}
      className={`absolute inset-0 select-none ${enabled ? "cursor-pointer" : "cursor-default"}`}
    >
      <OpenDreamerMainVideoView
        videoObjectFit={videoObjectFit}
        className="h-full w-full bg-black"
      />

      {/* Vignette so the glass HUD reads over any frame. */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ boxShadow: "inset 0 0 140px rgba(0,0,0,0.55)" }}
      />

      {enabled && !isLocked && (
        <div
          className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/25 text-center backdrop-blur-[2px]"
          style={{ textShadow: "0 2px 24px rgba(0,0,0,0.7)" }}
        >
          <div className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Click to play
          </div>
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-white/60">
            Esc to release · Tab to dream
          </div>
        </div>
      )}
    </div>
  );
}
