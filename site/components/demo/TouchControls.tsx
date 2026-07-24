"use client";

import { useCallback, useEffect, useRef } from "react";

import { useOpenDreamer } from "@/lib/open-dreamer/open-dreamer.gen.react";
import type { OpenDreamerSendKeyboardStateParams } from "@/lib/open-dreamer/open-dreamer.gen";

// Touch drag → camera delta. Touch pixels map roughly 1:1 to the raw
// movementX/Y pointer lock streams; a small boost keeps looking responsive.
const LOOK_SENSITIVITY = 1.4;

// The movement keys the on-screen pad drives (a subset of the full action
// space — touch play is WASD + jump + break/place + look).
type MoveKey = "w" | "a" | "s" | "d" | "space";

/** Drag-to-look handlers, bound to the transparent surface over the frame. */
export interface TouchLook {
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
  onPointerCancel: (e: React.PointerEvent) => void;
}

export interface TouchInput {
  look: TouchLook;
  setKey: (key: MoveKey, down: boolean) => void;
  setButton: (which: "left" | "right", down: boolean) => void;
}

/**
 * Touch input controller for phones/tablets, where pointer lock and a keyboard
 * aren't available. Dragging over the frame looks around; the on-screen pad
 * (rendered by {@link TouchControlBar}) drives WASD + jump and holds the
 * break/place mouse buttons. Returns the handlers both surfaces share so the
 * buttons can live in chrome below the frame instead of over the video.
 */
export function useTouchInput(enabled: boolean): TouchInput {
  const { sendKeyboardState, sendMouseState } = useOpenDreamer();

  const held = useRef<Set<MoveKey>>(new Set());
  const buttons = useRef({ left: false, right: false });
  const dx = useRef(0);
  const dy = useRef(0);
  const lookPointer = useRef<number | null>(null);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);

  const flushKeyboard = useCallback(() => {
    const state: OpenDreamerSendKeyboardStateParams = {
      w: held.current.has("w"),
      a: held.current.has("a"),
      s: held.current.has("s"),
      d: held.current.has("d"),
      space: held.current.has("space"),
    };
    void sendKeyboardState(state).catch(console.error);
  }, [sendKeyboardState]);

  const flushMouse = useCallback(() => {
    void sendMouseState({
      left: buttons.current.left,
      right: buttons.current.right,
      dx: dx.current,
      dy: dy.current,
    }).catch(console.error);
    dx.current = 0;
    dy.current = 0;
  }, [sendMouseState]);

  // Release everything whenever the world stops being playable underneath us.
  useEffect(() => {
    if (enabled) return;
    held.current.clear();
    buttons.current = { left: false, right: false };
    dx.current = 0;
    dy.current = 0;
    lookPointer.current = null;
    lastPoint.current = null;
  }, [enabled]);

  // Push accumulated look delta at ~20fps to match the game tick.
  useEffect(() => {
    if (!enabled) return;
    const id = window.setInterval(() => {
      if (dx.current !== 0 || dy.current !== 0) flushMouse();
    }, 50);
    return () => window.clearInterval(id);
  }, [enabled, flushMouse]);

  const setKey = useCallback(
    (key: MoveKey, down: boolean) => {
      if (down) held.current.add(key);
      else held.current.delete(key);
      flushKeyboard();
    },
    [flushKeyboard]
  );

  const setButton = useCallback(
    (which: "left" | "right", down: boolean) => {
      buttons.current[which] = down;
      flushMouse();
    },
    [flushMouse]
  );

  const look: TouchLook = {
    onPointerDown: (e) => {
      if (!enabled) return;
      lookPointer.current = e.pointerId;
      lastPoint.current = { x: e.clientX, y: e.clientY };
    },
    onPointerMove: (e) => {
      if (!enabled || e.pointerId !== lookPointer.current || !lastPoint.current)
        return;
      dx.current += (e.clientX - lastPoint.current.x) * LOOK_SENSITIVITY;
      dy.current += (e.clientY - lastPoint.current.y) * LOOK_SENSITIVITY;
      lastPoint.current = { x: e.clientX, y: e.clientY };
    },
    onPointerUp: (e) => {
      if (e.pointerId !== lookPointer.current) return;
      lookPointer.current = null;
      lastPoint.current = null;
    },
    onPointerCancel: (e) => {
      if (e.pointerId !== lookPointer.current) return;
      lookPointer.current = null;
      lastPoint.current = null;
    },
  };

  return { look, setKey, setButton };
}

/**
 * On-screen gamepad rendered in a chrome bar *below* the frame (never over the
 * video): a WASD cluster plus jump and the break/place actions.
 */
export function TouchControlBar({
  input,
  disabled,
}: {
  input: TouchInput;
  disabled: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-white/[0.06] bg-[#0c0c0c] px-3 py-3">
      <div className="grid grid-cols-3 grid-rows-2 gap-1.5">
        <HoldButton
          className="col-start-2"
          disabled={disabled}
          onHold={(d) => input.setKey("w", d)}
        >
          W
        </HoldButton>
        <HoldButton
          className="col-start-1 row-start-2"
          disabled={disabled}
          onHold={(d) => input.setKey("a", d)}
        >
          A
        </HoldButton>
        <HoldButton
          className="col-start-2 row-start-2"
          disabled={disabled}
          onHold={(d) => input.setKey("s", d)}
        >
          S
        </HoldButton>
        <HoldButton
          className="col-start-3 row-start-2"
          disabled={disabled}
          onHold={(d) => input.setKey("d", d)}
        >
          D
        </HoldButton>
      </div>

      <div className="flex items-end gap-2">
        <HoldButton disabled={disabled} onHold={(d) => input.setKey("space", d)}>
          Jump
        </HoldButton>
        <HoldButton
          tone="rose"
          disabled={disabled}
          onHold={(d) => input.setButton("left", d)}
        >
          Break
        </HoldButton>
        <HoldButton
          tone="sky"
          disabled={disabled}
          onHold={(d) => input.setButton("right", d)}
        >
          Place
        </HoldButton>
      </div>
    </div>
  );
}

/**
 * Press-and-hold button: fires `onHold(true)` while a pointer is down and
 * `onHold(false)` when it lifts or leaves, capturing the pointer so the release
 * always registers even if the finger drifts off the button.
 */
function HoldButton({
  children,
  className,
  tone = "default",
  disabled,
  onHold,
}: {
  children: React.ReactNode;
  className?: string;
  tone?: "default" | "rose" | "sky";
  disabled?: boolean;
  onHold: (down: boolean) => void;
}) {
  const toneClass =
    tone === "rose"
      ? "border-rose-400/40 bg-rose-500/25 text-rose-50 active:bg-rose-500/50"
      : tone === "sky"
        ? "border-sky-400/40 bg-sky-500/25 text-sky-50 active:bg-sky-500/50"
        : "border-white/20 bg-white/10 text-white active:bg-white/25";

  const down = (e: React.PointerEvent) => {
    if (disabled) return;
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    onHold(true);
  };
  const up = (e: React.PointerEvent) => {
    e.stopPropagation();
    onHold(false);
  };

  return (
    <button
      type="button"
      disabled={disabled}
      onPointerDown={down}
      onPointerUp={up}
      onPointerCancel={up}
      onPointerLeave={up}
      className={`flex h-11 w-11 items-center justify-center rounded-xl border font-mono text-[11px] uppercase tracking-wide backdrop-blur-md transition-colors disabled:opacity-30 ${toneClass} ${className ?? ""}`}
    >
      {children}
    </button>
  );
}
