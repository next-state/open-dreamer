"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useReactor } from "@reactor-team/js-sdk";
import { GameView } from "./GameView";

/**
 * Maps KeyboardEvent.code to the backend key names.
 * Matches the VPT action space from dreamer/actions.py.
 */
const KEY_CODE_TO_NAME: Record<string, string> = {
  // Movement (VPT indices 0-3)
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  // Modifiers (VPT indices 4-6)
  Space: "space",
  ShiftLeft: "shift",
  ShiftRight: "shift",
  ControlLeft: "ctrl",
  ControlRight: "ctrl",
  // Items (VPT indices 7-8, 10)
  KeyE: "e",
  KeyQ: "q",
  KeyF: "f",
  // Hotbar (VPT indices 11-19)
  Digit1: "n1",
  Digit2: "n2",
  Digit3: "n3",
  Digit4: "n4",
  Digit5: "n5",
  Digit6: "n6",
  Digit7: "n7",
  Digit8: "n8",
  Digit9: "n9",
  // Debug overlay — bound to G because the browser reserves F3
  KeyG: "f3",
};

/** Display labels for active keys. */
const KEY_DISPLAY: Record<string, string> = {
  w: "W", a: "A", s: "S", d: "D",
  space: "Jump", shift: "Sneak", ctrl: "Sprint",
  e: "Inv", q: "Drop", f: "Swap", f3: "Debug",
  n1: "1", n2: "2", n3: "3", n4: "4", n5: "5",
  n6: "6", n7: "7", n8: "8", n9: "9",
};

interface KeyboardControllerProps {
  enabled?: boolean;
  className?: string;
  onLockChange?: (isLocked: boolean) => void;
}

export function KeyboardController({
  enabled = true,
  className,
  onLockChange,
}: KeyboardControllerProps) {
  const { sendCommand, status } = useReactor((state) => ({
    sendCommand: state.sendCommand,
    status: state.status,
  }));

  const containerRef = useRef<HTMLDivElement>(null);
  const pressedKeys = useRef<Set<string>>(new Set());
  const mouseButtons = useRef({ left: false, right: false, middle: false });
  const mouseDx = useRef(0);
  const mouseDy = useRef(0);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [isLocked, setIsLocked] = useState(false);
  const [activeKeys, setActiveKeys] = useState<string[]>([]);
  const [activeButtons, setActiveButtons] = useState<string[]>([]);

  // --- Send commands to backend ---

  const sendKeyboardState = useCallback(() => {
    const keyState: Record<string, boolean> = {};
    for (const name of Object.values(KEY_CODE_TO_NAME)) {
      keyState[name] = false;
    }
    // Deduplicate: multiple codes can map to same name (e.g. ShiftLeft/ShiftRight)
    for (const code of pressedKeys.current) {
      const name = KEY_CODE_TO_NAME[code];
      if (name) keyState[name] = true;
    }
    sendCommand("send_keyboard_state", keyState);

    // Update display
    const active = Object.entries(keyState)
      .filter(([, v]) => v)
      .map(([k]) => KEY_DISPLAY[k] || k);
    setActiveKeys(active);
  }, [sendCommand]);

  const sendMouseState = useCallback(() => {
    sendCommand("send_mouse_state", {
      left: mouseButtons.current.left,
      right: mouseButtons.current.right,
      middle: mouseButtons.current.middle,
      dx: mouseDx.current,
      dy: mouseDy.current,
    });
    // Reset accumulated deltas after sending
    mouseDx.current = 0;
    mouseDy.current = 0;
  }, [sendCommand]);

  // --- Pointer lock management ---

  const handleClick = useCallback(() => {
    if (!enabled) return;
    const el = containerRef.current;
    if (el && document.pointerLockElement !== el) {
      el.requestPointerLock();
    }
  }, [enabled]);

  const handlePointerLockChange = useCallback(() => {
    const locked = document.pointerLockElement === containerRef.current;
    setIsLocked(locked);
    onLockChange?.(locked);
    if (!locked) {
      // Released — reset all input state
      pressedKeys.current.clear();
      mouseButtons.current = { left: false, right: false, middle: false };
      mouseDx.current = 0;
      mouseDy.current = 0;
      setActiveKeys([]);
      setActiveButtons([]);
      // Send zeroed state to backend
      sendKeyboardState();
      sendMouseState();
    }
  }, [sendKeyboardState, sendMouseState, onLockChange]);

  // --- Keyboard handlers (only active while pointer-locked) ---

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!isLocked) return;
      const code = event.code;
      if (code in KEY_CODE_TO_NAME) {
        event.preventDefault();
        event.stopPropagation();
        if (!pressedKeys.current.has(code)) {
          pressedKeys.current.add(code);
          sendKeyboardState();
        }
      }
    },
    [isLocked, sendKeyboardState]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent) => {
      if (!isLocked) return;
      const code = event.code;
      if (code in KEY_CODE_TO_NAME) {
        event.preventDefault();
        event.stopPropagation();
        pressedKeys.current.delete(code);
        sendKeyboardState();
      }
    },
    [isLocked, sendKeyboardState]
  );

  // --- Mouse handlers (only active while pointer-locked) ---

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!isLocked) return;
      // Accumulate deltas — will be sent on next tick
      mouseDx.current += event.movementX;
      mouseDy.current += event.movementY;
    },
    [isLocked]
  );

  const handleMouseDown = useCallback(
    (event: MouseEvent) => {
      if (!isLocked) return;
      event.preventDefault();
      if (event.button === 0) mouseButtons.current.left = true;
      if (event.button === 2) mouseButtons.current.right = true;
      if (event.button === 1) mouseButtons.current.middle = true;
      updateActiveButtons();
      sendMouseState();
    },
    [isLocked, sendMouseState]
  );

  const handleMouseUp = useCallback(
    (event: MouseEvent) => {
      if (!isLocked) return;
      event.preventDefault();
      if (event.button === 0) mouseButtons.current.left = false;
      if (event.button === 2) mouseButtons.current.right = false;
      if (event.button === 1) mouseButtons.current.middle = false;
      updateActiveButtons();
      sendMouseState();
    },
    [isLocked, sendMouseState]
  );

  const handleContextMenu = useCallback((event: Event) => {
    if (isLocked) event.preventDefault();
  }, [isLocked]);

  const handleWheel = useCallback(
    (event: WheelEvent) => {
      if (!isLocked) return;
      event.preventDefault();
      if (event.deltaY === 0) return;
      sendCommand("send_mouse_wheel", { dwheel: event.deltaY < 0 ? 1 : -1 });
    },
    [isLocked, sendCommand]
  );

  function updateActiveButtons() {
    const btns: string[] = [];
    if (mouseButtons.current.left) btns.push("Break");
    if (mouseButtons.current.right) btns.push("Place");
    if (mouseButtons.current.middle) btns.push("Pick");
    setActiveButtons(btns);
  }

  // --- Periodic mouse delta send (accumulate between frames) ---

  useEffect(() => {
    if (isLocked && enabled) {
      // Send accumulated mouse deltas at ~20fps to match game tick rate
      sendIntervalRef.current = setInterval(() => {
        if (mouseDx.current !== 0 || mouseDy.current !== 0) {
          sendMouseState();
        }
      }, 50);
    }
    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
    };
  }, [isLocked, enabled, sendMouseState]);

  // --- Event listener setup ---

  useEffect(() => {
    document.addEventListener("pointerlockchange", handlePointerLockChange);
    return () => {
      document.removeEventListener("pointerlockchange", handlePointerLockChange);
    };
  }, [handlePointerLockChange]);

  useEffect(() => {
    if (!enabled || !isLocked) return;

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("contextmenu", handleContextMenu);
    window.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("contextmenu", handleContextMenu);
      window.removeEventListener("wheel", handleWheel);
    };
  }, [enabled, isLocked, handleKeyDown, handleKeyUp, handleMouseMove, handleMouseDown, handleMouseUp, handleContextMenu, handleWheel]);

  // Exit pointer lock when disabled
  useEffect(() => {
    if (!enabled && isLocked) {
      document.exitPointerLock();
    }
  }, [enabled, isLocked]);

  const hasActiveInput = activeKeys.length > 0 || activeButtons.length > 0;

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      className={`relative h-full w-full select-none ${
        enabled ? "cursor-pointer" : "cursor-default"
      } ${className || ""}`}
    >
      <GameView className="h-full w-full" />

      {/* Vignette — fades the corners so the world feels deeper. */}
      <div
        className="pointer-events-none absolute inset-0 rounded-2xl"
        style={{
          boxShadow:
            "inset 0 0 120px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,255,255,0.06)",
        }}
      />

      {/* Click-to-play overlay — only once playable (the page owns the
          "preparing" screen for the connecting / world-generating states). */}
      {enabled && !isLocked && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center px-6 rounded-2xl bg-black/30 backdrop-blur-md backdrop-saturate-75"
          style={{ textShadow: "0 2px 24px rgba(0,0,0,0.7)" }}
        >
          <div className="text-white text-3xl sm:text-4xl font-semibold tracking-tight">
            Click to enter
          </div>
          <div className="text-white/65 text-sm sm:text-base">ESC to release</div>
        </div>
      )}

      {/* While locked, a faint reminder in the corner. */}
      {isLocked && (
        <div className="pointer-events-none absolute top-3 right-3 glass px-2.5 py-1 rounded-full text-[10px] font-mono tracking-wider text-white/70 uppercase">
          Esc to release
        </div>
      )}

      {/* Active-input chip cluster — anchored bottom-left so it doesn't
          occlude the hotbar at the bottom-center of the Minecraft viewport. */}
      {isLocked && hasActiveInput && (
        <div className="pointer-events-none absolute bottom-4 left-4 flex gap-1.5 flex-wrap max-w-[60%]">
          {[...activeKeys, ...activeButtons].map((label, idx) => (
            <span
              key={idx}
              className="px-2.5 py-1 rounded-md bg-emerald-400/20 text-emerald-200 border border-emerald-400/30 text-[11px] font-mono backdrop-blur"
            >
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
