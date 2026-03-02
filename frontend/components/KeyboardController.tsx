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
};

/** Display labels for active keys. */
const KEY_DISPLAY: Record<string, string> = {
  w: "W", a: "A", s: "S", d: "D",
  space: "Jump", shift: "Sneak", ctrl: "Sprint",
  e: "Inv", q: "Drop", f: "Swap",
  n1: "1", n2: "2", n3: "3", n4: "4", n5: "5",
  n6: "6", n7: "7", n8: "8", n9: "9",
};

interface KeyboardControllerProps {
  enabled?: boolean;
  className?: string;
}

export function KeyboardController({
  enabled = true,
  className,
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
  }, [sendKeyboardState, sendMouseState]);

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

  function updateActiveButtons() {
    const btns: string[] = [];
    if (mouseButtons.current.left) btns.push("Attack");
    if (mouseButtons.current.right) btns.push("Use");
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

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("contextmenu", handleContextMenu);
    };
  }, [enabled, isLocked, handleKeyDown, handleKeyUp, handleMouseMove, handleMouseDown, handleMouseUp, handleContextMenu]);

  // Exit pointer lock when disabled
  useEffect(() => {
    if (!enabled && isLocked) {
      document.exitPointerLock();
    }
  }, [enabled, isLocked]);

  const hasActiveInput = activeKeys.length > 0 || activeButtons.length > 0;

  return (
    <div className={className || ""}>
      {/* Game view with pointer lock target */}
      <div
        ref={containerRef}
        onClick={handleClick}
        className="relative cursor-pointer select-none"
      >
        <GameView className="w-full aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50 shadow-xl overflow-hidden" />

        {/* Overlay: click to play */}
        {enabled && !isLocked && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-xl transition-opacity">
            <div className="text-center">
              <div className="text-white text-lg font-medium mb-1">
                Click to play
              </div>
              <div className="text-gray-300 text-xs">
                ESC to release cursor
              </div>
            </div>
          </div>
        )}

        {/* Indicator: game is focused */}
        {isLocked && (
          <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded text-xs text-gray-300">
            ESC to release
          </div>
        )}
      </div>

      {/* Controls info panel */}
      <div className="border border-gray-700/30 bg-gray-900/40 p-3 rounded-lg mt-3">
        <div className="flex flex-col gap-2">
          {isLocked ? (
            <>
              <div className="text-xs text-gray-400 text-center">
                <kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">WASD</kbd> Move
                {" "}<kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">Space</kbd> Jump
                {" "}<kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">Mouse</kbd> Look
                {" "}<kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">LMB</kbd> Attack
                {" "}<kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">RMB</kbd> Use
              </div>
              {hasActiveInput && (
                <div className="flex items-center justify-center gap-2">
                  <span className="text-xs text-gray-500">Active:</span>
                  <div className="flex gap-1 flex-wrap justify-center">
                    {[...activeKeys, ...activeButtons].map((label, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded border border-emerald-500/30 text-xs font-medium"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-xs text-gray-500 text-center">
              {!enabled
                ? status === "disconnected"
                  ? "Connect to enable controls"
                  : status === "waiting"
                  ? "Waiting for connection..."
                  : "Connecting..."
                : "Click the game to start playing"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
