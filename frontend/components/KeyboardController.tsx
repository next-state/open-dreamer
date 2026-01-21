"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useReactor } from "@reactor-team/js-sdk";

/**
 * Maps keyboard codes to key names for the backend
 */
const KEY_CODE_TO_NAME: Record<string, string> = {
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  KeyQ: "q",
  KeyE: "e",
  ArrowUp: "w",
  ArrowLeft: "a",
  ArrowDown: "s",
  ArrowRight: "d",
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

  const pressedKeys = useRef<Set<string>>(new Set());
  const [activeKeys, setActiveKeys] = useState<string[]>([]);

  const updateKeyboardState = useCallback(() => {
    if (!enabled) return;

    // Build keyboard state object
    const keyState = {
      w: pressedKeys.current.has("KeyW") || pressedKeys.current.has("ArrowUp"),
      a: pressedKeys.current.has("KeyA") || pressedKeys.current.has("ArrowLeft"),
      s: pressedKeys.current.has("KeyS") || pressedKeys.current.has("ArrowDown"),
      d: pressedKeys.current.has("KeyD") || pressedKeys.current.has("ArrowRight"),
      q: pressedKeys.current.has("KeyQ"),
      e: pressedKeys.current.has("KeyE"),
    };

    // Send keyboard state to backend
    sendCommand("send_keyboard_state", keyState);

    // Update active keys display
    const active: string[] = [];
    if (keyState.w) active.push("↑");
    if (keyState.s) active.push("↓");
    if (keyState.a) active.push("←");
    if (keyState.d) active.push("→");
    if (keyState.q) active.push("Q");
    if (keyState.e) active.push("E");
    setActiveKeys(active);
  }, [sendCommand, enabled]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const code = event.code;
      if (KEY_CODE_TO_NAME.hasOwnProperty(code)) {
        event.preventDefault();
        pressedKeys.current.add(code);
        updateKeyboardState();
      }
    },
    [updateKeyboardState]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent) => {
      const code = event.code;
      if (KEY_CODE_TO_NAME.hasOwnProperty(code)) {
        event.preventDefault();
        pressedKeys.current.delete(code);
        updateKeyboardState();
      }
    },
    [updateKeyboardState]
  );

  useEffect(() => {
    if (!enabled) {
      // Reset to no keys pressed when disabled
      pressedKeys.current.clear();
      setActiveKeys([]);
      sendCommand("send_keyboard_state", { 
        w: false, a: false, s: false, d: false, q: false, e: false 
      });
      return;
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [enabled, handleKeyDown, handleKeyUp, sendCommand]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      pressedKeys.current.clear();
      setActiveKeys([]);
    };
  }, []);

  return (
    <div className={`border border-gray-700/30 bg-gray-900/40 p-3 rounded-lg ${className || ""}`}>
      <div className="flex flex-col gap-2">
        <div className="text-xs text-gray-400 text-center">
          Use <kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">WASD</kbd> or <kbd className="px-1.5 py-0.5 bg-gray-800/50 border border-gray-700/50 rounded text-gray-300">Arrow</kbd> keys to control
        </div>
        {activeKeys.length > 0 && enabled && (
          <div className="flex items-center justify-center gap-2">
            <span className="text-xs text-gray-500">Active:</span>
            <div className="flex gap-1">
              {activeKeys.map((key, idx) => (
                <span
                  key={idx}
                  className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded border border-emerald-500/30 text-sm font-medium animate-pulse"
                >
                  {key}
                </span>
              ))}
            </div>
          </div>
        )}
        {!enabled && status !== "ready" && (
          <div className="text-xs text-gray-600 text-center">
            {status === "disconnected" 
              ? "Connect to enable controls" 
              : status === "waiting" 
              ? "Waiting for connection..." 
              : "Connecting..."}
          </div>
        )}
      </div>
    </div>
  );
}

