"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useReactor } from "@reactor-team/js-sdk";

/**
 * Maps keyboard keys to coinrun actions:
 * 0 = up, 1 = down, 2 = left, 3 = right, 4 = null
 */
const KEY_TO_ACTION: Record<string, number> = {
  // WASD keys
  KeyW: 0, // up
  KeyS: 1, // down
  KeyA: 2, // left
  KeyD: 3, // right
  // Arrow keys
  ArrowUp: 0,
  ArrowDown: 1,
  ArrowLeft: 2,
  ArrowRight: 3,
};

interface KeyboardControllerProps {
  enabled?: boolean;
  className?: string;
}

export function KeyboardController({
  enabled = true,
  className,
}: KeyboardControllerProps) {
  const { sendMessage, status } = useReactor((state) => ({
    sendMessage: state.sendMessage,
    status: state.status,
  }));

  const pressedKeys = useRef<Set<string>>(new Set());
  const currentAction = useRef<number>(4); // Default to null action
  const [activeKeys, setActiveKeys] = useState<string[]>([]);

  const updateAction = useCallback(() => {
    // Priority: up > down > left > right
    let action = 4; // null

    if (pressedKeys.current.has("KeyW") || pressedKeys.current.has("ArrowUp")) {
      action = 0; // up
    } else if (
      pressedKeys.current.has("KeyS") ||
      pressedKeys.current.has("ArrowDown")
    ) {
      action = 1; // down
    } else if (
      pressedKeys.current.has("KeyA") ||
      pressedKeys.current.has("ArrowLeft")
    ) {
      action = 2; // left
    } else if (
      pressedKeys.current.has("KeyD") ||
      pressedKeys.current.has("ArrowRight")
    ) {
      action = 3; // right
    }

    // Only send if action changed
    if (action !== currentAction.current && enabled) {
      currentAction.current = action;
      sendMessage({ type: "send_keyboard_action", data: { action } });
    }

    // Update active keys display
    const active: string[] = [];
    if (pressedKeys.current.has("KeyW") || pressedKeys.current.has("ArrowUp")) {
      active.push("↑");
    }
    if (pressedKeys.current.has("KeyS") || pressedKeys.current.has("ArrowDown")) {
      active.push("↓");
    }
    if (pressedKeys.current.has("KeyA") || pressedKeys.current.has("ArrowLeft")) {
      active.push("←");
    }
    if (pressedKeys.current.has("KeyD") || pressedKeys.current.has("ArrowRight")) {
      active.push("→");
    }
    setActiveKeys(active);
  }, [sendMessage, enabled]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const code = event.code;
      if (KEY_TO_ACTION.hasOwnProperty(code)) {
        event.preventDefault();
        pressedKeys.current.add(code);
        updateAction();
      }
    },
    [updateAction]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent) => {
      const code = event.code;
      if (KEY_TO_ACTION.hasOwnProperty(code)) {
        event.preventDefault();
        pressedKeys.current.delete(code);
        updateAction();
      }
    },
    [updateAction]
  );

  useEffect(() => {
    if (!enabled) {
      // Reset to null action when disabled
      if (currentAction.current !== 4) {
        currentAction.current = 4;
        sendMessage({ type: "send_keyboard_action", data: { action: 4 } });
      }
      pressedKeys.current.clear();
      setActiveKeys([]);
      return;
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [enabled, handleKeyDown, handleKeyUp, sendMessage]);

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

