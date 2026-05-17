"use client";

import { useReactor } from "@reactor-team/js-sdk";

interface NewSceneButtonProps {
  className?: string;
}

export function NewSceneButton({ className }: NewSceneButtonProps) {
  const { sendCommand, status } = useReactor((state) => ({
    sendCommand: state.sendCommand,
    status: state.status,
  }));

  const isReady = status === "ready";

  const handleClick = () => {
    sendCommand("new_scene", {});
  };

  return (
    <div
      className={`border border-gray-700/30 bg-gray-900/40 p-3 rounded-lg ${className || ""}`}
    >
      <div className="flex flex-row justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Scene:</span>
          <span className="text-xs font-medium text-gray-200">
            Click to start a new generation with a fresh seed
          </span>
        </div>
        <button
          onClick={handleClick}
          disabled={!isReady}
          className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
            isReady
              ? "bg-emerald-600/80 text-white hover:bg-emerald-600 cursor-pointer"
              : "bg-gray-700/50 text-gray-400 border border-gray-600/50 opacity-50 cursor-not-allowed"
          }`}
        >
          New Scene
        </button>
      </div>
    </div>
  );
}
