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

  return (
    <button
      onClick={() => sendCommand("new_scene", {})}
      disabled={!isReady}
      className={`group flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors duration-200 ${
        isReady
          ? "text-white hover:bg-emerald-400/15 cursor-pointer"
          : "text-white/30 opacity-60 cursor-not-allowed"
      } ${className || ""}`}
    >
      <svg
        viewBox="0 0 16 16"
        className={`w-3.5 h-3.5 transition-transform duration-300 ${isReady ? "group-hover:rotate-180" : ""}`}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M14 8a6 6 0 1 1-1.76-4.24" />
        <path d="M14 2.5V6h-3.5" />
      </svg>
      <span>New dream</span>
    </button>
  );
}
