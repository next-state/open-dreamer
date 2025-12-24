"use client";

import { ReactorView } from "@reactor-team/js-sdk";

interface GameViewProps {
  className?: string;
}

export function GameView({ className }: GameViewProps) {
  return (
    <ReactorView
      className={`w-full aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50 shadow-xl overflow-hidden ${className || ""}`}
    />
  );
}

