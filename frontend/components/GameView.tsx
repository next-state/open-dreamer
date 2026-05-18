"use client";

import { ReactorView } from "@reactor-team/js-sdk";

interface GameViewProps {
  className?: string;
}

export function GameView({ className }: GameViewProps) {
  return (
    <ReactorView
      className={`h-full w-full bg-black rounded-2xl overflow-hidden ${className || ""}`}
    />
  );
}
