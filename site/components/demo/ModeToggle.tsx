"use client";

import { cn } from "@/lib/cn";
import { useOpenDreamer } from "@/lib/open-dreamer/open-dreamer.gen.react";

interface ModeToggleProps {
  /** True when the streamed video is the AI-generated ("dreamed") world. */
  dreaming: boolean;
  disabled?: boolean;
  /** Draw attention to the toggle before the visitor has ever switched. */
  highlight?: boolean;
  className?: string;
}

/**
 * Live ⟷ Dream switch for the Minecraft world. "Live" streams the real playable
 * game; "Dream" hands off to the world model, which continues generating from
 * exactly what the player was seeing. Reflected back via `world_status`, so the
 * parent keeps this in sync with the server's actual mode.
 */
export function ModeToggle({
  dreaming,
  disabled,
  highlight,
  className,
}: ModeToggleProps) {
  const { switchToPolicy } = useOpenDreamer();

  const toggle = () => {
    if (disabled) return;
    void switchToPolicy({ enable: !dreaming }).catch(console.error);
  };

  return (
    <div
      className={cn(
        "flex items-center gap-2.5",
        disabled && "opacity-50",
        className
      )}
    >
      <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/45">
        View
      </span>
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-pressed={dreaming}
        className={cn(
          "relative flex items-center rounded-full p-0.5 transition-colors duration-300",
          dreaming ? "bg-emerald-500/25" : "bg-white/10",
          disabled ? "cursor-not-allowed" : "cursor-pointer",
          highlight && "ring-2 ring-emerald-400/70 ring-offset-2 ring-offset-black"
        )}
        style={
          highlight ? { animation: "dreamxPulse 1.6s infinite" } : undefined
        }
      >
        <div className="flex text-[11px] font-medium">
          <span
            className={cn(
              "relative z-10 rounded-full px-3 py-1 transition-colors",
              !dreaming ? "text-white" : "text-white/50"
            )}
          >
            Game
          </span>
          <span
            className={cn(
              "relative z-10 rounded-full px-3 py-1 transition-colors",
              dreaming ? "text-white" : "text-white/50"
            )}
          >
            Dream
          </span>
        </div>
        <span
          className={cn(
            "absolute bottom-0.5 top-0.5 w-[calc(50%-2px)] rounded-full transition-all duration-300 ease-out",
            dreaming
              ? "left-[calc(50%+0px)] bg-gradient-to-r from-emerald-500 to-teal-400 shadow-[0_0_20px_rgba(52,211,153,0.45)]"
              : "left-0.5 bg-white/15"
          )}
        />
      </button>
    </div>
  );
}
