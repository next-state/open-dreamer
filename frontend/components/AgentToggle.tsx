"use client";

import { useState } from "react";
import { useReactor } from "@reactor-team/js-sdk";

interface AgentToggleProps {
  className?: string;
}

export function AgentToggle({ className }: AgentToggleProps) {
  const { sendCommand, status } = useReactor((state) => ({
    sendCommand: state.sendCommand,
    status: state.status,
  }));

  const [useAgent, setUseAgent] = useState(false);

  const isConnected = status === "ready" || status === "waiting";

  const handleToggle = () => {
    const newValue = !useAgent;
    setUseAgent(newValue);
    sendCommand("switch_to_policy", { enable: newValue });
  };

  return (
    <div
      className={`flex items-center gap-2.5 pl-2 pr-0.5 py-0.5 rounded-full ${
        !isConnected ? "opacity-50" : ""
      } ${className || ""}`}
    >
      <span className="eyebrow">Control</span>
      <button
        onClick={handleToggle}
        disabled={!isConnected}
        aria-pressed={useAgent}
        className={`relative flex items-center rounded-full p-0.5 transition-colors duration-300 ${
          useAgent ? "bg-blue-500/30" : "bg-white/10"
        } ${isConnected ? "cursor-pointer" : "cursor-not-allowed"}`}
      >
        <div className="flex text-[11px] font-medium">
          <span
            className={`relative z-10 px-3 py-1 rounded-full transition-colors duration-200 ${
              !useAgent ? "text-white" : "text-white/50"
            }`}
          >
            Manual
          </span>
          <span
            className={`relative z-10 px-3 py-1 rounded-full transition-colors duration-200 ${
              useAgent ? "text-white" : "text-white/50"
            }`}
          >
            Agent
          </span>
        </div>
        <span
          className={`absolute top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-full transition-all duration-300 ease-out ${
            useAgent
              ? "left-[calc(50%+0px)] bg-gradient-to-r from-blue-500 to-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.45)]"
              : "left-0.5 bg-white/15"
          }`}
        />
      </button>
    </div>
  );
}
