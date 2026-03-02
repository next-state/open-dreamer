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

  const handleToggle = () => {
    const newValue = !useAgent;
    setUseAgent(newValue);
    sendCommand("switch_to_agent", { enable: newValue });
  };

  const isConnected = status === "ready" || status === "waiting";

  return (
    <div
      className={`border border-gray-700/30 bg-gray-900/40 p-3 rounded-lg ${className || ""}`}
    >
      <div className="flex flex-row justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Control Mode:</span>
          <span className="text-xs font-medium text-gray-200">
            {useAgent ? "Agent (Policy)" : "Manual (Keyboard)"}
          </span>
        </div>
        <button
          onClick={handleToggle}
          disabled={!isConnected}
          className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
            useAgent
              ? "bg-blue-600/80 text-white hover:bg-blue-600"
              : "bg-gray-700/50 text-gray-300 border border-gray-600/50 hover:bg-gray-700 hover:text-white"
          } ${
            !isConnected
              ? "opacity-50 cursor-not-allowed"
              : "cursor-pointer"
          }`}
        >
          {useAgent ? "Switch to Manual" : "Switch to Agent"}
        </button>
      </div>
    </div>
  );
}

