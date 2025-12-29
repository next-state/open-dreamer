"use client";

import { useReactor } from "@reactor-team/js-sdk";

interface ReactorStatusProps {
  className?: string;
}

export function ReactorStatus({ className }: ReactorStatusProps) {
  const { status, waitingInfo, connect, disconnect } = useReactor((state) => ({
    status: state.status,
    waitingInfo: state.waitingInfo,
    connect: state.connect,
    disconnect: state.disconnect,
  }));

  return (
    <div
      className={`border border-gray-700/30 bg-gray-900/40 p-3 rounded-lg ${className || ""}`}
    >
      <div className="flex flex-row justify-between items-center">
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md border transition-all duration-200 ${
              status === "disconnected"
                ? "text-red-400 bg-red-500/10 border-red-500/20"
                : status === "waiting"
                ? "text-gray-300 bg-gray-700/30 border-gray-600/40"
                : "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
            }`}
          >
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                status === "disconnected"
                  ? "bg-red-500"
                  : status === "waiting"
                  ? "bg-gray-400 animate-pulse"
                  : "bg-emerald-500 animate-pulse"
              }`}
            />
            <span className="font-medium text-xs">
              {status === "disconnected"
                ? "Disconnected"
                : status === "waiting"
                ? "Waiting"
                : status === "connecting"
                ? "Connecting"
                : "Ready"}
            </span>
          </div>
          {status === "waiting" && waitingInfo?.position !== undefined && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-700/20 border border-gray-600/30">
              <span className="text-xs text-gray-400">Queue Position:</span>
              <span className="font-semibold text-xs text-gray-200">
                #{waitingInfo.position}
              </span>
            </div>
          )}
        </div>
        {status === "disconnected" ? (
          <button
            className="px-4 py-1.5 rounded-md bg-gray-700/50 text-gray-300 border border-gray-600/50 hover:bg-gray-700 hover:text-white transition-all duration-200 text-xs font-medium"
            onClick={() => connect()}
          >
            Connect
          </button>
        ) : (
          <button
            className="px-4 py-1.5 rounded-md bg-red-600/80 text-white hover:bg-red-600 transition-all duration-200 text-xs font-medium"
            onClick={() => disconnect()}
          >
            Disconnect
          </button>
        )}
      </div>
    </div>
  );
}

