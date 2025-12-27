"use client";

import { ReactorStatus } from "@/components/ReactorStatus";
import { GameView } from "@/components/GameView";
import { KeyboardController } from "@/components/KeyboardController";
import { AgentToggle } from "@/components/AgentToggle";
import { useReactor, ReactorProvider } from "@reactor-team/js-sdk";

function GameInterface() {
  const { status } = useReactor((state) => ({
    status: state.status,
  }));

  const isConnected = status === "ready" || status === "waiting";

  return (
    <div className="flex flex-col gap-3">
      <GameView className="w-full aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50 shadow-xl overflow-hidden" />
      <ReactorStatus />
      <AgentToggle />
      <KeyboardController enabled={isConnected && status === "ready"} />
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 p-4 sm:p-6 flex flex-col">
      <div className="w-full max-w-3xl md:min-w-[640px] lg:min-w-[768px] mx-auto flex flex-col gap-3 flex-1">
        <div className="text-center space-y-3 pt-8 pb-2">
          <h1 className="text-4xl sm:text-5xl font-bold leading-tight">
            <span className="text-gray-400 font-light">Dreamer</span>
            <span className="text-white"> World Model</span>
          </h1>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <p className="text-gray-300 text-sm font-light tracking-wide">
              Interactive coinrun world model with Reactor
            </p>
            <a
              href="https://github.com/reactor-team/js-sdk"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 hover:text-white rounded border border-gray-700/50 hover:border-gray-600 transition-colors"
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              View on GitHub
            </a>
          </div>
        </div>
        <ReactorProvider 
          modelName="procgen-coinrun" 
          local
        >
          <GameInterface />
        </ReactorProvider>
      </div>

      {/* Footer */}
      <footer className="w-full max-w-3xl md:min-w-[640px] lg:min-w-[768px] mx-auto mt-8 pt-6 pb-4 border-t border-gray-800/50">
        <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 text-xs text-gray-500">
          <span>© {new Date().getFullYear()} Dreamer World Model</span>
          <span className="hidden sm:inline">-</span>
          <a
            href="https://reactor.inc"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-gray-300 transition-colors"
          >
            reactor.inc
          </a>
          <span className="hidden sm:inline">-</span>
          <a
            href="https://docs.reactor.inc/"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-gray-300 transition-colors"
          >
            Documentation
          </a>
        </div>
      </footer>
    </div>
  );
}
