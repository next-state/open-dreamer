"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ReactorQueueProvider,
  useReactorQueue,
} from "@reactor-team/queue/react";

import {
  OpenDreamerProvider,
  useOpenDreamer,
} from "@/lib/open-dreamer/open-dreamer.gen.react";
import {
  COORDINATOR_URL,
  PARTYKIT_HOST,
  PARTYKIT_HOST_MISSING,
} from "@/lib/open-dreamer/config";
import { GameRoom } from "./GameRoom";

// End the run slightly before the queue's hard deadline so teardown (endSession
// + disconnect) beats the server reaper.
const CLIENT_END_SLACK_MS = 3_000;

function formatClock(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Manual connect instead of the provider's autoConnect: React StrictMode
// double-mounts, racing autoConnect against its cleanup disconnect. Delaying
// past the fake mount/unmount cycle sidesteps the race; the provider's getJwt /
// connectOptions are applied as connect() defaults.
function Connector() {
  const { connect, disconnect } = useOpenDreamer();
  useEffect(() => {
    let started = false;
    const id = window.setTimeout(() => {
      started = true;
      void connect().catch((err) =>
        console.error("[open-dreamer] connect failed", err)
      );
    }, 150);
    return () => {
      window.clearTimeout(id);
      if (started) void disconnect();
    };
  }, [connect, disconnect]);
  return null;
}

// Free the queue slot (and let the server delete the GPU session) when the
// session subtree unmounts. StrictMode fake-unmounts every effect once right
// after mount, so only treat an unmount as real if no remount follows quickly.
function SessionBridge() {
  const { endSession } = useReactorQueue();
  const mountedRef = useRef(false);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      window.setTimeout(() => {
        if (!mountedRef.current) endSession();
      }, 100);
    };
  }, [endSession]);
  return null;
}

// Adopts the queue-minted session: the queue owns the lifecycle (created the
// session on claim, deletes it on end), the browser only attaches via
// connect({ sessionId, connectionId }).
function Session({
  sessionId,
  connectionId,
  onExit,
  onExpire,
}: {
  sessionId: string;
  connectionId: number;
  onExit: () => void;
  onExpire: () => void;
}) {
  const { getJwt } = useReactorQueue();
  // Stable identity: an inline object would be new on every queue re-render,
  // rebuilding the provider store and churning the connect flow.
  const connectOptions = useMemo(
    () => ({ sessionId, connectionId }),
    [sessionId, connectionId]
  );

  return (
    <OpenDreamerProvider
      apiUrl={COORDINATOR_URL}
      getJwt={getJwt}
      connectOptions={connectOptions}
    >
      <Connector />
      <SessionBridge />
      <TimedGameRoom onExit={onExit} onExpire={onExpire} />
    </OpenDreamerProvider>
  );
}

// Wraps GameRoom with the server-truth session clock: sessionEndsAt arrives with
// session_ready and is refreshed by time_warning. Ends a bit early so teardown
// beats the reaper, and also finishes if the session drops on its own or the
// queue expires the slot mid-play.
function TimedGameRoom({
  onExit,
  onExpire,
}: {
  onExit: () => void;
  onExpire: () => void;
}) {
  const { status } = useOpenDreamer();
  const { sessionEndsAt, phase: queuePhase } = useReactorQueue();

  const [remainingMs, setRemainingMs] = useState<number | null>(null);
  const finishedRef = useRef(false);
  const wasReadyRef = useRef(false);

  const finish = useCallback(() => {
    if (finishedRef.current) return;
    finishedRef.current = true;
    onExpire();
  }, [onExpire]);

  useEffect(() => {
    if (sessionEndsAt === null) return;
    const id = window.setInterval(() => {
      const left = sessionEndsAt - CLIENT_END_SLACK_MS - Date.now();
      setRemainingMs(left);
      if (left <= 0) finish();
    }, 250);
    return () => window.clearInterval(id);
  }, [sessionEndsAt, finish]);

  useEffect(() => {
    if (status === "ready") wasReadyRef.current = true;
    if (status === "disconnected" && wasReadyRef.current) finish();
  }, [status, finish]);

  useEffect(() => {
    if (queuePhase === "expired") finish();
  }, [queuePhase, finish]);

  return <GameRoom onExit={onExit} remainingMs={remainingMs} />;
}

// The gate: the provider starts idle (autoConnect={false}) so nobody occupies a
// queue slot until they ask to. The square opens on a "Join the line" button;
// clicking connects, then shows queue position, the admitted countdown + Enter,
// and finally the live game once the session starts.
function OpenDreamerGate() {
  const q = useReactorQueue();
  const [left, setLeft] = useState(false);
  const [ended, setEnded] = useState(false);

  const joinLine = useCallback(() => {
    setLeft(false);
    setEnded(false);
    // `rejoin()` is `connect()` under the hood, covering a fresh visit, a
    // dropped socket, and a slot that expired while sitting in the lobby.
    if (
      q.phase === "idle" ||
      q.phase === "disconnected" ||
      q.phase === "expired"
    ) {
      q.rejoin();
    }
  }, [q]);

  const inSession =
    !ended &&
    q.phase === "active" &&
    q.sessionId !== null &&
    q.connectionId !== null;

  if (inSession) {
    return (
      <Session
        sessionId={q.sessionId!}
        connectionId={q.connectionId!}
        onExit={() => {
          setLeft(false);
          setEnded(false);
          q.endSession();
        }}
        onExpire={() => {
          q.endSession();
          setEnded(true);
        }}
      />
    );
  }

  return (
    <QueuePanel
      q={q}
      left={left}
      ended={ended}
      onLeave={() => {
        q.leave();
        setLeft(true);
      }}
      onJoin={joinLine}
    />
  );
}

// ── Queue panel (shown inside the square until the session starts) ───────────
function QueuePanel({
  q,
  left,
  ended,
  onLeave,
  onJoin,
}: {
  q: ReturnType<typeof useReactorQueue>;
  left: boolean;
  ended: boolean;
  onLeave: () => void;
  onJoin: () => void;
}) {
  return (
    <div className="relative flex aspect-[3/4] w-full flex-col items-center justify-center gap-3 overflow-hidden bg-[#0a0a0a] px-6 text-center sm:aspect-video">
      <FlowLines />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 60% 45% at 50% 50%, rgba(52,211,153,0.08) 0%, transparent 70%)",
        }}
      />
      <div className="relative z-10 flex flex-col items-center gap-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.28em] text-emerald-300/70">
          Open Dreamer · Live demo
        </span>
        <QueueStatus q={q} left={left} ended={ended} onLeave={onLeave} onJoin={onJoin} />
      </div>
    </div>
  );
}

function QueueStatus({
  q,
  left,
  ended,
  onLeave,
  onJoin,
}: {
  q: ReturnType<typeof useReactorQueue>;
  left: boolean;
  ended: boolean;
  onLeave: () => void;
  onJoin: () => void;
}) {
  if (left) {
    return (
      <>
        <Eyebrow>You left the line</Eyebrow>
        <PhaseText>Your spot was given to the next person.</PhaseText>
        <GhostButton onClick={onJoin}>Rejoin the line</GhostButton>
      </>
    );
  }

  if (ended) {
    return (
      <>
        <Eyebrow>Session ended</Eyebrow>
        <PhaseText>
          Your time ran out and the slot went to the next person in line.
        </PhaseText>
        <EmeraldButton onClick={onJoin}>Play again</EmeraldButton>
      </>
    );
  }

  switch (q.phase) {
    case "queued":
      return (
        <>
          <Badge dot>In line</Badge>
          <div className="text-4xl font-medium tracking-tight text-white">
            #{q.position}
          </div>
          <p className="font-mono text-[12px] text-white/45">
            {q.total} waiting · {q.active} / {q.capacity} sessions live
          </p>
          <GhostButton onClick={onLeave}>Leave queue</GhostButton>
        </>
      );

    case "admitted":
      return (
        <>
          <Badge tone="emerald">Your turn</Badge>
          <PhaseText>Enter before the timer runs out.</PhaseText>
          <Countdown to={q.sessionEndsAt} />
          <EmeraldButton onClick={q.claim}>Enter the world</EmeraldButton>
        </>
      );

    case "expired":
      return (
        <>
          <Eyebrow>Time&apos;s up</Eyebrow>
          <PhaseText>Your slot timed out before the session started.</PhaseText>
          <GhostButton onClick={onJoin}>Rejoin the line</GhostButton>
        </>
      );

    case "rejected":
      return (
        <>
          <Eyebrow>Already open elsewhere</Eyebrow>
          <PhaseText>
            This demo is open in another tab. Close it and this one reconnects
            automatically.
          </PhaseText>
        </>
      );

    // "starting" normally means the claim is in flight, but if the server
    // reported an error the phase never advances — surface the reason instead
    // of spinning until the grace timer fires with a misleading "time's up".
    case "starting":
      if (q.reason) {
        return (
          <>
            <Eyebrow>Couldn&apos;t start your session</Eyebrow>
            <PhaseText>
              Something went wrong on our end ({q.reason}). Rejoin the line to
              try again.
            </PhaseText>
            <GhostButton onClick={onJoin}>Rejoin the line</GhostButton>
          </>
        );
      }
      return <Spinner />;

    case "idle":
    case "disconnected":
      return <EmeraldButton onClick={onJoin}>Join the line</EmeraldButton>;

    // connecting / active-awaiting-session_ready — wordless spinner.
    default:
      return <Spinner />;
  }
}

/**
 * The live Open Dreamer demo, embedded as a contained square inside the
 * article. Waits for the visitor to join the shared queue and, once admitted,
 * streams the playable Minecraft world with a Live⟷Dream toggle.
 */
export function EmbeddedDemo() {
  return (
    <div className="od-demo my-8 flex flex-col items-center">
      <div className="relative w-full max-w-[900px] overflow-hidden rounded-2xl border border-white/10 bg-black shadow-[0_20px_60px_-20px_rgba(0,0,0,0.6)]">
        <DemoContent />
      </div>
      <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.16em] text-black/40">
        Interactive demo · generated live on Reactor
      </p>
    </div>
  );
}

function DemoContent() {
  if (PARTYKIT_HOST_MISSING) {
    return (
      <div className="flex aspect-[3/4] w-full flex-col items-center justify-center gap-2 bg-[#0a0a0a] px-6 text-center sm:aspect-video">
        <Eyebrow>Demo unavailable</Eyebrow>
        <PhaseText>
          This deployment is missing its queue configuration. Set{" "}
          <code>NEXT_PUBLIC_OPEN_DREAMER_PARTYKIT_HOST</code> and redeploy.
        </PhaseText>
      </div>
    );
  }

  return (
    <ReactorQueueProvider host={PARTYKIT_HOST} autoConnect={false}>
      <OpenDreamerGate />
    </ReactorQueueProvider>
  );
}

// ── Countdown ────────────────────────────────────────────────────────────────
function Countdown({ to }: { to: number | null }) {
  const left = useCountdown(to);
  if (left === null || !isFinite(left)) return null;
  const warn = left <= 10;
  return (
    <div
      className={`font-mono text-3xl tabular-nums ${
        warn ? "text-red-400" : "text-white"
      }`}
    >
      {formatClock(left * 1000)}
    </div>
  );
}

function useCountdown(to: number | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (to == null) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [to]);
  if (to == null || !isFinite(to)) return null;
  return Math.max(0, Math.round((to - now) / 1000));
}

// ── Building blocks ──────────────────────────────────────────────────────────
function PhaseText({ children }: { children: React.ReactNode }) {
  return (
    <p className="max-w-xs text-sm leading-relaxed text-white/40">{children}</p>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[11px] uppercase tracking-tight text-white/50">
      {children}
    </span>
  );
}

function Badge({
  children,
  tone = "default",
  dot = false,
}: {
  children: React.ReactNode;
  tone?: "default" | "emerald";
  dot?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] ${
        tone === "emerald"
          ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-100"
          : "border-white/[0.18] text-white/60"
      }`}
    >
      {dot && (
        <span
          className="h-1.5 w-1.5 rounded-full bg-emerald-400"
          style={{ animation: "dreamxPulse 1.6s infinite" }}
        />
      )}
      {children}
    </span>
  );
}

function Spinner() {
  return (
    <span className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
  );
}

function EmeraldButton({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center justify-center rounded-md bg-emerald-400 px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-black/80 transition-[transform,filter] duration-200 hover:scale-[1.03] hover:brightness-110 active:scale-[0.97]"
    >
      {children}
    </button>
  );
}

function GhostButton({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center justify-center rounded-md border border-white/15 px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-white/50 transition-colors hover:border-white/30 hover:text-white/80"
    >
      {children}
    </button>
  );
}

// Decorative animated flow lines behind the queue panel.
function FlowLines() {
  const lines = [12, 28, 44, 60, 76, 90];
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      {lines.map((leftPct, i) => (
        <div
          key={i}
          className="absolute bottom-0 top-0 w-px"
          style={{ left: `${leftPct}%`, background: "rgba(52,211,153,0.05)" }}
        >
          <div
            className="absolute inset-x-0"
            style={{
              height: "40%",
              background:
                "linear-gradient(180deg, transparent, rgba(52,211,153,0.45) 50%, transparent)",
              animation: `bg-flow-v ${4.5 + i * 0.9}s linear infinite ${i * 0.7}s`,
            }}
          />
        </div>
      ))}
    </div>
  );
}
