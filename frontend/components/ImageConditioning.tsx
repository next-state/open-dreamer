"use client";

import { useCallback, useRef, useState } from "react";
import { useReactor } from "@reactor-team/js-sdk";

// Built-in Minecraft scenes shipped under public/gallery. Selecting one seeds
// the world model from that image; users can also upload their own.
const GALLERY = [
  { name: "Overworld", src: "/gallery/overworld.png" },
  { name: "Cave", src: "/gallery/cave.png" },
  { name: "Nether", src: "/gallery/nether.png" },
  { name: "The End", src: "/gallery/end.png" },
];

interface ImageConditioningProps {
  className?: string;
}

// Seed the world model from an image: upload the file (or a gallery image) via
// the SDK, then send `set_conditioning_image` with the returned FileRef.
export function ImageConditioning({ className }: ImageConditioningProps) {
  const { uploadFile, sendCommand, status } = useReactor((state) => ({
    uploadFile: state.uploadFile,
    sendCommand: state.sendCommand,
    status: state.status,
  }));

  const isReady = status === "ready";
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const seed = useCallback(
    async (file: File | Blob, name: string) => {
      if (!isReady || busy) return;
      setBusy(true);
      try {
        const ref = await uploadFile(file, { name });
        await sendCommand("set_conditioning_image", { image: ref });
        setOpen(false);
      } catch (err) {
        console.error("[conditioning] upload failed", err);
      } finally {
        setBusy(false);
      }
    },
    [isReady, busy, uploadFile, sendCommand]
  );

  const onFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) seed(file, file.name);
    event.target.value = "";
  };

  const onGallery = async (item: (typeof GALLERY)[number]) => {
    const resp = await fetch(item.src);
    const blob = await resp.blob();
    seed(blob, `${item.name.toLowerCase().replace(/\s+/g, "-")}.png`);
  };

  return (
    <div className={`relative ${className || ""}`}>
      <button
        onClick={() => isReady && setOpen((o) => !o)}
        disabled={!isReady}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors duration-200 ${
          isReady
            ? "text-white hover:bg-emerald-400/15 cursor-pointer"
            : "text-white/30 opacity-60 cursor-not-allowed"
        }`}
      >
        <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <rect x="1.5" y="2.5" width="13" height="11" rx="1.5" />
          <path d="M1.5 11l3.5-3.5 3 3 2.5-2.5 4 4" />
          <circle cx="5.5" cy="6" r="1" />
        </svg>
        <span>Seed image</span>
      </button>

      {open && (
        <div className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-72 glass-strong rounded-2xl p-3 flex flex-col gap-2">
          <div className="eyebrow px-1">Start from a scene</div>
          <div className="grid grid-cols-2 gap-2">
            {GALLERY.map((item) => (
              <button
                key={item.name}
                onClick={() => onGallery(item)}
                disabled={busy}
                className="group relative overflow-hidden rounded-lg border border-white/10 hover:border-emerald-400/50 transition-colors disabled:opacity-50"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.src} alt={item.name} className="h-16 w-full object-cover" />
                <span className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 px-1.5 py-0.5 text-left text-[10px] font-medium text-white">
                  {item.name}
                </span>
              </button>
            ))}
          </div>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy}
            className="mt-0.5 rounded-lg border border-dashed border-white/20 px-3 py-2 text-xs text-white/70 hover:border-emerald-400/50 hover:text-white transition-colors disabled:opacity-50"
          >
            {busy ? "Uploading…" : "Upload your own image…"}
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
          <p className="px-1 text-[10px] leading-snug text-white/40">
            Minecraft-like images work best; others may look glitchy.
          </p>
        </div>
      )}
    </div>
  );
}
