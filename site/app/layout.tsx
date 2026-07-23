import type { Metadata } from "next";

import { asset } from "@/lib/site-config";
import "./globals.css";

export const metadata: Metadata = {
  title: "How to train a frontier-level world model",
  description:
    "How we trained and open-sourced a frontier-level world model — the lessons, failures, and fixes — with a live, playable demo running on Reactor.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {/* The Distill article's own stylesheet and KaTeX's CSS. Rendered as
            plain <link> tags so React 19 hoists them into <head>; loading the
            matching scripts happens after the article HTML is in the DOM (see
            components/Article.tsx). */}
        <link rel="stylesheet" href={asset("/website/style.css")} />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.css"
          integrity="sha384-zTROYFVGOfTw7JV7KUu8udsvW2fx4lWOsCEDqhBreBwlHI4ioVRtmIvEThzJHGET"
          crossOrigin="anonymous"
        />
        {children}
      </body>
    </html>
  );
}
