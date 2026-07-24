import type { Metadata } from "next";
import Script from "next/script";

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
      <head>
        {/* Distill owns the article's core grid and custom elements. Loading it
            before Next hydrates prevents the unstyled article from painting
            while the much larger live-demo bundle is still downloading. */}
        <Script
          id="distill-template"
          src={asset("/website/scripts/distill.pub_template.v2.js")}
          strategy="beforeInteractive"
        />
      </head>
      <body>
        {/* The Distill article's own stylesheet and KaTeX's CSS. Rendered as
            plain <link> tags so React 19 hoists them into <head>; loading the
            KaTeX scripts happens after the article HTML is in the DOM (see
            components/Article.tsx). Distill itself is loaded above because its
            layout styles are render-critical. */}
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
