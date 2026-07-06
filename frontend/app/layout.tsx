import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenDreamer | Interactive World Model",
  description: "A real-time, video-generative Minecraft world model. Step inside and play.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
