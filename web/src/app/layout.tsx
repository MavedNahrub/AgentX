import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentX v2.0 | Autonomous NSE Fund & TradingView Terminal",
  description:
    "Self-Calibrating Future-Basis Swing Ranker, Smart Loss-Recovery Vault & 375-Minute NSE Broker Workstation",
};

const FOUC_SCHEME_SCRIPT = `
try {
  var m = document.querySelector('meta[name="color-scheme"]');
  if (m) {
    m.content = localStorage.getItem("color-scheme") ?? "light dark";
  }
} catch (e) {}
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="light dark" />
        <script dangerouslySetInnerHTML={{ __html: FOUC_SCHEME_SCRIPT }} />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
