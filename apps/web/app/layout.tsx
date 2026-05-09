import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MIRA | Multimodal Mental Health Companion AI",
  description: "A privacy-first, multimodal mental health AI companion.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0 }}>
        {children}
      </body>
    </html>
  );
}
