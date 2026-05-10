import { ClerkProvider } from '@clerk/nextjs';
import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from '@/components/theme-provider';

export const metadata: Metadata = {
  title: "MIRA | Premium Mental Health AI",
  description: "A luxury, privacy-first multimodal mental health companion.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider>
      <html lang="en" suppressHydrationWarning>
        <head>
          <link rel="preconnect" href="https://fonts.googleapis.com" />
          <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        </head>
        <body className="antialiased selection:bg-accent/30">
          <ThemeProvider 
            attribute="data-theme" 
            defaultTheme="midnight" 
            enableSystem={false}
          >
            {children}
          </ThemeProvider>
        </body>
      </html>
    </ClerkProvider>
  );
}
