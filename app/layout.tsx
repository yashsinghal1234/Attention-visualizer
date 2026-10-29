import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UQURA NODES - AI Node Canvas",
  description: "Advanced Generative AI Node-based Canvas Workflow Interface",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-grid-pattern text-neutral-100 antialiased selection:bg-neutral-800 selection:text-white">
        {children}
      </body>
    </html>
  );
}

