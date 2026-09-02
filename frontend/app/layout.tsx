import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Grounded — RAG technical screening",
  description:
    "Resume-aware technical interviews generated from a role-specific textbook corpus.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <div className="shell">
          <header className="topnav">
            <div className="brand">
              <span className="brand-mark">G</span>
              <span>Grounded</span>
              <span className="muted" style={{ fontSize: 14, fontWeight: 400 }}>
                / technical screening
              </span>
            </div>
            <div className="nav-meta">
              <span>Retrieval-augmented interviewing</span>
            </div>
          </header>
          {children}
          <footer className="footer">
            Every question on this site is generated from passages retrieved out of the
            role&apos;s textbook corpus — expand “Why this question” to see the exact source.
          </footer>
        </div>
      </body>
    </html>
  );
}
