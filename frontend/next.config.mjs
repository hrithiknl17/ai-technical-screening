/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // STATIC_EXPORT=1 emits a plain static site into ./out, which the FastAPI app
  // serves from its own origin. Left off for `next dev`.
  output: process.env.STATIC_EXPORT ? "export" : undefined,
  images: { unoptimized: true },
  env: {
    // Same-origin by default; dev points at the separately running API.
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api",
  },
};

export default nextConfig;
