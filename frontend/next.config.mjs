/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The dashboard is a pure API client: it never reads the filesystem or a
  // database at request time, so there is nothing to render on the server that
  // the browser cannot render itself. Keeping typed-route checking on catches
  // broken <Link href> targets at build time.
  typedRoutes: true,
};

export default nextConfig;
