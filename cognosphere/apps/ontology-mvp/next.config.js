/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['@cognosphere/shared'],
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
}

module.exports = nextConfig