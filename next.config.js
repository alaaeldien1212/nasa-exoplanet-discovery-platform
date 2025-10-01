/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['python-shell']
  },
  webpack: (config) => {
    config.externals.push({
      'python-shell': 'commonjs python-shell'
    });
    return config;
  },
  env: {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  }
}

module.exports = nextConfig
