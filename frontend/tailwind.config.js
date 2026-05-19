/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#FF6B35',
        secondary: '#1A1A2E',
        surface: '#1E1E2E',
        'surface-elevated': '#252540',
        accent: {
          gold: '#FFD700',
          red: '#E94560',
          teal: '#4ECDC4',
        }
      },
      fontFamily: {
        display: ['Clash Display', 'sans-serif'],
        body: ['DM Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-up': 'fadeUp 0.5s ease forwards',
        'count-up': 'countUp 1.5s ease forwards',
        'pulse-glow': 'pulseGlow 2s infinite',
        'typing': 'typing 1s steps(3) infinite',
      }
    }
  },
  plugins: [],
}
