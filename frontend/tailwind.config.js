/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // High-end tailored color scheme
        bias: {
          left: '#3b82f6',    // Blue
          center: '#10b981',  // Emerald/Green
          right: '#ef4444',   // Red
          score: '#8b5cf6'    // Violet
        },
        darkBg: '#0f172a',    // Slate 900
        darkCard: '#1e293b'   // Slate 800
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
