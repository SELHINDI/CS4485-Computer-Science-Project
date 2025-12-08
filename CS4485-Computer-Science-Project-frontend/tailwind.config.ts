import type { Config } from "tailwindcss"

export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          light: "#f7f8fc",
          dark: "#0b0f19",
        },
        primary: "#5b8def",
        secondary: "#7bdcb5",
        neutral: {
          900: "#1b2333",
          800: "#2a3347",
          200: "#e5e7eb",
        },
      },
      borderRadius: {
        xl: "0.75rem",
      },
      boxShadow: {
        soft: "0 10px 25px -10px rgba(0,0,0,0.25)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  darkMode: "class",
  plugins: [],
} satisfies Config


