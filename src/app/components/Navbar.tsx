import { useApp } from "../context/AppContext"
import { Sun, Moon } from "lucide-react"
import { Link } from "react-router-dom"

export function Navbar() {
  const { country, setCountry, theme, toggleTheme } = useApp()
  return (
    <header className="sticky top-0 z-40 border-b border-neutral-200/70 dark:border-neutral-800 bg-white/60 dark:bg-neutral-900/60 backdrop-blur">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-4">
        <Link to="/" className="font-semibold">GDP Predictor</Link>
        <nav className="ml-auto flex items-center gap-6">
          <Link to="/" className="text-sm hover:text-primary">Dashboard</Link>
          <Link to="/multi" className="text-sm hover:text-primary">Multi-Step</Link>
          <Link to="/explorer" className="text-sm hover:text-primary">Explorer</Link>
          <Link to="/metrics" className="text-sm hover:text-primary">Metrics</Link>
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <label className="text-sm" htmlFor="country">Country</label>
          <select
            id="country"
            aria-label="Select country"
            className="px-2 py-1 rounded-md border bg-white dark:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-primary"
            value={country}
            onChange={(e) => setCountry(e.target.value)}
          >
            <option value="US">US</option>
            <option value="CA">CA</option>
            <option value="GB">UK</option>
            <option value="EU">EU</option>
          </select>
          <button
            aria-label="Toggle theme"
            className="p-2 rounded-md border bg-white dark:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-primary"
            onClick={toggleTheme}
          >
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
        </div>
      </div>
    </header>
  )
}


