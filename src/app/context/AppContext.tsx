import { createContext, useContext, useMemo, useState, ReactNode } from 'react'

export type UnitMode = 'level' | 'pct_qoq'

type AppState = {
  country: string
  setCountry: (c: string) => void
  asOfDate: string | null
  setAsOfDate: (d: string | null) => void
  unitMode: UnitMode
  setUnitMode: (m: UnitMode) => void
  horizons: number[]
  setHorizons: (h: number[]) => void
  theme: 'light' | 'dark'
  toggleTheme: () => void
}

const Ctx = createContext<AppState | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [country, setCountry] = useState<string>('US')
  const [asOfDate, setAsOfDate] = useState<string | null>(null)
  const [unitMode, setUnitMode] = useState<UnitMode>('level')
  const [horizons, setHorizons] = useState<number[]>([2, 4, 8, 12])
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'light'
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
  })

  const toggleTheme = () => {
    setTheme((t) => {
      const next = t === 'light' ? 'dark' : 'light'
      const root = document.documentElement
      root.classList.toggle('dark', next === 'dark')
      return next
    })
  }

  const value = useMemo<AppState>(() => ({
    country, setCountry,
    asOfDate, setAsOfDate,
    unitMode, setUnitMode,
    horizons, setHorizons,
    theme, toggleTheme,
  }), [country, asOfDate, unitMode, horizons, theme])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}


