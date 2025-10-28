import { useState } from 'react'
import { useApp } from '../app/context/AppContext'
import { useCountryData } from '../lib/hooks/useData'
import { ForecastChart } from '../app/components/ForecastChart'
import { Skeleton } from '../app/components/States'

// Type for each data point in the series
type SeriesPoint = { date: string; value: number }

export default function Explorer() {
  const [tab, setTab] = useState<'proxies' | 'macro'>('proxies')
  const { country, unitMode } = useApp()

  // Fetch historical series data
  const s = useCountryData(country, 2015, 2025)
  console.log("full hook results:", s)
  console.log("Series data from API:", s.data?.series)
  console.log("Full API response:", s.data?.data)

  type SeriesKey = 'exports' | 'imports' | 'gdp' | 'gdp_growth' | 'gdp_per_capita' | 'inflation' | 'unemployment' | 'population'

  // Helper to extract series points from API response
  const mk = (key: SeriesKey): SeriesPoint[] =>
    s.data?.data?.[key]?.map((p: any) => ({
      date: p.date ?? p.year ?? "Unknown",
      value: p.value ?? p.gdp ?? p.level ?? 0,
    })) ?? [];
  

    const proxiesData = [
      { label: 'Exports', data: mk('exports') },
      { label: 'Imports', data: mk('imports') },
      { label: 'GDP', data: mk('gdp') },
    ]
    
    const macroData = [
      { label: 'GDP Growth', data: mk('gdp_growth') },
      { label: 'GDP per Capita', data: mk('gdp_per_capita') },
      { label: 'Inflation', data: mk('inflation') },
      { label: 'Unemployment', data: mk('unemployment') },
    ]
  console.log("Proxies chart data:", proxiesData)
console.log("Macro chart data:", macroData)

  return (
    <div className="space-y-4">
      {/* Tab buttons */}
      <div className="card p-2 flex gap-2">
        <button
          className={`px-3 py-1 rounded-md border ${
            tab === 'proxies' ? 'bg-primary text-white border-primary' : ''
          }`}
          onClick={() => setTab('proxies')}
        >
          Yield Curve Proxies
        </button>
        <button
          className={`px-3 py-1 rounded-md border ${
            tab === 'macro' ? 'bg-primary text-white border-primary' : ''
          }`}
          onClick={() => setTab('macro')}
        >
          Macro
        </button>
      </div>

      {/* Charts */}
      {s.isLoading ? (
        <Skeleton className="h-40" />
      ) : tab === 'proxies' ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {proxiesData.map((sd) => (
            <ForecastChart key={sd.label} data={sd.data} unitMode={unitMode} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {macroData.map((sd) => (
            <ForecastChart key={sd.label} data={sd.data} unitMode={unitMode} />
          ))}
        </div>
      )}
    </div>
  )
}
