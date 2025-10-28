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

  type SeriesKey = 'level' | 'slope' | 'curvature' | 'inflation' | 'unemp' | 'caputil' | 'fedfunds'

  // Helper to extract series points from API response
  const mk = (key: SeriesKey): SeriesPoint[] =>
    (s.data?.series[key] ?? []).map((p: SeriesPoint) => ({ date: p.date, value: p.value }))

  const proxiesData = [
    { label: 'Level', data: mk('level') },
    { label: 'Slope', data: mk('slope') },
    { label: 'Curvature', data: mk('curvature') },
  ]

  const macroData = [
    { label: 'Inflation', data: mk('inflation') },
    { label: 'Unemployment', data: mk('unemp') },
    { label: 'Capacity Utilization', data: mk('caputil') },
    { label: 'Fed Funds', data: mk('fedfunds') },
  ]

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
