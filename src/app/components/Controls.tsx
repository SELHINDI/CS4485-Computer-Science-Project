import { useApp } from "../context/AppContext"
import type { UnitMode } from "../context/AppContext"

export function Controls({ showHorizons = false }: { showHorizons?: boolean }) {
  const { asOfDate, setAsOfDate, unitMode, setUnitMode, horizons, setHorizons } = useApp()

  const toggleHorizon = (h: number) => {
    setHorizons(horizons.includes(h) ? horizons.filter(x => x !== h) : [...horizons, h])
  }

  return (
    <div className="card p-4 flex flex-wrap items-center gap-3" role="group" aria-label="Controls">
      <div className="flex items-center gap-2">
        <label className="text-sm" htmlFor="as-of">As of</label>
        <input
          id="as-of"
          type="date"
          value={asOfDate ?? ''}
          onChange={(e) => setAsOfDate(e.target.value || null)}
          className="px-2 py-1 rounded-md border bg-white dark:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm">Unit</span>
        {(['level','pct_qoq'] as UnitMode[]).map(m => (
          <button
            key={m}
            aria-pressed={unitMode === m}
            onClick={() => setUnitMode(m)}
            className={`px-3 py-1 rounded-md border focus:outline-none focus:ring-2 focus:ring-primary ${unitMode===m? 'bg-primary text-white border-primary':'bg-white dark:bg-neutral-900'}`}
          >{m === 'level' ? 'Level' : '% QoQ'}</button>
        ))}
      </div>

      {showHorizons && (
        <div className="flex items-center gap-2">
          <span className="text-sm">Horizons</span>
          {[2,3,4,8,12].map(h => (
            <button
              key={h}
              aria-pressed={horizons.includes(h)}
              onClick={() => toggleHorizon(h)}
              className={`px-3 py-1 rounded-md border focus:outline-none focus:ring-2 focus:ring-primary ${horizons.includes(h)? 'bg-secondary text-neutral-900 border-secondary':'bg-white dark:bg-neutral-900'}`}
            >{h}</button>
          ))}
        </div>
      )}
    </div>
  )
}


