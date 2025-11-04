import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import { useRef } from 'react'
import { exportNodeToPng } from '../../lib/export/exportImage'

type SeriesPoint = { date: string; value?: number; predicted?: number }

type Props = {
  data: SeriesPoint[]
  unitMode: 'level' | 'pct_qoq'
  showLegend?: boolean
  lineDataKey?: string
  lineName?: string
  stroke?: string
}

export function ForecastChart({ data, unitMode, showLegend }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  // Check which lines exist in the data
  const hasActual = data.some(d => d.value !== undefined)
  const hasPredicted = data.some(d => d.predicted !== undefined)

  return (
    <div className="card p-4" role="group" aria-label="Forecast chart">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm text-neutral-600 dark:text-neutral-300">Chart</div>
        <button className="px-2 py-1 rounded-md border text-sm" onClick={() => ref.current && exportNodeToPng(ref.current, 'chart.png')}>Export PNG</button>
      </div>
      <div ref={ref} className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} accessibilityLayer aria-label="GDP with predictions">
            <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.3} />
            <XAxis dataKey="date" minTickGap={48} />
            <YAxis tickFormatter={(v) => unitMode === 'pct_qoq' ? `${v}%` : `${v}`} />
            <Tooltip formatter={(v) => typeof v === 'number' ? (unitMode === 'pct_qoq' ? `${v.toFixed(2)}%` : v.toLocaleString()) : v} />
            {showLegend && <Legend />}

            {/* Render lines only if data exists */}
            {hasActual && <Line type="monotone" dataKey="value" name="Actual" stroke="#5b8def" dot={false} strokeWidth={2} />}
            {hasPredicted && <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#7bdcb5" strokeDasharray="4 4" dot={{ r: 3 }} strokeWidth={2} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
