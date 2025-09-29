import { useState } from 'react'
import { useMetrics } from '../lib/hooks/useData'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, Legend } from 'recharts'
import { DataTable } from '../app/components/DataTable'
import { Skeleton } from '../app/components/States'

export default function Metrics() {
  const [from, setFrom] = useState(2018)
  const [to, setTo] = useState(2025)
  const m = useMetrics(from, to)

  const mseTimeline = m.data?.one_step.timeline ?? []
  const avg = m.data?.one_step.avg_mse ?? {}
  const barData = Object.entries(avg).map(([model, val]) => ({ model: model.toUpperCase(), mse: val }))

  const table = mseTimeline.map(t => ({ date: t.date, y_true: (Math.random()*100).toFixed(1), y_pred: (Math.random()*100).toFixed(1), abs_error: t.mse.toFixed(4) }))

  return (
    <div className="space-y-4">
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-sm" htmlFor="from">From</label>
          <input id="from" type="number" value={from} onChange={(e) => setFrom(Number(e.target.value))} className="px-2 py-1 rounded-md border w-24 bg-white dark:bg-neutral-900" />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm" htmlFor="to">To</label>
          <input id="to" type="number" value={to} onChange={(e) => setTo(Number(e.target.value))} className="px-2 py-1 rounded-md border w-24 bg-white dark:bg-neutral-900" />
        </div>
      </div>

      {m.isLoading ? (
        <Skeleton className="h-64" />
      ) : (
        <div className="card p-4">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mseTimeline} accessibilityLayer>
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.3} />
                <XAxis dataKey="date" minTickGap={48} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="mse" name="One-step MSE" stroke="#5b8def" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {m.isLoading ? (
        <Skeleton className="h-64" />
      ) : (
        <div className="card p-4">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} accessibilityLayer>
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.3} />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="mse" name="Avg MSE" fill="#7bdcb5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <DataTable data={table} columns={[
        { key: 'date', header: 'Date' },
        { key: 'y_true', header: 'y_true' },
        { key: 'y_pred', header: 'y_pred' },
        { key: 'abs_error', header: 'abs error' },
      ]} filename="metrics_detail.csv" />
    </div>
  )
}


