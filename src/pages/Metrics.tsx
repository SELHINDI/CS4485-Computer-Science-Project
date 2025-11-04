import { useState, useEffect } from 'react'
import { useAnalyzeCountry } from '../lib/hooks/useData'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from 'recharts'
import { DataTable } from '../app/components/DataTable'
import { Skeleton } from '../app/components/States'

type PerformanceMetric = { metric: string; value: number }

export default function Metrics() {
  const [from, setFrom] = useState(2018)
  const [to, setTo] = useState(2025)

  // Mutation hook for analysis
  const analyzeMutation = useAnalyzeCountry('USA', {
    start_year: from,
    end_year: to,
    prediction_years: 1
  })

  // Run analysis on mount and when years change
  useEffect(() => {
    analyzeMutation.mutate(undefined, {
      onSuccess: (res) => {
        console.log('Analyze response:', res)
      },
      onError: (err) => {
        console.error('Error analyzing:', err)
      }
    })
  }, [from, to])

  const isLoading = analyzeMutation.isPending
  const data = analyzeMutation.data?.data
  const perf = data?.model_performance

  console.log('Model performance data:', perf)

  // Transform model performance into chart-friendly format
  const barData: PerformanceMetric[] = perf
    ? Object.entries(perf).map(([metric, value]) => ({
        metric: metric.toUpperCase(),
        value: value as number
      }))
    : []

  return (
    <div className="space-y-4">
      {/* Year range inputs */}
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-sm" htmlFor="from">From</label>
          <input
            id="from"
            type="number"
            value={from}
            onChange={(e) => setFrom(Number(e.target.value))}
            className="px-2 py-1 rounded-md border w-24 bg-white dark:bg-neutral-900"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm" htmlFor="to">To</label>
          <input
            id="to"
            type="number"
            value={to}
            onChange={(e) => setTo(Number(e.target.value))}
            className="px-2 py-1 rounded-md border w-24 bg-white dark:bg-neutral-900"
          />
        </div>
      </div>

      {/* Model Performance Bar Chart */}
      {isLoading ? (
        <Skeleton className="h-64" />
      ) : (
        <div className="card p-4">
          <h3 className="text-lg font-semibold mb-2">Model Performance Metrics</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.3} />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Score" fill="#7bdcb5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Metrics Table */}
      {!isLoading && (
        <DataTable
          data={barData.map(b => ({
            metric: b.metric,
            value: b.value.toFixed(4)
          }))}
          columns={[
            { key: 'metric', header: 'Metric' },
            { key: 'value', header: 'Value' }
          ]}
          filename="metrics_summary.csv"
        />
      )}
    </div>
  )
}
