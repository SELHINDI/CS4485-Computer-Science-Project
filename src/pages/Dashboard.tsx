import { useEffect } from "react"
import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useCountryData, useAnalyzeCountry } from "../lib/hooks/useData"
import { MetricCard } from "../app/components/MetricCard"
import { ForecastChart } from "../app/components/ForecastChart"
import { Skeleton } from "../app/components/States"

// Type for each data point in the series chart
type SeriesPoint = { date: string; value: number; predicted?: number }

export default function Dashboard() {
  const { country, unitMode } = useApp()

  // Fetch historical GDP series
  const series = useCountryData(country, 2015, 2025)

  // Analyze country to get next quarter GDP
  const analyzeMutation = useAnalyzeCountry(country, {
    start_year: 2015,
    end_year: 2025,
    prediction_years: 1,
  })

  // Trigger analysis when country changes
  useEffect(() => {
    if (country) analyzeMutation.mutate()
  }, [country])

  const oneData = analyzeMutation.data

  // Prepare sparkline and chart data
  const spark = (series.data?.gdp.slice(-12) ?? []).map((p: SeriesPoint) => p.value)
  const chartData: SeriesPoint[] = (series.data?.gdp ?? []).map((p: SeriesPoint) => ({
    date: p.date,
    value: p.value,
  }))

  const lastDate = chartData.at(-1)?.date

  // Append predicted value from mutation
  if (oneData && lastDate) {
    const predictedValue = oneData.predictions?.[0] ?? null
    if (predictedValue !== null) {
      chartData.push({
        date: oneData.analysis_metadata.timestamp,
        value: predictedValue,
        predicted: predictedValue,
      })
    }
  }

  return (
    <div className="space-y-4">
      <Controls />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Metric card for next quarter GDP */}
        <div>
          {analyzeMutation.status === "pending" ? (
            <Skeleton className="h-28" />
          ) : analyzeMutation.status === "error" ? (
            <div className="text-red-500">Failed to load prediction</div>
          ) : (
            <MetricCard
              title="Next Quarter GDP"
              value={oneData?.predictions?.[0]?.toLocaleString() ?? "--"}
              delta={Math.random() * 2 - 1}
              spark={spark}
            />
          )}
        </div>

        {/* Forecast chart */}
        <div className="lg:col-span-2">
          {series.isLoading ? (
            <Skeleton className="h-80" />
          ) : (
            <ForecastChart data={chartData} unitMode={unitMode} />
          )}
        </div>
      </div>
    </div>
  )
}
