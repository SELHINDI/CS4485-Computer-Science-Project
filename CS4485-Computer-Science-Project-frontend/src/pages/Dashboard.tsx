import { useEffect } from "react"
import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useCountryData, useAnalyzeCountry } from "../lib/hooks/useData"
import { MetricCard } from "../app/components/MetricCard"
import { ForecastChart } from "../app/components/ForecastChart"
import { Skeleton } from "../app/components/States"
import type { AnalyzeResponse } from "../lib/api/type"

type SeriesPoint = { date: string; value: number; predicted?: number }
type AnalysisData = NonNullable<AnalyzeResponse["data"]>;

export default function Dashboard() {
  const { country, unitMode } = useApp()

  // Fetch historical GDP data
  const { data: series, isPending, isError } = useCountryData(country, 2015, 2025)
  
  // Trigger analysis when country changes
  const analyzeMutation = useAnalyzeCountry(country, {
    start_year: 2015,
    end_year: 2025,
    prediction_years: 1,
  })

  useEffect(() => {
    if (country) analyzeMutation.mutate()
  }, [country])

  if (isPending) return <p>Loading data...</p>
  if (isError) return <p>Failed to load data.</p>

  // Parse analysis API response
  const rawData = analyzeMutation.data
  const oneData: AnalysisData | undefined =
    (rawData && "data" in rawData ? rawData.data : rawData) as AnalysisData | undefined

  // Build base chart data
  let chartData: SeriesPoint[] = (series.data?.gdp ?? []).map((p: any) => ({
    date: p.year.toString(),
    value: p.value ?? p.gdp,
  }))

  // Sort in ascending order (smallest year → largest)
  chartData.sort((a: SeriesPoint, b: SeriesPoint) => Number(a.date) - Number(b.date))

  // If user selected %QoQ, compute percentage change
  if (unitMode === "pct_qoq") {
    chartData = chartData.map((point, i, arr) => {
      if (i === 0) return { ...point, value: 0 } // baseline
      const prev = arr[i - 1].value
      const pctChange = prev ? ((point.value - prev) / prev) * 100 : 0
      return { ...point, value: pctChange }
    })
  }

  // Append predicted GDP (from backend)
  const predictedValue =
    oneData?.predictions?.next_quarter_gdp ??
    oneData?.predictions?.predicted_gdp ??
    null

  const lastDate = chartData.at(-1)?.date

  if (oneData && predictedValue !== null && lastDate) {
    chartData.push({
      date: oneData.analysis_metadata.timestamp,
      value: predictedValue,
      predicted: predictedValue,
    })
  }

  // Sparkline data (for MetricCard)
  const spark = (series.data?.gdp ? series.data.gdp.slice(-12) : []).map(
    (p: SeriesPoint) => p.value
  )
  console.log("Series data:", series.data)

  return (
    <div className="space-y-4">
      <Controls />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Metric card */}
        <div>
          {analyzeMutation.status === "pending" ? (
            <Skeleton className="h-28" />
          ) : analyzeMutation.status === "error" ? (
            <div className="text-red-500">Failed to load prediction</div>
          ) : (
            <MetricCard
              title="Next Quarter GDP"
              value={
                predictedValue !== null
                  ? predictedValue.toLocaleString()
                  : "--"
              }
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
