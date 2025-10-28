import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useAnalyzeCountry } from "../lib/hooks/useData"
import { ForecastChart } from "../app/components/ForecastChart"
import { DataTable } from "../app/components/DataTable"
import { Skeleton } from "../app/components/States"
import { useMemo, useEffect } from "react"

export default function Multi() {
  const { country, horizons, unitMode } = useApp()

  // Prepare options for analysis
  const options = useMemo(() => ({
    prediction_years: horizons.length,
    start_year: 2015,
    end_year: 2025,
    model_type: "random_forest", // default model
  }), [horizons])

  // Mutation hook
  const multi = useAnalyzeCountry(country, options)

  // Trigger analysis when country or horizons change
  useEffect(() => {
    if (horizons.length > 0) {
      multi.mutate()
    }
  }, [country, horizons])

  // Historical series data
  const seriesData = useMemo(() => {
    const historical = multi.data?.historical_data?.gdp ?? []
    return historical.map((d: any) => ({ date: d.date, value: d.value }))
  }, [multi.data])

  // Add predicted points for chart
  const chartData = useMemo(() => {
    const data = [...seriesData]
    if (multi.data?.predictions) {
      Object.entries(multi.data.predictions).forEach(([h, v]) => {
        data.push({ date: `${multi.data.analysis_metadata?.end_year}+${h}`, predicted: v })
      })
    }
    return data
  }, [seriesData, multi.data])

  // Table rows for predicted GDP
  const tableRows = useMemo(() => {
    if (!multi.data?.predictions) return []
    return Object.entries(multi.data.predictions).map(([h, v]) => ({
      horizon: h,
      gdp: v
    }))
  }, [multi.data])

  // Type-safe loading flag
  const isLoading = multi.status === "pending"

  return (
    <div className="space-y-4">
      <Controls showHorizons />

      {isLoading ? (
        <Skeleton className="h-80" />
      ) : (
        <ForecastChart data={chartData} unitMode={unitMode} showLegend />
      )}

      <DataTable
        data={tableRows}
        columns={[
          { key: 'horizon', header: 'Horizon' },
          {
            key: 'gdp',
            header: 'Predicted GDP',
            render: (r) => Number(r.gdp).toLocaleString()
          }
        ]}
        filename="multi_forecast.csv"
      />
    </div>
  )
}
