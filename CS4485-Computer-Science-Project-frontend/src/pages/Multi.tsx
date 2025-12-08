import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useAnalyzeCountry } from "../lib/hooks/useData"
import { ForecastChart } from "../app/components/ForecastChart"
import { DataTable } from "../app/components/DataTable"
import { Skeleton } from "../app/components/States"
import { useMemo, useEffect } from "react"

interface ActualDataPoint {
  date: string
  value: number
}

interface PredictedDataPoint {
  date: string
  predicted: number
}

export default function Multi() {
  const { country, horizons, unitMode } = useApp()

  const options = useMemo(
    () => ({
      // Use the maximum selected horizon as the prediction range, defaulting to 5 if empty
      prediction_years: horizons.length > 0 ? Math.max(...horizons) : 5,
      start_year: 2015,
      end_year: 2030,
      model_type: "random_forest",
    }),
    [horizons]
  )

  const multi = useAnalyzeCountry(country, options)

  useEffect(() => {
    // Always trigger analysis when prerequisites change
    multi.mutate()
  }, [country, horizons])

  useEffect(() => {
    if (multi.data) {
      console.log("Analyze API Response:", multi.data)
    }
  }, [multi.data])

  // ACTUAL DATA
  const actualData: ActualDataPoint[] = useMemo(() => {
    const historical = multi.data?.data?.historical_data?.gdp ?? []
    let data: ActualDataPoint[] = historical
      .map((d: any) => ({
        date: String(d.year),
        value: d.value,
      }))
      .sort((a: ActualDataPoint, b: ActualDataPoint) => Number(a.date) - Number(b.date))

    if (unitMode === "pct_qoq") {
      data = data.map((point, i, arr) => {
        if (i === 0) return { ...point, value: 0 }
        const prev = arr[i - 1].value
        const pctChange = prev ? ((point.value - prev) / prev) * 100 : 0
        return { ...point, value: pctChange }
      })
    }

    return data
  }, [multi.data, unitMode])

  // PREDICTED DATA
  const predictedData: PredictedDataPoint[] = useMemo(() => {
    const predictions = multi.data?.data?.predictions ?? []
    let data: PredictedDataPoint[] = predictions
      .map((p: any) => ({
        date: String(p.year),
        predicted: p.predicted_gdp,
      }))
      .sort((a: PredictedDataPoint, b: PredictedDataPoint) => Number(a.date) - Number(b.date))


    if (unitMode === "pct_qoq") {
      data = data.map((point, i, arr) => {
        if (i === 0) return { ...point, predicted: 0 }
        const prev = arr[i - 1].predicted
        const pctChange = prev ? ((point.predicted - prev) / prev) * 100 : 0
        return { ...point, predicted: pctChange }
      })
    }

    return data
  }, [multi.data, unitMode])

  const tableRows = useMemo(() => {
    const predictions = multi.data?.data?.predictions ?? []
    // Filter table to show only selected horizons? Or show all up to max?
    // Let's show all for context, or maybe filter if specific horizons are key.
    // For now, showing all generated predictions is safer.
    return predictions.map((p: any) => ({
      horizon: p.year,
      gdp: p.predicted_gdp,
    }))
  }, [multi.data])

  const isLoading = multi.status === "pending"
  const isError = multi.status === "error"

  return (
    <div className="space-y-4">
      <Controls showHorizons />

      {isError && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-md border border-red-200 dark:border-red-800">
          Failed to load analysis data. Please try again or select fewer horizons.
        </div>
      )}

      {isLoading ? (
        <div className="flex flex-wrap gap-4">
          <Skeleton className="h-72 flex-1 min-w-[300px]" />
          <Skeleton className="h-72 flex-1 min-w-[300px]" />
        </div>
      ) : (
        <div className="flex flex-wrap gap-4">
          {/* Actual GDP chart */}
          <div className="flex-1 min-w-[300px]">
            {/* If we have no data but no error, maybe it's just empty? Render chart anyway so axes show up */}
            <ForecastChart data={actualData} unitMode={unitMode} showLegend />
          </div>

          {/* Predicted GDP chart */}
          <div className="flex-1 min-w-[300px]">
            <ForecastChart
              data={predictedData}
              unitMode={unitMode}
              showLegend
              lineDataKey="predicted"
              lineName="Predicted"
              stroke="#7bdcb5"
            />
          </div>
        </div>
      )}

      <DataTable
        data={tableRows}
        columns={[
          { key: "horizon", header: "Horizon" },
          {
            key: "gdp",
            header: "Predicted GDP",
            render: (r) => Number(r.gdp).toLocaleString(),
          },
        ]}
        filename="multi_forecast.csv"
      />
    </div>
  )
}
