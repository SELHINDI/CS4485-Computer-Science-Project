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
    end_year: 2030,
    model_type: "random_forest",
  }), [horizons])

  // Mutation hook
  const multi = useAnalyzeCountry(country, options)

  // Trigger analysis when country or horizons change
  useEffect(() => {
    if (horizons.length > 0) {
      multi.mutate()
    }
  }, [country, horizons])

  useEffect(() => {
    if (multi.data) {
      console.log("Analyze API Response:", multi.data)
    }
  }, [multi.data])

  // Separate arrays for actual and predicted data
  const actualData = useMemo(() => {
    const historical = multi.data?.data?.historical_data?.gdp ?? []
    return historical
      .map((d: any) => ({
        date: String(d.year),
        value: d.value
      }))
      .sort((a: { date: string }, b: { date: string }) => Number(a.date) - Number(b.date))
  }, [multi.data])
  
  const predictedData = useMemo(() => {
    const predictions = multi.data?.data?.predictions ?? []
    return predictions
      .map((p: any) => ({
        date: String(p.year),
        predicted: p.predicted_gdp
      }))
      .sort((a: { date: string }, b: { date: string }) => Number(a.date) - Number(b.date))
  }, [multi.data])

  // Table rows for predicted GDP
  const tableRows = useMemo(() => {
    const predictions = multi.data?.data?.predictions ?? []
    return predictions.map((p: any) => ({
      horizon: p.year,
      gdp: p.predicted_gdp
    }))
  }, [multi.data])

  console.log('Actual Data:', actualData)
  console.log('Predicted Data:', predictedData)

  const isLoading = multi.status === "pending"

  return (
    <div className="space-y-4">
      <Controls showHorizons />

      {isLoading ? (
        <div className="flex flex-wrap gap-4">
          <Skeleton className="h-72 flex-1 min-w-[300px]" />
          <Skeleton className="h-72 flex-1 min-w-[300px]" />
        </div>
      ) : (
        <div className="flex flex-wrap gap-4">
          {/* Actual GDP chart */}
          <div className="flex-1 min-w-[300px]">
            <ForecastChart data={actualData} unitMode={unitMode} showLegend />
          </div>

          {/* Predicted GDP chart */}
          <div className="flex-1 min-w-[300px]">
            <ForecastChart data={predictedData} unitMode={unitMode} showLegend lineDataKey="predicted" 
  lineName="Predicted" 
  stroke="#7bdcb5"/>
          </div>
        </div>
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
