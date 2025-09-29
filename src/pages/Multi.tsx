import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useMulti, useSeries } from "../lib/hooks/useData"
import { ForecastChart } from "../app/components/ForecastChart"
import { DataTable } from "../app/components/DataTable"
import { Skeleton } from "../app/components/States"

export default function Multi() {
  const { country, horizons, asOfDate, unitMode } = useApp()
  const multi = useMulti(country, horizons, asOfDate)
  const series = useSeries(country, 2015, 2025)

  const data = (series.data?.series.gdp ?? []).map(d => ({ date: d.date, value: d.value }))
  const baseDate = data.at(-1)?.date
  if (multi.data) {
    Object.entries(multi.data.preds).forEach(([h, v]) => {
      data.push({ date: `${multi.data.as_of}+${h}`, predicted: v })
    })
  }

  const tableRows = multi.data ? Object.entries(multi.data.preds).map(([h, v]) => ({ horizon: h, gdp: v })) : []

  return (
    <div className="space-y-4">
      <Controls showHorizons />
      {series.isLoading || multi.isLoading ? (
        <Skeleton className="h-80" />
      ) : (
        <ForecastChart data={data} unitMode={unitMode} showLegend />
      )}
      <DataTable data={tableRows} columns={[{ key: 'horizon', header: 'Horizon' }, { key: 'gdp', header: 'Predicted GDP', render: (r) => Number(r.gdp).toLocaleString() }]} filename="multi_forecast.csv" />
    </div>
  )
}


