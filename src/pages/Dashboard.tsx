import { Controls } from "../app/components/Controls"
import { useApp } from "../app/context/AppContext"
import { useOneStep, useSeries } from "../lib/hooks/useData"
import { MetricCard } from "../app/components/MetricCard"
import { ForecastChart } from "../app/components/ForecastChart"
import { Skeleton } from "../app/components/States"

export default function Dashboard() {
  const { country, asOfDate, unitMode } = useApp()
  const one = useOneStep(country, asOfDate)
  const series = useSeries(country, 2015, 2025)

  const spark = series.data?.series.gdp.slice(-12).map(p => p.value) ?? []
  const chartData = (series.data?.series.gdp ?? []).map(p => ({ date: p.date, value: p.value }))
  const lastDate = chartData.at(-1)?.date
  if (one.data && lastDate) {
    chartData.push({ date: one.data.as_of, predicted: one.data.gdp_pred })
  }

  return (
    <div className="space-y-4">
      <Controls />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div>
          {one.isLoading ? (
            <Skeleton className="h-28" />
          ) : (
            <MetricCard title="Next Quarter GDP" value={one.data ? one.data.gdp_pred.toLocaleString() : '--'} delta={Math.random()*2-1} spark={spark} />
          )}
        </div>
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


