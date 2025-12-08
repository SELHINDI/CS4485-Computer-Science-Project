// Simple mock API with small latency to simulate network
export type OneStep = { country: string; as_of: string; horizon: 1; gdp_pred: number; unit: string }
export type MultiStep = { country: string; as_of: string; preds: Record<string, number> }
export type SeriesPoint = { date: string; value: number }
export type SeriesResponse = {
  series: {
    gdp: SeriesPoint[]
    level: SeriesPoint[]
    slope: SeriesPoint[]
    curvature: SeriesPoint[]
    inflation: SeriesPoint[]
    unemp: SeriesPoint[]
    caputil: SeriesPoint[]
    fedfunds: SeriesPoint[]
  }
}
export type MetricsResponse = {
  one_step: { avg_mse: Record<string, number>; timeline: { date: string; mse: number }[] }
  multi: { avg_mse_by_h: Record<string, number> }
}

function delay<T>(data: T, ms = 300): Promise<T> {
  return new Promise((res) => setTimeout(() => res(data), ms))
}

function genSeries(startYear = 2015, endYear = 2025): SeriesPoint[] {
  const result: SeriesPoint[] = []
  const quarters = ["03-31", "06-30", "09-30", "12-31"]
  let base = 18000
  for (let y = startYear; y <= endYear; y++) {
    for (const q of quarters) {
      base += Math.random() * 80 - 10
      result.push({ date: `${y}-${q}`, value: Math.round(base * 10) / 10 })
    }
  }
  return result
}

export async function getOneStep(country: string, asOf: string | null): Promise<OneStep> {
  const as_of = asOf ?? '2024-12-31'
  return delay({ country, as_of, horizon: 1, gdp_pred: 20834.2, unit: 'billions_chained_2017' })
}

export async function getMulti(country: string, horizons: number[], asOf: string | null): Promise<MultiStep> {
  const as_of = asOf ?? '2024-12-31'
  const preds: Record<string, number> = {}
  horizons.forEach((h) => preds[String(h)] = 20834.2 + h * 50 + Math.random() * 40)
  return delay({ country, as_of, preds })
}

export async function getSeries(country: string, fromYear = 2015, toYear = 2025): Promise<SeriesResponse> {
  const gdp = genSeries(fromYear, toYear)
  const level = genSeries(fromYear, toYear)
  const slope = genSeries(fromYear, toYear)
  const curvature = genSeries(fromYear, toYear)
  const inflation = genSeries(fromYear, toYear)
  const unemp = genSeries(fromYear, toYear)
  const caputil = genSeries(fromYear, toYear)
  const fedfunds = genSeries(fromYear, toYear)
  return delay({ series: { gdp, level, slope, curvature, inflation, unemp, caputil, fedfunds } })
}

export async function getMetrics(fromYear = 2018, toYear = 2025): Promise<MetricsResponse> {
  const timeline = [] as { date: string; mse: number }[]
  for (let y = fromYear; y <= toYear; y++) {
    timeline.push({ date: `${y}-01-01`, mse: Math.random() * 0.003 })
  }
  return delay({
    one_step: { avg_mse: { knn: 0.0012, lr: 0.0018, sarimax: 0.0021, sarima: 0.0030 }, timeline },
    multi: { avg_mse_by_h: { '2': 0.0007, '4': 0.0012, '8': 0.0027, '12': 0.0039 } },
  })
}


