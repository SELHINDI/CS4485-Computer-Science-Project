import { useQuery } from '@tanstack/react-query'
import { getOneStep, getMulti, getSeries, getMetrics } from '../api/mock'

export function useOneStep(country: string, asOf: string | null) {
  return useQuery({
    queryKey: ['one-step', country, asOf],
    queryFn: () => getOneStep(country, asOf),
  })
}

export function useMulti(country: string, horizons: number[], asOf: string | null) {
  return useQuery({
    queryKey: ['multi', country, horizons.sort().join(','), asOf],
    queryFn: () => getMulti(country, horizons, asOf),
    enabled: horizons.length > 0,
  })
}

export function useSeries(country: string, fromYear?: number, toYear?: number) {
  return useQuery({
    queryKey: ['series', country, fromYear, toYear],
    queryFn: () => getSeries(country, fromYear, toYear),
  })
}

export function useMetrics(fromYear?: number, toYear?: number) {
  return useQuery({
    queryKey: ['metrics', fromYear, toYear],
    queryFn: () => getMetrics(fromYear, toYear),
  })
}


