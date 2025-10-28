import { useQuery, useMutation } from '@tanstack/react-query'
import { getCountries, getCountryData, analyzeCountry, compareCountries, getHealth } from '../api/api'

export function useCountries() {
  return useQuery({
    queryKey: ['countries'],
    queryFn: () => getCountries(),
  })
}

export function useCountryData(country: string, startYear?:number, endYear?: number) {
  return useQuery({
    queryKey: ['country-data', country, startYear, endYear],
    queryFn: () => getCountryData(country, startYear, endYear),
    enabled: !!country,
  })
}

export function useAnalyzeCountry(
  country: string,
  options?: {
    start_year?: number;
    end_year?: number;
    prediction_years?: number;
    model_type?: string;
  }
) {
  return useMutation({
    mutationFn: () => analyzeCountry(country, options),
  });
}

/** Compare multiple countries */
export function useCompareCountries(
  countries: string[],
  startYear?: number,
  endYear?: number
) {
  return useMutation({
    mutationFn: () => compareCountries(countries, startYear, endYear),
  });
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => getHealth(),
  })
}


