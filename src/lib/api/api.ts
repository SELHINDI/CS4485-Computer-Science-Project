

export type CountryData = Record<string, any>;
export type AnalyzeResponse = {
  historical_data: any;
  predictions: any;
  model_performance: any;
  feature_importance?: any;
  analysis_metadata: any;
};
export type CompareResponse = Record<string, any>;
export type HealthResponse = {
  status: string;
  timestamp: string;
  cache_size: { data: number; models: number };
  version: string;
};

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5001/api";

// Helper to handle fetch + error parsing
async function handleResponse(res: Response) {
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Request failed: ${res.status} - ${errorText}`);
  }
  const json = await res.json();
  if (json.status === "error") throw new Error(json.message);
  return json.data ?? json;
}

/** Get list of available countries */
export async function getCountries(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/countries`);
  return handleResponse(res);
}

/** Get historical data for a specific country */
export async function getCountryData(
  country: string,
  startYear = 2010,
  endYear = 2023
): Promise<CountryData> {
  const params = new URLSearchParams({
    start_year: startYear.toString(),
    end_year: endYear.toString(),
  });
  const res = await fetch(`${BASE_URL}/data/${country}?${params}`);
  return handleResponse(res);
}

/** Perform analysis (train + predict) for a specific country */
export async function analyzeCountry(
  country: string,
  options: {
    start_year?: number;
    end_year?: number;
    prediction_years?: number;
    model_type?: string;
  } = {}
): Promise<AnalyzeResponse> {
  const body = JSON.stringify(options);
  const res = await fetch(`${BASE_URL}/analyze/${country}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  return handleResponse(res);
}

/** Compare multiple countries */
export async function compareCountries(
  countries: string[],
  startYear = 2010,
  endYear = 2023
): Promise<CompareResponse> {
  const body = JSON.stringify({ countries, start_year: startYear, end_year: endYear });
  const res = await fetch(`${BASE_URL}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  return handleResponse(res);
}

/** Health check */
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}
