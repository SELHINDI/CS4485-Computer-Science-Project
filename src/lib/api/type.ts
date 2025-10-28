

export interface AnalyzeResponse {
    status: "success" | "error"
    data?: {
      historical_data: any
      predictions: Record<string, any>
      model_performance: Record<string, any>
      feature_importance?: Record<string, number>
      analysis_metadata: {
        country: string
        start_year: number
        end_year: number
        prediction_years: number
        model_type: string
        timestamp: string
      }
    }
    message?: string
  }
  