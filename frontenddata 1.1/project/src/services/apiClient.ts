/**
 * Garuda ML Pipeline API Client
 * 
 * Provides TypeScript client for all 26 API endpoints
 * Handles authentication, error handling, and type safety
 */

import { getAPIBaseURL, getTimeout } from './apiConfig'

// ===================================
// TYPE DEFINITIONS
// ===================================

export interface APIResponse<T = any> {
  success: boolean
  message: string
  output_key?: string
  meta?: Record<string, any>
  logs?: string[]
  execution_time?: number
  timestamp?: string
  data?: T
}

export interface UploadRequest {
  user_id: string
  data_type: 'numeric' | 'text' | 'mixed'
  description?: string
}

export interface UploadResponse extends APIResponse {
  dataset_id?: string
  bucket_key?: string
  file_size?: number
  validation_results?: Record<string, any>
}

export interface PipelineRequest {
  user_id: string
  dataset_key: string
  params?: Record<string, any>
}

export interface PreprocessRequest extends PipelineRequest {
  operation: string
}

export interface PreprocessResponse extends APIResponse {
  preprocessing_results?: Record<string, any>
}

export interface AugmentRequest extends PipelineRequest {
  technique: string
  target_size?: number
}

export interface AugmentResponse extends APIResponse {
  augmentation_results?: Record<string, any>
}

export interface MixupRequest {
  texts: string[]
  alpha?: number
  mix_labels?: boolean
}

export interface MixupResponse extends APIResponse {
  augmented_samples?: any[]
  total_samples?: number
}

export interface EDARequest extends PipelineRequest {
  analysis_type: string
  output_format?: string
}

export interface EDAResponse extends APIResponse {
  analysis_results?: Record<string, any>
  visualizations?: string[]
  insights?: string[]
}

export interface ModelRequest extends PipelineRequest {
  model_type: string
  hyperparameters?: Record<string, any>
  validation_split?: number
}

export interface ModelResponse extends APIResponse {
  model_id?: string
  metrics?: Record<string, number>
  model_size?: number
  training_time?: number
}

// ===================================
// API CLIENT CLASS
// ===================================

export class GarudaAPIClient {
  private baseURL: string
  private timeout: number

  constructor(baseURL?: string, timeout?: number) {
    this.baseURL = baseURL || getAPIBaseURL()
    this.timeout = timeout || getTimeout('default')
  }

  /**
   * Generic API call method with error handling
   */
  private async apiCall<T>(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'POST',
    data?: any,
    isFormData: boolean = false,
    customTimeout?: number
  ): Promise<T> {
    const timeout = customTimeout || this.timeout
    
    try {
      const url = `${this.baseURL}${endpoint}`
      
      const headers: Record<string, string> = {}
      let body: any = undefined

      if (data) {
        if (isFormData) {
          body = data // FormData object
        } else {
          headers['Content-Type'] = 'application/json'
          body = JSON.stringify(data)
        }
      }

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeout)

      const response = await fetch(url, {
        method,
        headers,
        body,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }

      return await response.json()
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error(`Request timeout after ${timeout}ms`)
        }
        throw error
      }
      throw new Error('Unknown error occurred')
    }
  }

  // ===================================
  // UPLOAD ENDPOINTS
  // ===================================

  async uploadDataset(file: File, request: UploadRequest): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('user_id', request.user_id)
    formData.append('data_type', request.data_type)
    if (request.description) {
      formData.append('description', request.description)
    }

    // Use 2 minutes timeout for file uploads
    return this.apiCall<UploadResponse>('/api/upload/dataset', 'POST', formData, true, 120000)
  }

  // ===================================
  // PREPROCESSING ENDPOINTS
  // ===================================

  async handleMissingValues(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/handle-missing-values', 'POST', request)
  }

  async handleOutliers(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/handle-outliers', 'POST', request)
  }

  async scaleNumericFeatures(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/scale-numeric-features', 'POST', request)
  }

  async transformFeatures(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/transform-features', 'POST', request)
  }

  async compareData(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/comparison', 'POST', request)
  }

  async tokenizeText(request: PreprocessRequest): Promise<PreprocessResponse> {
    return this.apiCall<PreprocessResponse>('/api/preprocess/tokenization', 'POST', request)
  }

  // ===================================
  // AUGMENTATION ENDPOINTS
  // ===================================

  async augmentWithMixup(request: MixupRequest): Promise<MixupResponse> {
    return this.apiCall<MixupResponse>('/api/augment/mixup', 'POST', request)
  }

  async augmentWithSMOTE(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/smote', 'POST', request)
  }

  async augmentWithNoise(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/noise', 'POST', request)
  }

  async augmentWithScale(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/scale', 'POST', request)
  }

  async augmentWithSynonym(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/synonym', 'POST', request)
  }

  async augmentWithMLM(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/mlm', 'POST', request)
  }

  async augmentWithRandom(request: AugmentRequest): Promise<AugmentResponse> {
    return this.apiCall<AugmentResponse>('/api/augment/random', 'POST', request)
  }

  // ===================================
  // EDA ENDPOINTS
  // ===================================

  async runNumericEDAManager(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/numeric-manager', 'POST', request)
  }

  async runEDAManager(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/eda-manager', 'POST', request)
  }

  async runCorrelationAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/correlation-analysis', 'POST', request)
  }

  async runStatisticalAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/statistical-analysis', 'POST', request)
  }

  async runAdvancedVisualization(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/advanced-visualization', 'POST', request)
  }

  async runSentimentAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/sentiment-analysis', 'POST', request)
  }

  async runWordFrequencyAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/word-frequency', 'POST', request)
  }

  async runTextLengthAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/text-length', 'POST', request)
  }

  async runTopicModeling(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/topic-modeling', 'POST', request)
  }

  async runNgramAnalysis(request: EDARequest): Promise<EDAResponse> {
    return this.apiCall<EDAResponse>('/api/eda/ngram-analysis', 'POST', request)
  }

  // ===================================
  // MODEL TRAINING ENDPOINTS
  // ===================================

  async trainLogisticRegression(request: ModelRequest): Promise<ModelResponse> {
    return this.apiCall<ModelResponse>('/api/model/logistic-regression', 'POST', request, false, 180000) // 3 minute timeout
  }

  async trainRandomForest(request: ModelRequest): Promise<ModelResponse> {
    return this.apiCall<ModelResponse>('/api/model/random-forest', 'POST', request, false, 300000) // 5 minute timeout
  }

  async trainGradientBoosting(request: ModelRequest): Promise<ModelResponse> {
    return this.apiCall<ModelResponse>('/api/model/gradient-boosting', 'POST', request, false, 300000) // 5 minute timeout
  }

  async trainXGBoost(request: ModelRequest): Promise<ModelResponse> {
    return this.apiCall<ModelResponse>('/api/model/xgboost', 'POST', request, false, 300000) // 5 minute timeout
  }

  async trainDistilBERT(request: ModelRequest): Promise<ModelResponse> {
    return this.apiCall<ModelResponse>('/api/model/distilbert-finetune', 'POST', request, false, 600000) // 10 minute timeout
  }

  // ===================================
  // UTILITY METHODS
  // ===================================

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.apiCall<{ status: string; timestamp: string }>('/health', 'GET')
  }

  /**
   * Test API connectivity
   */
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck()
      return true
    } catch (error) {
      console.error('API connection test failed:', error)
      return false
    }
  }

  /**
   * Update base URL for different environments
   */
  setBaseURL(url: string): void {
    this.baseURL = url
  }

  /**
   * Update timeout for long-running operations
   */
  setTimeout(timeout: number): void {
    this.timeout = timeout
  }

  /**
   * Download a file from storage
   */
  async downloadFile(
    bucketName: string, 
    fileKey: string, 
    userId: string
  ): Promise<Blob> {
    const url = `${this.baseURL}/download/${bucketName}/${fileKey}?user_id=${userId}`
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream'
      }
    })

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status} ${response.statusText}`)
    }

    return await response.blob()
  }
}

// ===================================
// SINGLETON INSTANCE
// ===================================

// Create default API client instance
export const apiClient = new GarudaAPIClient()

// Export for direct use
export default apiClient
