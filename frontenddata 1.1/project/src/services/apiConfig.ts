/**
 * API Configuration for Garuda ML Pipeline
 */

// Environment-based configuration
const config = {
  // API Base URLs for different environments
  api: {
    development: 'http://localhost:8000',
    staging: 'http://localhost:8000', 
    production: 'https://your-production-api.com' // Update when deploying
  },
  
  // API timeouts for different operations (in milliseconds)
  timeouts: {
    upload: 60000,        // 60 seconds for file uploads
    preprocessing: 120000, // 2 minutes for preprocessing
    augmentation: 180000,  // 3 minutes for augmentation
    eda: 300000,          // 5 minutes for EDA (can generate many visualizations)
    training: 600000,     // 10 minutes for model training
    default: 30000        // 30 seconds for other operations
  },
  
  // File upload limits
  upload: {
    maxSizeBytes: 100 * 1024 * 1024, // 100MB
    allowedTypes: [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/plain',
      'application/json'
    ]
  }
}

// Get current environment
const getEnvironment = (): keyof typeof config.api => {
  // For now, always use development since we're in local development
  // This can be updated when deploying to production
  return 'development'
}

// Get API base URL for current environment
export const getAPIBaseURL = (): string => {
  const env = getEnvironment()
  return config.api[env]
}

// Get timeout for specific operation type
export const getTimeout = (operation: keyof typeof config.timeouts): number => {
  return config.timeouts[operation] || config.timeouts.default
}

// Validate file before upload
export const validateFile = (file: File): { valid: boolean; error?: string } => {
  // Check file size
  if (file.size > config.upload.maxSizeBytes) {
    return {
      valid: false,
      error: `File size exceeds ${config.upload.maxSizeBytes / (1024 * 1024)}MB limit`
    }
  }
  
  // Check file type
  if (!config.upload.allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: `File type ${file.type} is not supported. Allowed types: ${config.upload.allowedTypes.join(', ')}`
    }
  }
  
  return { valid: true }
}

// Export configuration
export default config
