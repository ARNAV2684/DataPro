import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate, useLocation } from 'react-router-dom'
import { apiClient } from '../services/apiClient'
import { downloadPreprocessingResult } from '../services/downloadService'
import { 
  ArrowLeft,
  ArrowRight,
  Brain,
  Settings,
  CheckCircle2,
  AlertTriangle,
  Database,
  FileText,
  Image as ImageIcon,
  BarChart3,
  Filter,
  Zap,
  Eye,
  Download,
  Play,
  Pause,
  RotateCcw,
  User,
  LogOut,
  Layers,
  Target,
  Sliders
} from 'lucide-react'
import { DatasetType } from './DatasetSelectionPage'

interface PreprocessingStep {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  status: 'pending' | 'running' | 'completed' | 'error'
  enabled: boolean
  parameters?: any
}

interface PreprocessingConfig {
  title: string
  description: string
  steps: PreprocessingStep[]
  tips: string[]
}

const PreprocessingPage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [datasetType, setDatasetType] = useState<DatasetType>('mixed')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [preprocessingSteps, setPreprocessingSteps] = useState<PreprocessingStep[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState<string | null>(null)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [showPreview, setShowPreview] = useState(false)

  useEffect(() => {
    if (location.state) {
      setDatasetType(location.state.datasetType || 'mixed')
      setUploadedFiles(location.state.uploadedFiles || [])
    } else {
      navigate('/home')
    }
  }, [location.state, navigate])

  const preprocessingConfigs: Record<DatasetType, PreprocessingConfig> = {
    numeric: {
      title: 'Numeric Data Preprocessing',
      description: 'Clean and prepare your numeric datasets for analysis and machine learning',
      steps: [
        {
          id: 'data-validation',
          name: 'Data Validation',
          description: 'Check data types, formats, and basic structure',
          icon: <CheckCircle2 className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'missing-values',
          name: 'Handle Missing Values',
          description: 'Detect and handle missing or null values',
          icon: <Target className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            strategy: 'mean'
          }
        },
        {
          id: 'outlier-detection',
          name: 'Outlier Detection',
          description: 'Identify and handle statistical outliers',
          icon: <AlertTriangle className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            threshold: 1.5
          }
        },
        {
          id: 'feature-scaling',
          name: 'Feature Scaling',
          description: 'Normalize or standardize numeric features',
          icon: <Sliders className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            scaler: 'standard'
          }
        },
        {
          id: 'feature-encoding',
          name: 'Feature Encoding',
          description: 'Encode categorical variables',
          icon: <Layers className="w-5 h-5" />,
          status: 'pending',
          enabled: false,
          parameters: {
            method: 'onehot',
            dropFirst: true
          }
        }
      ],
      tips: [
        'Check for data quality issues early',
        'Consider domain knowledge when handling outliers',
        'Choose scaling method based on your algorithm',
        'Validate preprocessing steps with sample data'
      ]
    },
    text: {
      title: 'Text Data Preprocessing',
      description: 'Clean and prepare your text data for natural language processing',
      steps: [
        {
          id: 'text-cleaning',
          name: 'Text Cleaning',
          description: 'Remove special characters, HTML tags, and formatting',
          icon: <Filter className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'tokenization',
          name: 'Tokenization',
          description: 'Split text into words, sentences, or subwords',
          icon: <Layers className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            method: 'word',
            language: 'english'
          }
        },
        {
          id: 'normalization',
          name: 'Text Normalization',
          description: 'Convert to lowercase, remove punctuation',
          icon: <Sliders className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            lowercase: true,
            removePunctuation: true,
            removeNumbers: false
          }
        },
        {
          id: 'stopword-removal',
          name: 'Stopword Removal',
          description: 'Remove common words that add little meaning',
          icon: <Target className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            language: 'english',
            customStopwords: []
          }
        },
        {
          id: 'stemming-lemmatization',
          name: 'Stemming/Lemmatization',
          description: 'Reduce words to their root forms',
          icon: <RotateCcw className="w-5 h-5" />,
          status: 'pending',
          enabled: false,
          parameters: {
            method: 'lemmatization',
            language: 'english'
          }
        }
      ],
      tips: [
        'Preserve important punctuation for sentiment analysis',
        'Consider domain-specific stopwords',
        'Choose between stemming and lemmatization carefully',
        'Validate text quality after preprocessing'
      ]
    },
    image: {
      title: 'Image Data Preprocessing',
      description: 'Prepare your images for computer vision and deep learning models',
      steps: [
        {
          id: 'image-validation',
          name: 'Image Validation',
          description: 'Check image formats, sizes, and integrity',
          icon: <CheckCircle2 className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'resize-normalize',
          name: 'Resize & Normalize',
          description: 'Standardize image dimensions and pixel values',
          icon: <Sliders className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            targetSize: [224, 224],
            normalization: 'imagenet'
          }
        },
        {
          id: 'color-correction',
          name: 'Color Correction',
          description: 'Adjust brightness, contrast, and color balance',
          icon: <Filter className="w-5 h-5" />,
          status: 'pending',
          enabled: false,
          parameters: {
            brightness: 0,
            contrast: 1,
            saturation: 1
          }
        },
        {
          id: 'noise-reduction',
          name: 'Noise Reduction',
          description: 'Remove noise and artifacts from images',
          icon: <Target className="w-5 h-5" />,
          status: 'pending',
          enabled: false,
          parameters: {
            method: 'gaussian',
            strength: 0.5
          }
        },
        {
          id: 'format-conversion',
          name: 'Format Conversion',
          description: 'Convert images to consistent format',
          icon: <Layers className="w-5 h-5" />,
          status: 'pending',
          enabled: true,
          parameters: {
            format: 'RGB',
            quality: 95
          }
        }
      ],
      tips: [
        'Maintain aspect ratio when resizing',
        'Choose normalization based on your model',
        'Consider image quality vs. processing speed',
        'Validate preprocessing with sample images'
      ]
    },
    mixed: {
      title: 'Mixed Data Preprocessing',
      description: 'Comprehensive preprocessing for multimodal datasets',
      steps: [
        {
          id: 'data-type-detection',
          name: 'Data Type Detection',
          description: 'Automatically detect and categorize data types',
          icon: <Database className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'modality-separation',
          name: 'Modality Separation',
          description: 'Separate different data modalities',
          icon: <Layers className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'cross-modal-validation',
          name: 'Cross-Modal Validation',
          description: 'Validate consistency across modalities',
          icon: <CheckCircle2 className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'unified-preprocessing',
          name: 'Unified Preprocessing',
          description: 'Apply appropriate preprocessing to each modality',
          icon: <Zap className="w-5 h-5" />,
          status: 'pending',
          enabled: true
        },
        {
          id: 'feature-alignment',
          name: 'Feature Alignment',
          description: 'Align features across different modalities',
          icon: <Target className="w-5 h-5" />,
          status: 'pending',
          enabled: false,
          parameters: {
            method: 'temporal',
            tolerance: 0.1
          }
        }
      ],
      tips: [
        'Understand relationships between modalities',
        'Maintain temporal alignment for time-series data',
        'Consider modality-specific preprocessing needs',
        'Validate cross-modal consistency'
      ]
    }
  }

  useEffect(() => {
    const config = preprocessingConfigs[datasetType]
    setPreprocessingSteps(config.steps)
  }, [datasetType])

  const currentConfig = preprocessingConfigs[datasetType]

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const handleBackToHome = () => {
    navigate('/home', {
      state: {
        datasetType,
        uploadedFiles
      }
    })
  }

  const handleDownload = async () => {
    if (!user?.id) {
      alert('User not authenticated')
      return
    }

    // Find the last completed step with an output_key
    const completedSteps = preprocessingSteps.filter(step => 
      step.status === 'completed' && step.parameters?.output_key
    )

    if (completedSteps.length === 0) {
      alert('No processed files available for download. Please run preprocessing first.')
      return
    }

    // Get the output key from the last completed step
    const lastStep = completedSteps[completedSteps.length - 1]
    const outputKey = lastStep.parameters.output_key

    try {
      await downloadPreprocessingResult(
        outputKey,
        user.id,
        `${datasetType}_preprocessed_${lastStep.name.toLowerCase().replace(/\s+/g, '_')}.csv`
      )
    } catch (error) {
      console.error('Download failed:', error)
      alert('Download failed. Please try again.')
    }
  }

  const handleProceedToAugmentation = () => {
    navigate('/data-augmentation', {
      state: {
        datasetType,
        uploadedFiles
      }
    })
  }

  const toggleStep = (stepId: string) => {
    setPreprocessingSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, enabled: !step.enabled } : step
    ))
  }

  const runPreprocessing = async () => {
    const enabledSteps = preprocessingSteps.filter(step => step.enabled)
    if (enabledSteps.length === 0) return

    // Check if user is available
    if (!user?.id) {
      console.error('User not authenticated')
      return
    }

    // Get the first uploaded file's dataset_key from the uploadedFiles
    if (!uploadedFiles || uploadedFiles.length === 0) {
      console.error('No uploaded files available for preprocessing')
      return
    }

    // Debug: Log the uploaded files structure to understand what properties are available
    console.log('Uploaded files structure:', uploadedFiles[0])
    console.log('All uploaded files:', uploadedFiles)

    // Try different possible property names for the dataset key
    const firstFile = uploadedFiles[0]
    let datasetKey = firstFile.dataset_key || 
                    firstFile.output_key || 
                    firstFile.bucket_key ||
                    firstFile.storage_key ||
                    firstFile.key ||
                    firstFile.file_key

    // If no dataset key found and file wasn't uploaded, try uploading it now
    if (!datasetKey && !firstFile.uploaded) {
      console.log('No dataset key found, attempting to upload file now...')
      try {
        const uploadResponse = await apiClient.uploadDataset(firstFile.file, {
          user_id: user.id,
          data_type: 'numeric', // Default to numeric for now
          description: 'Uploaded during preprocessing'
        })
        
        console.log('Upload response:', uploadResponse)
        datasetKey = uploadResponse.output_key || uploadResponse.bucket_key || uploadResponse.dataset_id
        
        // Update the uploaded files array
        setUploadedFiles(prev => prev.map(file => 
          file === firstFile ? {
            ...file,
            dataset_key: datasetKey,
            output_key: uploadResponse.output_key,
            bucket_key: uploadResponse.bucket_key,
            dataset_id: uploadResponse.dataset_id,
            uploaded: true
          } : file
        ))
      } catch (uploadError) {
        console.error('Failed to upload file during preprocessing:', uploadError)
        alert('Failed to upload file. Please go back and upload the file again.')
        return
      }
    }

    if (!datasetKey) {
      console.error('No dataset_key found in uploaded files. Available properties:', Object.keys(firstFile))
      alert('No dataset key found. Please upload files again.')
      return
    }

    console.log('Using dataset key:', datasetKey)

    setIsProcessing(true)
    setProcessingProgress(0)

    try {
      for (let i = 0; i < enabledSteps.length; i++) {
        const step = enabledSteps[i]
        setCurrentStep(step.id)
        
        // Update step status to running
        setPreprocessingSteps(prev => prev.map(s => 
          s.id === step.id ? { ...s, status: 'running' } : s
        ))

        try {
          let response

          // Call appropriate API based on step ID
          switch (step.id) {
            case 'missing-values':
            case 'handle-missing':
              response = await apiClient.handleMissingValues({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'missing_values',
                params: step.parameters || {}
              })
              break

            case 'outlier-detection':
            case 'handle-outliers':
              response = await apiClient.handleOutliers({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'outliers',
                params: step.parameters || {}
              })
              break

            case 'feature-scaling':
            case 'scale-features':
            case 'scale-numeric-features':
              response = await apiClient.scaleNumericFeatures({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'scaling',
                params: step.parameters || {}
              })
              break

            case 'feature-encoding':
            case 'transform-features':
              response = await apiClient.transformFeatures({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'transformation',
                params: step.parameters || {}
              })
              break

            case 'tokenization':
              response = await apiClient.tokenizeText({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'tokenization',
                params: step.parameters || {}
              })
              break

            case 'comparison':
              response = await apiClient.compareData({
                user_id: user.id,
                dataset_key: datasetKey,
                operation: 'comparison',
                params: step.parameters || {}
              })
              break

            default:
              console.warn(`Unknown preprocessing step: ${step.id}`)
              continue
          }

          console.log(`Step ${step.id} response:`, response)

          if (response.success) {
            // Update step status to completed and store output_key for potential download
            setPreprocessingSteps(prev => prev.map(s => 
              s.id === step.id ? { 
                ...s, 
                status: 'completed',
                parameters: { 
                  ...s.parameters, 
                  output_key: response.output_key 
                }
              } : s
            ))
          } else {
            throw new Error(response.message || 'Processing failed')
          }

        } catch (error) {
          console.error(`Error in step ${step.id}:`, error)
          
          // Update step status to error
          setPreprocessingSteps(prev => prev.map(s => 
            s.id === step.id ? { ...s, status: 'error' } : s
          ))
          
          // Stop processing on error
          break
        }

        // Update progress
        const progress = ((i + 1) / enabledSteps.length) * 100
        setProcessingProgress(progress)
      }

    } catch (error) {
      console.error('Preprocessing error:', error)
    } finally {
      setIsProcessing(false)
      setCurrentStep(null)
    }
  }

  const enabledSteps = preprocessingSteps.filter(step => step.enabled)
  const completedSteps = preprocessingSteps.filter(step => step.status === 'completed')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToHome}
                className="flex items-center space-x-2 px-3 py-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span className="text-sm">Back</span>
              </button>
              <div className="w-10 h-10 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">DataLab Pro</h1>
                <p className="text-xs text-gray-400">Data Preprocessing</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-300">
                <User className="w-5 h-5" />
                <span className="text-sm">{user?.email}</span>
              </div>
              <button
                onClick={handleSignOut}
                className="flex items-center space-x-2 px-4 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Sign Out</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-4">{currentConfig.title}</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">{currentConfig.description}</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            {/* Processing Summary */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Processing Summary</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Steps</span>
                  <span className="text-white font-semibold">{preprocessingSteps.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Enabled</span>
                  <span className="text-white font-semibold">{enabledSteps.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Completed</span>
                  <span className="text-white font-semibold">{completedSteps.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Status</span>
                  <span className={`font-semibold flex items-center space-x-1 ${
                    isProcessing ? 'text-yellow-400' : 
                    completedSteps.length === enabledSteps.length && enabledSteps.length > 0 ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    {isProcessing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-yellow-400/30 border-t-yellow-400 rounded-full animate-spin"></div>
                        <span>Processing</span>
                      </>
                    ) : completedSteps.length === enabledSteps.length && enabledSteps.length > 0 ? (
                      <>
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Complete</span>
                      </>
                    ) : (
                      <>
                        <Settings className="w-4 h-4" />
                        <span>Ready</span>
                      </>
                    )}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              {isProcessing && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-300">Progress</span>
                    <span className="text-sm text-white">{Math.round(processingProgress)}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-accent-500 to-secondary-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${processingProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Run Processing Button */}
              <button
                onClick={runPreprocessing}
                disabled={enabledSteps.length === 0 || isProcessing}
                className={`w-full mt-4 py-3 px-4 rounded-lg font-semibold transition-all transform ${
                  enabledSteps.length > 0 && !isProcessing
                    ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105'
                    : 'bg-gray-600/50 text-gray-400 cursor-not-allowed'
                } flex items-center justify-center space-x-2`}
              >
                {isProcessing ? (
                  <>
                    <Pause className="w-5 h-5" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start Processing</span>
                  </>
                )}
              </button>

              {/* Skip Preprocessing Button */}
              <button
                onClick={handleProceedToAugmentation}
                disabled={isProcessing}
                className={`w-full mt-3 py-2 px-4 rounded-lg font-medium transition-all ${
                  !isProcessing
                    ? 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/20 hover:border-white/30'
                    : 'bg-gray-600/30 text-gray-500 cursor-not-allowed border border-gray-600/30'
                } flex items-center justify-center space-x-2`}
              >
                <ArrowRight className="w-4 h-4" />
                <span>Skip Preprocessing</span>
              </button>
            </div>

            {/* Tips */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-accent-400" />
                <span>Pro Tips</span>
              </h3>
              <div className="space-y-3 text-sm text-gray-300">
                {currentConfig.tips.map((tip, index) => (
                  <p key={index} className="flex items-start space-x-2">
                    <span className="text-accent-400 mt-1">•</span>
                    <span>{tip}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Preprocessing Steps */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-6">Preprocessing Pipeline</h3>

              <div className="space-y-4">
                {preprocessingSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className={`bg-white/5 rounded-lg p-4 border transition-all ${
                      step.enabled 
                        ? step.status === 'completed' 
                          ? 'border-green-400 bg-green-500/10'
                          : step.status === 'running'
                          ? 'border-yellow-400 bg-yellow-500/10'
                          : 'border-accent-400 bg-accent-500/10'
                        : 'border-white/10'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        {/* Step Number */}
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                          step.status === 'completed' ? 'bg-green-500 text-white' :
                          step.status === 'running' ? 'bg-yellow-500 text-white' :
                          step.enabled ? 'bg-accent-500 text-white' : 'bg-gray-600 text-gray-300'
                        }`}>
                          {step.status === 'completed' ? (
                            <CheckCircle2 className="w-4 h-4" />
                          ) : step.status === 'running' ? (
                            <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                          ) : (
                            index + 1
                          )}
                        </div>

                        {/* Step Icon */}
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          step.enabled ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white' : 'bg-gray-600 text-gray-300'
                        }`}>
                          {step.icon}
                        </div>

                        {/* Step Info */}
                        <div>
                          <h4 className="text-white font-semibold">{step.name}</h4>
                          <p className="text-gray-300 text-sm">{step.description}</p>
                          {step.status === 'running' && currentStep === step.id && (
                            <p className="text-yellow-400 text-xs mt-1">Currently processing...</p>
                          )}
                        </div>
                      </div>

                      {/* Toggle Switch */}
                      <button
                        onClick={() => toggleStep(step.id)}
                        disabled={isProcessing}
                        className={`w-12 h-6 rounded-full transition-colors ${
                          step.enabled ? 'bg-accent-500' : 'bg-gray-600'
                        } relative ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                          step.enabled ? 'translate-x-6' : 'translate-x-0.5'
                        }`}></div>
                      </button>
                    </div>

                    {/* Parameters (if any) */}
                    {step.enabled && step.parameters && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <h5 className="text-sm font-semibold text-gray-300 mb-2">Parameters</h5>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          {Object.entries(step.parameters).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}:</span>
                              <span className="text-white">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Data Preview */}
            {completedSteps.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white">Data Preview</h3>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setShowPreview(!showPreview)}
                      className="flex items-center space-x-2 px-4 py-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors"
                    >
                      <Eye className="w-4 h-4" />
                      <span>{showPreview ? 'Hide' : 'Show'} Preview</span>
                    </button>
                    <button 
                      onClick={handleDownload}
                      className="p-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      disabled={!preprocessingSteps.some(step => step.status === 'completed' && step.parameters?.output_key)}
                      title="Download processed data"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {showPreview && (
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-white font-semibold mb-2">Before Processing</h4>
                        <div className="bg-gradient-to-br from-red-500/20 to-orange-500/20 rounded-lg p-4 h-32 flex items-center justify-center">
                          <span className="text-gray-300">Raw Data Sample</span>
                        </div>
                      </div>
                      <div>
                        <h4 className="text-white font-semibold mb-2">After Processing</h4>
                        <div className="bg-gradient-to-br from-green-500/20 to-blue-500/20 rounded-lg p-4 h-32 flex items-center justify-center">
                          <span className="text-gray-300">Processed Data Sample</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Proceed Button */}
            <div className="text-center">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={handleProceedToAugmentation}
                  className="inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all transform bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105 shadow-lg hover:shadow-accent-500/25"
                >
                  <span>Proceed to Data Augmentation</span>
                  <ArrowRight className="w-6 h-6" />
                </button>
              </div>
              
              <div className="mt-4 text-center">
                {completedSteps.length > 0 ? (
                  <p className="text-green-400 text-sm">
                    ✅ {completedSteps.length} preprocessing step{completedSteps.length !== 1 ? 's' : ''} completed
                  </p>
                ) : (
                  <p className="text-gray-400 text-sm">
                    You can proceed with raw data or complete preprocessing steps first
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PreprocessingPage