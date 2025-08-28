import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate, useLocation } from 'react-router-dom'
import { apiClient } from '../services/apiClient'
import type { ModelRequest, ModelResponse } from '../services/apiClient'
import { 
  ArrowLeft,
  ArrowRight,
  Brain,
  Zap,
  Target,
  BarChart3,
  Settings,
  Play,
  Pause,
  Square,
  Download,
  Eye,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  Activity,
  Layers,
  Database,
  FileText,
  Image as ImageIcon,
  User,
  LogOut,
  Cpu,
  Clock,
  Award,
  LineChart,
  PieChart,
  Grid,
  List,
  RefreshCw,
  Save,
  Share2,
  Sparkles
} from 'lucide-react'
import { DatasetType } from './DatasetSelectionPage'

interface ModelConfig {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  category: string
  color: string
  complexity: 'beginner' | 'intermediate' | 'advanced'
  trainingTime: string
  accuracy: string
  parameters: Array<{
    name: string
    type: 'slider' | 'select' | 'toggle'
    value: any
    min?: number
    max?: number
    step?: number
    options?: string[]
  }>
  enabled: boolean
}

interface TrainingResult {
  id: string
  modelName: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  trainingTime: number
  status: 'training' | 'completed' | 'failed'
}

interface ModelTrainingConfig {
  title: string
  description: string
  models: ModelConfig[]
  metrics: string[]
  tips: string[]
}

const ModelTrainingPage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [datasetType, setDatasetType] = useState<DatasetType>('mixed')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [augmentedSamples, setAugmentedSamples] = useState<any[]>([])
  const [analysisResults, setAnalysisResults] = useState<any[]>([])
  const [models, setModels] = useState<ModelConfig[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingResults, setTrainingResults] = useState<TrainingResult[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [datasetKey, setDatasetKey] = useState<string | null>(null)
  const [trainingErrors, setTrainingErrors] = useState<string[]>([])

  useEffect(() => {
    if (location.state) {
      setDatasetType(location.state.datasetType || 'mixed')
      setUploadedFiles(location.state.uploadedFiles || [])
      setAugmentedSamples(location.state.augmentedSamples || [])
      setAnalysisResults(location.state.analysisResults || [])
      
      // Extract dataset key from the most recent analysis result or uploaded file
      const latestResult = location.state.analysisResults?.[location.state.analysisResults.length - 1]
      const latestFile = location.state.uploadedFiles?.[location.state.uploadedFiles.length - 1]
      
      const key = latestResult?.output_key || latestFile?.output_key || latestFile?.dataset_key
      setDatasetKey(key)
      
      console.log('Dataset key for training:', key)
    } else {
      navigate('/home')
    }
  }, [location.state, navigate])

  const trainRealModels = async (modelIds: string[]) => {
    if (!datasetKey) {
      setTrainingErrors(['No dataset available for training. Please upload and process a dataset first.'])
      return
    }

    if (!user?.email) {
      setTrainingErrors(['User not authenticated'])
      return
    }

    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingErrors([])
    
    const results: TrainingResult[] = []
    const errors: string[] = []

    for (let i = 0; i < modelIds.length; i++) {
      const modelId = modelIds[i]
      setCurrentModel(modelId)
      
      try {
        // Update progress
        setTrainingProgress((i / modelIds.length) * 100)

        // Get model configuration to extract parameters
        const modelConfig = models.find(m => m.id === modelId)
        const hyperparameters: Record<string, any> = {}
        
        if (modelConfig) {
          modelConfig.parameters.forEach(param => {
            hyperparameters[param.name] = param.value
          })
        }

        // Prepare request
        const request: ModelRequest = {
          user_id: user.email,
          dataset_key: datasetKey,
          model_type: modelId,
          params: {
            target_column: 'target' // Default target column - could be made configurable
          },
          hyperparameters,
          validation_split: 0.2
        }

        console.log(`Training ${modelId} with request:`, request)

        // Call appropriate API endpoint
        let response: ModelResponse
        switch (modelId) {
          case 'logistic-regression':
            response = await apiClient.trainLogisticRegression(request)
            break
          case 'random-forest':
            response = await apiClient.trainRandomForest(request)
            break
          case 'gradient-boosting':
            response = await apiClient.trainGradientBoosting(request)
            break
          case 'xgboost':
            response = await apiClient.trainXGBoost(request)
            break
          case 'distilbert':
            response = await apiClient.trainDistilBERT(request)
            break
          default:
            throw new Error(`Unknown model type: ${modelId}`)
        }

        console.log(`Training result for ${modelId}:`, response)

        if (response.success) {
          // Convert API response to TrainingResult format
          const result: TrainingResult = {
            id: response.model_id || `result-${Date.now()}-${Math.random()}`,
            modelName: modelConfig?.name || modelId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            accuracy: response.metrics?.accuracy || 0,
            precision: response.metrics?.precision || 0,
            recall: response.metrics?.recall || 0,
            f1Score: response.metrics?.f1_score || 0,
            trainingTime: response.training_time || 0,
            status: 'completed' as const
          }
          results.push(result)
        } else {
          errors.push(`${modelConfig?.name || modelId}: ${response.message}`)
        }

      } catch (error) {
        console.error(`Training failed for ${modelId}:`, error)
        errors.push(`${modelId}: ${error instanceof Error ? error.message : 'Unknown error'}`)
      }
      
      // Update progress
      setTrainingProgress(((i + 1) / modelIds.length) * 100)
    }

    setIsTraining(false)
    setCurrentModel(null)
    setTrainingResults(prev => [...prev, ...results])
    
    if (errors.length > 0) {
      setTrainingErrors(errors)
    }
  }

  const downloadModel = async (result: TrainingResult) => {
    if (!user?.email) return
    
    try {
      // The model ID should contain information about the output key
      // For now, we'll construct the download path based on the model ID
      const parts = result.id.split('_')
      if (parts.length >= 4) {
        const modelType = parts[0]
        const userId = parts[1] + '_' + parts[2]
        const datasetId = parts[3]
        
        // Construct the file path for the model
        const fileName = `${modelType}_model.joblib`
        const filePath = `${userId}/${datasetId}/${fileName}`
        
        console.log('Downloading model:', filePath)
        
        const blob = await apiClient.downloadFile('models', filePath, user.email)
        
        // Create download link
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = fileName
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
        
        console.log('Model downloaded successfully')
      }
    } catch (error) {
      console.error('Download failed:', error)
      alert(`Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const stopTraining = () => {
    setIsTraining(false)
    setCurrentModel(null)
    setTrainingProgress(0)
  }

  const modelConfigs: Record<DatasetType, ModelTrainingConfig> = {
    numeric: {
      title: 'Numeric Data Model Training',
      description: 'Train machine learning models on your numeric datasets',
      models: [
        {
          id: 'logistic-regression',
          name: 'Logistic Regression',
          description: 'Fast probabilistic classifier with mixed features',
          icon: <LineChart className="w-5 h-5" />,
          category: 'classical',
          color: 'text-blue-400',
          complexity: 'beginner',
          trainingTime: '< 3 min',
          accuracy: '70-85%',
          parameters: [
            { name: 'C', type: 'slider', value: 1.0, min: 0.1, max: 100, step: 0.1 },
            { name: 'penalty', type: 'select', value: 'l2', options: ['l1', 'l2'] },
            { name: 'max_iter', type: 'slider', value: 1000, min: 100, max: 5000, step: 100 }
          ],
          enabled: false
        },
        {
          id: 'random-forest',
          name: 'Random Forest',
          description: 'Ensemble method with mixed numeric/text features',
          icon: <Layers className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-green-400',
          complexity: 'intermediate',
          trainingTime: '2-5 min',
          accuracy: '75-88%',
          parameters: [
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 500, step: 25 },
            { name: 'max_depth', type: 'slider', value: 10, min: 5, max: 30, step: 1 },
            { name: 'min_samples_split', type: 'slider', value: 2, min: 2, max: 10, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'gradient-boosting',
          name: 'Gradient Boosting',
          description: 'Scikit-learn gradient boosting with mixed features',
          icon: <TrendingUp className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-purple-400',
          complexity: 'intermediate',
          trainingTime: '3-8 min',
          accuracy: '80-90%',
          parameters: [
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 300, step: 25 },
            { name: 'learning_rate', type: 'slider', value: 0.1, min: 0.01, max: 0.3, step: 0.01 },
            { name: 'max_depth', type: 'slider', value: 3, min: 3, max: 10, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'xgboost',
          name: 'XGBoost',
          description: 'High-performance gradient boosting with early stopping',
          icon: <Brain className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-orange-400',
          complexity: 'advanced',
          trainingTime: '5-15 min',
          accuracy: '82-92%',
          parameters: [
            { name: 'learning_rate', type: 'slider', value: 0.1, min: 0.01, max: 0.3, step: 0.01 },
            { name: 'max_depth', type: 'slider', value: 6, min: 3, max: 15, step: 1 },
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 500, step: 25 }
          ],
          enabled: false
        }
      ],
      metrics: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
      tips: [
        'Start with Logistic Regression for quick baseline results',
        'Random Forest works well with mixed numeric and text features',
        'Gradient Boosting often provides good performance-speed balance',
        'XGBoost achieves the best performance but requires more time'
      ]
    },
    text: {
      title: 'Text Data Model Training',
      description: 'Train production-ready NLP models for text classification and analysis',
      models: [
        {
          id: 'logistic-regression',
          name: 'Logistic Regression',
          description: 'Fast probabilistic classifier with mixed features',
          icon: <BarChart3 className="w-5 h-5" />,
          category: 'classical',
          color: 'text-blue-400',
          complexity: 'beginner',
          trainingTime: '< 3 min',
          accuracy: '70-85%',
          parameters: [
            { name: 'C', type: 'slider', value: 1.0, min: 0.1, max: 100, step: 0.1 },
            { name: 'penalty', type: 'select', value: 'l2', options: ['l1', 'l2'] },
            { name: 'max_iter', type: 'slider', value: 1000, min: 100, max: 5000, step: 100 }
          ],
          enabled: false
        },
        {
          id: 'random-forest',
          name: 'Random Forest',
          description: 'Ensemble method with mixed numeric/text features',
          icon: <Target className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-green-400',
          complexity: 'intermediate',
          trainingTime: '2-5 min',
          accuracy: '75-88%',
          parameters: [
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 500, step: 25 },
            { name: 'max_depth', type: 'slider', value: 10, min: 5, max: 30, step: 1 },
            { name: 'min_samples_split', type: 'slider', value: 2, min: 2, max: 10, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'gradient-boosting',
          name: 'Gradient Boosting',
          description: 'Scikit-learn gradient boosting with mixed features',
          icon: <TrendingUp className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-purple-400',
          complexity: 'intermediate',
          trainingTime: '3-8 min',
          accuracy: '80-90%',
          parameters: [
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 300, step: 25 },
            { name: 'learning_rate', type: 'slider', value: 0.1, min: 0.01, max: 0.3, step: 0.01 },
            { name: 'max_depth', type: 'slider', value: 3, min: 3, max: 10, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'xgboost',
          name: 'XGBoost',
          description: 'High-performance gradient boosting with early stopping',
          icon: <Activity className="w-5 h-5" />,
          category: 'ensemble',
          color: 'text-orange-400',
          complexity: 'advanced',
          trainingTime: '5-15 min',
          accuracy: '82-92%',
          parameters: [
            { name: 'learning_rate', type: 'slider', value: 0.1, min: 0.01, max: 0.3, step: 0.01 },
            { name: 'max_depth', type: 'slider', value: 6, min: 3, max: 15, step: 1 },
            { name: 'n_estimators', type: 'slider', value: 100, min: 50, max: 500, step: 25 }
          ],
          enabled: false
        },
        {
          id: 'distilbert',
          name: 'DistilBERT Fine-tuning',
          description: 'State-of-the-art transformer model fine-tuning',
          icon: <Sparkles className="w-5 h-5" />,
          category: 'transformer',
          color: 'text-orange-400',
          complexity: 'advanced',
          trainingTime: '15-60 min',
          accuracy: '88-96%',
          parameters: [
            { name: 'Learning Rate', type: 'slider', value: 2e-5, min: 1e-5, max: 5e-5, step: 1e-5 },
            { name: 'Batch Size', type: 'slider', value: 16, min: 8, max: 32, step: 8 },
            { name: 'Epochs', type: 'slider', value: 3, min: 1, max: 10, step: 1 }
          ],
          enabled: false
        }
      ],
      metrics: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time'],
      tips: [
        'Start with Logistic Regression for quick baseline results',
        'Random Forest works well with mixed numeric and text features',
        'XGBoost often provides the best performance for structured data',
        'DistilBERT achieves state-of-the-art results but requires more time'
      ]
    },
    image: {
      title: 'Image Data Model Training',
      description: 'Train computer vision models for image classification and detection',
      models: [
        {
          id: 'cnn-basic',
          name: 'Basic CNN',
          description: 'Convolutional Neural Network from scratch',
          icon: <Grid className="w-5 h-5" />,
          category: 'deep-learning',
          color: 'text-blue-400',
          complexity: 'intermediate',
          trainingTime: '20-60 min',
          accuracy: '75-85%',
          parameters: [
            { name: 'Conv Layers', type: 'slider', value: 3, min: 2, max: 6, step: 1 },
            { name: 'Filters', type: 'slider', value: 32, min: 16, max: 128, step: 16 },
            { name: 'Dropout', type: 'slider', value: 0.25, min: 0, max: 0.5, step: 0.05 }
          ],
          enabled: false
        },
        {
          id: 'resnet',
          name: 'ResNet',
          description: 'Residual Network with skip connections',
          icon: <Layers className="w-5 h-5" />,
          category: 'deep-learning',
          color: 'text-green-400',
          complexity: 'advanced',
          trainingTime: '45-120 min',
          accuracy: '85-92%',
          parameters: [
            { name: 'Architecture', type: 'select', value: 'resnet50', options: ['resnet18', 'resnet34', 'resnet50'] },
            { name: 'Pretrained', type: 'toggle', value: true },
            { name: 'Fine Tune', type: 'toggle', value: true }
          ],
          enabled: false
        },
        {
          id: 'efficientnet',
          name: 'EfficientNet',
          description: 'Efficient scaling of CNN architectures',
          icon: <Zap className="w-5 h-5" />,
          category: 'deep-learning',
          color: 'text-purple-400',
          complexity: 'advanced',
          trainingTime: '30-90 min',
          accuracy: '88-94%',
          parameters: [
            { name: 'Model Variant', type: 'select', value: 'b0', options: ['b0', 'b1', 'b2', 'b3'] },
            { name: 'Input Size', type: 'slider', value: 224, min: 224, max: 512, step: 32 },
            { name: 'Augmentation', type: 'toggle', value: true }
          ],
          enabled: false
        },
        {
          id: 'vision-transformer',
          name: 'Vision Transformer',
          description: 'Transformer architecture for images',
          icon: <Brain className="w-5 h-5" />,
          category: 'deep-learning',
          color: 'text-orange-400',
          complexity: 'advanced',
          trainingTime: '60-180 min',
          accuracy: '90-96%',
          parameters: [
            { name: 'Patch Size', type: 'slider', value: 16, min: 8, max: 32, step: 8 },
            { name: 'Attention Heads', type: 'slider', value: 12, min: 6, max: 16, step: 2 },
            { name: 'Layers', type: 'slider', value: 12, min: 6, max: 24, step: 2 }
          ],
          enabled: false
        }
      ],
      metrics: ['Accuracy', 'Top-5 Accuracy', 'Precision', 'Recall', 'mAP'],
      tips: [
        'Use data augmentation to improve generalization',
        'Start with pre-trained models for transfer learning',
        'Monitor GPU memory usage during training',
        'Use learning rate scheduling for better convergence'
      ]
    },
    mixed: {
      title: 'Multimodal Model Training',
      description: 'Train models that handle multiple data types simultaneously',
      models: [
        {
          id: 'multimodal-fusion',
          name: 'Early Fusion Model',
          description: 'Combine features at input level',
          icon: <Layers className="w-5 h-5" />,
          category: 'multimodal',
          color: 'text-blue-400',
          complexity: 'intermediate',
          trainingTime: '15-45 min',
          accuracy: '80-88%',
          parameters: [
            { name: 'Fusion Strategy', type: 'select', value: 'concatenate', options: ['concatenate', 'add', 'multiply'] },
            { name: 'Hidden Layers', type: 'slider', value: 3, min: 2, max: 6, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'late-fusion',
          name: 'Late Fusion Model',
          description: 'Combine predictions from separate models',
          icon: <Target className="w-5 h-5" />,
          category: 'multimodal',
          color: 'text-green-400',
          complexity: 'advanced',
          trainingTime: '30-90 min',
          accuracy: '85-92%',
          parameters: [
            { name: 'Ensemble Method', type: 'select', value: 'weighted', options: ['average', 'weighted', 'voting'] },
            { name: 'Text Weight', type: 'slider', value: 0.5, min: 0, max: 1, step: 0.1 },
            { name: 'Image Weight', type: 'slider', value: 0.5, min: 0, max: 1, step: 0.1 }
          ],
          enabled: false
        },
        {
          id: 'attention-fusion',
          name: 'Attention Fusion',
          description: 'Cross-modal attention mechanisms',
          icon: <Brain className="w-5 h-5" />,
          category: 'multimodal',
          color: 'text-purple-400',
          complexity: 'advanced',
          trainingTime: '45-120 min',
          accuracy: '88-94%',
          parameters: [
            { name: 'Attention Heads', type: 'slider', value: 8, min: 4, max: 16, step: 2 },
            { name: 'Cross Attention', type: 'toggle', value: true },
            { name: 'Temperature', type: 'slider', value: 1.0, min: 0.1, max: 2.0, step: 0.1 }
          ],
          enabled: false
        },
        {
          id: 'transformer-multimodal',
          name: 'Multimodal Transformer',
          description: 'Unified transformer for all modalities',
          icon: <Sparkles className="w-5 h-5" />,
          category: 'multimodal',
          color: 'text-orange-400',
          complexity: 'advanced',
          trainingTime: '60-180 min',
          accuracy: '90-96%',
          parameters: [
            { name: 'Model Layers', type: 'slider', value: 12, min: 6, max: 24, step: 2 },
            { name: 'Hidden Size', type: 'slider', value: 768, min: 256, max: 1024, step: 128 },
            { name: 'Modality Dropout', type: 'slider', value: 0.1, min: 0, max: 0.3, step: 0.05 }
          ],
          enabled: false
        }
      ],
      metrics: ['Overall Accuracy', 'Per-Modality Accuracy', 'Cross-Modal Correlation', 'Fusion Effectiveness'],
      tips: [
        'Balance data from different modalities',
        'Consider modality-specific preprocessing',
        'Use cross-validation across modalities',
        'Monitor individual modality contributions'
      ]
    }
  }

  useEffect(() => {
    const config = modelConfigs[datasetType]
    setModels(config.models)
  }, [datasetType])

  const currentConfig = modelConfigs[datasetType]

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const handleBackToEDA = () => {
    navigate('/exploratory-data-analysis', {
      state: {
        datasetType,
        uploadedFiles,
        augmentedSamples,
        analysisResults
      }
    })
  }

  const toggleModel = (modelId: string) => {
    setModels(prev => prev.map(model => 
      model.id === modelId ? { ...model, enabled: !model.enabled } : model
    ))
  }

  const updateParameter = (modelId: string, paramName: string, value: any) => {
    setModels(prev => prev.map(model => 
      model.id === modelId 
        ? {
            ...model,
            parameters: model.parameters.map(p => 
              p.name === paramName ? { ...p, value } : p
            )
          }
        : model
    ))
  }

  const trainModels = async () => {
    const enabledModels = models.filter(model => model.enabled)
    if (enabledModels.length === 0) {
      alert('Please select at least one model to train')
      return
    }

    if (!datasetKey) {
      alert('No dataset available for training. Please upload and process a dataset first.')
      return
    }

    // Extract model IDs for training
    const modelIds = enabledModels.map(model => model.id)
    
    // Start real training via API
    await trainRealModels(modelIds)
  }

  const enabledModels = models.filter(model => model.enabled)
  const categories = ['all', ...new Set(models.map(m => m.category))]
  const filteredModels = filterCategory === 'all' 
    ? models 
    : models.filter(m => m.category === filterCategory)

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'beginner': return 'text-green-400'
      case 'intermediate': return 'text-yellow-400'
      case 'advanced': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToEDA}
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
                <p className="text-xs text-gray-400">Model Training</p>
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
            {/* Training Summary */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Training Summary</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Available Models</span>
                  <span className="text-white font-semibold">{models.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Selected</span>
                  <span className="text-white font-semibold">{enabledModels.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Trained</span>
                  <span className="text-white font-semibold">{trainingResults.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Status</span>
                  <span className={`font-semibold flex items-center space-x-1 ${
                    isTraining ? 'text-yellow-400' : 
                    trainingResults.length > 0 ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    {isTraining ? (
                      <>
                        <div className="w-4 h-4 border-2 border-yellow-400/30 border-t-yellow-400 rounded-full animate-spin"></div>
                        <span>Training</span>
                      </>
                    ) : trainingResults.length > 0 ? (
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
              {isTraining && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-300">Progress</span>
                    <span className="text-sm text-white">{Math.round(trainingProgress)}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-accent-500 to-secondary-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                  {currentModel && (
                    <p className="text-xs text-gray-400 mt-2">Training {models.find(m => m.id === currentModel)?.name}...</p>
                  )}
                </div>
              )}

              {/* Train Models Button */}
              <div className="space-y-2">
                <button
                  onClick={trainModels}
                  disabled={enabledModels.length === 0 || isTraining || !datasetKey}
                  className={`w-full py-3 px-4 rounded-lg font-semibold transition-all transform ${
                    enabledModels.length > 0 && !isTraining && datasetKey
                      ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105'
                      : 'bg-gray-600/50 text-gray-400 cursor-not-allowed'
                  } flex items-center justify-center space-x-2`}
                >
                  {isTraining ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Training...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span>Start Training</span>
                    </>
                  )}
                </button>
                
                {!datasetKey && (
                  <p className="text-xs text-red-400 text-center">
                    No dataset available. Please upload and process data first.
                  </p>
                )}
                
                {datasetKey && (
                  <p className="text-xs text-gray-400 text-center">
                    Dataset: {datasetKey.split('/').pop()}
                  </p>
                )}

                {/* Stop Training Button */}
                {isTraining && (
                  <button
                    onClick={stopTraining}
                    className="w-full py-2 px-4 rounded-lg font-semibold transition-all bg-red-500/20 text-red-300 hover:bg-red-500/30 border border-red-500/30 flex items-center justify-center space-x-2"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop Training</span>
                  </button>
                )}
              </div>

              {/* Training Errors */}
              {trainingErrors.length > 0 && (
                <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
                  <h4 className="text-red-300 font-semibold mb-2 flex items-center space-x-2">
                    <AlertCircle className="w-4 h-4" />
                    <span>Training Errors</span>
                  </h4>
                  <div className="space-y-1">
                    {trainingErrors.map((error, index) => (
                      <p key={index} className="text-red-200 text-sm">{error}</p>
                    ))}
                  </div>
                  <button
                    onClick={() => setTrainingErrors([])}
                    className="mt-2 text-xs text-red-300 hover:text-red-200 transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>

            {/* Model Controls */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Controls</h3>
              
              {/* View Mode Toggle */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">View Mode</label>
                <div className="flex bg-white/10 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md transition-colors ${
                      viewMode === 'grid' ? 'bg-accent-500 text-white' : 'text-gray-300 hover:text-white'
                    }`}
                  >
                    <Grid className="w-4 h-4" />
                    <span className="text-sm">Grid</span>
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md transition-colors ${
                      viewMode === 'list' ? 'bg-accent-500 text-white' : 'text-gray-300 hover:text-white'
                    }`}
                  >
                    <List className="w-4 h-4" />
                    <span className="text-sm">List</span>
                  </button>
                </div>
              </div>

              {/* Category Filter */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">Category</label>
                <select
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-500"
                >
                  {categories.map(category => (
                    <option key={category} value={category} className="bg-slate-800">
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-accent-400" />
                <span>Training Tips</span>
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
            {/* Models Grid */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-6">Available Models</h3>

              <div className={`grid gap-4 ${
                viewMode === 'grid' 
                  ? 'grid-cols-1 md:grid-cols-2' 
                  : 'grid-cols-1'
              }`}>
                {filteredModels.map((model) => (
                  <div
                    key={model.id}
                    className={`bg-white/5 rounded-lg p-4 border transition-all ${
                      model.enabled 
                        ? 'border-accent-400 bg-accent-500/10' 
                        : 'border-white/10 hover:border-white/30'
                    } ${viewMode === 'list' ? 'flex items-center space-x-4' : ''}`}
                  >
                    <div className={`${viewMode === 'list' ? 'flex items-center space-x-4 flex-1' : ''}`}>
                      <div className={`w-12 h-12 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-lg flex items-center justify-center ${model.color} ${
                        viewMode === 'list' ? 'flex-shrink-0' : 'mb-4'
                      }`}>
                        {model.icon}
                      </div>
                      
                      <div className={viewMode === 'list' ? 'flex-1' : ''}>
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="text-white font-semibold">{model.name}</h4>
                          <div className="flex items-center space-x-2">
                            <span className={`text-xs px-2 py-1 rounded ${getComplexityColor(model.complexity)} bg-white/10`}>
                              {model.complexity}
                            </span>
                            <button
                              onClick={() => toggleModel(model.id)}
                              disabled={isTraining}
                              className={`w-10 h-5 rounded-full transition-colors ${
                                model.enabled ? 'bg-accent-500' : 'bg-gray-600'
                              } relative ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                              <div className={`w-4 h-4 bg-white rounded-full absolute top-0.5 transition-transform ${
                                model.enabled ? 'translate-x-5' : 'translate-x-0.5'
                              }`}></div>
                            </button>
                          </div>
                        </div>
                        
                        <p className="text-gray-300 text-sm mb-3">{model.description}</p>
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-400 mb-4">
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{model.trainingTime}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Award className="w-3 h-3" />
                            <span>{model.accuracy}</span>
                          </div>
                        </div>

                        {/* Parameters */}
                        {model.enabled && (
                          <div className="space-y-3 pt-4 border-t border-white/10">
                            {model.parameters.map((param, paramIndex) => (
                              <div key={paramIndex}>
                                <label className="block text-sm font-medium text-gray-300 mb-1">
                                  {param.name}
                                </label>
                                
                                {param.type === 'slider' && (
                                  <div className="flex items-center space-x-3">
                                    <input
                                      type="range"
                                      min={param.min}
                                      max={param.max}
                                      step={param.step}
                                      value={param.value}
                                      onChange={(e) => updateParameter(model.id, param.name, parseFloat(e.target.value))}
                                      className="flex-1 h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                      disabled={isTraining}
                                    />
                                    <span className="text-white text-sm w-16 text-right">
                                      {param.value}
                                    </span>
                                  </div>
                                )}
                                
                                {param.type === 'toggle' && (
                                  <button
                                    onClick={() => updateParameter(model.id, param.name, !param.value)}
                                    disabled={isTraining}
                                    className={`w-10 h-5 rounded-full transition-colors ${
                                      param.value ? 'bg-accent-500' : 'bg-gray-600'
                                    } relative ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                                  >
                                    <div className={`w-4 h-4 bg-white rounded-full absolute top-0.5 transition-transform ${
                                      param.value ? 'translate-x-5' : 'translate-x-0.5'
                                    }`}></div>
                                  </button>
                                )}
                                
                                {param.type === 'select' && (
                                  <select
                                    value={param.value}
                                    onChange={(e) => updateParameter(model.id, param.name, e.target.value)}
                                    disabled={isTraining}
                                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:opacity-50"
                                  >
                                    {param.options?.map(option => (
                                      <option key={option} value={option} className="bg-slate-800">
                                        {option}
                                      </option>
                                    ))}
                                  </select>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Training Results */}
            {trainingResults.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white">Training Results</h3>
                  <div className="flex items-center space-x-2">
                    <button className="p-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                    <button className="p-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors">
                      <Share2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {trainingResults.map((result) => (
                    <div key={result.id} className="bg-white/5 rounded-lg p-4 border border-white/10">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-white font-semibold">{result.modelName}</h4>
                        <span className="text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded">
                          Completed
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-accent-400">
                            {(result.accuracy * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-400">Accuracy</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-secondary-400">
                            {(result.f1Score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-400">F1-Score</div>
                        </div>
                      </div>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Precision:</span>
                          <span className="text-white">{(result.precision * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Recall:</span>
                          <span className="text-white">{(result.recall * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Training Time:</span>
                          <span className="text-white">{Math.floor(result.trainingTime / 60)}m {result.trainingTime % 60}s</span>
                        </div>
                      </div>
                      
                      <div className="mt-4 flex space-x-2">
                        <button 
                          onClick={() => downloadModel(result)}
                          className="flex-1 bg-accent-500/20 text-accent-300 py-2 px-3 rounded-lg hover:bg-accent-500/30 transition-colors text-sm flex items-center justify-center space-x-1"
                        >
                          <Download className="w-3 h-3" />
                          <span>Download</span>
                        </button>
                        <button className="flex-1 bg-secondary-500/20 text-secondary-300 py-2 px-3 rounded-lg hover:bg-secondary-500/30 transition-colors text-sm">
                          View Details
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Next Steps */}
            <div className="text-center">
              <button
                disabled={trainingResults.length === 0}
                className={`inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all transform ${
                  trainingResults.length > 0
                    ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105 shadow-lg hover:shadow-accent-500/25'
                    : 'bg-gray-600/50 text-gray-400 cursor-not-allowed'
                }`}
              >
                <span>Deploy Best Model</span>
                <ArrowRight className="w-6 h-6" />
              </button>
              
              {trainingResults.length === 0 && (
                <p className="text-gray-400 text-sm mt-3">Train at least one model to continue</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelTrainingPage