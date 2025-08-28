import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate, useLocation } from 'react-router-dom'
import { FileService } from '../services/fileService'
import { apiClient } from '../services/apiClient'
import { downloadAugmentationResult } from '../services/downloadService'
import { 
  ArrowLeft,
  ArrowRight,
  Brain,
  Shuffle,
  RotateCw,
  Zap,
  Copy,
  Layers,
  Sliders,
  Wand2,
  Settings,
  Play,
  Pause,
  Download,
  Eye,
  RefreshCw,
  Target,
  Sparkles,
  Image as ImageIcon,
  FileText,
  Database,
  User,
  LogOut,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  Filter,
  Grid,
  List
} from 'lucide-react'
import { DatasetType } from './DatasetSelectionPage'

// Interfaces
interface AugmentationTechnique {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  category: string
  color: string
  parameters: Array<{
    name: string
    type: 'slider' | 'toggle' | 'select'
    value: any
    min?: number
    max?: number
    step?: number
    options?: string[]
  }>
  enabled: boolean
}

interface AugmentationConfig {
  title: string
  description: string
  techniques: AugmentationTechnique[]
  benefits: string[]
}

// Remove the custom API definitions since we're using apiClient
const DataAugmentationPage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [datasetType, setDatasetType] = useState<DatasetType>('mixed')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [techniques, setTechniques] = useState<AugmentationTechnique[]>([])
  const [isAugmenting, setIsAugmenting] = useState(false)
  const [augmentationProgress, setAugmentationProgress] = useState(0)
  const [previewMode, setPreviewMode] = useState<'before' | 'after'>('before')
  const [selectedTechnique, setSelectedTechnique] = useState<string | null>(null)
  const [augmentedSamples, setAugmentedSamples] = useState<any[]>([])

  useEffect(() => {
    if (location.state) {
      setDatasetType(location.state.datasetType || 'mixed')
      setUploadedFiles(location.state.uploadedFiles || [])
    } else {
      navigate('/home')
    }
  }, [location.state, navigate])

  const augmentationConfigs: Record<DatasetType, AugmentationConfig> = {
    numeric: {
      title: 'Numeric Data Augmentation',
      description: 'Enhance your numeric datasets with advanced augmentation techniques',
      techniques: [
        {
          id: 'gaussian-noise',
          name: 'Gaussian Noise',
          description: 'Add random noise to improve model robustness',
          icon: <Shuffle className="w-5 h-5" />,
          category: 'noise',
          color: 'text-blue-400',
          parameters: [
            { name: 'noise_factor', type: 'slider', value: 0.1, min: 0, max: 1, step: 0.01 },
            { name: 'augmentation_factor', type: 'slider', value: 2, min: 1, max: 10, step: 1 }
          ],
          enabled: true
        },
        {
          id: 'scale-jitter',
          name: 'Scale and Jitter',
          description: 'Apply random scaling with noise augmentation',
          icon: <Sliders className="w-5 h-5" />,
          category: 'transformation',
          color: 'text-green-400',
          parameters: [
            { name: 'scale_min', type: 'slider', value: 0.8, min: 0.5, max: 1.0, step: 0.05 },
            { name: 'scale_max', type: 'slider', value: 1.2, min: 1.0, max: 2.0, step: 0.05 },
            { name: 'jitter', type: 'slider', value: 0.1, min: 0.01, max: 0.5, step: 0.01 }
          ],
          enabled: false
        },
        {
          id: 'smote',
          name: 'SMOTE Synthesis',
          description: 'Generate synthetic samples using SMOTE (requires classification target)',
          icon: <Copy className="w-5 h-5" />,
          category: 'synthesis',
          color: 'text-purple-400',
          parameters: [
            { name: 'target', type: 'select', value: 'Close', options: ['Close', 'High', 'Low', 'Open', 'Volume'] },
            { name: 'k_neighbors', type: 'slider', value: 5, min: 1, max: 10, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'outlier-injection',
          name: 'Outlier Injection',
          description: 'Inject controlled outliers for robustness',
          icon: <Target className="w-5 h-5" />,
          category: 'robustness',
          color: 'text-red-400',
          parameters: [
            { name: 'Outlier Ratio', type: 'slider', value: 0.05, min: 0, max: 0.2, step: 0.01 },
            { name: 'Outlier Strength', type: 'slider', value: 3, min: 1, max: 5, step: 0.5 }
          ],
          enabled: false
        }
      ],
      benefits: [
        'Improve model generalization',
        'Handle class imbalance',
        'Increase dataset size',
        'Enhance robustness to noise'
      ]
    },
    text: {
      title: 'Text Data Augmentation',
      description: 'Expand your text datasets with natural language augmentation methods',
      techniques: [
        {
          id: 'mixup',
          name: 'Mixup',
          description: 'Combine two text samples with weighted interpolation',
          icon: <Shuffle className="w-5 h-5" />,
          category: 'interpolation',
          color: 'text-blue-400',
          parameters: [
            { name: 'Alpha', type: 'slider', value: 0.2, min: 0.1, max: 1.0, step: 0.1 },
            { name: 'Mix Labels', type: 'toggle', value: true }
          ],
          enabled: false
        },
        {
          id: 'mlm',
          name: 'Masked Language Model',
          description: 'Use pre-trained models to replace masked tokens',
          icon: <RefreshCw className="w-5 h-5" />,
          category: 'generation',
          color: 'text-green-400',
          parameters: [
            { name: 'Mask Probability', type: 'slider', value: 0.15, min: 0.1, max: 0.3, step: 0.05 },
            { name: 'Model Type', type: 'select', value: 'bert', options: ['bert', 'roberta', 'distilbert'] }
          ],
          enabled: false
        },
        {
          id: 'random',
          name: 'Random Augmentation',
          description: 'Apply random text modifications',
          icon: <Wand2 className="w-5 h-5" />,
          category: 'modification',
          color: 'text-purple-400',
          parameters: [
            { name: 'Modification Rate', type: 'slider', value: 0.1, min: 0.05, max: 0.25, step: 0.05 },
            { name: 'Operations', type: 'select', value: 'all', options: ['all', 'deletion', 'insertion', 'substitution'] }
          ],
          enabled: false
        },
        {
          id: 'synonym',
          name: 'Synonym Replacement',
          description: 'Replace words with their synonyms using WordNet',
          icon: <Layers className="w-5 h-5" />,
          category: 'lexical',
          color: 'text-orange-400',
          parameters: [
            { name: 'Replacement Rate', type: 'slider', value: 0.2, min: 0.1, max: 0.5, step: 0.05 },
            { name: 'Preserve POS', type: 'toggle', value: true }
          ],
          enabled: false
        }
      ],
      benefits: [
        'Increase vocabulary diversity',
        'Improve model robustness',
        'Handle domain adaptation',
        'Reduce overfitting'
      ]
    },
    image: {
      title: 'Image Data Augmentation',
      description: 'Transform your image datasets with computer vision augmentation techniques',
      techniques: [
        {
          id: 'rotation',
          name: 'Random Rotation',
          description: 'Rotate images by random angles',
          icon: <RotateCw className="w-5 h-5" />,
          category: 'geometric',
          color: 'text-blue-400',
          parameters: [
            { name: 'Max Angle', type: 'slider', value: 30, min: 0, max: 180, step: 5 },
            { name: 'Fill Mode', type: 'select', value: 'reflect', options: ['reflect', 'constant', 'wrap'] }
          ],
          enabled: false
        },
        {
          id: 'color-jitter',
          name: 'Color Jittering',
          description: 'Randomly adjust brightness, contrast, saturation',
          icon: <Sparkles className="w-5 h-5" />,
          category: 'color',
          color: 'text-green-400',
          parameters: [
            { name: 'Brightness', type: 'slider', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'Contrast', type: 'slider', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'Saturation', type: 'slider', value: 0.2, min: 0, max: 1, step: 0.1 }
          ],
          enabled: false
        },
        {
          id: 'elastic-transform',
          name: 'Elastic Transform',
          description: 'Apply elastic deformations',
          icon: <Wand2 className="w-5 h-5" />,
          category: 'deformation',
          color: 'text-purple-400',
          parameters: [
            { name: 'Alpha', type: 'slider', value: 1, min: 0, max: 5, step: 0.5 },
            { name: 'Sigma', type: 'slider', value: 0.5, min: 0.1, max: 2, step: 0.1 }
          ],
          enabled: false
        },
        {
          id: 'cutout',
          name: 'Cutout',
          description: 'Randomly mask rectangular regions',
          icon: <Target className="w-5 h-5" />,
          category: 'occlusion',
          color: 'text-red-400',
          parameters: [
            { name: 'Hole Size', type: 'slider', value: 16, min: 4, max: 64, step: 4 },
            { name: 'Number of Holes', type: 'slider', value: 1, min: 1, max: 5, step: 1 }
          ],
          enabled: false
        },
        {
          id: 'mixup',
          name: 'MixUp',
          description: 'Blend images and labels',
          icon: <Layers className="w-5 h-5" />,
          category: 'blending',
          color: 'text-cyan-400',
          parameters: [
            { name: 'Alpha', type: 'slider', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'Beta', type: 'slider', value: 0.2, min: 0, max: 1, step: 0.1 }
          ],
          enabled: false
        }
      ],
      benefits: [
        'Improve model generalization',
        'Reduce overfitting',
        'Handle lighting variations',
        'Increase dataset diversity'
      ]
    },
    mixed: {
      title: 'Multimodal Data Augmentation',
      description: 'Comprehensive augmentation strategies for mixed data types',
      techniques: [
        {
          id: 'cross-modal-synthesis',
          name: 'Cross-Modal Synthesis',
          description: 'Generate data across modalities',
          icon: <Layers className="w-5 h-5" />,
          category: 'synthesis',
          color: 'text-blue-400',
          parameters: [
            { name: 'Synthesis Rate', type: 'slider', value: 0.2, min: 0, max: 0.5, step: 0.05 },
            { name: 'Quality Threshold', type: 'slider', value: 0.8, min: 0.5, max: 1, step: 0.05 }
          ],
          enabled: false
        },
        {
          id: 'modality-dropout',
          name: 'Modality Dropout',
          description: 'Randomly drop modalities during training',
          icon: <Filter className="w-5 h-5" />,
          category: 'regularization',
          color: 'text-green-400',
          parameters: [
            { name: 'Dropout Rate', type: 'slider', value: 0.1, min: 0, max: 0.5, step: 0.05 },
            { name: 'Preserve Primary', type: 'toggle', value: true }
          ],
          enabled: false
        },
        {
          id: 'feature-mixing',
          name: 'Feature Mixing',
          description: 'Mix features across different samples',
          icon: <Shuffle className="w-5 h-5" />,
          category: 'mixing',
          color: 'text-purple-400',
          parameters: [
            { name: 'Mix Ratio', type: 'slider', value: 0.3, min: 0, max: 1, step: 0.1 },
            { name: 'Preserve Correlations', type: 'toggle', value: true }
          ],
          enabled: false
        },
        {
          id: 'adaptive-augmentation',
          name: 'Adaptive Augmentation',
          description: 'Automatically adjust augmentation based on data',
          icon: <Brain className="w-5 h-5" />,
          category: 'adaptive',
          color: 'text-orange-400',
          parameters: [
            { name: 'Adaptation Rate', type: 'slider', value: 0.1, min: 0, max: 0.5, step: 0.05 },
            { name: 'Learning Window', type: 'slider', value: 100, min: 50, max: 500, step: 50 }
          ],
          enabled: false
        }
      ],
      benefits: [
        'Improve cross-modal learning',
        'Handle missing modalities',
        'Increase data diversity',
        'Enhance model robustness'
      ]
    }
  }

  useEffect(() => {
    const config = augmentationConfigs[datasetType]
    setTechniques(config.techniques)
  }, [datasetType])

  const currentConfig = augmentationConfigs[datasetType]

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const handleDownload = async () => {
    if (!user?.id) {
      alert('User not authenticated')
      return
    }

    // Find the latest augmented sample with an output_key
    const samplesWithOutputKey = augmentedSamples.filter(sample => 
      sample.metadata?.output_key
    )

    if (samplesWithOutputKey.length === 0) {
      alert('No augmented files available for download. Please run augmentation first.')
      return
    }

    // Get the output key from the last sample
    const lastSample = samplesWithOutputKey[samplesWithOutputKey.length - 1]
    const outputKey = lastSample.metadata.output_key

    try {
      await downloadAugmentationResult(
        outputKey,
        user.id,
        `${datasetType}_augmented_${lastSample.techniques.toLowerCase().replace(/\s+/g, '_')}.csv`
      )
    } catch (error) {
      console.error('Download failed:', error)
      alert('Download failed. Please try again.')
    }
  }

  const handleBackToPreprocessing = () => {
    navigate('/preprocessing', {
      state: {
        datasetType,
        uploadedFiles
      }
    })
  }

  const handleProceedToEDA = () => {
    navigate('/exploratory-data-analysis', {
      state: {
        datasetType,
        uploadedFiles,
        augmentedSamples
      }
    })
  }

  const toggleTechnique = (techniqueId: string) => {
    setTechniques(prev => prev.map(t => 
      t.id === techniqueId ? { ...t, enabled: !t.enabled } : t
    ))
  }

  const updateParameter = (techniqueId: string, paramName: string, value: any) => {
    setTechniques(prev => prev.map(t => 
      t.id === techniqueId 
        ? {
            ...t,
            parameters: t.parameters.map(p => 
              p.name === paramName ? { ...p, value } : p
            )
          }
        : t
    ))
  }

  const runAugmentation = async () => {
    const enabledTechniques = techniques.filter(t => t.enabled)
    if (enabledTechniques.length === 0) return

    setIsAugmenting(true)
    setAugmentationProgress(0)

    try {
      let allAugmentedSamples: any[] = []

      // Check if mixup is enabled
      const mixupTechnique = enabledTechniques.find(t => t.id === 'mixup')
      
      if (mixupTechnique) {
        console.log('Mixup technique enabled, calling API...')
        
        // Progress update
        setAugmentationProgress(20)

        // Get real user files from Supabase instead of demo data
        const userFiles = await FileService.getUserFiles(user?.id || '', 'uploaded')
        setAugmentationProgress(20)

        let realTexts: string[] = []

        // Download and parse actual file content
        for (const file of userFiles.slice(0, 3)) { // Limit to 3 files for demo
          const fileContent = await FileService.downloadFileContent(file.storagePath)
          
          if (file.fileType.includes('csv')) {
            // Parse CSV and extract text columns
            const parsedData = FileService.parseCSV(fileContent)
            const extractedTexts = FileService.extractTextFromCSV(parsedData)
            realTexts = [...realTexts, ...extractedTexts.slice(0, 5)] // Limit to 5 texts per file
          } else if (file.fileType.includes('text')) {
            // For plain text files
            realTexts.push(fileContent)
          }
        }

        // Fallback to demo data if no real files or texts found
        if (realTexts.length < 2) {
          console.log('Using demo data as fallback')
          realTexts = [
            "Machine learning is a powerful tool for data analysis.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can process complex patterns in data.",
            "Artificial intelligence is transforming various industries.",
            "Data augmentation improves model performance by increasing dataset diversity."
          ]
        }

        // Get alpha parameter from technique settings
        const alphaParam = mixupTechnique.parameters.find(p => p.name === 'Alpha')
        const mixLabelsParam = mixupTechnique.parameters.find(p => p.name === 'Mix Labels')
        
        // Get dataset_key from uploaded files
        const datasetKey = uploadedFiles?.[0]?.dataset_key
        if (!datasetKey) {
          console.error('No dataset_key found in uploaded files')
          return
        }

        const mixupRequest = {
          texts: realTexts,
          alpha: alphaParam?.value || 0.2,
          mix_labels: mixLabelsParam?.value || true
        }

        setAugmentationProgress(40)

        try {
          const mixupResponse = await apiClient.augmentWithMixup(mixupRequest)
          console.log('Mixup API Response:', mixupResponse)
          
          if (mixupResponse.success) {
            // Convert API response to our format - apiClient returns different structure
            const mixupSamples = (mixupResponse.meta?.augmented_samples || []).map((sample: any, index: number) => ({
              id: `mixup-${index}`,
              original: sample.original || `Original text ${index}`,
              augmented: sample.augmented || sample.text || sample,
              techniques: 'Mixup',
              metadata: {
                alpha_used: sample.alpha_used || mixupRequest.alpha,
                mixed_with_index: sample.mixed_with_index || index
              }
            }))
            
            allAugmentedSamples = [...allAugmentedSamples, ...mixupSamples]
          }
        } catch (apiError) {
          console.error('Mixup API call failed, falling back to mock data:', apiError)
          // Fallback to mock data if API fails
          const fallbackSamples = realTexts.map((text, i) => ({
            id: `mixup-fallback-${i}`,
            original: text,
            augmented: `[Mock Mixup] ${text}`,
            techniques: 'Mixup (Fallback)',
            metadata: { error: 'API unavailable' }
          }))
          allAugmentedSamples = [...allAugmentedSamples, ...fallbackSamples]
        }

        setAugmentationProgress(70)
      }

      // Handle other techniques with API calls
      const otherTechniques = enabledTechniques.filter(t => t.id !== 'mixup')
      if (otherTechniques.length > 0) {
        setAugmentationProgress(80)
        
        // Get dataset_key from uploaded files
        const datasetKey = uploadedFiles?.[0]?.dataset_key || uploadedFiles?.[0]?.output_key
        if (!datasetKey) {
          console.error('No dataset_key found in uploaded files for augmentation')
          throw new Error('No dataset available for augmentation')
        }

        // Process each enabled technique
        for (const technique of otherTechniques) {
          try {
            let response: any = null
            const params: Record<string, any> = {}
            
            // Convert technique parameters to API format
            technique.parameters.forEach(param => {
              params[param.name] = param.value
            })

            const baseRequest = {
              user_id: user?.id || '',
              dataset_key: datasetKey,
              technique: technique.id,
              params: params
            }

            // Call appropriate API based on technique
            switch (technique.id) {
              case 'scale-jitter':
                response = await apiClient.augmentWithScale(baseRequest)
                break
              case 'smote':
                response = await apiClient.augmentWithSMOTE(baseRequest)
                break
              case 'gaussian-noise':
                response = await apiClient.augmentWithNoise(baseRequest)
                break
              default:
                console.warn(`Technique ${technique.id} not implemented yet`)
                continue
            }

            if (response && response.success) {
              console.log(`${technique.name} augmentation completed:`, response)
              // For numeric augmentation, we don't have individual samples to show
              // but we can show the overall result
              const augmentationSample = {
                id: `${technique.id}-result`,
                original: `Original dataset (${response.original_size || 'unknown'} samples)`,
                augmented: `Augmented dataset (${response.augmented_size || 'unknown'} samples)`,
                techniques: technique.name,
                metadata: {
                  technique: technique.id,
                  original_size: response.original_size,
                  augmented_size: response.augmented_size,
                  augmentation_ratio: response.augmentation_ratio,
                  output_key: response.output_key
                }
              }
              allAugmentedSamples.push(augmentationSample)
            }
          } catch (error) {
            console.error(`${technique.name} augmentation failed:`, error)
            // Add error sample
            const errorSample = {
              id: `${technique.id}-error`,
              original: 'Augmentation failed',
              augmented: `Error: ${error}`,
              techniques: technique.name,
              metadata: { error: true }
            }
            allAugmentedSamples.push(errorSample)
          }
        }
      }

      // Final progress update
      setAugmentationProgress(100)

      // Set all augmented samples
      setAugmentedSamples(allAugmentedSamples)
      
    } catch (error) {
      console.error('Augmentation failed:', error)
      // Fallback to original mock behavior
      const mockSamples = Array.from({ length: 3 }, (_, i) => ({
        id: `error-fallback-${i}`,
        original: `Original Sample ${i + 1}`,
        augmented: `Error: Could not augment sample ${i + 1}`,
        techniques: 'Error'
      }))
      setAugmentedSamples(mockSamples)
    } finally {
      setIsAugmenting(false)
    }
  }

  const enabledTechniques = techniques.filter(t => t.enabled)
  const categories = [...new Set(techniques.map(t => t.category))]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToPreprocessing}
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
                <p className="text-xs text-gray-400">Data Augmentation</p>
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
            {/* Augmentation Summary */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Augmentation Summary</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Enabled Techniques</span>
                  <span className="text-white font-semibold">{enabledTechniques.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Original Samples</span>
                  <span className="text-white font-semibold">{uploadedFiles.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Expected Output</span>
                  <span className="text-white font-semibold">
                    {uploadedFiles.length * (enabledTechniques.length + 1)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Status</span>
                  <span className={`font-semibold flex items-center space-x-1 ${
                    isAugmenting ? 'text-yellow-400' : 
                    augmentedSamples.length > 0 ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    {isAugmenting ? (
                      <>
                        <div className="w-4 h-4 border-2 border-yellow-400/30 border-t-yellow-400 rounded-full animate-spin"></div>
                        <span>Processing</span>
                      </>
                    ) : augmentedSamples.length > 0 ? (
                      <>
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Complete</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-4 h-4" />
                        <span>Ready</span>
                      </>
                    )}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              {isAugmenting && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-300">Progress</span>
                    <span className="text-sm text-white">{augmentationProgress}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-accent-500 to-secondary-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${augmentationProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Run Augmentation Button */}
              <button
                onClick={runAugmentation}
                disabled={enabledTechniques.length === 0 || isAugmenting}
                className={`w-full mt-4 py-3 px-4 rounded-lg font-semibold transition-all transform ${
                  enabledTechniques.length > 0 && !isAugmenting
                    ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105'
                    : 'bg-gray-600/50 text-gray-400 cursor-not-allowed'
                } flex items-center justify-center space-x-2`}
              >
                {isAugmenting ? (
                  <>
                    <Pause className="w-5 h-5" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Run Augmentation</span>
                  </>
                )}
              </button>

              {/* Skip Augmentation Button */}
              <button
                onClick={handleProceedToEDA}
                disabled={isAugmenting}
                className={`w-full mt-3 py-2 px-4 rounded-lg font-medium transition-all ${
                  !isAugmenting
                    ? 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/20 hover:border-white/30'
                    : 'bg-gray-600/30 text-gray-500 cursor-not-allowed border border-gray-600/30'
                } flex items-center justify-center space-x-2`}
              >
                <ArrowRight className="w-4 h-4" />
                <span>Skip Augmentation</span>
              </button>
            </div>

            {/* Benefits */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-accent-400" />
                <span>Benefits</span>
              </h3>
              <div className="space-y-3 text-sm text-gray-300">
                {currentConfig.benefits.map((benefit, index) => (
                  <p key={index} className="flex items-start space-x-2">
                    <span className="text-accent-400 mt-1">•</span>
                    <span>{benefit}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Techniques Grid */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-6">Augmentation Techniques</h3>

              {categories.map(category => (
                <div key={category} className="mb-8">
                  <h4 className="text-md font-semibold text-gray-300 mb-4 capitalize">
                    {category} Techniques
                  </h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {techniques
                      .filter(t => t.category === category)
                      .map((technique) => (
                        <div
                          key={technique.id}
                          className={`bg-white/5 rounded-lg p-4 border transition-all ${
                            technique.enabled 
                              ? 'border-accent-400 bg-accent-500/10' 
                              : 'border-white/10 hover:border-white/30'
                          }`}
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center space-x-3">
                              <div className={`w-10 h-10 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-lg flex items-center justify-center ${technique.color}`}>
                                {technique.icon}
                              </div>
                              <div>
                                <h5 className="text-white font-semibold">{technique.name}</h5>
                                <p className="text-gray-300 text-sm">{technique.description}</p>
                              </div>
                            </div>
                            
                            <button
                              onClick={() => toggleTechnique(technique.id)}
                              className={`w-12 h-6 rounded-full transition-colors ${
                                technique.enabled ? 'bg-accent-500' : 'bg-gray-600'
                              } relative`}
                            >
                              <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                                technique.enabled ? 'translate-x-6' : 'translate-x-0.5'
                              }`}></div>
                            </button>
                          </div>

                          {/* Parameters */}
                          {technique.enabled && (
                            <div className="space-y-3 pt-4 border-t border-white/10">
                              {technique.parameters.map((param, paramIndex) => (
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
                                        onChange={(e) => updateParameter(technique.id, param.name, parseFloat(e.target.value))}
                                        className="flex-1 h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                      />
                                      <span className="text-white text-sm w-12 text-right">
                                        {param.value}
                                      </span>
                                    </div>
                                  )}
                                  
                                  {param.type === 'toggle' && (
                                    <button
                                      onClick={() => updateParameter(technique.id, param.name, !param.value)}
                                      className={`w-10 h-5 rounded-full transition-colors ${
                                        param.value ? 'bg-accent-500' : 'bg-gray-600'
                                      } relative`}
                                    >
                                      <div className={`w-4 h-4 bg-white rounded-full absolute top-0.5 transition-transform ${
                                        param.value ? 'translate-x-5' : 'translate-x-0.5'
                                      }`}></div>
                                    </button>
                                  )}
                                  
                                  {param.type === 'select' && (
                                    <select
                                      value={param.value}
                                      onChange={(e) => updateParameter(technique.id, param.name, e.target.value)}
                                      className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-500"
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
                      ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Results Preview */}
            {augmentedSamples.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white">Augmentation Results</h3>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setPreviewMode(previewMode === 'before' ? 'after' : 'before')}
                      className="flex items-center space-x-2 px-4 py-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors"
                    >
                      <Eye className="w-4 h-4" />
                      <span>{previewMode === 'before' ? 'Show Augmented' : 'Show Original'}</span>
                    </button>
                    <button 
                      onClick={handleDownload}
                      className="p-2 bg-white/10 text-gray-300 rounded-lg hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      disabled={!augmentedSamples.some(sample => sample.metadata?.output_key)}
                      title="Download augmented data"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {augmentedSamples.map((sample) => (
                    <div key={sample.id} className="bg-white/5 rounded-lg p-4 border border-white/10">
                      <div className="bg-gradient-to-br from-accent-500/20 to-secondary-500/20 rounded-lg p-6 mb-4">
                        <div className="flex items-center justify-center h-24 text-gray-300">
                          {datasetType === 'image' ? (
                            <ImageIcon className="w-8 h-8" />
                          ) : datasetType === 'text' ? (
                            <FileText className="w-8 h-8" />
                          ) : (
                            <Database className="w-8 h-8" />
                          )}
                        </div>
                      </div>
                      
                      <h4 className="text-white font-semibold mb-2">
                        {previewMode === 'before' ? sample.original : sample.augmented}
                      </h4>
                      <p className="text-gray-300 text-sm mb-2">
                        Techniques: {sample.techniques}
                      </p>
                      <span className="text-xs text-gray-400 bg-white/10 px-2 py-1 rounded">
                        {previewMode === 'before' ? 'Original' : 'Augmented'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Next Steps */}
            <div className="text-center">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={handleProceedToEDA}
                  className="inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all transform bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105 shadow-lg hover:shadow-accent-500/25"
                >
                  <span>Proceed to Data Exploration</span>
                  <ArrowRight className="w-6 h-6" />
                </button>
              </div>
              
              <div className="mt-4 text-center">
                {augmentedSamples.length > 0 ? (
                  <p className="text-green-400 text-sm">
                    ✅ {augmentedSamples.length} augmentation sample{augmentedSamples.length !== 1 ? 's' : ''} generated
                  </p>
                ) : (
                  <p className="text-gray-400 text-sm">
                    You can proceed with current data or complete augmentation steps first
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

export default DataAugmentationPage