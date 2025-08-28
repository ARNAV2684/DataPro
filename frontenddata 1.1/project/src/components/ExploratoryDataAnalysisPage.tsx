import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate, useLocation } from 'react-router-dom'
import { apiClient } from '../services/apiClient'
import { 
  ArrowLeft,
  ArrowRight,
  Brain,
  BarChart3,
  PieChart,
  TrendingUp,
  Database,
  FileText,
  Image as ImageIcon,
  Download,
  Eye,
  Zap,
  Settings,
  Filter,
  Search,
  Grid,
  List,
  Play,
  Pause,
  RotateCcw,
  Share2,
  BookOpen,
  Target,
  Layers,
  Activity,
  User,
  LogOut
} from 'lucide-react'
import { DatasetType } from './DatasetSelectionPage'

interface AnalysisResult {
  id: string
  title: string
  description: string
  type: 'chart' | 'table' | 'metric' | 'insight' | 'visualization'
  data: any
  visualization?: string
  imageUrls?: string[]
}

interface AnalysisParameters {
  // Statistical Analysis
  no_plots?: boolean
  plots_only?: boolean
  
  // Correlation Analysis
  threshold?: number
  method?: 'pearson' | 'spearman' | 'both'
  
  // Advanced Visualization
  pca_components?: number
  sample_size?: number
  skip_pca?: boolean
  skip_pairs?: boolean
  
  // EDA Manager
  technique?: string
  all?: boolean
  
  // Common
  output_format?: string
}

interface EDAConfig {
  title: string
  description: string
  analyses: Array<{
    name: string
    icon: React.ReactNode
    description: string
    color: string
    category: string
  }>
  insights: string[]
}

const ExploratoryDataAnalysisPage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [datasetType, setDatasetType] = useState<DatasetType>('mixed')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [augmentedSamples, setAugmentedSamples] = useState<any[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([])
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [showParameterConfig, setShowParameterConfig] = useState(false)
  const [currentAnalysisName, setCurrentAnalysisName] = useState<string>('')
  const [analysisParameters, setAnalysisParameters] = useState<AnalysisParameters>({
    threshold: 0.7,
    method: 'both',
    pca_components: undefined,
    sample_size: 1000,
    skip_pca: false,
    skip_pairs: false,
    no_plots: false,
    plots_only: false,
    output_format: 'json'
  })

  useEffect(() => {
    if (location.state) {
      setDatasetType(location.state.datasetType || 'mixed')
      setUploadedFiles(location.state.uploadedFiles || [])
      setAugmentedSamples(location.state.augmentedSamples || [])
    } else {
      // Fallback: If no state, try to get the last dataset from local storage or redirect to home
      const lastDatasetType = localStorage.getItem('lastDatasetType')
      const lastUploadedFiles = localStorage.getItem('lastUploadedFiles')
      
      if (lastDatasetType && lastUploadedFiles) {
        try {
          setDatasetType(lastDatasetType as DatasetType)
          setUploadedFiles(JSON.parse(lastUploadedFiles))
          console.log('Restored dataset from localStorage:', JSON.parse(lastUploadedFiles))
        } catch (error) {
          console.error('Failed to restore dataset from localStorage:', error)
          navigate('/home')
        }
      } else {
        console.log('No uploaded files found, redirecting to home')
        navigate('/home')
      }
    }
  }, [location.state, navigate])

  const edaConfigs: Record<DatasetType, EDAConfig> = {
    numeric: {
      title: 'Numeric Data Exploration',
      description: 'Comprehensive statistical analysis and visualization of your numeric datasets using CLI tools',
      analyses: [
        {
          name: 'Statistical Analysis',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Comprehensive statistical analysis with descriptive statistics, distributions, and summary metrics',
          color: 'text-blue-400',
          category: 'statistics'
        },
        {
          name: 'Correlation Analysis',
          icon: <Grid className="w-5 h-5" />,
          description: 'Feature relationships and dependencies with correlation matrices and heatmaps',
          color: 'text-purple-400',
          category: 'correlation'
        },
        {
          name: 'Advanced Visualization',
          icon: <TrendingUp className="w-5 h-5" />,
          description: 'Advanced plots including PCA analysis, pair plots, and distribution visualizations',
          color: 'text-green-400',
          category: 'visualization'
        },
        {
          name: 'Comprehensive EDA',
          icon: <Target className="w-5 h-5" />,
          description: 'Complete EDA suite running all available analysis techniques',
          color: 'text-red-400',
          category: 'comprehensive'
        }
      ],
      insights: [
        'Comprehensive statistical summaries and descriptive analytics',
        'Correlation analysis with customizable thresholds and methods',
        'Advanced visualizations including PCA and pair plots',
        'Complete EDA pipeline with all techniques in one analysis'
      ]
    },
    text: {
      title: 'Text Data Exploration',
      description: 'Natural language processing and linguistic analysis of your text data',
      analyses: [
        {
          name: 'Word Frequency',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Most common words and phrases',
          color: 'text-blue-400',
          category: 'frequency'
        },
        {
          name: 'Sentiment Distribution',
          icon: <TrendingUp className="w-5 h-5" />,
          description: 'Emotional tone analysis',
          color: 'text-green-400',
          category: 'sentiment'
        },
        {
          name: 'Topic Modeling',
          icon: <Brain className="w-5 h-5" />,
          description: 'Hidden themes and topics',
          color: 'text-purple-400',
          category: 'topics'
        },
        {
          name: 'Text Length Analysis',
          icon: <FileText className="w-5 h-5" />,
          description: 'Document and sentence statistics',
          color: 'text-orange-400',
          category: 'structure'
        },
        {
          name: 'Language Detection',
          icon: <Search className="w-5 h-5" />,
          description: 'Identify languages in text',
          color: 'text-cyan-400',
          category: 'language'
        },
        {
          name: 'Named Entity Recognition',
          icon: <Target className="w-5 h-5" />,
          description: 'Extract people, places, organizations',
          color: 'text-red-400',
          category: 'entities'
        }
      ],
      insights: [
        'Understand vocabulary richness and diversity',
        'Identify dominant themes and topics',
        'Analyze sentiment patterns across documents',
        'Discover linguistic patterns and structures'
      ]
    },
    image: {
      title: 'Image Data Exploration',
      description: 'Computer vision analysis and visual pattern discovery in your image datasets',
      analyses: [
        {
          name: 'Image Statistics',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Resolution, format, size distribution',
          color: 'text-blue-400',
          category: 'metadata'
        },
        {
          name: 'Color Analysis',
          icon: <PieChart className="w-5 h-5" />,
          description: 'Color distribution and palettes',
          color: 'text-green-400',
          category: 'color'
        },
        {
          name: 'Feature Extraction',
          icon: <Brain className="w-5 h-5" />,
          description: 'Visual features and patterns',
          color: 'text-purple-400',
          category: 'features'
        },
        {
          name: 'Object Detection',
          icon: <Eye className="w-5 h-5" />,
          description: 'Identify objects in images',
          color: 'text-orange-400',
          category: 'objects'
        },
        {
          name: 'Similarity Analysis',
          icon: <Grid className="w-5 h-5" />,
          description: 'Find similar images',
          color: 'text-cyan-400',
          category: 'similarity'
        },
        {
          name: 'Quality Assessment',
          icon: <Target className="w-5 h-5" />,
          description: 'Blur, noise, brightness analysis',
          color: 'text-red-400',
          category: 'quality'
        }
      ],
      insights: [
        'Understand image quality and consistency',
        'Identify dominant colors and visual themes',
        'Discover patterns in image composition',
        'Assess dataset balance and diversity'
      ]
    },
    mixed: {
      title: 'Multimodal Data Exploration',
      description: 'Comprehensive analysis across multiple data types and modalities',
      analyses: [
        {
          name: 'Data Type Distribution',
          icon: <PieChart className="w-5 h-5" />,
          description: 'Breakdown by data modality',
          color: 'text-blue-400',
          category: 'overview'
        },
        {
          name: 'Cross-Modal Patterns',
          icon: <Grid className="w-5 h-5" />,
          description: 'Relationships between data types',
          color: 'text-green-400',
          category: 'correlation'
        },
        {
          name: 'Unified Statistics',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Combined statistical overview',
          color: 'text-purple-400',
          category: 'statistics'
        },
        {
          name: 'Quality Assessment',
          icon: <Target className="w-5 h-5" />,
          description: 'Data quality across modalities',
          color: 'text-orange-400',
          category: 'quality'
        },
        {
          name: 'Feature Alignment',
          icon: <Layers className="w-5 h-5" />,
          description: 'Feature correspondence analysis',
          color: 'text-cyan-400',
          category: 'alignment'
        },
        {
          name: 'Multimodal Insights',
          icon: <Brain className="w-5 h-5" />,
          description: 'Cross-modal discoveries',
          color: 'text-red-400',
          category: 'insights'
        }
      ],
      insights: [
        'Understand data distribution across modalities',
        'Identify complementary information sources',
        'Discover cross-modal correlations',
        'Assess integration opportunities'
      ]
    }
  }

  const currentConfig = edaConfigs[datasetType]

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const handleBackToAugmentation = () => {
    navigate('/data-augmentation', {
      state: {
        datasetType,
        uploadedFiles
      }
    })
  }

  const handleProceedToModelTraining = () => {
    navigate('/model-training', {
      state: {
        datasetType,
        uploadedFiles,
        augmentedSamples,
        analysisResults
      }
    })
  }

  const showAnalysisConfig = (analysisName: string) => {
    setCurrentAnalysisName(analysisName)
    setShowParameterConfig(true)
  }

  const runAnalysis = async (analysisName: string, customParams?: AnalysisParameters) => {
    setIsAnalyzing(true)
    setSelectedAnalysis(analysisName)
    setShowParameterConfig(false)

    // Check if user is available
    if (!user?.id) {
      console.error('User not authenticated')
      setIsAnalyzing(false)
      setSelectedAnalysis(null)
      return
    }

    // Get dataset_key from uploaded files
    if (!uploadedFiles || uploadedFiles.length === 0) {
      console.error('No uploaded files available for EDA')
      setIsAnalyzing(false)
      setSelectedAnalysis(null)
      return
    }

    // Debug: Log the uploaded files structure
    console.log('Uploaded files structure:', uploadedFiles[0])

    // Try different possible property names for the dataset key
    const datasetKey = uploadedFiles[0].dataset_key || 
                      uploadedFiles[0].output_key || 
                      uploadedFiles[0].storage_key ||
                      uploadedFiles[0].key
    
    if (!datasetKey) {
      console.error('No dataset_key found in uploaded files. Available properties:', Object.keys(uploadedFiles[0]))
      setIsAnalyzing(false)
      setSelectedAnalysis(null)
      return
    }

    console.log('Using dataset key:', datasetKey)

    try {
      let response

      // Merge default parameters with custom parameters
      const finalParams = { ...analysisParameters, ...customParams }

      // Call appropriate API based on analysis name
      const request = {
        user_id: user.id,
        dataset_key: datasetKey,
        analysis_type: analysisName,
        params: finalParams
      }

      switch (analysisName.toLowerCase().replace(/\s+/g, '-')) {
        case 'statistical-analysis':
          response = await apiClient.runStatisticalAnalysis(request)
          break

        case 'correlation-analysis':
          response = await apiClient.runCorrelationAnalysis(request)
          break

        case 'advanced-visualization':
          response = await apiClient.runAdvancedVisualization(request)
          break

        case 'comprehensive-eda':
          response = await apiClient.runEDAManager(request)
          break

        case 'numeric-summary':
        case 'numeric-manager':
          response = await apiClient.runNumericEDAManager(request)
          break

        case 'sentiment-analysis':
          response = await apiClient.runSentimentAnalysis(request)
          break

        case 'word-frequency':
          response = await apiClient.runWordFrequencyAnalysis(request)
          break

        case 'text-length':
          response = await apiClient.runTextLengthAnalysis(request)
          break

        case 'topic-modeling':
          response = await apiClient.runTopicModeling(request)
          break

        case 'ngram-analysis':
          response = await apiClient.runNgramAnalysis(request)
          break

        default:
          console.log(`Analysis ${analysisName} - simulating processing (no API endpoint available)`)
          await new Promise(resolve => setTimeout(resolve, 3000))
          response = {
            success: true,
            message: `${analysisName} completed`,
            meta: {
              analysis_results: {
                values: Array.from({ length: 10 }, () => Math.floor(Math.random() * 100)),
                labels: Array.from({ length: 10 }, (_, i) => `Item ${i + 1}`)
              }
            }
          }
      }

      if (response.success) {
        // Extract visualization URLs from the response
        const visualizationUrls = response.meta?.visualization_urls || []
        const hasVisualizations = visualizationUrls.length > 0
        
        const mockResult: AnalysisResult = {
          id: `analysis-${Date.now()}`,
          title: analysisName,
          description: response.message || `Completed ${analysisName} analysis`,
          type: hasVisualizations ? 'visualization' : 'chart',
          data: response.meta?.analysis_results || response.meta || {
            values: Array.from({ length: 10 }, () => Math.floor(Math.random() * 100)),
            labels: Array.from({ length: 10 }, (_, i) => `Item ${i + 1}`)
          },
          visualization: 'bar-chart',
          imageUrls: hasVisualizations ? visualizationUrls.map((viz: any) => viz.url) : undefined
        }

        setAnalysisResults(prev => [...prev, mockResult])
        console.log(`✅ ${analysisName} completed successfully:`, response.message)
        if (hasVisualizations) {
          console.log(`📊 Generated ${visualizationUrls.length} visualizations:`, visualizationUrls)
        }
      } else {
        throw new Error(response.message || 'Analysis failed')
      }

    } catch (error) {
      console.error(`❌ ${analysisName} failed:`, error)
      // Still add a result indicating the error
      const errorResult: AnalysisResult = {
        id: `analysis-error-${Date.now()}`,
        title: analysisName,
        description: `Failed to complete ${analysisName}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'metric',
        data: { error: true }
      }
      setAnalysisResults(prev => [...prev, errorResult])
    } finally {
      setIsAnalyzing(false)
      setSelectedAnalysis(null)
    }
  }

  const categories = ['all', ...new Set(currentConfig.analyses.map(a => a.category))]
  const filteredAnalyses = filterCategory === 'all' 
    ? currentConfig.analyses 
    : currentConfig.analyses.filter(a => a.category === filterCategory)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToAugmentation}
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
                <p className="text-xs text-gray-400">Exploratory Data Analysis</p>
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
          {/* Analysis Controls */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Analysis Controls</h3>
              
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

              {/* Dataset Info */}
              <div className="bg-white/5 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-white mb-2">Dataset Info</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Files:</span>
                    <span className="text-white">{uploadedFiles.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Type:</span>
                    <span className="text-white capitalize">{datasetType}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Augmented:</span>
                    <span className="text-white">{augmentedSamples.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Analyses:</span>
                    <span className="text-white">{analysisResults.length}</span>
                  </div>
                </div>
              </div>

              {/* Skip Analysis Button */}
              <button
                onClick={handleProceedToModelTraining}
                disabled={isAnalyzing}
                className={`w-full mt-4 py-2 px-4 rounded-lg font-medium transition-all ${
                  !isAnalyzing
                    ? 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/20 hover:border-white/30'
                    : 'bg-gray-600/30 text-gray-500 cursor-not-allowed border border-gray-600/30'
                } flex items-center justify-center space-x-2`}
              >
                <ArrowRight className="w-4 h-4" />
                <span>Skip Analysis</span>
              </button>
            </div>

            {/* Key Insights */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                <BookOpen className="w-5 h-5 text-accent-400" />
                <span>Key Insights</span>
              </h3>
              <div className="space-y-3 text-sm text-gray-300">
                {currentConfig.insights.map((insight, index) => (
                  <p key={index} className="flex items-start space-x-2">
                    <span className="text-accent-400 mt-1">•</span>
                    <span>{insight}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Analysis Grid */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">Available Analyses</h3>
                <div className="flex items-center space-x-2">
                  {isAnalyzing && (
                    <div className="flex items-center space-x-2 text-accent-400">
                      <div className="w-4 h-4 border-2 border-accent-400/30 border-t-accent-400 rounded-full animate-spin"></div>
                      <span className="text-sm">Running {selectedAnalysis}...</span>
                    </div>
                  )}
                </div>
              </div>

              <div className={`grid gap-4 ${
                viewMode === 'grid' 
                  ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' 
                  : 'grid-cols-1'
              }`}>
                {filteredAnalyses.map((analysis, index) => (
                  <div
                    key={index}
                    className={`bg-white/5 rounded-lg p-4 border border-white/10 hover:border-white/30 transition-all group ${
                      viewMode === 'list' ? 'flex items-center space-x-4' : ''
                    }`}
                  >
                    <div className={`w-12 h-12 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-lg flex items-center justify-center ${analysis.color} group-hover:scale-110 transition-transform ${
                      viewMode === 'list' ? 'flex-shrink-0' : 'mb-4'
                    }`}>
                      {analysis.icon}
                    </div>
                    
                    <div className={viewMode === 'list' ? 'flex-1' : ''}>
                      <h4 className="text-white font-semibold mb-2">{analysis.name}</h4>
                      <p className="text-gray-300 text-sm mb-4">{analysis.description}</p>
                      
                      <div className="flex space-x-2">
                        <button
                          onClick={() => runAnalysis(analysis.name)}
                          disabled={isAnalyzing}
                          className="flex-1 bg-gradient-to-r from-accent-500 to-secondary-500 text-white py-2 px-4 rounded-lg hover:from-accent-600 hover:to-secondary-600 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                        >
                          <Play className="w-4 h-4" />
                          <span>Run</span>
                        </button>
                        
                        {datasetType === 'numeric' && (
                          <button
                            onClick={() => showAnalysisConfig(analysis.name)}
                            disabled={isAnalyzing}
                            className="bg-white/10 text-gray-300 py-2 px-3 rounded-lg hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Configure Parameters"
                          >
                            <Settings className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Results Section */}
            {analysisResults.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white">Analysis Results</h3>
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
                  {analysisResults.map((result) => (
                    <div key={result.id} className="bg-white/5 rounded-lg p-4 border border-white/10">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-white font-semibold">{result.title}</h4>
                        <span className="text-xs text-gray-400 bg-white/10 px-2 py-1 rounded">
                          {result.type}
                        </span>
                      </div>
                      
                      {/* Visualization display */}
                      {result.type === 'visualization' && result.imageUrls && result.imageUrls.length > 0 ? (
                        <div className="space-y-4 mb-4">
                          {result.imageUrls.map((imageUrl, index) => (
                            <div key={index} className="bg-white rounded-lg p-2">
                              <img 
                                src={imageUrl} 
                                alt={`${result.title} visualization ${index + 1}`}
                                className="w-full h-auto rounded"
                                onError={(e) => {
                                  console.error('Failed to load image:', imageUrl)
                                  e.currentTarget.style.display = 'none'
                                }}
                              />
                            </div>
                          ))}
                        </div>
                      ) : (
                        /* Mock visualization for non-image results */
                        <div className="bg-gradient-to-br from-accent-500/20 to-secondary-500/20 rounded-lg p-6 mb-4">
                          <div className="flex items-center justify-center h-32 text-gray-300">
                            <BarChart3 className="w-12 h-12" />
                          </div>
                        </div>
                      )}
                      
                      <p className="text-gray-300 text-sm">{result.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Proceed Button */}
            <div className="text-center">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={handleProceedToModelTraining}
                  className="inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all transform bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105 shadow-lg hover:shadow-accent-500/25"
                >
                  <span>Proceed to Model Training</span>
                  <ArrowRight className="w-6 h-6" />
                </button>
              </div>
              
              <div className="mt-4 text-center">
                {analysisResults.length > 0 ? (
                  <p className="text-green-400 text-sm">
                    ✅ {analysisResults.length} analysis result{analysisResults.length !== 1 ? 's' : ''} completed
                  </p>
                ) : (
                  <p className="text-gray-400 text-sm">
                    You can proceed with current data or complete analysis steps first
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Parameter Configuration Modal */}
      {showParameterConfig && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-2xl border border-white/20 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white">
                  Configure {currentAnalysisName} Parameters
                </h3>
                <button
                  onClick={() => setShowParameterConfig(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <span className="sr-only">Close</span>
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {currentAnalysisName.toLowerCase().includes('statistical') && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-medium text-white">Statistical Analysis Options</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={analysisParameters.no_plots}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, no_plots: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Skip plots generation</span>
                      </label>
                      
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={analysisParameters.plots_only}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, plots_only: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Generate only plots</span>
                      </label>
                    </div>
                  </div>
                )}

                {currentAnalysisName.toLowerCase().includes('correlation') && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-medium text-white">Correlation Analysis Options</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Correlation Threshold
                        </label>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.1"
                          value={analysisParameters.threshold}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, threshold: parseFloat(e.target.value) }))}
                          className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Correlation Method
                        </label>
                        <select
                          value={analysisParameters.method}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, method: e.target.value as 'pearson' | 'spearman' | 'both' }))}
                          className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                        >
                          <option value="both">Both (Pearson & Spearman)</option>
                          <option value="pearson">Pearson</option>
                          <option value="spearman">Spearman</option>
                        </select>
                      </div>
                      
                      <label className="flex items-center space-x-3 md:col-span-2">
                        <input
                          type="checkbox"
                          checked={analysisParameters.no_plots}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, no_plots: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Skip plots generation</span>
                      </label>
                    </div>
                  </div>
                )}

                {currentAnalysisName.toLowerCase().includes('visualization') && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-medium text-white">Advanced Visualization Options</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          PCA Components
                        </label>
                        <input
                          type="number"
                          min="1"
                          value={analysisParameters.pca_components || ''}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, pca_components: e.target.value ? parseInt(e.target.value) : undefined }))}
                          placeholder="Auto"
                          className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Sample Size for Pair Plots
                        </label>
                        <input
                          type="number"
                          min="100"
                          value={analysisParameters.sample_size}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, sample_size: parseInt(e.target.value) }))}
                          className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                        />
                      </div>
                      
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={analysisParameters.skip_pca}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, skip_pca: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Skip PCA analysis</span>
                      </label>
                      
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={analysisParameters.skip_pairs}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, skip_pairs: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Skip pair plots</span>
                      </label>
                    </div>
                  </div>
                )}

                {currentAnalysisName.toLowerCase().includes('comprehensive') && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-medium text-white">Comprehensive EDA Options</h4>
                    <div className="space-y-4">
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          checked={analysisParameters.all}
                          onChange={(e) => setAnalysisParameters(prev => ({ ...prev, all: e.target.checked }))}
                          className="rounded border-gray-600 bg-gray-700 text-accent-500 focus:ring-accent-500"
                        />
                        <span className="text-gray-300">Run all available techniques</span>
                      </label>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Correlation Threshold
                          </label>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={analysisParameters.threshold}
                            onChange={(e) => setAnalysisParameters(prev => ({ ...prev, threshold: parseFloat(e.target.value) }))}
                            className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Sample Size
                          </label>
                          <input
                            type="number"
                            min="100"
                            value={analysisParameters.sample_size}
                            onChange={(e) => setAnalysisParameters(prev => ({ ...prev, sample_size: parseInt(e.target.value) }))}
                            className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-accent-500 focus:ring-1 focus:ring-accent-500"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex justify-end space-x-4 mt-8">
                <button
                  onClick={() => setShowParameterConfig(false)}
                  className="px-6 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => runAnalysis(currentAnalysisName, analysisParameters)}
                  disabled={isAnalyzing}
                  className="px-6 py-2 bg-gradient-to-r from-accent-500 to-secondary-500 text-white rounded-lg hover:from-accent-600 hover:to-secondary-600 transition-all disabled:opacity-50"
                >
                  Run Analysis
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ExploratoryDataAnalysisPage