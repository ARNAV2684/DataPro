import React, { useState, useRef, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate, useLocation } from 'react-router-dom'
import { DatabaseService } from '../lib/database'
import apiClient from '../services/apiClient'
import { 
  Upload, 
  FolderOpen, 
  FileText, 
  BarChart3, 
  Database, 
  Settings, 
  LogOut, 
  User,
  Brain,
  Zap,
  CheckCircle2,
  AlertCircle,
  Image as ImageIcon,
  X,
  Eye,
  Download,
  ArrowLeft,
  Sparkles,
  ArrowRight
} from 'lucide-react'
import { DatasetType } from './DatasetSelectionPage'

interface UploadedFile {
  file: File
  type: 'data' | 'image'
  preview?: string
  dataset_key?: string  // Add dataset_key from API response
  uploaded?: boolean    // Track upload status
  // All possible dataset key properties from API response
  output_key?: string
  bucket_key?: string
  dataset_id?: string
  storage_key?: string
  key?: string
  file_key?: string
  error?: string        // Track upload errors
}

interface DataTypeConfig {
  title: string
  description: string
  acceptedFormats: string[]
  tools: Array<{
    name: string
    icon: React.ReactNode
    description: string
    color: string
  }>
  tips: string[]
}

const HomePage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [selectedImagePreview, setSelectedImagePreview] = useState<string | null>(null)
  const [datasetType, setDatasetType] = useState<DatasetType>('mixed')
  const [isLoadingPreference, setIsLoadingPreference] = useState(true)

  // Get dataset type from navigation state or database
  useEffect(() => {
    const loadDatasetType = async () => {
      try {
        // First check if we have it from navigation state
        if (location.state?.datasetType) {
          setDatasetType(location.state.datasetType)
          setIsLoadingPreference(false)
          return
        }

        // Otherwise, load from database
        const { data: preference, error } = await DatabaseService.getUserDatasetTypePreference()
        
        if (error) {
          console.error('Error loading user preference:', error)
          // If no preference found, redirect to selection page
          navigate('/dataset-selection')
          return
        }

        if (preference) {
          setDatasetType(preference.dataset_type)
        } else {
          // No preference found, redirect to selection page
          navigate('/dataset-selection')
          return
        }
      } catch (error) {
        console.error('Error loading dataset type:', error)
        navigate('/dataset-selection')
      } finally {
        setIsLoadingPreference(false)
      }
    }

    loadDatasetType()
  }, [location.state, navigate])

  const dataTypeConfigs: Record<DatasetType, DataTypeConfig> = {
    numeric: {
      title: 'Numeric Data Analysis',
      description: 'Upload your structured datasets for statistical analysis and machine learning',
      acceptedFormats: ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv'],
      tools: [
        {
          name: 'Statistical Analysis',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Descriptive statistics and distributions',
          color: 'text-blue-400'
        },
        {
          name: 'ML Models',
          icon: <Brain className="w-5 h-5" />,
          description: 'Regression, classification, clustering',
          color: 'text-purple-400'
        },
        {
          name: 'Data Visualization',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Charts, plots, and dashboards',
          color: 'text-green-400'
        },
        {
          name: 'Feature Engineering',
          icon: <Zap className="w-5 h-5" />,
          description: 'Transform and optimize features',
          color: 'text-yellow-400'
        }
      ],
      tips: [
        '• Include column headers for better analysis',
        '• Clean data produces more accurate results',
        '• CSV format is processed fastest',
        '• Remove missing values for ML models'
      ]
    },
    text: {
      title: 'Text Data Processing',
      description: 'Upload documents and text files for NLP analysis and text mining',
      acceptedFormats: ['.txt', '.docx', '.pdf', '.json', '.csv'],
      tools: [
        {
          name: 'Sentiment Analysis',
          icon: <Brain className="w-5 h-5" />,
          description: 'Analyze emotional tone and sentiment',
          color: 'text-green-400'
        },
        {
          name: 'Text Classification',
          icon: <FileText className="w-5 h-5" />,
          description: 'Categorize and label text content',
          color: 'text-blue-400'
        },
        {
          name: 'Topic Modeling',
          icon: <Sparkles className="w-5 h-5" />,
          description: 'Discover hidden topics and themes',
          color: 'text-purple-400'
        },
        {
          name: 'Text Summarization',
          icon: <Zap className="w-5 h-5" />,
          description: 'Generate concise summaries',
          color: 'text-orange-400'
        }
      ],
      tips: [
        '• Plain text files work best for analysis',
        '• Remove special characters for cleaner results',
        '• Larger text corpora provide better insights',
        '• Consider language preprocessing steps'
      ]
    },
    image: {
      title: 'Image Data Analysis',
      description: 'Upload images and visual datasets for computer vision and deep learning',
      acceptedFormats: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'],
      tools: [
        {
          name: 'Image Classification',
          icon: <ImageIcon className="w-5 h-5" />,
          description: 'Classify and categorize images',
          color: 'text-purple-400'
        },
        {
          name: 'Object Detection',
          icon: <Eye className="w-5 h-5" />,
          description: 'Identify objects within images',
          color: 'text-blue-400'
        },
        {
          name: 'Feature Extraction',
          icon: <Brain className="w-5 h-5" />,
          description: 'Extract visual features and patterns',
          color: 'text-green-400'
        },
        {
          name: 'Image Enhancement',
          icon: <Sparkles className="w-5 h-5" />,
          description: 'Improve image quality and clarity',
          color: 'text-yellow-400'
        }
      ],
      tips: [
        '• Upload folders to maintain organization',
        '• Higher resolution images provide better results',
        '• Consistent image sizes improve processing',
        '• Label folders by categories for classification'
      ]
    },
    mixed: {
      title: 'Mixed Data Analysis',
      description: 'Upload any combination of data types for comprehensive multimodal analysis',
      acceptedFormats: ['.csv', '.json', '.xlsx', '.jpg', '.png', '.txt', '.pdf'],
      tools: [
        {
          name: 'Multimodal Analysis',
          icon: <Database className="w-5 h-5" />,
          description: 'Analyze multiple data types together',
          color: 'text-orange-400'
        },
        {
          name: 'Cross-Modal Insights',
          icon: <Brain className="w-5 h-5" />,
          description: 'Find patterns across data types',
          color: 'text-purple-400'
        },
        {
          name: 'Unified Visualization',
          icon: <BarChart3 className="w-5 h-5" />,
          description: 'Combined charts and dashboards',
          color: 'text-blue-400'
        },
        {
          name: 'Auto Processing',
          icon: <Zap className="w-5 h-5" />,
          description: 'Automatic data type detection',
          color: 'text-green-400'
        }
      ],
      tips: [
        '• Organize files by type in separate folders',
        '• Mixed datasets enable richer insights',
        '• Maintain consistent naming conventions',
        '• Consider relationships between data types'
      ]
    }
  }

  const currentConfig = dataTypeConfigs[datasetType]

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const handleBackToSelection = () => {
    navigate('/dataset-selection')
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    handleFiles(files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      handleFiles(files)
    }
  }

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      handleFiles(files)
    }
  }

  const isImageFile = (file: File): boolean => {
    return file.type.startsWith('image/')
  }

  const isDataFile = (file: File): boolean => {
    const dataExtensions = ['.csv', '.json', '.xlsx', '.xls', '.txt', '.parquet', '.tsv', '.pdf', '.docx']
    return dataExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
  }

  const isAcceptedFormat = (file: File): boolean => {
    return currentConfig.acceptedFormats.some(format => 
      file.name.toLowerCase().endsWith(format.toLowerCase())
    ) || file.type.startsWith('image/') || isDataFile(file)
  }

  const createImagePreview = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.readAsDataURL(file)
    })
  }

  const handleFiles = async (files: File[]) => {
    setIsUploading(true)
    
    const processedFiles: UploadedFile[] = []
    
    for (const file of files) {
      if (!isAcceptedFormat(file)) continue
      
      try {
        console.log(`🔄 Uploading file: ${file.name}`)
        console.log('File type:', file.type)
        console.log('Is image file:', isImageFile(file))
        console.log('Is data file:', isDataFile(file))
        
        if (isImageFile(file)) {
          const preview = await createImagePreview(file)
          processedFiles.push({
            file,
            type: 'image',
            preview,
            uploaded: false  // Will upload to API later when needed
          })
        } else if (isDataFile(file)) {
          // Upload data files immediately to get dataset_key
          const uploadResponse = await apiClient.uploadDataset(file, {
            user_id: user?.id || 'anonymous',
            data_type: datasetType === 'mixed' ? 'numeric' : datasetType as any,
            description: `Uploaded from ${datasetType} workspace`
          })
          
          console.log(`✅ File uploaded successfully:`, uploadResponse)
          
          processedFiles.push({
            file,
            type: 'data',
            // Ensure we capture the dataset key correctly - prioritize output_key from API response
            dataset_key: uploadResponse.output_key || uploadResponse.bucket_key || uploadResponse.dataset_id,
            output_key: uploadResponse.output_key,
            bucket_key: uploadResponse.bucket_key,
            dataset_id: uploadResponse.dataset_id,
            uploaded: true
          })
        } else {
          // Fallback: treat as data file and try to upload
          console.log('File not detected as image or data, treating as data file')
          const uploadResponse = await apiClient.uploadDataset(file, {
            user_id: user?.id || 'anonymous',
            data_type: datasetType === 'mixed' ? 'numeric' : datasetType as any,
            description: `Uploaded from ${datasetType} workspace`
          })
          
          console.log(`✅ File uploaded successfully (fallback):`, uploadResponse)
          
          processedFiles.push({
            file,
            type: 'data',
            dataset_key: uploadResponse.output_key || uploadResponse.bucket_key || uploadResponse.dataset_id,
            output_key: uploadResponse.output_key,
            bucket_key: uploadResponse.bucket_key,
            dataset_id: uploadResponse.dataset_id,
            uploaded: true
          })
        }
      } catch (error) {
        console.error(`❌ Failed to upload ${file.name}:`, error)
        // Still add to list but mark as failed
        processedFiles.push({
          file,
          type: isImageFile(file) ? 'image' : 'data',
          preview: isImageFile(file) ? await createImagePreview(file) : undefined,
          uploaded: false,
          error: error instanceof Error ? error.message : 'Upload failed'
        })
      }
    }
    
    const newUploadedFiles = [...uploadedFiles, ...processedFiles]
    setUploadedFiles(newUploadedFiles)
    
    // Save to localStorage for fallback access
    localStorage.setItem('lastUploadedFiles', JSON.stringify(newUploadedFiles))
    localStorage.setItem('lastDatasetType', datasetType)
    
    setIsUploading(false)
  }

  const removeFile = (index: number) => {
    const newUploadedFiles = uploadedFiles.filter((_, i) => i !== index)
    setUploadedFiles(newUploadedFiles)
    
    // Update localStorage
    localStorage.setItem('lastUploadedFiles', JSON.stringify(newUploadedFiles))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handleProceedToPreprocessing = () => {
    if (uploadedFiles.length === 0) return
    
    // Save to localStorage before navigation
    localStorage.setItem('lastUploadedFiles', JSON.stringify(uploadedFiles))
    localStorage.setItem('lastDatasetType', datasetType)
    
    navigate('/preprocessing', {
      state: {
        datasetType,
        uploadedFiles
      }
    })
  }

  const dataFiles = uploadedFiles.filter(f => f.type === 'data')
  const imageFiles = uploadedFiles.filter(f => f.type === 'image')

  // Show loading state while fetching preference
  if (isLoadingPreference) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-accent-500/30 border-t-accent-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading your workspace...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToSelection}
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
                <p className="text-xs text-gray-400 capitalize">{datasetType} Data Mode</p>
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Upload Area */}
          <div className="lg:col-span-2">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-white mb-2">{currentConfig.title}</h2>
                <p className="text-gray-300">{currentConfig.description}</p>
              </div>
              
              {/* Upload Zone */}
              <div
                className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all ${
                  isDragOver 
                    ? 'border-accent-400 bg-accent-500/20' 
                    : 'border-white/30 hover:border-white/50'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept={currentConfig.acceptedFormats.join(',')}
                />
                
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto">
                    <Upload className="w-8 h-8 text-white" />
                  </div>
                  
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Drop your {datasetType} files or folders here
                    </h3>
                    <p className="text-gray-300 mb-4">
                      Supports: {currentConfig.acceptedFormats.join(', ')}
                    </p>
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-3 justify-center">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="bg-gradient-to-r from-accent-500 to-secondary-500 text-white px-6 py-3 rounded-lg font-medium hover:from-accent-600 hover:to-secondary-600 transition-all transform hover:scale-105 flex items-center space-x-2"
                    >
                      <FileText className="w-5 h-5" />
                      <span>Select Files</span>
                    </button>
                    
                    <button
                      onClick={() => folderInputRef.current?.click()}
                      className="bg-gradient-to-r from-primary-500 to-secondary-500 text-white px-6 py-3 rounded-lg font-medium hover:from-primary-600 hover:to-secondary-600 transition-all transform hover:scale-105 flex items-center space-x-2"
                    >
                      <FolderOpen className="w-5 h-5" />
                      <span>Select Folder</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Hidden folder input */}
              <input
                ref={folderInputRef}
                type="file"
                multiple
                {...({ webkitdirectory: "" } as any)}
                onChange={handleFolderSelect}
                className="hidden"
              />

              {/* Upload Progress */}
              {isUploading && (
                <div className="mt-6 bg-white/5 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 border-2 border-accent-500/30 border-t-accent-500 rounded-full animate-spin"></div>
                    <span className="text-white">Processing your {datasetType} files...</span>
                  </div>
                </div>
              )}

              {/* Data Files Section */}
              {dataFiles.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <Database className="w-5 h-5 text-secondary-400" />
                    <span>Data Files ({dataFiles.length})</span>
                  </h3>
                  <div className="space-y-3">
                    {dataFiles.map((item, index) => (
                      <div key={`data-${index}`} className="bg-white/5 rounded-lg p-4 flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <FileText className="w-6 h-6 text-secondary-400" />
                          <div>
                            <p className="text-white font-medium">{item.file.name}</p>
                            <p className="text-gray-400 text-sm">{formatFileSize(item.file.size)}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <CheckCircle2 className="w-6 h-6 text-green-400" />
                          <button
                            onClick={() => removeFile(uploadedFiles.indexOf(item))}
                            className="text-red-400 hover:text-red-300 transition-colors"
                          >
                            <X className="w-5 h-5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Image Files Section */}
              {imageFiles.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <ImageIcon className="w-5 h-5 text-accent-400" />
                    <span>Images ({imageFiles.length})</span>
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {imageFiles.map((item, index) => (
                      <div key={`image-${index}`} className="bg-white/5 rounded-lg p-3 group">
                        <div className="relative aspect-square mb-2">
                          <img
                            src={item.preview}
                            alt={item.file.name}
                            className="w-full h-full object-cover rounded-lg"
                          />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center space-x-2">
                            <button
                              onClick={() => setSelectedImagePreview(item.preview!)}
                              className="p-2 bg-white/20 rounded-full hover:bg-white/30 transition-colors"
                            >
                              <Eye className="w-4 h-4 text-white" />
                            </button>
                            <button
                              onClick={() => removeFile(uploadedFiles.indexOf(item))}
                              className="p-2 bg-red-500/20 rounded-full hover:bg-red-500/30 transition-colors"
                            >
                              <X className="w-4 h-4 text-white" />
                            </button>
                          </div>
                        </div>
                        <p className="text-white text-sm font-medium truncate">{item.file.name}</p>
                        <p className="text-gray-400 text-xs">{formatFileSize(item.file.size)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Proceed Button */}
              {uploadedFiles.length > 0 && (
                <div className="mt-8 text-center">
                  <button
                    onClick={handleProceedToPreprocessing}
                    className="inline-flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-accent-500 to-secondary-500 text-white font-semibold text-lg rounded-xl hover:from-accent-600 hover:to-secondary-600 transition-all transform hover:scale-105 shadow-lg hover:shadow-accent-500/25"
                  >
                    <span>Proceed to Preprocessing</span>
                    <ArrowRight className="w-6 h-6" />
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Stats</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Data Files</span>
                  <span className="text-white font-semibold">{dataFiles.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Images</span>
                  <span className="text-white font-semibold">{imageFiles.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Size</span>
                  <span className="text-white font-semibold">
                    {formatFileSize(uploadedFiles.reduce((acc, item) => acc + item.file.size, 0))}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Status</span>
                  <span className="text-green-400 font-semibold flex items-center space-x-1">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Ready</span>
                  </span>
                </div>
              </div>
            </div>

            {/* Specialized Tools */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4">
                {datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Tools
              </h3>
              <div className="space-y-3">
                {currentConfig.tools.map((tool, index) => (
                  <button 
                    key={index}
                    className="w-full flex items-start space-x-3 p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors text-left group"
                    disabled={uploadedFiles.length === 0}
                  >
                    <div className={`${tool.color} group-hover:scale-110 transition-transform`}>
                      {tool.icon}
                    </div>
                    <div className="flex-1">
                      <span className="text-white font-medium block">{tool.name}</span>
                      <span className="text-gray-400 text-sm">{tool.description}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Tips */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-accent-400" />
                <span>Pro Tips</span>
              </h3>
              <div className="space-y-3 text-sm text-gray-300">
                {currentConfig.tips.map((tip, index) => (
                  <p key={index}>{tip}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      {selectedImagePreview && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={() => setSelectedImagePreview(null)}
              className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors"
            >
              <X className="w-8 h-8" />
            </button>
            <img
              src={selectedImagePreview}
              alt="Preview"
              className="max-w-full max-h-full object-contain rounded-lg"
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage