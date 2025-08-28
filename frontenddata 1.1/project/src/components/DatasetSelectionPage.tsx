import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { DatabaseService } from '../lib/database'
import { 
  Database, 
  FileText, 
  Image as ImageIcon, 
  BarChart3, 
  Brain, 
  ArrowRight,
  LogOut,
  User,
  Zap,
  FileSpreadsheet,
  Code,
  Camera
} from 'lucide-react'

export type DatasetType = 'numeric' | 'text' | 'image' | 'mixed'

interface DatasetOption {
  id: DatasetType
  title: string
  description: string
  icon: React.ReactNode
  examples: string[]
  color: string
  gradient: string
}

const DatasetSelectionPage: React.FC = () => {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const [selectedType, setSelectedType] = useState<DatasetType | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  const datasetOptions: DatasetOption[] = [
    {
      id: 'numeric',
      title: 'Numeric Data',
      description: 'Structured datasets with numerical values for statistical analysis and machine learning',
      icon: <BarChart3 className="w-8 h-8" />,
      examples: ['CSV files', 'Excel spreadsheets', 'Financial data', 'Sensor readings', 'Survey results'],
      color: 'text-blue-400',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'text',
      title: 'Text Data',
      description: 'Textual content for natural language processing, sentiment analysis, and text mining',
      icon: <FileText className="w-8 h-8" />,
      examples: ['Documents', 'Social media posts', 'Reviews', 'Articles', 'Chat logs'],
      color: 'text-green-400',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      id: 'image',
      title: 'Image Data',
      description: 'Visual datasets for computer vision, image classification, and deep learning models',
      icon: <ImageIcon className="w-8 h-8" />,
      examples: ['Photos', 'Medical scans', 'Satellite imagery', 'Product images', 'Artwork'],
      color: 'text-purple-400',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      id: 'mixed',
      title: 'Mixed Data',
      description: 'Combination of multiple data types for comprehensive analysis and multimodal learning',
      icon: <Database className="w-8 h-8" />,
      examples: ['Research datasets', 'E-commerce data', 'Social platforms', 'IoT sensors', 'Multimedia content'],
      color: 'text-orange-400',
      gradient: 'from-orange-500 to-red-500'
    }
  ]

  const handleContinue = async () => {
    if (!selectedType) return
    
    setIsAnimating(true)
    setIsSaving(true)
    setError('')

    try {
      // Save the dataset type preference to database
      const { error: saveError } = await DatabaseService.saveDatasetTypePreference(selectedType)
      
      if (saveError) {
        throw saveError
      }

      // Navigate to home page with the selected type
      setTimeout(() => {
        navigate('/home', { state: { datasetType: selectedType } })
      }, 1000)
    } catch (error: any) {
      setError(error.message || 'Failed to save preference')
      setIsAnimating(false)
      setIsSaving(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-800">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold text-white">DataLab Pro</h1>
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

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-lg rounded-full px-4 py-2 mb-6">
            <Zap className="w-5 h-5 text-accent-400" />
            <span className="text-accent-400 font-medium">Step 1 of 2</span>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
            What type of data are you
            <span className="block bg-gradient-to-r from-accent-400 to-secondary-400 bg-clip-text text-transparent">
              working with today?
            </span>
          </h1>
          
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Choose your data type to get personalized tools and optimized workflows. 
            We'll customize your experience and save your preference for future sessions.
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6">
            <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
              <p className="text-red-200 text-center">{error}</p>
            </div>
          </div>
        )}

        {/* Dataset Options Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {datasetOptions.map((option) => (
            <div
              key={option.id}
              onClick={() => setSelectedType(option.id)}
              className={`relative group cursor-pointer transition-all duration-300 transform hover:scale-105 ${
                selectedType === option.id 
                  ? 'ring-2 ring-accent-400 scale-105' 
                  : 'hover:ring-2 hover:ring-white/30'
              }`}
            >
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 h-full">
                {/* Selection Indicator */}
                {selectedType === option.id && (
                  <div className="absolute top-4 right-4">
                    <div className="w-6 h-6 bg-gradient-to-r from-accent-500 to-secondary-500 rounded-full flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                  </div>
                )}

                {/* Icon */}
                <div className={`w-16 h-16 bg-gradient-to-r ${option.gradient} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <div className="text-white">
                    {option.icon}
                  </div>
                </div>

                {/* Content */}
                <h3 className="text-2xl font-bold text-white mb-3">{option.title}</h3>
                <p className="text-gray-300 mb-6 leading-relaxed">{option.description}</p>

                {/* Examples */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Examples:</h4>
                  <div className="flex flex-wrap gap-2">
                    {option.examples.map((example, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-white/10 rounded-full text-sm text-gray-300 border border-white/20"
                      >
                        {example}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Hover Effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-accent-500/10 to-secondary-500/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
              </div>
            </div>
          ))}
        </div>

        {/* Continue Button */}
        <div className="text-center">
          <button
            onClick={handleContinue}
            disabled={!selectedType || isAnimating || isSaving}
            className={`inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all transform ${
              selectedType && !isAnimating && !isSaving
                ? 'bg-gradient-to-r from-accent-500 to-secondary-500 text-white hover:from-accent-600 hover:to-secondary-600 hover:scale-105 shadow-lg hover:shadow-accent-500/25'
                : 'bg-gray-600/50 text-gray-400 cursor-not-allowed'
            }`}
          >
            {isAnimating || isSaving ? (
              <>
                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>
                  {isSaving ? 'Saving your preference...' : 'Preparing your workspace...'}
                </span>
              </>
            ) : (
              <>
                <span>Continue to Upload</span>
                <ArrowRight className="w-6 h-6" />
              </>
            )}
          </button>
          
          {!selectedType && (
            <p className="text-gray-400 text-sm mt-3">Please select a data type to continue</p>
          )}
        </div>

        {/* Features Preview */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <FileSpreadsheet className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Smart Processing</h3>
            <p className="text-gray-400 text-sm">Automatic data type detection and optimized processing pipelines</p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Code className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Custom Tools</h3>
            <p className="text-gray-400 text-sm">Tailored analysis tools based on your specific data type</p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Camera className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Advanced Analytics</h3>
            <p className="text-gray-400 text-sm">Machine learning models optimized for your data format</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DatasetSelectionPage