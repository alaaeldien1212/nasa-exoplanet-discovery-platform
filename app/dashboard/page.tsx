'use client'

import { useState, useEffect } from 'react'
import { BarChart3, TrendingUp, Database, Brain, Star, Activity } from 'lucide-react'

interface ModelInfo {
  metadata?: {
    model_type?: string
    training_samples?: number
    exoplanet_ratio?: number
    best_accuracy?: number
    features_used?: string[]
  }
  feature_names?: string[]
  model_loaded?: boolean
  error?: string
}

export default function DashboardPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModelInfo()
  }, [])

  const fetchModelInfo = async () => {
    try {
      const response = await fetch('/api/predict')
      const data = await response.json()
      
      if (data.error) {
        setError(data.error)
      } else {
        setModelInfo(data)
      }
    } catch (err) {
      setError('Failed to load model information')
    } finally {
      setIsLoading(false)
    }
  }

  const stats = [
    {
      title: "Training Samples",
      value: modelInfo?.metadata?.training_samples?.toLocaleString() || "N/A",
      icon: <Database className="w-6 h-6" />,
      color: "text-blue-400"
    },
    {
      title: "Model Accuracy",
      value: modelInfo?.metadata?.best_accuracy ? `${Math.round(modelInfo.metadata.best_accuracy * 100)}%` : "N/A",
      icon: <TrendingUp className="w-6 h-6" />,
      color: "text-green-400"
    },
    {
      title: "Features Used",
      value: modelInfo?.feature_names?.length?.toString() || "N/A",
      icon: <BarChart3 className="w-6 h-6" />,
      color: "text-purple-400"
    },
    {
      title: "Exoplanet Ratio",
      value: modelInfo?.metadata?.exoplanet_ratio ? `${Math.round(modelInfo.metadata.exoplanet_ratio * 100)}%` : "N/A",
      icon: <Star className="w-6 h-6" />,
      color: "text-yellow-400"
    }
  ]

  const features = modelInfo?.feature_names || []

  return (
    <div className="min-h-screen bg-black text-white py-12">
      <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="p-4 rounded-full bg-white/10 backdrop-blur-sm glow-effect">
              <BarChart3 className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Model Dashboard
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Monitor AI model performance and training statistics
          </p>
        </div>

        {isLoading && (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
            <p className="text-gray-400">Loading model information...</p>
          </div>
        )}

        {error && (
          <div className="max-w-2xl mx-auto">
            <div className="p-6 bg-red-500/20 border border-red-500/50 rounded-lg">
              <div className="flex items-center">
                <Activity className="w-5 h-5 text-red-400 mr-2" />
                <span className="text-red-400">{error}</span>
              </div>
            </div>
          </div>
        )}

        {modelInfo && !isLoading && (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              {stats.map((stat, index) => (
                <div 
                  key={index}
                  className="p-6 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 card-hover"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color}`}>
                      {stat.icon}
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {stat.value}
                    </div>
                  </div>
                  <div className="text-sm text-gray-300">
                    {stat.title}
                  </div>
                </div>
              ))}
            </div>

            {/* Model Information */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
              {/* Model Details */}
              <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
                <h2 className="text-2xl font-semibold mb-6 text-white">
                  Model Information
                </h2>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-3 border-b border-white/10">
                    <span className="text-gray-300">Model Type</span>
                    <span className="text-white font-medium">
                      {modelInfo.metadata?.model_type || 'Random Forest'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-3 border-b border-white/10">
                    <span className="text-gray-300">Status</span>
                    <span className="text-green-400 font-medium">
                      {modelInfo.model_loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-3 border-b border-white/10">
                    <span className="text-gray-300">Training Samples</span>
                    <span className="text-white font-medium">
                      {modelInfo.metadata?.training_samples?.toLocaleString() || 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-3 border-b border-white/10">
                    <span className="text-gray-300">Best Accuracy</span>
                    <span className="text-green-400 font-medium">
                      {modelInfo.metadata?.best_accuracy ? `${Math.round(modelInfo.metadata.best_accuracy * 100)}%` : 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-3">
                    <span className="text-gray-300">Exoplanet Ratio</span>
                    <span className="text-yellow-400 font-medium">
                      {modelInfo.metadata?.exoplanet_ratio ? `${Math.round(modelInfo.metadata.exoplanet_ratio * 100)}%` : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Features Used */}
              <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
                <h2 className="text-2xl font-semibold mb-6 text-white">
                  Features Used
                </h2>
                
                <div className="space-y-2">
                  {features.map((feature, index) => (
                    <div 
                      key={index}
                      className="flex items-center justify-between py-2 px-3 bg-white/5 rounded-lg"
                    >
                      <span className="text-gray-300 text-sm">{feature}</span>
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    </div>
                  ))}
                </div>
                
                <div className="mt-6 pt-4 border-t border-white/10">
                  <div className="text-sm text-gray-400">
                    Total Features: <span className="text-white font-medium">{features.length}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Data Sources */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
              <h2 className="text-2xl font-semibold mb-6 text-white">
                NASA Data Sources
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-white/5 rounded-lg">
                  <div className="text-3xl font-bold text-blue-400 mb-2">Kepler</div>
                  <div className="text-gray-300 text-sm">Primary mission data</div>
                </div>
                
                <div className="text-center p-6 bg-white/5 rounded-lg">
                  <div className="text-3xl font-bold text-green-400 mb-2">K2</div>
                  <div className="text-gray-300 text-sm">Extended mission data</div>
                </div>
                
                <div className="text-center p-6 bg-white/5 rounded-lg">
                  <div className="text-3xl font-bold text-purple-400 mb-2">TESS</div>
                  <div className="text-gray-300 text-sm">Transit survey data</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
