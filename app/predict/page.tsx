'use client'

import { useState } from 'react'
import { Upload, Brain, AlertCircle, CheckCircle, Star, Zap, Target, Thermometer, Clock, Activity, Globe, Sparkles, Info, BarChart3 } from 'lucide-react'

interface PredictionResult {
  prediction: number
  classification: string
  confidence: number
  features_used: string[]
  model_info?: any
  ai_analysis?: string
  error?: string
}

export default function PredictPage() {
  const [formData, setFormData] = useState({
    koi_period: '',
    koi_duration: '',
    koi_depth: '',
    koi_prad: '',
    koi_teq: '',
    koi_insol: '',
    koi_model_snr: '',
    koi_steff: '',
    koi_slogg: '',
    koi_srad: '',
    koi_kepmag: '',
    mission: 'Kepler'
  })
  
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Sample datasets for testing
  const sampleDatasets = {
    confirmed_exoplanet: {
      name: "Confirmed Exoplanet (Kepler-22b)",
      description: "A confirmed exoplanet similar to Earth",
      data: {
        koi_period: "289.9",
        koi_duration: "10.5",
        koi_depth: "0.0001",
        koi_prad: "2.4",
        koi_teq: "262",
        koi_insol: "1.1",
        koi_model_snr: "15.2",
        koi_steff: "5518",
        koi_slogg: "4.4",
        koi_srad: "0.98",
        koi_kepmag: "11.7",
        mission: "Kepler"
      }
    },
    false_positive: {
      name: "False Positive Candidate",
      description: "A candidate likely to be a false positive",
      data: {
        koi_period: "0.5",
        koi_duration: "0.8",
        koi_depth: "0.00001",
        koi_prad: "0.1",
        koi_teq: "2000",
        koi_insol: "100",
        koi_model_snr: "3.2",
        koi_steff: "4000",
        koi_slogg: "4.8",
        koi_srad: "0.5",
        koi_kepmag: "14.2",
        mission: "Kepler"
      }
    },
    high_confidence: {
      name: "High Confidence Exoplanet",
      description: "Strong exoplanet candidate with high confidence",
      data: {
        koi_period: "45.2",
        koi_duration: "8.3",
        koi_depth: "0.0005",
        koi_prad: "1.8",
        koi_teq: "400",
        koi_insol: "2.5",
        koi_model_snr: "25.7",
        koi_steff: "5800",
        koi_slogg: "4.2",
        koi_srad: "1.1",
        koi_kepmag: "10.5",
        mission: "Kepler"
      }
    }
  }

  const loadSampleData = (sampleType: keyof typeof sampleDatasets) => {
    const sample = sampleDatasets[sampleType]
    setFormData(sample.data)
    setResult(null)
    setError(null)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert form data to numbers
      const numericData = {
        ...formData,
        koi_period: parseFloat(formData.koi_period) || 0,
        koi_duration: parseFloat(formData.koi_duration) || 0,
        koi_depth: parseFloat(formData.koi_depth) || 0,
        koi_prad: parseFloat(formData.koi_prad) || 0,
        koi_teq: parseFloat(formData.koi_teq) || 0,
        koi_insol: parseFloat(formData.koi_insol) || 0,
        koi_model_snr: parseFloat(formData.koi_model_snr) || 0,
        koi_steff: parseFloat(formData.koi_steff) || 0,
        koi_slogg: parseFloat(formData.koi_slogg) || 0,
        koi_srad: parseFloat(formData.koi_srad) || 0,
        koi_kepmag: parseFloat(formData.koi_kepmag) || 0,
        [`mission_${formData.mission}`]: 1
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(numericData),
      })

      const data = await response.json()

      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
        
        // Generate AI analysis if not already provided
        if (!data.ai_analysis) {
          try {
            const aiResponse = await fetch('/api/ai-description', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ 
                predictionData: {
                  prediction: data.prediction,
                  classification: data.classification,
                  confidence: data.confidence,
                  features: numericData
                }
              }),
            })
            
            const aiData = await aiResponse.json()
            if (aiData.success) {
              setResult(prev => prev ? { ...prev, ai_analysis: aiData.ai_description } : null)
            }
          } catch (aiError) {
            console.error('AI analysis failed:', aiError)
          }
        }
      }
    } catch (err) {
      setError('Failed to make prediction. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence'
    if (confidence >= 0.6) return 'Medium Confidence'
    return 'Low Confidence'
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent"></div>
        <div className="relative max-w-7xl mx-auto px-2 sm:px-4 lg:px-8 pt-8 pb-16">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="p-4 rounded-full bg-white/10 backdrop-blur-sm glow-effect">
                <Brain className="w-12 h-12 text-white" />
              </div>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold mb-4 gradient-text">
              Exoplanet Detection
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
              Use NASA PALD AI to analyze exoplanet candidate data and get instant classification results
            </p>
            
            {/* Quick Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto mb-8">
              <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                <div className="text-2xl font-bold text-white">6,247</div>
                <div className="text-sm text-gray-400">Training Samples</div>
              </div>
              <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                <div className="text-2xl font-bold text-white">95.2%</div>
                <div className="text-sm text-gray-400">Accuracy Rate</div>
              </div>
              <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                <div className="text-2xl font-bold text-white">3</div>
                <div className="text-sm text-gray-400">NASA Missions</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8 pb-16">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* Left Column - Input Form */}
          <div className="xl:col-span-2">
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8">
              <div className="flex items-center mb-6">
                <Target className="w-6 h-6 text-white mr-3" />
                <h2 className="text-2xl font-semibold text-white">Candidate Data Input</h2>
              </div>

              {/* Sample Data Section */}
              <div className="mb-8 p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-blue-500/20">
                <div className="flex items-center mb-4">
                  <Sparkles className="w-5 h-5 text-blue-400 mr-2" />
                  <h3 className="text-lg font-semibold text-white">Quick Test with Sample Data</h3>
                </div>
                <p className="text-sm text-gray-300 mb-4">Try these pre-loaded examples to see how the AI works</p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => loadSampleData('confirmed_exoplanet')}
                    className="p-4 bg-green-500/20 border border-green-500/50 rounded-xl text-green-400 hover:bg-green-500/30 transition-all duration-300 hover:scale-105"
                  >
                    <div className="flex items-center mb-2">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      <span className="font-semibold">Confirmed Exoplanet</span>
                    </div>
                    <div className="text-xs text-green-300">Kepler-22b - Earth-like</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => loadSampleData('high_confidence')}
                    className="p-4 bg-blue-500/20 border border-blue-500/50 rounded-xl text-blue-400 hover:bg-blue-500/30 transition-all duration-300 hover:scale-105"
                  >
                    <div className="flex items-center mb-2">
                      <Zap className="w-4 h-4 mr-2" />
                      <span className="font-semibold">High Confidence</span>
                    </div>
                    <div className="text-xs text-blue-300">Strong candidate</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => loadSampleData('false_positive')}
                    className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-400 hover:bg-red-500/30 transition-all duration-300 hover:scale-105"
                  >
                    <div className="flex items-center mb-2">
                      <AlertCircle className="w-4 h-4 mr-2" />
                      <span className="font-semibold">False Positive</span>
                    </div>
                    <div className="text-xs text-red-300">Likely noise</div>
                  </button>
                </div>
              </div>
              
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* Required Fields */}
                <div className="space-y-6">
                  <div className="flex items-center">
                    <Star className="w-5 h-5 text-yellow-400 mr-2" />
                    <h3 className="text-xl font-semibold text-white">Essential Parameters</h3>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="flex items-center text-sm font-medium text-gray-300">
                        <Clock className="w-4 h-4 mr-2" />
                        Orbital Period (days)
                        <span className="text-red-400 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        name="koi_period"
                        value={formData.koi_period}
                        onChange={handleInputChange}
                        required
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 365.25"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-center text-sm font-medium text-gray-300">
                        <Globe className="w-4 h-4 mr-2" />
                        Planetary Radius (R⊕)
                        <span className="text-red-400 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        name="koi_prad"
                        value={formData.koi_prad}
                        onChange={handleInputChange}
                        required
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 1.0"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-center text-sm font-medium text-gray-300">
                        <Thermometer className="w-4 h-4 mr-2" />
                        Stellar Temperature (K)
                        <span className="text-red-400 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        name="koi_steff"
                        value={formData.koi_steff}
                        onChange={handleInputChange}
                        required
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 5778"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-center text-sm font-medium text-gray-300">
                        <Activity className="w-4 h-4 mr-2" />
                        Mission
                        <span className="text-red-400 ml-1">*</span>
                      </label>
                      <select
                        name="mission"
                        value={formData.mission}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                      >
                        <option value="Kepler">Kepler</option>
                        <option value="K2">K2</option>
                        <option value="TESS">TESS</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Optional Fields */}
                <div className="space-y-6">
                  <div className="flex items-center">
                    <Info className="w-5 h-5 text-blue-400 mr-2" />
                    <h3 className="text-xl font-semibold text-white">Additional Parameters</h3>
                    <span className="text-sm text-gray-400 ml-2">(Optional - improves accuracy)</span>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Transit Duration (hours)</label>
                      <input
                        type="number"
                        name="koi_duration"
                        value={formData.koi_duration}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 2.5"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Transit Depth (ppm)</label>
                      <input
                        type="number"
                        name="koi_depth"
                        value={formData.koi_depth}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 1000"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Equilibrium Temperature (K)</label>
                      <input
                        type="number"
                        name="koi_teq"
                        value={formData.koi_teq}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 288"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Signal-to-Noise Ratio</label>
                      <input
                        type="number"
                        name="koi_model_snr"
                        value={formData.koi_model_snr}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50 focus:border-white/50 transition-all"
                        placeholder="e.g., 15.2"
                      />
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full px-8 py-4 bg-gradient-to-r from-white to-gray-200 text-black font-semibold rounded-xl hover:from-gray-200 hover:to-white transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed card-hover flex items-center justify-center"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-black mr-3"></div>
                      Analyzing with NASA PALD...
                    </div>
                  ) : (
                    <div className="flex items-center">
                      <Brain className="w-5 h-5 mr-3" />
                      Detect Exoplanet
                    </div>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="xl:col-span-1">
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8 sticky top-8">
              <div className="flex items-center mb-6">
                <BarChart3 className="w-6 h-6 text-white mr-3" />
                <h2 className="text-2xl font-semibold text-white">Analysis Results</h2>
              </div>

              {error && (
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl mb-6">
                  <div className="flex items-center">
                    <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
                    <span className="text-red-400">{error}</span>
                  </div>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Main Result Card */}
                  <div className="text-center p-6 bg-gradient-to-br from-white/10 to-white/5 rounded-xl border border-white/20">
                    <div className="flex justify-center mb-4">
                      {result.prediction === 1 ? (
                        <div className="p-3 bg-green-500/20 rounded-full">
                          <CheckCircle className="w-8 h-8 text-green-400" />
                        </div>
                      ) : (
                        <div className="p-3 bg-red-500/20 rounded-full">
                          <AlertCircle className="w-8 h-8 text-red-400" />
                        </div>
                      )}
                    </div>
                    
                    <h3 className="text-xl font-bold mb-2 text-white">
                      {result.classification === 'exoplanet' ? 'EXOPLANET DETECTED' : 'FALSE POSITIVE'}
                    </h3>
                    
                    <div className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
                      {getConfidenceLabel(result.confidence)}
                    </div>
                    
                    <div className={`text-2xl font-bold ${getConfidenceColor(result.confidence)} mt-2`}>
                      {Math.round(result.confidence * 100)}%
                    </div>
                  </div>

                  {/* Confidence Meter */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-medium text-gray-400">Confidence Level</h4>
                    <div className="w-full bg-gray-700 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ${
                          result.confidence >= 0.8 ? 'bg-green-400' : 
                          result.confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-400">
                      <span>Low</span>
                      <span>Medium</span>
                      <span>High</span>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  {result.ai_analysis && (
                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-gray-400">AI Analysis</h4>
                      <div className="text-sm text-gray-300 bg-white/5 p-4 rounded-xl border border-white/10">
                        {result.ai_analysis}
                      </div>
                    </div>
                  )}

                  {/* Model Info */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-medium text-gray-400">Model Information</h4>
                    <div className="space-y-2 text-xs text-gray-300">
                      <div className="flex justify-between">
                        <span>Model:</span>
                        <span className="text-white">NASA PALD</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Training Data:</span>
                        <span className="text-white">{result.model_info?.training_samples || '6,247'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Algorithm:</span>
                        <span className="text-white">Random Forest</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Features:</span>
                        <span className="text-white">{result.features_used?.length || 15}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {!result && !error && (
                <div className="text-center py-12">
                  <div className="p-4 bg-white/5 rounded-full w-fit mx-auto mb-4">
                    <Star className="w-8 h-8 text-gray-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-400 mb-2">Ready for Analysis</h3>
                  <p className="text-sm text-gray-500">
                    Enter candidate data and click "Detect Exoplanet" to get AI-powered analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
