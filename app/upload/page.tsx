'use client'

import { useState, useRef } from 'react'
import { Upload, FileText, Brain, BarChart3, Download, AlertCircle, CheckCircle, ChevronLeft, ChevronRight, Search } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'

interface UploadResult {
  filename: string
  totalRows: number
  processedRows: number
  predictions: Array<{
    row: number
    prediction: number
    classification: string
    confidence: number
    features: any
    database_match?: {
      planet_name?: string
      status?: string
      mission?: string
      db_confidence?: number
      is_known_planet: boolean
    }
    ai_description?: string
  }>
  summary: {
    exoplanets: number
    falsePositives: number
    avgConfidence: number
  }
  totalPredictions?: number
}

export default function UploadPage() {
  const [isUploading, setIsUploading] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [isGeneratingAI, setIsGeneratingAI] = useState(false)
  const [aiProgress, setAiProgress] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [searchTerm, setSearchTerm] = useState('')
  const [itemsPerPage] = useState(20)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Helper functions for data processing
  const getFilteredPredictions = () => {
    if (!uploadResult) return []
    return uploadResult.predictions.filter(pred => 
      pred.features.koi_period?.toString().includes(searchTerm) ||
      pred.features.koi_prad?.toString().includes(searchTerm) ||
      pred.classification.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }

  const getPaginatedPredictions = () => {
    const filtered = getFilteredPredictions()
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    return filtered.slice(startIndex, endIndex)
  }

  const getTotalPages = () => {
    const filtered = getFilteredPredictions()
    return Math.ceil(filtered.length / itemsPerPage)
  }

  const getChartData = () => {
    if (!uploadResult) return []
    
    // Group by confidence ranges
    const ranges = [
      { range: '90-100%', min: 0.9, max: 1.0, count: 0 },
      { range: '80-90%', min: 0.8, max: 0.9, count: 0 },
      { range: '70-80%', min: 0.7, max: 0.8, count: 0 },
      { range: '60-70%', min: 0.6, max: 0.7, count: 0 },
      { range: '50-60%', min: 0.5, max: 0.6, count: 0 },
      { range: '<50%', min: 0.0, max: 0.5, count: 0 }
    ]

    uploadResult.predictions.forEach(pred => {
      const confidence = pred.confidence
      for (let range of ranges) {
        if (confidence >= range.min && confidence < range.max) {
          range.count++
          break
        }
      }
    })

    return ranges
  }

  const getPeriodDistribution = () => {
    if (!uploadResult) return []
    
    const periods = uploadResult.predictions.map(pred => pred.features.koi_period).filter(p => p)
    const bins = [
      { period: '<10 days', min: 0, max: 10, count: 0 },
      { period: '10-50 days', min: 10, max: 50, count: 0 },
      { period: '50-100 days', min: 50, max: 100, count: 0 },
      { period: '100-365 days', min: 100, max: 365, count: 0 },
      { period: '>365 days', min: 365, max: Infinity, count: 0 }
    ]

    periods.forEach(period => {
      for (let bin of bins) {
        if (period >= bin.min && period < bin.max) {
          bin.count++
          break
        }
      }
    })

    return bins
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file')
      return
    }

    setIsUploading(true)
    setError(null)
    setUploadResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (result.error) {
        setError(result.error)
      } else {
        setUploadResult(result)
        setCurrentPage(1) // Reset to first page
        setSearchTerm('') // Clear search
      }
    } catch (err) {
      setError('Failed to upload and analyze file')
    } finally {
      setIsUploading(false)
    }
  }

  const downloadResults = () => {
    if (!uploadResult) return

    const csvContent = [
      'Row,Classification,Confidence,Orbital Period,Planetary Radius,Stellar Temperature',
      ...uploadResult.predictions.map(p => 
        `${p.row},${p.classification},${p.confidence.toFixed(3)},${p.features.koi_period || ''},${p.features.koi_prad || ''},${p.features.koi_steff || ''}`
      )
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `exoplanet_analysis_${uploadResult.filename}`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const generateAIDescriptions = async () => {
    if (!uploadResult) return

    setIsGeneratingAI(true)
    setAiProgress(0)

    const predictions = uploadResult.predictions
    const totalPredictions = predictions.length
    let completed = 0

    for (let i = 0; i < predictions.length; i++) {
      const pred = predictions[i]
      
      try {
        const response = await fetch('/api/ai-description', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ predictionData: pred }),
        })

        const result = await response.json()
        
        if (result.success) {
          // Update the prediction with AI description
          const updatedPredictions = [...predictions]
          updatedPredictions[i] = {
            ...pred,
            ai_description: result.ai_description
          }
          
          setUploadResult({
            ...uploadResult,
            predictions: updatedPredictions
          })
        }
        
        completed++
        setAiProgress((completed / totalPredictions) * 100)
        
        // Small delay to prevent overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 100))
        
      } catch (error) {
        console.error('Error generating AI description:', error)
        completed++
        setAiProgress((completed / totalPredictions) * 100)
      }
    }

    setIsGeneratingAI(false)
  }

  return (
    <div className="min-h-screen bg-black text-white py-12">
      <div className="max-w-8xl mx-auto px-2 sm:px-4 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="p-4 rounded-full bg-white/10 backdrop-blur-sm glow-effect">
              <Upload className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
            Upload NASA Data
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Upload NASA exoplanet datasets for AI-powered analysis using Llama and our trained model
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
            <h2 className="text-2xl font-semibold mb-6 text-white">
              Upload CSV File
            </h2>
            
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-white bg-white/5' 
                  : 'border-white/30 hover:border-white/50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              
              {isUploading ? (
                <div className="space-y-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
                  <p className="text-gray-300">Uploading and analyzing...</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <p className="text-lg text-white mb-2">Drop your CSV file here</p>
                    <p className="text-gray-400">or</p>
                  </div>
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-6 py-3 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    Choose File
                  </button>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                  
                  <p className="text-sm text-gray-400">
                    Supports NASA Kepler, K2, and TESS datasets
                  </p>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
                  <span className="text-red-400">{error}</span>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
            <h2 className="text-2xl font-semibold mb-6 text-white">
              Analysis Results
            </h2>

            {uploadResult ? (
              <div className="space-y-8">
                {/* Summary Stats */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="text-center p-6 bg-black/40 backdrop-blur-xl rounded-xl border border-white/20 shadow-xl shadow-green-500/20">
                    <div className="text-4xl font-bold text-green-400 mb-2 drop-shadow-lg">
                      {uploadResult.summary.exoplanets}
                    </div>
                    <div className="text-lg text-gray-200 font-semibold">Exoplanets</div>
                  </div>
                  
                  <div className="text-center p-6 bg-black/40 backdrop-blur-xl rounded-xl border border-white/20 shadow-xl shadow-red-500/20">
                    <div className="text-4xl font-bold text-red-400 mb-2 drop-shadow-lg">
                      {uploadResult.summary.falsePositives}
                    </div>
                    <div className="text-lg text-gray-200 font-semibold">False Positives</div>
                  </div>
                </div>

                {/* File Info */}
                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/15 p-6 shadow-xl shadow-black/50">
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-semibold text-lg">File:</span>
                      <span className="text-white font-bold text-lg">{uploadResult.filename}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-semibold text-lg">Total Rows:</span>
                      <span className="text-white font-bold text-lg">{uploadResult.totalRows}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-semibold text-lg">Processed:</span>
                      <span className="text-white font-bold text-lg">{uploadResult.processedRows}</span>
                    </div>
                    {uploadResult.totalPredictions && uploadResult.totalPredictions > uploadResult.predictions.length && (
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 font-semibold text-lg">Total Predictions:</span>
                        <span className="text-white font-bold text-lg">{uploadResult.totalPredictions}</span>
                      </div>
                    )}
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-semibold text-lg">Avg Confidence:</span>
                      <span className="text-white font-bold text-lg">{(uploadResult.summary.avgConfidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                {/* Download Button */}
                <button
                  onClick={downloadResults}
                  className="w-full px-8 py-4 bg-gradient-to-r from-white to-gray-200 text-black font-bold text-lg rounded-xl hover:from-gray-200 hover:to-white transition-all duration-300 flex items-center justify-center shadow-xl shadow-white/20 backdrop-blur-lg"
                >
                  <Download className="w-6 h-6 mr-3" />
                  Download Results
                </button>

                {/* AI Description Button */}
                <button
                  onClick={generateAIDescriptions}
                  disabled={isGeneratingAI}
                  className="w-full px-8 py-4 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-bold text-lg rounded-xl hover:from-purple-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center shadow-xl shadow-purple-500/20 backdrop-blur-lg"
                >
                  <Brain className="w-6 h-6 mr-3" />
                  {isGeneratingAI ? `Generating AI Analysis... ${aiProgress.toFixed(0)}%` : 'Generate AI Analysis'}
                </button>

                {/* Database Stats */}
                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/15 p-6 shadow-xl shadow-black/50">
                  <h3 className="text-xl font-bold text-white mb-4 text-center">Database Statistics</h3>
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div className="p-4 bg-blue-500/20 rounded-lg border border-blue-500/30">
                      <div className="text-2xl font-bold text-blue-400">19,894</div>
                      <div className="text-sm text-gray-300">Total Planets</div>
                    </div>
                    <div className="p-4 bg-green-500/20 rounded-lg border border-green-500/30">
                      <div className="text-2xl font-bold text-green-400">1,746</div>
                      <div className="text-sm text-gray-300">Confirmed</div>
                    </div>
                    <div className="p-4 bg-yellow-500/20 rounded-lg border border-yellow-500/30">
                      <div className="text-2xl font-bold text-yellow-400">13,475</div>
                      <div className="text-sm text-gray-300">Candidates</div>
                    </div>
                    <div className="p-4 bg-purple-500/20 rounded-lg border border-purple-500/30">
                      <div className="text-2xl font-bold text-purple-400">3</div>
                      <div className="text-sm text-gray-300">Missions</div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">
                  Upload a CSV file to see AI-powered analysis results
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Dashboard Section */}
        {uploadResult && (
          <div className="mt-12 space-y-8">
            {/* Charts Section */}
            <div className="bg-black/40 backdrop-blur-xl rounded-2xl border border-white/20 p-8 shadow-2xl shadow-white/10">
              <h2 className="text-3xl font-bold mb-8 text-white text-center gradient-text">
                Analysis Dashboard
              </h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Confidence Distribution Chart */}
                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/15 p-6 shadow-xl shadow-black/50">
                  <h3 className="text-xl font-bold text-white mb-6 text-center">Confidence Distribution</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" opacity={0.3} />
                      <XAxis 
                        dataKey="range" 
                        stroke="#E5E7EB" 
                        fontSize={12}
                        fontWeight={600}
                      />
                      <YAxis 
                        stroke="#E5E7EB" 
                        fontSize={12}
                        fontWeight={600}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#000000', 
                          border: '2px solid #FFFFFF',
                          borderRadius: '12px',
                          color: '#FFFFFF',
                          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
                          backdropFilter: 'blur(16px)'
                        }} 
                      />
                      <Bar 
                        dataKey="count" 
                        fill="#00FF88"
                        radius={[4, 4, 0, 0]}
                        stroke="#00FF88"
                        strokeWidth={2}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Classification Pie Chart */}
                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/15 p-6 shadow-xl shadow-black/50">
                  <h3 className="text-xl font-bold text-white mb-6 text-center">Classification Distribution</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Exoplanets', value: uploadResult.summary.exoplanets, color: '#00FF88' },
                          { name: 'False Positives', value: uploadResult.summary.falsePositives, color: '#FF4444' }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        stroke="#FFFFFF"
                        strokeWidth={3}
                      >
                        {[
                          { name: 'Exoplanets', value: uploadResult.summary.exoplanets, color: '#00FF88' },
                          { name: 'False Positives', value: uploadResult.summary.falsePositives, color: '#FF4444' }
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#000000', 
                          border: '2px solid #FFFFFF',
                          borderRadius: '12px',
                          color: '#FFFFFF',
                          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
                          backdropFilter: 'blur(16px)'
                        }} 
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Period Distribution Chart */}
                <div className="lg:col-span-2 bg-black/30 backdrop-blur-lg rounded-xl border border-white/15 p-6 shadow-xl shadow-black/50">
                  <h3 className="text-xl font-bold text-white mb-6 text-center">Orbital Period Distribution</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={getPeriodDistribution()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" opacity={0.3} />
                      <XAxis 
                        dataKey="period" 
                        stroke="#E5E7EB" 
                        fontSize={12}
                        fontWeight={600}
                      />
                      <YAxis 
                        stroke="#E5E7EB" 
                        fontSize={12}
                        fontWeight={600}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#000000', 
                          border: '2px solid #FFFFFF',
                          borderRadius: '12px',
                          color: '#FFFFFF',
                          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
                          backdropFilter: 'blur(16px)'
                        }} 
                      />
                      <Bar 
                        dataKey="count" 
                        fill="#0088FF"
                        radius={[4, 4, 0, 0]}
                        stroke="#0088FF"
                        strokeWidth={2}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Data Table Section */}
            <div className="bg-black/40 backdrop-blur-xl rounded-2xl border border-white/20 p-8 shadow-2xl shadow-white/10 max-w-full overflow-hidden">
              <div className="flex justify-between items-center mb-8">
                <h2 className="text-3xl font-bold text-white gradient-text">
                  Detailed Results
                  {uploadResult.totalPredictions && uploadResult.totalPredictions > uploadResult.predictions.length && (
                    <span className="text-lg text-gray-300 ml-3 font-normal">
                      (showing {uploadResult.predictions.length} of {uploadResult.totalPredictions} predictions)
                    </span>
                  )}
                </h2>
                
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-300 w-5 h-5" />
                  <input
                    type="text"
                    placeholder="Search by period, radius, or classification..."
                    value={searchTerm}
                    onChange={(e) => {
                      setSearchTerm(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="pl-12 pr-6 py-3 bg-black/50 border-2 border-white/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-white/60 focus:shadow-lg focus:shadow-white/20 backdrop-blur-lg transition-all duration-300"
                  />
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto bg-black/20 backdrop-blur-lg rounded-xl border border-white/15 shadow-xl shadow-black/50">
                <table className="w-full text-xs sm:text-sm min-w-[1200px]">
                  <thead>
                    <tr className="border-b-2 border-white/20 bg-black/30">
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Row</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Classification</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Confidence</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Coordinates</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Period</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Radius</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Temp</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Duration</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">Depth</th>
                      <th className="text-left py-2 px-2 sm:py-4 sm:px-6 text-gray-200 font-bold text-sm sm:text-lg">AI Analysis</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getPaginatedPredictions().map((pred, index) => (
                      <tr key={index} className="border-b border-white/10 hover:bg-white/10 transition-all duration-300 hover:shadow-lg hover:shadow-white/10">
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">{pred.row}</td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6">
                          <div className="space-y-2">
                            <div className="flex items-center space-x-3">
                              {pred.prediction === 1 ? (
                                <CheckCircle className="w-6 h-6 text-green-400 drop-shadow-lg" />
                              ) : (
                                <AlertCircle className="w-6 h-6 text-red-400 drop-shadow-lg" />
                              )}
                              <span className={`font-bold text-lg ${
                                pred.prediction === 1 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                {pred.classification}
                              </span>
                            </div>
                            {pred.database_match?.is_known_planet && (
                              <div className="ml-9 space-y-1">
                                <div className="flex items-center space-x-2">
                                  <span className="text-xs text-blue-400 font-semibold bg-blue-400/20 px-2 py-1 rounded">
                                    {pred.database_match.mission}
                                  </span>
                                  <span className="text-xs text-yellow-400 font-semibold bg-yellow-400/20 px-2 py-1 rounded">
                                    {pred.database_match.status}
                                  </span>
                                </div>
                                {pred.database_match.planet_name && (
                                  <div className="text-sm text-white font-bold">
                                    🌟 {pred.database_match.planet_name}
                                  </div>
                                )}
                                {pred.features.koi_id && (
                                  <div className="text-xs text-gray-400">
                                    KOI: {pred.features.koi_id}
                                  </div>
                                )}
                              </div>
                            )}
                            {!pred.database_match?.is_known_planet && pred.features.koi_id && (
                              <div className="ml-9">
                                <div className="text-xs text-orange-400 font-semibold bg-orange-400/20 px-2 py-1 rounded">
                                  New Discovery
                                </div>
                                <div className="text-xs text-gray-400 mt-1">
                                  KOI: {pred.features.koi_id}
                                </div>
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {(pred.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          <div className="space-y-1">
                            {pred.features.ra && pred.features.dec ? (
                              <>
                                <div className="text-xs sm:text-sm">RA: {pred.features.ra.toFixed(4)}°</div>
                                <div className="text-xs sm:text-sm">Dec: {pred.features.dec.toFixed(4)}°</div>
                              </>
                            ) : (
                              <div className="text-xs sm:text-sm text-gray-500">N/A</div>
                            )}
                          </div>
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {pred.features.koi_period?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {pred.features.koi_prad?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {pred.features.koi_steff?.toFixed(0) || 'N/A'}
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {pred.features.koi_duration?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold">
                          {pred.features.koi_depth?.toFixed(0) || 'N/A'}
                        </td>
                        <td className="py-2 px-2 sm:py-4 sm:px-6 text-gray-300 font-semibold max-w-xs">
                          <div className="text-xs sm:text-sm leading-relaxed">
                            {pred.ai_description || 'Analysis pending...'}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="flex justify-between items-center mt-8">
                <div className="text-gray-300 text-lg font-semibold">
                  Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, getFilteredPredictions().length)} of {getFilteredPredictions().length} results
                </div>
                
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="p-3 rounded-xl bg-black/50 border-2 border-white/30 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 hover:border-white/60 transition-all duration-300 shadow-lg shadow-black/50 backdrop-blur-lg"
                  >
                    <ChevronLeft className="w-6 h-6 text-white" />
                  </button>
                  
                  <span className="px-6 py-3 text-white font-bold text-lg bg-black/30 rounded-xl border border-white/20 backdrop-blur-lg">
                    Page {currentPage} of {getTotalPages()}
                  </span>
                  
                  <button
                    onClick={() => setCurrentPage(Math.min(getTotalPages(), currentPage + 1))}
                    disabled={currentPage === getTotalPages()}
                    className="p-3 rounded-xl bg-black/50 border-2 border-white/30 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 hover:border-white/60 transition-all duration-300 shadow-lg shadow-black/50 backdrop-blur-lg"
                  >
                    <ChevronRight className="w-6 h-6 text-white" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Process Flow */}
        <div className="mt-16 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-8">
          <h2 className="text-2xl font-semibold mb-8 text-center text-white">
            Analysis Process
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="p-4 rounded-full bg-blue-500/20 border border-blue-500/50 mx-auto mb-4 w-fit">
                <Upload className="w-8 h-8 text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">1. Upload Data</h3>
              <p className="text-gray-300 text-sm">
                Upload NASA exoplanet CSV files (Kepler, K2, TESS)
              </p>
            </div>
            
            <div className="text-center">
              <div className="p-4 rounded-full bg-purple-500/20 border border-purple-500/50 mx-auto mb-4 w-fit">
                <Brain className="w-8 h-8 text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">2. Llama Analysis</h3>
              <p className="text-gray-300 text-sm">
                AI analyzes data structure and extracts features
              </p>
            </div>
            
            <div className="text-center">
              <div className="p-4 rounded-full bg-green-500/20 border border-green-500/50 mx-auto mb-4 w-fit">
                <BarChart3 className="w-8 h-8 text-green-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">3. Model Prediction</h3>
              <p className="text-gray-300 text-sm">
                Trained model classifies each candidate
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
