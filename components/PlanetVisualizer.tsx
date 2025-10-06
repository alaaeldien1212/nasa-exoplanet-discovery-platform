'use client'

import { useEffect, useRef, useState } from 'react'
import { Brain, Sparkles, Download, Eye, Zap } from 'lucide-react'

interface PlanetData {
  radius: number
  temperature: number
  period: number
  depth: number
  classification: string
  confidence: number
}

interface PlanetVisualizerProps {
  planetData: PlanetData | null
  className?: string
}

export default function PlanetVisualizer({ planetData, className = '' }: PlanetVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [planetTexture, setPlanetTexture] = useState<string | null>(null)
  const [is3DMode, setIs3DMode] = useState(false)

  // Generate planet color based on temperature and classification
  const generatePlanetColor = (temperature: number, classification: string) => {
    if (classification === 'false positive') {
      return '#666666' // Gray for false positives
    }

    // Color based on temperature (Kelvin)
    if (temperature < 200) {
      return '#4a90e2' // Blue - very cold
    } else if (temperature < 300) {
      return '#7ed321' // Green - habitable zone
    } else if (temperature < 500) {
      return '#f5a623' // Orange - warm
    } else if (temperature < 1000) {
      return '#d0021b' // Red - hot
    } else {
      return '#ff6b6b' // Bright red - very hot
    }
  }

  // Generate planet surface texture using AI
  const generatePlanetTexture = async (planetData: PlanetData) => {
    setIsGenerating(true)
    
    try {
      const response = await fetch('/api/generate-planet-texture', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(planetData),
      })

      const result = await response.json()
      
      if (result.success) {
        setPlanetTexture(result.texture)
      } else {
        console.error('Failed to generate texture:', result.error)
        // Fallback to canvas generation
        generateCanvasTexture(planetData)
      }
    } catch (error) {
      console.error('Error generating AI texture:', error)
      // Fallback to canvas generation
      generateCanvasTexture(planetData)
    } finally {
      setIsGenerating(false)
    }
  }

  // Fallback canvas texture generation
  const generateCanvasTexture = (planetData: PlanetData) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')!
    const size = 400
    canvas.width = size
    canvas.height = size

    // Clear canvas
    ctx.clearRect(0, 0, size, size)

    // Create gradient background (space)
    const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2)
    gradient.addColorStop(0, '#000011')
    gradient.addColorStop(1, '#000000')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, size, size)

    // Add stars
    ctx.fillStyle = '#ffffff'
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * size
      const y = Math.random() * size
      const starSize = Math.random() * 2
      ctx.beginPath()
      ctx.arc(x, y, starSize, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw planet
    const centerX = size / 2
    const centerY = size / 2
    const planetRadius = Math.min(120, Math.max(60, planetData.radius * 30))
    
    // Planet base
    const planetColor = generatePlanetColor(planetData.temperature, planetData.classification)
    ctx.fillStyle = planetColor
    ctx.beginPath()
    ctx.arc(centerX, centerY, planetRadius, 0, Math.PI * 2)
    ctx.fill()

    // Add planet features based on type
    if (planetData.classification === 'exoplanet') {
      addPlanetFeatures(ctx, centerX, centerY, planetRadius, planetData)
    }

    // Add atmosphere glow
    if (planetData.classification === 'exoplanet' && planetData.radius > 1.5) {
      const glowGradient = ctx.createRadialGradient(centerX, centerY, planetRadius, centerX, centerY, planetRadius + 20)
      glowGradient.addColorStop(0, planetColor + '40')
      glowGradient.addColorStop(1, 'transparent')
      ctx.fillStyle = glowGradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, planetRadius + 20, 0, Math.PI * 2)
      ctx.fill()
    }

    // Add lighting effect
    const lightGradient = ctx.createRadialGradient(
      centerX - planetRadius/3, centerY - planetRadius/3, 0,
      centerX - planetRadius/3, centerY - planetRadius/3, planetRadius/2
    )
    lightGradient.addColorStop(0, '#ffffff40')
    lightGradient.addColorStop(1, 'transparent')
    ctx.fillStyle = lightGradient
    ctx.beginPath()
    ctx.arc(centerX - planetRadius/3, centerY - planetRadius/3, planetRadius/2, 0, Math.PI * 2)
    ctx.fill()

    // Convert to data URL
    setPlanetTexture(canvas.toDataURL('image/png'))
  }

  // Add planet surface features
  const addPlanetFeatures = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, planetData: PlanetData) => {
    if (planetData.radius < 1.5) {
      // Rocky planet features
      addRockyFeatures(ctx, centerX, centerY, radius)
    } else {
      // Gas giant features
      addGasGiantFeatures(ctx, centerX, centerY, radius, planetData.temperature)
    }
  }

  const addRockyFeatures = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number) => {
    // Add craters
    for (let i = 0; i < 8; i++) {
      const angle = Math.random() * Math.PI * 2
      const distance = Math.random() * (radius * 0.7)
      const x = centerX + Math.cos(angle) * distance
      const y = centerY + Math.sin(angle) * distance
      const craterRadius = Math.random() * 8 + 3
      
      // Crater shadow
      ctx.fillStyle = '#00000040'
      ctx.beginPath()
      ctx.arc(x + 2, y + 2, craterRadius, 0, Math.PI * 2)
      ctx.fill()
      
      // Crater
      ctx.fillStyle = '#333333'
      ctx.beginPath()
      ctx.arc(x, y, craterRadius, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  const addGasGiantFeatures = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, temperature: number) => {
    // Add atmospheric bands
    const numBands = 6
    const bandHeight = (radius * 2) / numBands
    
    for (let i = 0; i < numBands; i++) {
      const y = centerY - radius + i * bandHeight
      const bandWidth = Math.sqrt(radius * radius - (y - centerY) * (y - centerY)) * 2
      
      if (bandWidth > 0) {
        const bandColor = temperature > 500 ? '#ff6b6b' : '#4a90e2'
        ctx.strokeStyle = bandColor + '80'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(centerX - bandWidth/2, y)
        ctx.lineTo(centerX + bandWidth/2, y)
        ctx.stroke()
      }
    }
  }

  // Initialize canvas when planetData changes
  useEffect(() => {
    if (planetData && canvasRef.current) {
      generateCanvasTexture(planetData)
    }
  }, [planetData])

  const downloadPlanetImage = () => {
    if (!canvasRef.current) return
    
    const link = document.createElement('a')
    link.download = `exoplanet-${planetData?.classification}-${Date.now()}.png`
    link.href = canvasRef.current.toDataURL()
    link.click()
  }

  if (!planetData) {
    return (
      <div className={`bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8 ${className}`}>
        <div className="text-center text-gray-400">
          <Eye className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>Enter exoplanet data to generate 3D visualization</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Sparkles className="w-6 h-6 text-white mr-3" />
          <h2 className="text-2xl font-semibold text-white">Planet Visualization</h2>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => generatePlanetTexture(planetData)}
            disabled={isGenerating}
            className="px-4 py-2 bg-blue-500/20 border border-blue-500/50 rounded-lg text-blue-400 hover:bg-blue-500/30 transition-colors disabled:opacity-50 flex items-center"
          >
            {isGenerating ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400 mr-2"></div>
            ) : (
              <Brain className="w-4 h-4 mr-2" />
            )}
            {isGenerating ? 'Generating...' : 'AI Texture'}
          </button>
          <button
            onClick={downloadPlanetImage}
            className="px-4 py-2 bg-green-500/20 border border-green-500/50 rounded-lg text-green-400 hover:bg-green-500/30 transition-colors flex items-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Download
          </button>
        </div>
      </div>

      {/* Planet Canvas */}
      <div className="flex justify-center mb-6">
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="rounded-xl border border-white/20 bg-black"
            style={{ maxWidth: '100%', height: 'auto' }}
          />
          {planetTexture && (
            <div className="absolute top-2 right-2 bg-green-500/20 border border-green-500/50 rounded-full p-2">
              <Zap className="w-4 h-4 text-green-400" />
            </div>
          )}
        </div>
      </div>

      {/* Planet Information */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white/5 rounded-lg p-4">
          <div className="text-sm text-gray-400">Classification</div>
          <div className={`text-lg font-semibold ${
            planetData.classification === 'exoplanet' ? 'text-green-400' : 'text-red-400'
          }`}>
            {planetData.classification === 'exoplanet' ? 'Exoplanet' : 'False Positive'}
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <div className="text-sm text-gray-400">Confidence</div>
          <div className="text-lg font-semibold text-white">
            {Math.round(planetData.confidence * 100)}%
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <div className="text-sm text-gray-400">Radius</div>
          <div className="text-lg font-semibold text-white">
            {planetData.radius.toFixed(2)} R⊕
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-4">
          <div className="text-sm text-gray-400">Temperature</div>
          <div className="text-lg font-semibold text-white">
            {planetData.temperature.toFixed(0)} K
          </div>
        </div>
      </div>

      {/* Planet Type Description */}
      <div className="bg-white/5 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-2">Planet Characteristics</div>
        <div className="text-sm text-gray-300">
          {planetData.classification === 'exoplanet' ? (
            planetData.radius < 1.5 ? (
              <span>Rocky planet with surface features and potential for geological activity</span>
            ) : planetData.radius < 4.0 ? (
              <span>Gas giant with atmospheric bands and storm systems</span>
            ) : (
              <span>Super gas giant with complex atmospheric dynamics</span>
            )
          ) : (
            <span>False positive - likely stellar variability or instrumental noise</span>
          )}
        </div>
      </div>

      {/* Controls Info */}
      <div className="mt-4 text-center text-sm text-gray-400">
        <p>🎨 AI-generated texture • 📊 Based on real planetary data • 🚀 NASA-quality visualization</p>
      </div>
    </div>
  )
}