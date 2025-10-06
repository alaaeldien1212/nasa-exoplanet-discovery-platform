'use client'

import { useRef, useState, useEffect, Suspense } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, Stars, Environment } from '@react-three/drei'
import { Mesh, TextureLoader } from 'three'
import { Brain, Download, RotateCcw, Eye, Zap } from 'lucide-react'

interface PlanetData {
  radius: number
  temperature: number
  period: number
  depth: number
  classification: string
  confidence: number
}

interface Planet3DProps {
  planetData: PlanetData | null
  className?: string
}

// 3D Planet Component
function PlanetSphere({ planetData, textureUrl }: { planetData: PlanetData, textureUrl?: string }) {
  const meshRef = useRef<Mesh>(null)
  const [texture, setTexture] = useState<any>(null)
  
  // Load texture if provided
  useEffect(() => {
    if (textureUrl) {
      const loader = new TextureLoader()
      loader.load(textureUrl, (loadedTexture) => {
        setTexture(loadedTexture)
      })
    }
  }, [textureUrl])
  
  // Auto-rotation
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.5
    }
  })
  
  // Get planet color based on characteristics with dynamic color variations
  const getPlanetColor = () => {
    if (planetData.classification === 'false positive') {
      return '#666666'
    }
    
    const temp = planetData.temperature
    return getDynamicPlanetColor(temp)
  }
  
  // Calculate planet scale
  const planetScale = Math.max(0.5, Math.min(planetData.radius / 2, 3))
  
  return (
    <mesh ref={meshRef} scale={planetScale}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial 
        color={getPlanetColor()}
        map={texture}
        roughness={0.7}
        metalness={0.1}
      />
    </mesh>
  )
}

// Atmosphere component for gas giants
function PlanetAtmosphere({ planetData }: { planetData: PlanetData }) {
  const meshRef = useRef<Mesh>(null)
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.3
    }
  })
  
  if (planetData.classification !== 'exoplanet' || planetData.radius <= 1.5) {
    return null
  }
  
  const planetScale = Math.max(0.5, Math.min(planetData.radius / 2, 3))
  
  return (
    <mesh ref={meshRef} scale={planetScale * 1.1}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshBasicMaterial 
        color={getPlanetColor(planetData.temperature, planetData.classification)}
        transparent
        opacity={0.1}
        side={2} // BackSide
      />
    </mesh>
  )
}

// Main 3D Scene
function Scene3D({ planetData, textureUrl }: { planetData: PlanetData, textureUrl?: string }) {
  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 50 }}
      style={{ background: 'transparent' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={1} />
      <pointLight position={[-5, -5, -5]} intensity={0.5} />
      
      <Stars radius={100} depth={50} count={1000} factor={4} />
      
      <Suspense fallback={null}>
        <PlanetSphere planetData={planetData} textureUrl={textureUrl} />
        <PlanetAtmosphere planetData={planetData} />
      </Suspense>
      
      <OrbitControls 
        enablePan={false}
        enableZoom={true}
        enableRotate={true}
        minDistance={2}
        maxDistance={10}
        autoRotate={false}
        autoRotateSpeed={1}
      />
    </Canvas>
  )
}

// Dynamic planet color function with smooth temperature-based transitions
function getDynamicPlanetColor(temperature: number): string {
  // Create smooth color transitions based on temperature
  // Temperature ranges: 0K - 50K (Deep Space Blue) to 2000K+ (Molten White)
  
  if (temperature <= 50) {
    // Deep space blue to dark blue (0K - 50K)
    const ratio = temperature / 50
    return interpolateColor('#000033', '#1a237e', ratio)
  } else if (temperature <= 150) {
    // Dark blue to medium blue (50K - 150K)
    const ratio = (temperature - 50) / 100
    return interpolateColor('#1a237e', '#1976d2', ratio)
  } else if (temperature <= 250) {
    // Medium blue to cyan-blue (150K - 250K)
    const ratio = (temperature - 150) / 100
    return interpolateColor('#1976d2', '#0288d1', ratio)
  } else if (temperature <= 350) {
    // Cyan-blue to teal (250K - 350K)
    const ratio = (temperature - 250) / 100
    return interpolateColor('#0288d1', '#0097a7', ratio)
  } else if (temperature <= 450) {
    // Teal to green (350K - 450K)
    const ratio = (temperature - 350) / 100
    return interpolateColor('#0097a7', '#388e3c', ratio)
  } else if (temperature <= 550) {
    // Green to yellow-green (450K - 550K)
    const ratio = (temperature - 450) / 100
    return interpolateColor('#388e3c', '#689f38', ratio)
  } else if (temperature <= 650) {
    // Yellow-green to yellow (550K - 650K)
    const ratio = (temperature - 550) / 100
    return interpolateColor('#689f38', '#fbc02d', ratio)
  } else if (temperature <= 750) {
    // Yellow to orange (650K - 750K)
    const ratio = (temperature - 650) / 100
    return interpolateColor('#fbc02d', '#ff8f00', ratio)
  } else if (temperature <= 850) {
    // Orange to red-orange (750K - 850K)
    const ratio = (temperature - 750) / 100
    return interpolateColor('#ff8f00', '#f57c00', ratio)
  } else if (temperature <= 950) {
    // Red-orange to red (850K - 950K)
    const ratio = (temperature - 850) / 100
    return interpolateColor('#f57c00', '#d32f2f', ratio)
  } else if (temperature <= 1100) {
    // Red to dark red (950K - 1100K)
    const ratio = (temperature - 950) / 150
    return interpolateColor('#d32f2f', '#b71c1c', ratio)
  } else if (temperature <= 1300) {
    // Dark red to deep red (1100K - 1300K)
    const ratio = (temperature - 1100) / 200
    return interpolateColor('#b71c1c', '#8d1b1b', ratio)
  } else if (temperature <= 1500) {
    // Deep red to purple-red (1300K - 1500K)
    const ratio = (temperature - 1300) / 200
    return interpolateColor('#8d1b1b', '#6a1b9a', ratio)
  } else if (temperature <= 1700) {
    // Purple-red to purple (1500K - 1700K)
    const ratio = (temperature - 1500) / 200
    return interpolateColor('#6a1b9a', '#7b1fa2', ratio)
  } else if (temperature <= 1900) {
    // Purple to pink (1700K - 1900K)
    const ratio = (temperature - 1700) / 200
    return interpolateColor('#7b1fa2', '#c2185b', ratio)
  } else if (temperature <= 2100) {
    // Pink to hot pink (1900K - 2100K)
    const ratio = (temperature - 1900) / 200
    return interpolateColor('#c2185b', '#e91e63', ratio)
  } else {
    // Hot pink to white (2100K+)
    const ratio = Math.min((temperature - 2100) / 500, 1)
    return interpolateColor('#e91e63', '#ffffff', ratio)
  }
}

// Helper function to interpolate between two hex colors
function interpolateColor(color1: string, color2: string, ratio: number): string {
  const hex1 = color1.replace('#', '')
  const hex2 = color2.replace('#', '')
  
  const r1 = parseInt(hex1.substr(0, 2), 16)
  const g1 = parseInt(hex1.substr(2, 2), 16)
  const b1 = parseInt(hex1.substr(4, 2), 16)
  
  const r2 = parseInt(hex2.substr(0, 2), 16)
  const g2 = parseInt(hex2.substr(2, 2), 16)
  const b2 = parseInt(hex2.substr(4, 2), 16)
  
  const r = Math.round(r1 + (r2 - r1) * ratio)
  const g = Math.round(g1 + (g2 - g1) * ratio)
  const b = Math.round(b1 + (b2 - b1) * ratio)
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// Helper function to get planet color (backward compatibility)
function getPlanetColor(temperature: number, classification: string): string {
  if (classification === 'false positive') {
    return '#666666'
  }
  return getDynamicPlanetColor(temperature)
}

export default function Planet3D({ planetData, className = '' }: Planet3DProps) {
  const [isGenerating, setIsGenerating] = useState(false)
  const [aiTexture, setAiTexture] = useState<string | null>(null)
  const [is3DMode, setIs3DMode] = useState(true)
  
  // Generate AI texture
  const generateAITexture = async () => {
    if (!planetData) return
    
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
        setAiTexture(`data:image/png;base64,${result.texture}`)
      }
    } catch (error) {
      console.error('Error generating AI texture:', error)
    } finally {
      setIsGenerating(false)
    }
  }
  
  // Download planet image
  const downloadPlanetImage = () => {
    if (!planetData) return
    
    // Create a canvas to capture the 3D scene
    const canvas = document.createElement('canvas')
    canvas.width = 1024
    canvas.height = 1024
    const ctx = canvas.getContext('2d')!
    
    // Draw planet representation
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const radius = Math.min(200, Math.max(100, planetData.radius * 50))
    
    // Background
    ctx.fillStyle = '#000011'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // Stars
    ctx.fillStyle = '#ffffff'
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * canvas.width
      const y = Math.random() * canvas.height
      const starSize = Math.random() * 2
      ctx.beginPath()
      ctx.arc(x, y, starSize, 0, Math.PI * 2)
      ctx.fill()
    }
    
    // Planet
    const planetColor = getPlanetColor(planetData.temperature, planetData.classification)
    ctx.fillStyle = planetColor
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.fill()
    
    // Add planet info text
    ctx.fillStyle = '#ffffff'
    ctx.font = '24px Arial'
    ctx.fillText(`${planetData.classification.toUpperCase()}`, 50, 50)
    ctx.fillText(`Radius: ${planetData.radius.toFixed(2)} R⊕`, 50, 80)
    ctx.fillText(`Temperature: ${planetData.temperature.toFixed(0)} K`, 50, 110)
    ctx.fillText(`Confidence: ${Math.round(planetData.confidence * 100)}%`, 50, 140)
    
    // Download
    const link = document.createElement('a')
    link.download = `exoplanet-${planetData.classification}-${Date.now()}.png`
    link.href = canvas.toDataURL()
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
          <Brain className="w-6 h-6 text-white mr-3" />
          <h2 className="text-2xl font-semibold text-white">3D Planet Visualization</h2>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIs3DMode(!is3DMode)}
            className="px-4 py-2 bg-purple-500/20 border border-purple-500/50 rounded-lg text-purple-400 hover:bg-purple-500/30 transition-colors flex items-center"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            {is3DMode ? '2D View' : '3D View'}
          </button>
          <button
            onClick={generateAITexture}
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

      {/* 3D Renderer */}
      <div className="w-full h-96 rounded-xl overflow-hidden bg-gradient-to-br from-gray-900 to-black mb-6">
        {is3DMode ? (
          <Scene3D planetData={planetData} textureUrl={aiTexture || undefined} />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <div className="text-6xl mb-4">🌍</div>
              <p>2D View Mode</p>
              <p className="text-sm">Switch to 3D for interactive model</p>
            </div>
          </div>
        )}
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
      <div className="bg-white/5 rounded-lg p-4 mb-4">
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
      <div className="text-center text-sm text-gray-400">
        <p>🖱️ Drag to rotate • 🔄 Scroll to zoom • 🎨 AI-generated textures • 🚀 True 3D NASA-quality visualization</p>
      </div>
    </div>
  )
}