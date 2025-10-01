'use client'

import { useState, useEffect } from 'react'
import { Code2, Star, Rocket, Loader2 } from 'lucide-react'

interface SplashScreenProps {
  onComplete: () => void
}

export default function SplashScreen({ onComplete }: SplashScreenProps) {
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState(0)

  const phases = [
    { text: "Initializing NASA AI Engine...", duration: 1000 },
    { text: "Loading Exoplanet Database...", duration: 1000 },
    { text: "Preparing Llama Analysis...", duration: 1000 },
    { text: "Calibrating Detection Models...", duration: 1000 },
    { text: "Ready for Launch! ⚫", duration: 1000 }
  ]

  useEffect(() => {
    const totalDuration = phases.reduce((sum, phase) => sum + phase.duration, 0)
    const interval = 50
    const increment = (100 / totalDuration) * interval

    const timer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(timer)
          setTimeout(onComplete, 500)
          return 100
        }
        return prev + increment
      })
    }, interval)

    // Phase transitions
    let currentTime = 0
    phases.forEach((phase, index) => {
      setTimeout(() => {
        setPhase(index)
      }, currentTime)
      currentTime += phase.duration
    })

    return () => clearInterval(timer)
  }, [onComplete])

  return (
    <div className="fixed inset-0 bg-black z-50 flex items-center justify-center">
      {/* Static Black Background */}
      <div className="absolute inset-0 bg-black"></div>

      {/* Main Content */}
      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        {/* Logo */}
        <div className="flex items-center justify-center space-x-4 mb-8">
          <div className="bg-white p-3 rounded-xl flex-shrink-0 shadow-2xl">
            <Code2 className="h-8 w-8 text-black" />
          </div>
          <div className="text-left">
            <div className="text-2xl font-bold text-white">RhythmLab</div>
            <div className="text-sm text-gray-300">.dev</div>
          </div>
        </div>

        {/* Title */}
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
          NASA Exoplanet AI
        </h1>
        
        <p className="text-xl text-gray-300 mb-8">
          AI-Powered Space Discovery Platform
        </p>

        {/* Circular Loading Spinner */}
        <div className="mb-6 flex flex-col items-center">
          <div className="relative w-16 h-16 mb-4">
            <div className="absolute inset-0 rounded-full border-4 border-gray-800"></div>
            <div 
              className="absolute inset-0 rounded-full border-4 border-white border-t-transparent animate-spin"
              style={{ 
                animationDuration: '1s',
                transform: `rotate(${progress * 3.6}deg)`
              }}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-white text-sm font-bold">{Math.round(progress)}%</span>
            </div>
          </div>
        </div>

        {/* Phase Text */}
        <div className="text-lg text-white mb-8 min-h-[2rem] flex items-center justify-center">
          <div className="flex items-center space-x-2">
            <Rocket className="w-5 h-5 text-white" />
            <span>{phases[phase]?.text}</span>
          </div>
        </div>

        {/* Built with Love */}
        <div className="text-sm text-gray-400">
          Built with ❤️ by{' '}
          <a 
            href="https://rhythmlab.dev" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-white hover:text-gray-300 transition-colors"
          >
            @rhythmlab.dev
          </a>
        </div>

        {/* NASA Badge */}
        <div className="mt-6 inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-full px-4 py-2 border border-white/20">
          <Star className="w-4 h-4 text-white" />
          <span className="text-sm text-white font-semibold">NASA Space Apps Challenge 2025</span>
        </div>
      </div>
    </div>
  )
}
