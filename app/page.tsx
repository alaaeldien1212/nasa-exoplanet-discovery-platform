'use client'

import { useState, useEffect, useRef } from 'react'
import { Upload, Brain, BarChart3, Star, Database, FileText, Play, ArrowRight, X, CheckCircle, Rocket, ChevronRight } from 'lucide-react'
import Link from 'next/link'
import { useLanguage } from '../contexts/LanguageContext'

interface GuideStep {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  action: string
  href: string
  tips: string[]
  ref: React.RefObject<HTMLElement>
  position: 'top' | 'bottom' | 'left' | 'right'
}

export default function Home() {
  const { t, language, isInitialized } = useLanguage()
  const [showGuide, setShowGuide] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [isFirstVisit, setIsFirstVisit] = useState(false)
  
      // Refs for guide elements
      const detectButtonRef = useRef<HTMLAnchorElement>(null)
      const uploadButtonRef = useRef<HTMLAnchorElement>(null)
      const dashboardButtonRef = useRef<HTMLAnchorElement>(null)

  // Don't render until initialized to prevent hydration issues
  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-black text-white">
        <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8 py-16">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 gradient-text">
              NASA Exoplanet AI
            </h1>
            <p className="text-lg md:text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Discover exoplanets using NASA PALD (Planetary Analysis & Detection) AI model trained on Kepler, K2, and TESS mission data
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <div className="px-8 py-4 bg-white text-black font-semibold rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 mr-2" />
                Start Detection
              </div>
              <div className="px-8 py-4 border-2 border-white text-white font-semibold rounded-lg flex items-center justify-center">
                <Upload className="w-5 h-5 mr-2" />
                Upload Data
              </div>
              <div className="inline-flex items-center px-8 py-4 border-2 border-white text-white font-semibold rounded-lg">
                <BarChart3 className="w-5 h-5 mr-2" />
                View Analytics
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  useEffect(() => {
    // Check if this is the user's first visit
    const hasVisited = localStorage.getItem('nasa-ai-visited')
    if (!hasVisited) {
      setIsFirstVisit(true)
      setShowGuide(true)
      localStorage.setItem('nasa-ai-visited', 'true')
    }
  }, [])

  useEffect(() => {
    if (showGuide && guideSteps[currentStep]) {
      const element = guideSteps[currentStep].ref.current
      if (element) {
        const rect = element.getBoundingClientRect()
        const tooltip = document.querySelector('.guide-tooltip') as HTMLElement
        if (tooltip) {
          tooltip.style.left = `${rect.left + rect.width / 2}px`
          tooltip.style.top = `${rect.bottom + 10}px`
        }
      }
    }
  }, [showGuide, currentStep])

  const guideSteps: GuideStep[] = [
    {
      id: 'detect',
      title: 'Exoplanet Detection',
      description: 'Test our AI model with sample data or your own exoplanet candidate data',
      icon: <Brain className="w-6 h-6" />,
      action: 'Try Detection',
      href: '/predict',
      tips: [
        'Use our sample data buttons for quick testing',
        'Enter orbital period, planetary radius, and stellar temperature',
        'Get instant AI analysis with confidence scores',
        'View detailed explanations of why it\'s classified as planet or not'
      ],
      ref: detectButtonRef,
      position: 'bottom'
    },
    {
      id: 'upload',
      title: 'Upload NASA Data',
      description: 'Upload CSV files from NASA missions for bulk analysis',
      icon: <Upload className="w-6 h-6" />,
      action: 'Upload Files',
      href: '/upload',
      tips: [
        'Supports Kepler, K2, and TESS mission data',
        'Drag and drop CSV files or click to browse',
        'Get comprehensive analysis dashboard with charts',
        'Download results and view detailed predictions'
      ],
      ref: uploadButtonRef,
      position: 'bottom'
    },
    {
      id: 'dashboard',
      title: 'Model Analytics',
      description: 'View model performance, accuracy metrics, and training statistics',
      icon: <BarChart3 className="w-6 h-6" />,
      action: 'View Analytics',
      href: '/dashboard',
      tips: [
        'See model accuracy and performance metrics',
        'View training data statistics',
        'Monitor detection confidence trends',
        'Access model information and features'
      ],
      ref: dashboardButtonRef,
      position: 'top'
    }
  ]

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: t('features.aiModel.title'),
      description: t('features.aiModel.description'),
      stats: t('features.aiModel.stats')
    },
    {
      icon: <FileText className="w-8 h-8" />,
      title: t('features.llama.title'),
      description: t('features.llama.description'),
      stats: "AI-Powered Insights"
    },
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: t('features.dashboard.title'),
      description: t('features.dashboard.description'),
      stats: "Live Analytics"
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: t('features.database.title'),
      description: t('features.database.description'),
      stats: "5,000+ Planets"
    }
  ]

  const nextStep = () => {
    if (currentStep < guideSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      setShowGuide(false)
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleButtonClick = (href: string, stepIndex: number) => {
    if (showGuide && currentStep === stepIndex) {
      // If this is the current step in the guide, advance to next step
      if (currentStep < guideSteps.length - 1) {
        setCurrentStep(currentStep + 1)
      } else {
        setShowGuide(false)
      }
    }
    // Navigation will happen automatically via Link component
  }

  const skipGuide = () => {
    setShowGuide(false)
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Guide Overlay */}
      {showGuide && (
        <>
          {/* Dark overlay */}
          <div className="fixed inset-0 bg-black/60 z-40" />
          
          {/* Guide Tooltip */}
          <div className="fixed z-50 pointer-events-none guide-tooltip">
            {guideSteps[currentStep] && (
              <div className="relative">
                {/* Tooltip */}
                <div className="bg-white text-black rounded-xl p-4 shadow-2xl max-w-sm transform -translate-x-1/2">
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-blue-100 rounded-lg flex-shrink-0">
                      {guideSteps[currentStep].icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-lg mb-1">
                        {guideSteps[currentStep].title}
                      </h3>
                      <p className="text-gray-600 text-sm mb-3">
                        {guideSteps[currentStep].description}
                      </p>
                      <div className="space-y-1 mb-3">
                        {guideSteps[currentStep].tips.slice(0, 2).map((tip, index) => (
                          <div key={index} className="flex items-start space-x-2 text-xs text-gray-500">
                            <CheckCircle className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                            <span>{tip}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  {/* Arrow */}
                  <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full w-0 h-0 border-4 border-transparent border-b-white" />
                </div>
              </div>
            )}
          </div>

          {/* Guide Controls */}
          <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-4">
              <div className="flex items-center space-x-4">
                <div className="flex space-x-2">
                  {guideSteps.map((_, index) => (
                    <div
                      key={index}
                      className={`w-2 h-2 rounded-full ${
                        index === currentStep ? 'bg-white' : 'bg-white/30'
                      }`}
                    />
                  ))}
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="px-3 py-1 text-white/70 hover:text-white transition-colors disabled:opacity-50 text-sm"
                  >
                    Previous
                  </button>
                  <button
                    onClick={nextStep}
                    className="px-4 py-1 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-colors text-sm"
                  >
                    {currentStep === guideSteps.length - 1 ? 'Finish' : 'Next'}
                  </button>
                  <button
                    onClick={skipGuide}
                    className="px-3 py-1 text-white/70 hover:text-white transition-colors text-sm"
                  >
                    Skip
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent"></div>
        <div className="relative max-w-7xl mx-auto px-2 sm:px-4 lg:px-8 pt-20 pb-16">
          <div className="text-center">
            <div className="flex justify-center mb-8">
              <div className="p-4 rounded-full bg-white/10 backdrop-blur-sm glow-effect">
                <Star className="w-16 h-16 text-white" />
              </div>
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold mb-6 gradient-text">
              NASA Exoplanet AI
            </h1>
            
            <p className="text-lg md:text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              {t('home.subtitle')}
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <Link 
                ref={detectButtonRef}
                href="/predict"
                onClick={() => handleButtonClick('/predict', 0)}
                className={`
                  px-8 py-4 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-all duration-300 card-hover flex items-center justify-center relative
                  ${showGuide && currentStep === 0 ? 'ring-4 ring-blue-400 ring-opacity-75 shadow-2xl shadow-blue-400/50 scale-105' : ''}
                `}
              >
                <Brain className={`w-5 h-5 ${language === 'ar' ? 'ml-2' : 'mr-2'}`} />
                {t('home.startDetection')}
                {showGuide && currentStep === 0 && (
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center animate-pulse">
                    <span className="text-white text-xs font-bold">1</span>
                  </div>
                )}
              </Link>
              <Link 
                ref={uploadButtonRef}
                href="/upload"
                onClick={() => handleButtonClick('/upload', 1)}
                className={`
                  px-8 py-4 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-black transition-all duration-300 card-hover flex items-center justify-center relative
                  ${showGuide && currentStep === 1 ? 'ring-4 ring-blue-400 ring-opacity-75 shadow-2xl shadow-blue-400/50 scale-105' : ''}
                `}
              >
                <Upload className={`w-5 h-5 ${language === 'ar' ? 'ml-2' : 'mr-2'}`} />
                {t('home.uploadData')}
                {showGuide && currentStep === 1 && (
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center animate-pulse">
                    <span className="text-white text-xs font-bold">2</span>
                  </div>
                )}
              </Link>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              <div className="text-center p-4 bg-white/5 rounded-xl">
                <div className="text-2xl font-bold text-white">6,247</div>
                <div className="text-xs text-gray-400">{t('stats.trainingSamples')}</div>
              </div>
              <div className="text-center p-4 bg-white/5 rounded-xl">
                  <div className="text-2xl font-bold text-white">83%+</div>
                <div className="text-xs text-gray-400">{t('stats.accuracy')}</div>
              </div>
              <div className="text-center p-4 bg-white/5 rounded-xl">
                <div className="text-2xl font-bold text-white">3</div>
                <div className="text-xs text-gray-400">{t('stats.missions')}</div>
              </div>
              <div className="text-center p-4 bg-white/5 rounded-xl">
                <div className="text-2xl font-bold text-white">15</div>
                <div className="text-xs text-gray-400">Features</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 bg-gradient-to-b from-transparent to-white/5">
        <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 gradient-text">
              NASA PALD AI Platform
            </h2>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto">
              Advanced machine learning meets NASA's extensive exoplanet datasets
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="p-6 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 card-hover"
              >
                <div className="flex items-start space-x-4">
                  <div className="text-white p-3 bg-white/10 rounded-xl">
                    {feature.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-xl font-semibold text-white">
                        {feature.title}
                      </h3>
                      <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                        {feature.stats}
                      </span>
                    </div>
                    <p className="text-gray-300">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-20 bg-gradient-to-t from-white/10 to-transparent">
        <div className="max-w-4xl mx-auto text-center px-2 sm:px-4 lg:px-8">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 gradient-text">
            {t('cta.title')}
          </h2>
          <p className="text-lg text-gray-300 mb-8">
            {t('cta.subtitle')}
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link 
              href="/predict"
              className="inline-flex items-center px-8 py-4 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-colors card-hover"
            >
              <Play className={`w-5 h-5 ${language === 'ar' ? 'ml-2' : 'mr-2'}`} />
              {t('cta.trySample')}
            </Link>
            <Link 
              href="/upload"
              className="inline-flex items-center px-8 py-4 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-black transition-colors card-hover"
            >
              <FileText className={`w-5 h-5 ${language === 'ar' ? 'ml-2' : 'mr-2'}`} />
              {t('cta.uploadNasa')}
            </Link>
            <Link 
              ref={dashboardButtonRef}
              href="/dashboard"
              onClick={() => handleButtonClick('/dashboard', 2)}
              className={`
                inline-flex items-center px-8 py-4 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-black transition-all duration-300 card-hover relative
                ${showGuide && currentStep === 2 ? 'ring-4 ring-blue-400 ring-opacity-75 shadow-2xl shadow-blue-400/50 scale-105' : ''}
              `}
            >
                <BarChart3 className={`w-5 h-5 ${language === 'ar' ? 'ml-2' : 'mr-2'}`} />
                {t('home.viewAnalytics')}
              {showGuide && currentStep === 2 && (
                <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center animate-pulse">
                  <span className="text-white text-xs font-bold">3</span>
                </div>
              )}
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="py-12 pb-32 lg:pb-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Star className="w-5 h-5 text-white" />
            <span className="text-white font-semibold">NASA PALD</span>
          </div>
          <p className="text-gray-400">
            {t('footer.platform')}
          </p>
          <p className="text-gray-500 text-sm mt-2">
            {t('footer.builtWith')} <a href="https://rhythmlab.dev" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-300">@rhythmlab.dev</a>
          </p>
        </div>
      </footer>
    </div>
  )
}
