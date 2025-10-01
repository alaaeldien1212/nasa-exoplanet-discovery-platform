'use client'

import Image from 'next/image'
import Link from 'next/link'
import { Star, Languages } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'

export default function Header() {
  const { language, toggleLanguage, t, isInitialized } = useLanguage()

  // Don't render until initialized to prevent hydration issues
  if (!isInitialized) {
    return (
      <header className="bg-black/80 backdrop-blur-xl border-b border-white/10 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="relative w-10 h-10 flex-shrink-0">
                <Image
                  src="/logo.png"
                  alt="NASA Exoplanet AI Logo"
                  fill
                  className="object-contain"
                  priority
                />
              </div>
              <div className="hidden sm:block">
                <div className="text-lg font-bold text-white">NASA PALD</div>
                <div className="text-xs text-gray-400">Exoplanet AI Detection</div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-full px-3 py-1 border border-white/20">
                <Languages className="w-4 h-4 text-white" />
                <span className="text-sm text-white font-semibold">العربية</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-full px-3 py-1 border border-white/20">
                <Star className="w-4 h-4 text-white" />
                <span className="text-sm text-white font-semibold hidden sm:block">NASA Space Apps 2025</span>
                <span className="text-sm text-white font-semibold sm:hidden">Space Apps</span>
              </div>
            </div>
          </div>
        </div>
      </header>
    )
  }

  return (
    <header className="bg-black/80 backdrop-blur-xl border-b border-white/10 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <Link href="/" className={`flex items-center hover:opacity-80 transition-opacity ${language === 'ar' ? 'space-x-reverse space-x-3' : 'space-x-3'}`}>
            <div className="relative w-10 h-10 flex-shrink-0">
              <Image
                src="/logo.png"
                alt="NASA Exoplanet AI Logo"
                fill
                className="object-contain"
                priority
              />
            </div>
            <div className="hidden sm:block">
              <div className="text-lg font-bold text-white">{t('header.title')}</div>
              <div className="text-xs text-gray-400">{t('header.subtitle')}</div>
            </div>
          </Link>

          {/* Right side - Language Toggle and NASA Badge */}
          <div className={`flex items-center ${language === 'ar' ? 'space-x-reverse space-x-3' : 'space-x-3'}`}>
            {/* Language Toggle */}
            <button
              onClick={toggleLanguage}
              className={`flex items-center bg-white/10 backdrop-blur-sm rounded-full px-3 py-1 border border-white/20 hover:bg-white/20 transition-colors ${language === 'ar' ? 'space-x-reverse space-x-2' : 'space-x-2'}`}
              title={language === 'en' ? 'Switch to Arabic' : 'التبديل إلى الإنجليزية'}
            >
              <Languages className="w-4 h-4 text-white" />
              <span className="text-sm text-white font-semibold">
                {language === 'en' ? 'العربية' : 'English'}
              </span>
            </button>

            {/* NASA Badge */}
            <div className={`flex items-center bg-white/10 backdrop-blur-sm rounded-full px-3 py-1 border border-white/20 ${language === 'ar' ? 'space-x-reverse space-x-2' : 'space-x-2'}`}>
              <Star className="w-4 h-4 text-white" />
              <span className="text-sm text-white font-semibold hidden sm:block">{t('header.badge')}</span>
              <span className="text-sm text-white font-semibold sm:hidden">Space Apps</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
