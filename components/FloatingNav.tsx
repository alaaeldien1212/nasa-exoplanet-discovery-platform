'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Home, 
  Brain, 
  Upload, 
  BarChart3, 
  Menu,
  X
} from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'

export default function FloatingNav() {
  const pathname = usePathname()
  const [isOpen, setIsOpen] = useState(false)
  const { language, isInitialized } = useLanguage()

  // Don't render until initialized to prevent hydration issues
  if (!isInitialized) {
    return (
      <>
        <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
          <div className="floating-nav bg-black/80 border border-white/20 rounded-3xl px-6 py-3">
            <div className="flex items-center space-x-2">
              <button className="lg:hidden p-2 rounded-xl bg-white/10 hover:bg-white/20 transition-colors">
                <Menu className="w-5 h-5 text-white" />
              </button>
              <div className="hidden lg:flex items-center space-x-1">
                <div className="flex items-center space-x-2 px-4 py-2 rounded-2xl text-white">
                  <Home className="w-5 h-5" />
                  <span className="font-medium">Home</span>
                </div>
                <div className="flex items-center space-x-2 px-4 py-2 rounded-2xl text-white">
                  <Brain className="w-5 h-5" />
                  <span className="font-medium">Detect</span>
                </div>
                <div className="flex items-center space-x-2 px-4 py-2 rounded-2xl text-white">
                  <Upload className="w-5 h-5" />
                  <span className="font-medium">Upload</span>
                </div>
                <div className="flex items-center space-x-2 px-4 py-2 rounded-2xl text-white">
                  <BarChart3 className="w-5 h-5" />
                  <span className="font-medium">Dashboard</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="h-20 lg:h-0"></div>
      </>
    )
  }

  const navItems = [
    {
      name: 'Home',
      href: '/',
      icon: Home,
      description: 'Main Dashboard'
    },
    {
      name: 'Detect',
      href: '/predict',
      icon: Brain,
      description: 'Exoplanet Detection'
    },
    {
      name: 'Upload',
      href: '/upload',
      icon: Upload,
      description: 'Upload Data'
    },
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: BarChart3,
      description: 'Model Analytics'
    }
  ]

  const isActive = (href: string) => {
    if (href === '/') {
      return pathname === '/'
    }
    return pathname.startsWith(href)
  }

  return (
    <>
      {/* Floating Navigation Bar */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
        <div className="floating-nav bg-black/80 border border-white/20 rounded-3xl px-6 py-3">
          <div className={`flex items-center ${language === 'ar' ? 'space-x-reverse space-x-2' : 'space-x-2'}`}>
            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="lg:hidden p-2 rounded-xl bg-white/10 hover:bg-white/20 transition-colors"
            >
              {isOpen ? (
                <X className="w-5 h-5 text-white" />
              ) : (
                <Menu className="w-5 h-5 text-white" />
              )}
            </button>

            {/* Desktop Navigation */}
            <div className={`hidden lg:flex items-center ${language === 'ar' ? 'space-x-reverse space-x-1' : 'space-x-1'}`}>
              {navItems.map((item) => {
                const Icon = item.icon
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`
                      flex items-center px-4 py-2 rounded-2xl transition-all duration-300
                      ${isActive(item.href) 
                        ? 'nav-item-active text-black' 
                        : 'text-white nav-item-hover hover:text-white'
                      }
                      ${language === 'ar' ? 'space-x-reverse space-x-2' : 'space-x-2'}
                    `}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.name}</span>
                  </Link>
                )
              })}
            </div>
          </div>

          {/* Mobile Navigation Menu */}
          {isOpen && (
            <div className="lg:hidden mt-4 pt-4 border-t border-white/20">
              <div className="grid grid-cols-2 gap-2">
                {navItems.map((item) => {
                  const Icon = item.icon
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setIsOpen(false)}
                      className={`
                        flex flex-col items-center space-y-1 p-3 rounded-xl transition-all duration-300
                        ${isActive(item.href) 
                          ? 'nav-item-active text-black' 
                          : 'text-white nav-item-hover hover:text-white'
                        }
                      `}
                    >
                      <Icon className="w-5 h-5" />
                      <span className="text-xs font-medium">{item.name}</span>
                    </Link>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Spacing for Mobile */}
      <div className="h-20 lg:h-0"></div>
    </>
  )
}
