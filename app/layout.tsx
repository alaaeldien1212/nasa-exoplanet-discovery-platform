'use client'

import type { Metadata } from 'next'
import { Inter, Cairo } from 'next/font/google'
import { useState, useEffect } from 'react'
import SplashScreen from '../components/SplashScreen'
import FloatingNav from '../components/FloatingNav'
import Header from '../components/Header'
import { LanguageProvider } from '../contexts/LanguageContext'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })
const cairo = Cairo({ 
  subsets: ['arabic', 'latin'],
  variable: '--font-cairo',
  display: 'swap',
  fallback: ['Noto Sans Arabic', 'Segoe UI', 'Tahoma', 'Arial', 'sans-serif']
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [showSplash, setShowSplash] = useState(true)

  useEffect(() => {
    // Show splash screen for 5 seconds
    const timer = setTimeout(() => {
      setShowSplash(false)
    }, 5000)

    return () => clearTimeout(timer)
  }, [])

  return (
    <html lang="en" className="dark">
      <head>
        <title>NASA Exoplanet AI Detection Platform</title>
        <meta name="description" content="AI-powered exoplanet detection using NASA datasets - 2025 Space Apps Challenge" />
        <meta name="keywords" content="NASA, exoplanet, AI, machine learning, space, astronomy" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        <meta name="theme-color" content="#000000" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@200;300;400;500;600;700;800;900&family=Noto+Sans+Arabic:wght@100;200;300;400;500;600;700;800;900&family=Amiri:wght@400;700&display=swap" rel="stylesheet" />
      </head>
      <body className={`${inter.className} ${cairo.variable} min-h-screen bg-black text-white overflow-x-hidden`}>
        <LanguageProvider>
          {showSplash ? (
            <SplashScreen onComplete={() => setShowSplash(false)} />
          ) : (
            <div className="min-h-screen bg-black text-white">
              <Header />
              {children}
              <FloatingNav />
            </div>
          )}
        </LanguageProvider>
      </body>
    </html>
  )
}