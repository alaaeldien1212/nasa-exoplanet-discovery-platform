'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

type Language = 'en' | 'ar'

interface LanguageContextType {
  language: Language
  toggleLanguage: () => void
  t: (key: string) => string
  isInitialized: boolean
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

// Translation keys
const translations = {
  en: {
    // Header
    'header.title': 'NASA PALD',
    'header.subtitle': 'Exoplanet AI Detection',
    'header.badge': 'NASA Space Apps 2025',
    
    // Home page
    'home.title': 'NASA Exoplanet AI Detection Platform',
    'home.subtitle': 'Discover exoplanets using advanced AI trained on NASA mission data',
    'home.startDetection': 'Start Detection',
    'home.uploadData': 'Upload Data',
    'home.viewAnalytics': 'View Analytics',
    
    // Quick Stats
    'stats.trainingSamples': 'Training Samples',
    'stats.accuracy': 'Accuracy',
    'stats.missions': 'NASA Missions',
    
    // Features
    'features.aiModel.title': 'NASA PALD AI Model',
    'features.aiModel.description': 'Advanced Random Forest classifier trained on 6,247+ exoplanet samples from Kepler, K2, and TESS missions',
    'features.aiModel.stats': '83%+ Accuracy',
    
    'features.llama.title': 'Llama AI Analysis',
    'features.llama.description': 'Get detailed explanations using Ollama\'s Llama model for why data suggests planet or false positive',
    
    'features.database.title': 'Exoplanet Database',
    'features.database.description': 'Comprehensive database with coordinate matching and planet name lookup from NASA missions',
    
    'features.upload.title': 'Bulk Data Upload',
    'features.upload.description': 'Upload CSV files from Kepler, K2, and TESS missions for comprehensive analysis',
    
    'features.dashboard.title': 'Analytics Dashboard',
    'features.dashboard.description': 'Interactive charts, detailed predictions, and comprehensive analysis results',
    
    'features.responsive.title': 'Mobile Responsive',
    'features.responsive.description': 'Optimized for all devices with touch-friendly interface and responsive design',
    
    // CTA Section
    'cta.title': 'Ready to Discover Exoplanets?',
    'cta.subtitle': 'Start with sample data or upload your own NASA mission data',
    'cta.trySample': 'Try Sample Data',
    'cta.uploadNasa': 'Upload NASA Data',
    
    // Footer
    'footer.builtWith': 'Built with ❤️ by',
    'footer.platform': 'NASA Space Apps Challenge 2025 - Exoplanet AI Detection Platform',
  },
  ar: {
    // Header
    'header.title': 'ناسا بالد',
    'header.subtitle': 'كشف الكواكب الخارجية بالذكاء الاصطناعي',
    'header.badge': 'تحدي ناسا الفضائي 2025',
    
    // Home page
    'home.title': 'منصة ناسا لكشف الكواكب الخارجية بالذكاء الاصطناعي',
    'home.subtitle': 'اكتشف الكواكب الخارجية باستخدام الذكاء الاصطناعي المتقدم المدرب على بيانات بعثات ناسا',
    'home.startDetection': 'بدء الكشف',
    'home.uploadData': 'رفع البيانات',
    'home.viewAnalytics': 'عرض التحليلات',
    
    // Quick Stats
    'stats.trainingSamples': 'عينات التدريب',
    'stats.accuracy': 'الدقة',
    'stats.missions': 'بعثات ناسا',
    
    // Features
    'features.aiModel.title': 'نموذج ناسا بالد للذكاء الاصطناعي',
    'features.aiModel.description': 'مصنف غابة عشوائية متقدم مدرب على 6,247+ عينة من الكواكب الخارجية من بعثات كيبلر وك2 وتيس',
    'features.aiModel.stats': '83%+ دقة',
    
    'features.llama.title': 'تحليل لاما للذكاء الاصطناعي',
    'features.llama.description': 'احصل على تفسيرات مفصلة باستخدام نموذج لاما من أولاما لسبب اقتراح البيانات لكوكب أو إيجابية خاطئة',
    
    'features.database.title': 'قاعدة بيانات الكواكب الخارجية',
    'features.database.description': 'قاعدة بيانات شاملة مع مطابقة الإحداثيات والبحث عن أسماء الكواكب من بعثات ناسا',
    
    'features.upload.title': 'رفع البيانات المجمعة',
    'features.upload.description': 'ارفع ملفات CSV من بعثات كيبلر وك2 وتيس للتحليل الشامل',
    
    'features.dashboard.title': 'لوحة التحليلات',
    'features.dashboard.description': 'رسوم بيانية تفاعلية وتوقعات مفصلة ونتائج تحليل شاملة',
    
    'features.responsive.title': 'متجاوب مع الهواتف',
    'features.responsive.description': 'محسن لجميع الأجهزة مع واجهة سهلة اللمس وتصميم متجاوب',
    
    // CTA Section
    'cta.title': 'مستعد لاكتشاف الكواكب الخارجية؟',
    'cta.subtitle': 'ابدأ بالبيانات التجريبية أو ارفع بيانات بعثة ناسا الخاصة بك',
    'cta.trySample': 'جرب البيانات التجريبية',
    'cta.uploadNasa': 'رفع بيانات ناسا',
    
    // Footer
    'footer.builtWith': 'تم البناء بـ ❤️ بواسطة',
    'footer.platform': 'تحدي ناسا الفضائي 2025 - منصة كشف الكواكب الخارجية بالذكاء الاصطناعي',
  }
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>('en')
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => {
    // Load language preference from localStorage (only in browser)
    if (typeof window !== 'undefined') {
      const savedLanguage = localStorage.getItem('nasa-ai-language') as Language
      if (savedLanguage && (savedLanguage === 'en' || savedLanguage === 'ar')) {
        setLanguage(savedLanguage)
      }
    }
    setIsInitialized(true)
  }, [])

  useEffect(() => {
    // Save language preference to localStorage (only in browser)
    if (typeof window !== 'undefined') {
      localStorage.setItem('nasa-ai-language', language)
      
      // Update document direction and language
      document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr'
      document.documentElement.lang = language
    }
  }, [language])

  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'ar' : 'en')
  }

  const t = (key: string): string => {
    return translations[language][key as keyof typeof translations[typeof language]] || key
  }

  return (
    <LanguageContext.Provider value={{ language, toggleLanguage, t, isInitialized }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider')
  }
  return context
}
