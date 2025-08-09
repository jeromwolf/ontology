'use client'

import dynamic from 'next/dynamic'
import { Suspense } from 'react'
import { 
  Factory, Settings, Cpu, Eye, Bot, Shield, Activity, Gauge, Cog, Clock,
  TrendingUp, Globe, MapPin, DollarSign, Zap, Database, Wrench, 
  BarChart3, Users, Target, Building, Lightbulb, Rocket, Brain, 
  Network, Wifi, Cloud, HardDrive, AlertTriangle, Lock, Monitor,
  Smartphone, Server, Code, TestTube, ChevronRight
} from 'lucide-react'
import CodeEditor from '@/components/common/CodeEditor'

// Dynamic imports for All Chapters 1-16
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })
const Chapter11 = dynamic(() => import('./chapters/Chapter11'), { ssr: false })
const Chapter12 = dynamic(() => import('./chapters/Chapter12'), { ssr: false })
const Chapter13 = dynamic(() => import('./chapters/Chapter13'), { ssr: false })
const Chapter14 = dynamic(() => import('./chapters/Chapter14'), { ssr: false })
const Chapter15 = dynamic(() => import('./chapters/Chapter15'), { ssr: false })
const Chapter16 = dynamic(() => import('./chapters/Chapter16'), { ssr: false })

// Loading component
const LoadingChapter = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-slate-600"></div>
    <span className="ml-3 text-slate-600 dark:text-slate-400">챕터를 로딩 중...</span>
  </div>
)

export default function ChapterContent({ chapterId }: { chapterId: string }) {
  // All chapters (1-16) use separated components
  const getChapterComponent = () => {
    switch (chapterId) {
      // WHY Part (1-4)
      case 'why-smart-factory':
        return <Chapter1 />
      case 'global-trends-cases':
        return <Chapter2 />
      case 'digital-transformation-roadmap':
        return <Chapter3 />
      case 'business-case-roi':
        return <Chapter4 />
      // WHAT Part (5-12)
      case 'iot-sensor-networks':
        return <Chapter5 />
      case 'ai-data-analytics':
        return <Chapter6 />
      case 'robotics-automation':
        return <Chapter7 />
      case 'digital-twin-simulation':
        return <Chapter8 />
      case 'predictive-maintenance':
        return <Chapter9 />
      case 'quality-management-ai':
        return <Chapter10 />
      case 'mes-erp-integration':
        return <Chapter11 />
      case 'cybersecurity-standards':
        return <Chapter13 />  // Chapter13.tsx has cybersecurity content
      // HOW Part (13-16)
      case 'implementation-methodology':
        return <Chapter12 />  // Chapter12.tsx has implementation content
      case 'system-architecture-design':
        return <Chapter14 />
      case 'change-management-training':
        return <Chapter15 />
      case 'future-outlook-strategy':
        return <Chapter16 />
      default:
        return null
    }
  }
  
  const component = getChapterComponent()
  if (component) {
    return (
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <Suspense fallback={<LoadingChapter />}>
          {component}
        </Suspense>
      </div>
    )
  }
  
  // If no component found, show error
  return (
    <div className="text-center py-12">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-4">
        챕터를 찾을 수 없습니다
      </h2>
      <p className="text-slate-600 dark:text-slate-400">
        요청하신 챕터 ID '{chapterId}'에 해당하는 콘텐츠가 존재하지 않습니다.
      </p>
    </div>
  )
}

