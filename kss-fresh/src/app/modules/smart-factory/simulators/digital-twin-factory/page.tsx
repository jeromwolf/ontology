'use client'

import dynamic from 'next/dynamic'
import { useSearchParams } from 'next/navigation'

const InteractiveDigitalTwin = dynamic(() => import('./InteractiveDigitalTwin'), {
  ssr: false,
  loading: () => <div className="min-h-screen flex items-center justify-center">로딩 중...</div>
})

export default function DigitalTwinFactoryPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  
  return <InteractiveDigitalTwin backUrl={backUrl} />
}