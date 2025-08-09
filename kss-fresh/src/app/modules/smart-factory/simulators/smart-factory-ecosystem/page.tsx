'use client'

import dynamic from 'next/dynamic'
import { useSearchParams } from 'next/navigation'

const SmartFactoryEcosystem = dynamic(() => import('./SmartFactoryEcosystem'), {
  ssr: false,
  loading: () => <div className="min-h-screen flex items-center justify-center">로딩 중...</div>
})

export default function SmartFactoryEcosystemPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  
  return <SmartFactoryEcosystem backUrl={backUrl} />
}