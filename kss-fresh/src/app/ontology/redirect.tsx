'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function OntologyRedirect() {
  const router = useRouter()
  
  useEffect(() => {
    router.replace('/modules/ontology')
  }, [router])
  
  return (
    <div className="min-h-screen flex items-center justify-center">
      <p className="text-gray-500">리다이렉트 중...</p>
    </div>
  )
}