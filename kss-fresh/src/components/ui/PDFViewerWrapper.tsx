'use client'

import dynamic from 'next/dynamic'
import { useState } from 'react'

// PDF Viewer를 동적으로 로드 (SSR 비활성화)
const PDFViewer = dynamic(() => import('./PDFViewer'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center p-8">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div>
      <span className="ml-2 text-gray-600 dark:text-gray-400">PDF 뷰어 로딩 중...</span>
    </div>
  ),
})

// extractTextFromPDF도 동적으로 로드
const PDFTextExtractor = dynamic(
  () => import('./PDFViewer').then(mod => ({ default: mod.extractTextFromPDF })),
  { ssr: false }
)

interface PDFViewerWrapperProps {
  file: File | string
  className?: string
  onError?: (error: Error) => void
  onTextExtracted?: (text: string, pageIndex: number) => void
}

export default function PDFViewerWrapper(props: PDFViewerWrapperProps) {
  return <PDFViewer {...props} />
}

// 텍스트 추출 래퍼 함수
export const extractTextFromPDFWrapper = async (file: File): Promise<string> => {
  // 동적으로 함수를 로드하고 실행
  const module = await import('./PDFViewer')
  return module.extractTextFromPDF(file)
}