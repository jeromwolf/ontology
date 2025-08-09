'use client'

import { useState } from 'react'
import { HelpCircle } from 'lucide-react'

interface TermTooltipProps {
  term: string
  definition: string
  example?: string
}

export default function TermTooltip({ term, definition, example }: TermTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <span className="relative inline-flex items-center gap-1">
      <span className="font-medium border-b border-dotted border-blue-500 cursor-help">
        {term}
      </span>
      <button
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onClick={() => setIsVisible(!isVisible)}
        className="text-blue-500 hover:text-blue-600"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      
      {isVisible && (
        <div className="absolute z-10 bottom-full left-0 mb-2 w-64 p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
          <div className="text-sm">
            <div className="font-semibold text-gray-900 dark:text-white mb-1">
              {term}
            </div>
            <div className="text-gray-600 dark:text-gray-400 mb-2">
              {definition}
            </div>
            {example && (
              <div className="text-xs text-blue-600 dark:text-blue-400 italic">
                예: {example}
              </div>
            )}
          </div>
        </div>
      )}
    </span>
  )
}

// 자주 사용되는 용어들
export const smartFactoryTerms = {
  'Six Sigma': {
    definition: '품질 관리 방법론. 100만개 중 3.4개만 불량 허용',
    example: '자동차 부품은 99.99966% 정확도 필요'
  },
  'PPM': {
    definition: 'Parts Per Million, 백만분의 1 단위',
    example: '불량률 10PPM = 100만개 중 10개 불량'
  },
  'FDA': {
    definition: '미국 식품의약품안전처',
    example: '의료기기는 FDA 승인 필수'
  },
  'OEE': {
    definition: 'Overall Equipment Effectiveness, 설비종합효율',
    example: 'OEE 85% = 가동률×성능×품질'
  },
  'ROI': {
    definition: 'Return on Investment, 투자수익률',
    example: 'ROI 200% = 투자금의 2배 수익'
  },
  'TCO': {
    definition: 'Total Cost of Ownership, 총소유비용',
    example: '구매비 + 운영비 + 유지보수비'
  },
  'MES': {
    definition: 'Manufacturing Execution System, 생산관리시스템',
    example: '실시간 생산 현황 모니터링'
  },
  'ERP': {
    definition: 'Enterprise Resource Planning, 전사적자원관리',
    example: '회계, 인사, 생산을 통합 관리'
  },
  'PLC': {
    definition: 'Programmable Logic Controller, 프로그래머블 논리 제어기',
    example: '공장 기계 자동 제어 장치'
  },
  'SCADA': {
    definition: 'Supervisory Control and Data Acquisition, 감시제어 데이터수집',
    example: '공장 전체를 중앙에서 모니터링'
  }
}