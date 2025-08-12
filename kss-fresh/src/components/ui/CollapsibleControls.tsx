'use client'

import React, { useState, useRef, useEffect } from 'react'
import { 
  ChevronDown, 
  ChevronRight, 
  Settings, 
  Eye, 
  EyeOff, 
  Pin, 
  PinOff,
  X,
  HelpCircle 
} from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ControlSection {
  id: string
  title: string
  icon?: React.ReactNode
  content: React.ReactNode
  defaultExpanded?: boolean
  badge?: string | number
  disabled?: boolean
  tooltip?: string
}

export interface CollapsibleControlsProps {
  sections: ControlSection[]
  className?: string
  defaultCollapsed?: boolean
  persistent?: boolean
  showHeader?: boolean
  title?: string
  maxHeight?: number
  onSectionToggle?: (sectionId: string, expanded: boolean) => void
  onVisibilityToggle?: (visible: boolean) => void
}

/**
 * 접이식 제어판 컴포넌트
 * 
 * 기존 문제점 해결:
 * ❌ 고정 제어판: 항상 공간 점유
 * ❌ 섹션 구분 없음: 모든 설정이 한 곳에
 * ❌ 우선순위 없음: 중요하지 않은 설정도 항상 표시
 * ❌ 스크롤 문제: 긴 제어판으로 인한 UI 밀림
 * 
 * 새로운 기능:
 * ✅ 섹션별 접기/펴기: 필요한 설정만 표시
 * ✅ 전체 숨기기/보이기: 시각화 공간 최대화
 * ✅ 고정 모드: 자주 쓰는 설정 항상 표시
 * ✅ 스마트 높이: 컨텐츠에 맞춰 자동 조정
 * ✅ 접근성: 키보드 네비게이션 완벽 지원
 */
const CollapsibleControls: React.FC<CollapsibleControlsProps> = ({
  sections,
  className,
  defaultCollapsed = false,
  persistent = false,
  showHeader = true,
  title = '제어판',
  maxHeight = 600,
  onSectionToggle,
  onVisibilityToggle,
}) => {
  const [isVisible, setIsVisible] = useState(!defaultCollapsed)
  const [isPinned, setIsPinned] = useState(persistent)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(sections.filter(s => s.defaultExpanded).map(s => s.id))
  )
  
  const controlsRef = useRef<HTMLDivElement>(null)

  // 섹션 토글
  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId)
    } else {
      newExpanded.add(sectionId)
    }
    setExpandedSections(newExpanded)
    onSectionToggle?.(sectionId, newExpanded.has(sectionId))
  }

  // 전체 표시/숨김 토글
  const toggleVisibility = () => {
    const newVisible = !isVisible
    setIsVisible(newVisible)
    onVisibilityToggle?.(newVisible)
  }

  // 고정 토글
  const togglePin = () => {
    setIsPinned(!isPinned)
  }

  // 키보드 단축키
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ctrl + . 으로 제어판 토글
      if (event.ctrlKey && event.key === '.') {
        event.preventDefault()
        toggleVisibility()
      }
      // Ctrl + ; 으로 고정 토글
      if (event.ctrlKey && event.key === ';') {
        event.preventDefault()
        togglePin()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isVisible, isPinned])

  // 컨텐츠 영역 외부 클릭시 숨기기 (고정되지 않은 경우)
  useEffect(() => {
    if (isPinned || !isVisible) return

    const handleClickOutside = (event: MouseEvent) => {
      if (controlsRef.current && !controlsRef.current.contains(event.target as Node)) {
        setIsVisible(false)
        onVisibilityToggle?.(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isPinned, isVisible, onVisibilityToggle])

  return (
    <div
      ref={controlsRef}
      className={cn(
        'relative transition-all duration-300 ease-in-out',
        isVisible ? 'opacity-100' : 'opacity-90 hover:opacity-100',
        className
      )}
      role="region"
      aria-label="Controls panel"
    >
      {/* 헤더 */}
      {showHeader && (
        <div className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-600">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-gray-600 dark:text-gray-300" />
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {title}
            </h3>
          </div>
          
          <div className="flex items-center gap-1">
            {/* 고정 버튼 */}
            <button
              onClick={togglePin}
              className={cn(
                'p-1 rounded transition-colors',
                isPinned 
                  ? 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900' 
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              )}
              title={`제어판 ${isPinned ? '고정 해제' : '고정'} (Ctrl+;)`}
              aria-label={`제어판 ${isPinned ? '고정 해제' : '고정'}`}
            >
              {isPinned ? <Pin className="w-4 h-4" /> : <PinOff className="w-4 h-4" />}
            </button>

            {/* 표시/숨김 버튼 */}
            <button
              onClick={toggleVisibility}
              className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 rounded transition-colors"
              title={`제어판 ${isVisible ? '숨기기' : '보이기'} (Ctrl+.)`}
              aria-label={`제어판 ${isVisible ? '숨기기' : '보이기'}`}
            >
              {isVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
      )}

      {/* 컨텐츠 영역 */}
      <div
        className={cn(
          'transition-all duration-300 ease-in-out overflow-hidden',
          'bg-white dark:bg-gray-800 rounded-b-lg',
          !showHeader && 'rounded-lg',
          isVisible ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
        )}
        style={{ maxHeight: isVisible ? `${maxHeight}px` : 0 }}
      >
        <div className="overflow-y-auto max-h-full">
          {sections.map((section, index) => {
            const isExpanded = expandedSections.has(section.id)
            const isDisabled = section.disabled

            return (
              <div
                key={section.id}
                className={cn(
                  'border-b border-gray-200 dark:border-gray-600 last:border-b-0',
                  isDisabled && 'opacity-50 pointer-events-none'
                )}
              >
                {/* 섹션 헤더 */}
                <button
                  onClick={() => !isDisabled && toggleSection(section.id)}
                  className={cn(
                    'w-full flex items-center justify-between p-3 text-left',
                    'hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset',
                    isDisabled && 'cursor-not-allowed'
                  )}
                  aria-expanded={isExpanded}
                  aria-controls={`section-${section.id}`}
                  disabled={isDisabled}
                  title={section.tooltip}
                >
                  <div className="flex items-center gap-2">
                    {section.icon && (
                      <span className="text-gray-500 dark:text-gray-400">
                        {section.icon}
                      </span>
                    )}
                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {section.title}
                    </span>
                    {section.badge && (
                      <span className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full">
                        {section.badge}
                      </span>
                    )}
                    {section.tooltip && (
                      <HelpCircle className="w-3 h-3 text-gray-400" />
                    )}
                  </div>
                  
                  <span className="text-gray-400">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </span>
                </button>

                {/* 섹션 컨텐츠 */}
                <div
                  id={`section-${section.id}`}
                  className={cn(
                    'transition-all duration-200 ease-in-out overflow-hidden',
                    isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                  )}
                  role="region"
                  aria-labelledby={`header-${section.id}`}
                >
                  <div className="p-3 pt-0 space-y-3">
                    {section.content}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 키보드 단축키 힌트 */}
      <div className="sr-only" aria-live="polite">
        Ctrl+.: 제어판 토글, Ctrl+;: 고정 토글
      </div>
    </div>
  )
}

export default CollapsibleControls

// 편의를 위한 섹션 빌더 헬퍼
export const createControlSection = (
  id: string,
  title: string,
  content: React.ReactNode,
  options: Partial<Omit<ControlSection, 'id' | 'title' | 'content'>> = {}
): ControlSection => ({
  id,
  title,
  content,
  ...options,
})

// 일반적인 섹션 타입들
export const ControlSectionTypes = {
  BASIC: 'basic',
  ADVANCED: 'advanced',
  VISUALIZATION: 'visualization',
  EXPORT: 'export',
  HELP: 'help',
} as const