'use client'

import React, { useState, useEffect } from 'react'
import { 
  Split, 
  PanelRightClose, 
  PanelRightOpen, 
  Palette, 
  FileText, 
  Grid3X3, 
  Settings,
  Download,
  Upload,
  Save,
  Share2,
  History,
  Moon,
  Sun
} from 'lucide-react'
import AdaptiveLayout from '@/components/ui/AdaptiveLayout'
import ResponsiveCanvas from '@/components/ui/ResponsiveCanvas'
import CollapsibleControls, { createControlSection } from '@/components/ui/CollapsibleControls'
import SpaceOptimizedButton, { ButtonGroup } from '@/components/ui/SpaceOptimizedButton'
import MermaidEditor from '@/components/ui/MermaidEditor'
import MermaidPreview from '@/components/ui/MermaidPreview'
import MermaidTemplates, { type MermaidTemplate } from '@/components/ui/MermaidTemplates'
import { cn } from '@/lib/utils'

/**
 * 전문급 Mermaid 다이어그램 에디터
 * 
 * 🎯 System Design 모듈의 핵심 시뮬레이터
 * 
 * 특징:
 * ✅ 완전한 공간 최적화: 새로운 UI 컴포넌트 활용
 * ✅ 실시간 미리보기: 코드 입력과 동시에 다이어그램 업데이트
 * ✅ 전문 템플릿: 실무에서 사용 가능한 아키텍처 템플릿
 * ✅ 고급 내보내기: SVG, PNG, 코드 공유
 * ✅ 협업 기능: 템플릿 공유, 히스토리 관리
 * ✅ 접근성: 완벽한 키보드 단축키 지원
 */
const MermaidDiagramEditor: React.FC = () => {
  // 상태 관리
  const [code, setCode] = useState(`graph TD
    A[사용자 요청] --> B{인증 확인}
    B -->|성공| C[API 게이트웨이]
    B -->|실패| D[로그인 페이지]
    C --> E[마이크로서비스]
    E --> F[데이터베이스]
    F --> G[응답 전송]
    G --> A`)
  
  const [theme, setTheme] = useState<'light' | 'dark' | 'forest' | 'base' | 'neutral'>('light')
  const [showTemplates, setShowTemplates] = useState(false)
  const [showEditor, setShowEditor] = useState(true)
  const [history, setHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [lastError, setLastError] = useState<string | null>(null)

  // 히스토리 관리
  useEffect(() => {
    if (code && code !== history[historyIndex]) {
      const newHistory = history.slice(0, historyIndex + 1)
      newHistory.push(code)
      if (newHistory.length > 50) { // 최대 50개 히스토리
        newHistory.shift()
      }
      setHistory(newHistory)
      setHistoryIndex(newHistory.length - 1)
    }
  }, [code])

  // 실행취소/다시실행
  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1)
      setCode(history[historyIndex - 1])
    }
  }

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1)
      setCode(history[historyIndex + 1])
    }
  }

  // 템플릿 선택
  const handleSelectTemplate = (template: MermaidTemplate) => {
    setCode(template.code)
    setShowTemplates(false)
  }

  // 파일 저장
  const handleSave = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'diagram.mmd'
    a.click()
    URL.revokeObjectURL(url)
  }

  // 파일 로드
  const handleLoad = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target?.result as string
      setCode(content)
    }
    reader.readAsText(file)
  }

  // 공유 기능
  const handleShare = async () => {
    const shareData = {
      title: 'Mermaid 다이어그램',
      text: 'KSS에서 생성된 다이어그램입니다.',
      url: window.location.href,
    }
    
    if (navigator.share) {
      try {
        await navigator.share(shareData)
      } catch (err) {
        console.log('공유 취소됨')
      }
    } else {
      // 클립보드에 복사
      await navigator.clipboard.writeText(code)
      alert('다이어그램 코드가 클립보드에 복사되었습니다!')
    }
  }

  // 제어 섹션들
  const controlSections = [
    createControlSection(
      'templates',
      '템플릿',
      <div className="space-y-3">
        <SpaceOptimizedButton
          variant={showTemplates ? 'secondary' : 'primary'}
          size="sm"
          fullWidth
          icon={<Grid3X3 className="w-4 h-4" />}
          onClick={() => setShowTemplates(!showTemplates)}
        >
          {showTemplates ? '에디터 보기' : '템플릿 보기'}
        </SpaceOptimizedButton>
        
        <div className="text-xs text-gray-600 dark:text-gray-400">
          실무에서 사용 가능한 아키텍처 템플릿을 제공합니다.
        </div>
      </div>,
      { 
        defaultExpanded: true, 
        icon: <Grid3X3 className="w-4 h-4" />,
        badge: '6+' 
      }
    ),

    createControlSection(
      'appearance',
      '테마 설정',
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-2">다이어그램 테마</label>
          <div className="grid grid-cols-2 gap-2">
            {(['light', 'dark', 'forest', 'base', 'neutral'] as const).map(t => (
              <SpaceOptimizedButton
                key={t}
                variant={theme === t ? 'primary' : 'outline'}
                size="xs"
                compact
                onClick={() => setTheme(t)}
              >
                {t === 'light' ? '라이트' : 
                 t === 'dark' ? '다크' : 
                 t === 'forest' ? '포레스트' :
                 t === 'base' ? '베이스' : '뉴트럴'}
              </SpaceOptimizedButton>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="showEditor"
            checked={showEditor}
            onChange={(e) => setShowEditor(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="showEditor" className="text-sm">에디터 표시</label>
        </div>
      </div>,
      { 
        icon: <Palette className="w-4 h-4" /> 
      }
    ),

    createControlSection(
      'history',
      '히스토리',
      <div className="space-y-3">
        <ButtonGroup>
          <SpaceOptimizedButton
            variant="outline"
            size="sm"
            onClick={undo}
            disabled={historyIndex <= 0}
          >
            실행취소
          </SpaceOptimizedButton>
          
          <SpaceOptimizedButton
            variant="outline"
            size="sm"
            onClick={redo}
            disabled={historyIndex >= history.length - 1}
          >
            다시실행
          </SpaceOptimizedButton>
        </ButtonGroup>

        <div className="text-xs text-gray-600 dark:text-gray-400">
          히스토리: {historyIndex + 1} / {history.length}
        </div>
      </div>,
      { 
        icon: <History className="w-4 h-4" /> 
      }
    ),

    createControlSection(
      'export',
      '내보내기 & 공유',
      <div className="space-y-2">
        <SpaceOptimizedButton
          variant="outline"
          size="sm"
          fullWidth
          icon={<Save className="w-4 h-4" />}
          onClick={handleSave}
        >
          Mermaid 파일 저장
        </SpaceOptimizedButton>

        <SpaceOptimizedButton
          variant="outline"
          size="sm"
          fullWidth
          icon={<Share2 className="w-4 h-4" />}
          onClick={handleShare}
        >
          다이어그램 공유
        </SpaceOptimizedButton>

        <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
          💡 미리보기에서 PNG/SVG 다운로드 가능
        </div>
      </div>,
      { 
        icon: <Download className="w-4 h-4" /> 
      }
    ),
  ]

  return (
    <div className="w-full h-screen bg-gray-50 dark:bg-gray-900">
      {/* 헤더 */}
      <div className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <FileText className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Mermaid 다이어그램 에디터
            </h1>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              전문급 시스템 설계 도구
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {lastError && (
            <div className="text-xs text-red-600 dark:text-red-400 max-w-48 truncate">
              ⚠️ {lastError}
            </div>
          )}
          
          <ButtonGroup>
            <SpaceOptimizedButton
              variant="ghost"
              size="sm"
              icon={<Upload className="w-4 h-4" />}
              tooltip="파일 업로드"
            >
              업로드
            </SpaceOptimizedButton>
            
            <SpaceOptimizedButton
              variant="ghost"
              size="sm"
              icon={<Settings className="w-4 h-4" />}
              tooltip="설정"
            >
              설정
            </SpaceOptimizedButton>
          </ButtonGroup>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="h-[calc(100vh-64px)]">
        <AdaptiveLayout
          controls={
            <CollapsibleControls
              sections={controlSections}
              title="도구"
              defaultCollapsed={false}
              persistent={true}
              onSectionToggle={(sectionId, expanded) => {
                console.log(`${sectionId} ${expanded ? 'expanded' : 'collapsed'}`)
              }}
            />
          }
          config={{
            mode: 'visualization-focused',
            allowModeSwitch: true,
            showModeToggle: true,
          }}
          onLayoutChange={(config) => {
            console.log('Layout changed:', config)
          }}
        >
          {/* 메인 작업 영역 */}
          <div className="h-full flex flex-col">
            {showTemplates ? (
              /* 템플릿 라이브러리 */
              <MermaidTemplates
                onSelectTemplate={handleSelectTemplate}
                onPreviewTemplate={(template) => {
                  // 미리보기 모달 또는 사이드패널
                  console.log('Preview template:', template)
                }}
              />
            ) : (
              /* 에디터 + 미리보기 */
              <div className="h-full flex flex-col lg:flex-row gap-2">
                {/* 코드 에디터 */}
                {showEditor && (
                  <div className="flex-1 lg:w-1/2">
                    <MermaidEditor
                      value={code}
                      onChange={setCode}
                      onSave={handleSave}
                      onLoad={handleLoad}
                      theme="light"
                      showLineNumbers={true}
                    />
                  </div>
                )}

                {/* 미리보기 */}
                <div className={cn(
                  'flex-1',
                  showEditor ? 'lg:w-1/2' : 'w-full'
                )}>
                  <MermaidPreview
                    code={code}
                    theme={theme}
                    onError={setLastError}
                    onSuccess={() => setLastError(null)}
                    enableZoom={true}
                    enablePan={true}
                    autoFit={true}
                  />
                </div>
              </div>
            )}
          </div>
        </AdaptiveLayout>
      </div>
    </div>
  )
}

export default MermaidDiagramEditor