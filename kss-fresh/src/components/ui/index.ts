// 공간 최적화 UI 컴포넌트 라이브러리
// KSS 프로젝트 전용 - 확장 가능한 아키텍처

// 핵심 컴포넌트들
export { default as ResponsiveCanvas } from './ResponsiveCanvas'
export { default as AdaptiveLayout } from './AdaptiveLayout'
export { default as CollapsibleControls } from './CollapsibleControls'
export { default as SpaceOptimizedButton, ButtonGroup, SimulationControls } from './SpaceOptimizedButton'

// Mermaid 전용 컴포넌트들
export { default as MermaidEditor } from './MermaidEditor'
export { default as MermaidPreview } from './MermaidPreview'
export { default as MermaidTemplates, searchTemplates, TEMPLATES, CATEGORIES } from './MermaidTemplates'

// 완성된 시뮬레이터 템플릿
export { default as SpaceOptimizedSimulator } from './SpaceOptimizedSimulator'

// 타입 정의들
export type { 
  ResponsiveCanvasProps, 
  ResponsiveCanvasRef, 
  CanvasConfig 
} from './ResponsiveCanvas'

export type { 
  LayoutMode, 
  LayoutOrientation, 
  LayoutConfig, 
  AdaptiveLayoutProps 
} from './AdaptiveLayout'

export type { 
  ControlSection, 
  CollapsibleControlsProps 
} from './CollapsibleControls'

export type { 
  SpaceOptimizedButtonProps, 
  ButtonVariant, 
  ButtonSize 
} from './SpaceOptimizedButton'

// 훅들
export { useResponsiveCanvas } from '@/hooks/useResponsiveCanvas'
export { useAdaptiveLayout } from './AdaptiveLayout'

// 헬퍼 함수들
export { createControlSection, ControlSectionTypes } from './CollapsibleControls'