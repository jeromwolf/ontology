'use client'

import React, { useState } from 'react'
import { 
  Grid3X3, 
  ArrowRight, 
  Database, 
  Network, 
  GitBranch, 
  Calendar, 
  Users, 
  Workflow, 
  Activity,
  Search,
  Star,
  Copy,
  Eye
} from 'lucide-react'
import SpaceOptimizedButton, { ButtonGroup } from './SpaceOptimizedButton'
import { cn } from '@/lib/utils'

export interface MermaidTemplate {
  id: string
  title: string
  description: string
  category: string
  tags: string[]
  code: string
  icon: React.ReactNode
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  useCase: string
  preview?: string
}

export interface MermaidTemplatesProps {
  onSelectTemplate: (template: MermaidTemplate) => void
  onPreviewTemplate?: (template: MermaidTemplate) => void
  className?: string
  searchQuery?: string
  selectedCategory?: string
}

/**
 * 전문급 Mermaid 템플릿 라이브러리
 * 
 * 특징:
 * ✅ 체계적인 분류: 시스템 설계별 템플릿 구분
 * ✅ 실용적인 템플릿: 실제 업무에서 사용 가능한 구조
 * ✅ 난이도별 구분: 초급/중급/고급 템플릿
 * ✅ 검색 및 필터링: 빠른 템플릿 검색
 * ✅ 미리보기: 선택 전 템플릿 미리보기
 */

// 실용적인 아키텍처 템플릿들
const TEMPLATES: MermaidTemplate[] = [
  // 🏗️ 시스템 아키텍처
  {
    id: 'microservices',
    title: '마이크로서비스 아키텍처',
    description: 'MSA 기반 분산 시스템 구조',
    category: '시스템 아키텍처',
    tags: ['microservices', 'distributed', 'api-gateway', 'kubernetes'],
    difficulty: 'advanced',
    useCase: '대규모 웹 서비스, 분산 시스템 설계',
    icon: <Network className="w-4 h-4" />,
    code: `graph TB
    subgraph "Client Layer"
        WEB[Web App]
        MOBILE[Mobile App]
        API_DOC[API Documentation]
    end
    
    subgraph "API Gateway"
        GATEWAY[API Gateway<br/>Authentication & Routing]
    end
    
    subgraph "Microservices"
        USER_SVC[User Service<br/>Spring Boot]
        ORDER_SVC[Order Service<br/>Node.js]
        PAYMENT_SVC[Payment Service<br/>Python]
        INVENTORY_SVC[Inventory Service<br/>Go]
    end
    
    subgraph "Data Layer"
        USER_DB[(User DB<br/>PostgreSQL)]
        ORDER_DB[(Order DB<br/>MongoDB)]
        PAYMENT_DB[(Payment DB<br/>MySQL)]
        INVENTORY_DB[(Inventory DB<br/>Redis)]
    end
    
    subgraph "Infrastructure"
        QUEUE[Message Queue<br/>RabbitMQ]
        CACHE[Cache<br/>Redis]
        LOG[Logging<br/>ELK Stack]
        MONITOR[Monitoring<br/>Prometheus]
    end
    
    WEB --> GATEWAY
    MOBILE --> GATEWAY
    
    GATEWAY --> USER_SVC
    GATEWAY --> ORDER_SVC
    GATEWAY --> PAYMENT_SVC
    GATEWAY --> INVENTORY_SVC
    
    USER_SVC --> USER_DB
    ORDER_SVC --> ORDER_DB
    PAYMENT_SVC --> PAYMENT_DB
    INVENTORY_SVC --> INVENTORY_DB
    
    ORDER_SVC --> QUEUE
    PAYMENT_SVC --> QUEUE
    INVENTORY_SVC --> QUEUE
    
    USER_SVC --> CACHE
    ORDER_SVC --> CACHE
    
    USER_SVC --> LOG
    ORDER_SVC --> LOG
    PAYMENT_SVC --> LOG
    INVENTORY_SVC --> LOG
    
    ALL_SERVICES --> MONITOR`
  },
  
  {
    id: 'ci-cd-pipeline',
    title: 'CI/CD 파이프라인',
    description: 'DevOps 자동화 배포 플로우',
    category: 'DevOps',
    tags: ['ci-cd', 'devops', 'automation', 'deployment'],
    difficulty: 'intermediate',
    useCase: '소프트웨어 배포 자동화, DevOps 워크플로우',
    icon: <GitBranch className="w-4 h-4" />,
    code: `graph LR
    subgraph "Development"
        DEV[Developer]
        IDE[IDE/VS Code]
        GIT[Git Repository]
    end
    
    subgraph "CI Pipeline"
        TRIGGER[Webhook Trigger]
        BUILD[Build & Test<br/>Docker]
        SCAN[Security Scan<br/>SonarQube]
        ARTIFACT[Artifact Storage<br/>Nexus/JFrog]
    end
    
    subgraph "CD Pipeline"
        DEPLOY_DEV[Deploy to Dev<br/>Kubernetes]
        TEST_AUTO[Automated Tests<br/>Selenium/Jest]
        DEPLOY_STAGE[Deploy to Staging<br/>Blue-Green]
        MANUAL_TEST[Manual Testing<br/>QA Team]
        DEPLOY_PROD[Deploy to Production<br/>Canary Release]
    end
    
    subgraph "Monitoring"
        HEALTH[Health Checks<br/>Prometheus]
        LOGS[Log Aggregation<br/>ELK Stack]
        ALERT[Alerting<br/>PagerDuty]
        ROLLBACK[Auto Rollback<br/>Argo CD]
    end
    
    DEV --> IDE
    IDE --> GIT
    GIT --> TRIGGER
    
    TRIGGER --> BUILD
    BUILD --> SCAN
    SCAN --> ARTIFACT
    
    ARTIFACT --> DEPLOY_DEV
    DEPLOY_DEV --> TEST_AUTO
    TEST_AUTO --> DEPLOY_STAGE
    DEPLOY_STAGE --> MANUAL_TEST
    MANUAL_TEST --> DEPLOY_PROD
    
    DEPLOY_PROD --> HEALTH
    HEALTH --> LOGS
    LOGS --> ALERT
    ALERT --> ROLLBACK
    
    ROLLBACK -.-> DEPLOY_STAGE`
  },

  {
    id: 'database-sharding',
    title: '데이터베이스 샤딩',
    description: '수평 분할을 통한 데이터베이스 확장',
    category: '데이터베이스',
    tags: ['database', 'sharding', 'scalability', 'performance'],
    difficulty: 'advanced',
    useCase: '대용량 데이터 처리, 데이터베이스 성능 최적화',
    icon: <Database className="w-4 h-4" />,
    code: `graph TB
    subgraph "Application Layer"
        APP1[App Server 1]
        APP2[App Server 2]
        APP3[App Server 3]
    end
    
    subgraph "Database Router"
        ROUTER[Shard Router<br/>Consistent Hashing]
        METADATA[Metadata Store<br/>Shard Mapping]
    end
    
    subgraph "Shard 1 - Users A-F"
        SHARD1_M[(Master DB 1<br/>Users A-F)]
        SHARD1_S1[(Slave DB 1-1)]
        SHARD1_S2[(Slave DB 1-2)]
    end
    
    subgraph "Shard 2 - Users G-M"
        SHARD2_M[(Master DB 2<br/>Users G-M)]
        SHARD2_S1[(Slave DB 2-1)]
        SHARD2_S2[(Slave DB 2-2)]
    end
    
    subgraph "Shard 3 - Users N-Z"
        SHARD3_M[(Master DB 3<br/>Users N-Z)]
        SHARD3_S1[(Slave DB 3-1)]
        SHARD3_S2[(Slave DB 3-2)]
    end
    
    subgraph "Cross-Shard Operations"
        AGG[Aggregation Service]
        QUEUE[Message Queue<br/>Cross-Shard Events]
    end
    
    APP1 --> ROUTER
    APP2 --> ROUTER
    APP3 --> ROUTER
    
    ROUTER --> METADATA
    
    ROUTER --> SHARD1_M
    ROUTER --> SHARD2_M
    ROUTER --> SHARD3_M
    
    SHARD1_M --> SHARD1_S1
    SHARD1_M --> SHARD1_S2
    SHARD2_M --> SHARD2_S1
    SHARD2_M --> SHARD2_S2
    SHARD3_M --> SHARD3_S1
    SHARD3_M --> SHARD3_S2
    
    ROUTER --> AGG
    AGG --> QUEUE
    
    QUEUE --> SHARD1_M
    QUEUE --> SHARD2_M
    QUEUE --> SHARD3_M`
  },

  // 🔄 플로우차트
  {
    id: 'user-onboarding',
    title: '사용자 온보딩 플로우',
    description: '신규 사용자 등록 및 인증 과정',
    category: '비즈니스 플로우',
    tags: ['user-flow', 'onboarding', 'authentication', 'ux'],
    difficulty: 'beginner',
    useCase: 'UX 설계, 사용자 경험 최적화',
    icon: <Users className="w-4 h-4" />,
    code: `flowchart TD
    START([사용자 방문]) --> LANDING[랜딩 페이지]
    LANDING --> SIGNUP_BTN{회원가입 클릭}
    SIGNUP_BTN -->|Yes| EMAIL_FORM[이메일 입력]
    SIGNUP_BTN -->|No| BROWSE[서비스 둘러보기]
    
    EMAIL_FORM --> EMAIL_VALID{이메일 유효성}
    EMAIL_VALID -->|Invalid| EMAIL_ERROR[오류 메시지]
    EMAIL_ERROR --> EMAIL_FORM
    EMAIL_VALID -->|Valid| SEND_CODE[인증코드 발송]
    
    SEND_CODE --> VERIFY_FORM[인증코드 입력]
    VERIFY_FORM --> CODE_VALID{코드 유효성}
    CODE_VALID -->|Invalid| CODE_ERROR[오류 메시지]
    CODE_ERROR --> VERIFY_FORM
    CODE_VALID -->|Valid| PROFILE_FORM[프로필 입력]
    
    PROFILE_FORM --> TERMS[약관 동의]
    TERMS --> COMPLETE[회원가입 완료]
    COMPLETE --> WELCOME[환영 메시지]
    WELCOME --> TUTORIAL[튜토리얼 시작]
    
    TUTORIAL --> FEATURE1[핵심 기능 1]
    FEATURE1 --> FEATURE2[핵심 기능 2]
    FEATURE2 --> FEATURE3[핵심 기능 3]
    FEATURE3 --> DASHBOARD[대시보드]
    
    BROWSE --> LOGIN{로그인 필요}
    LOGIN -->|Yes| LOGIN_FORM[로그인 폼]
    LOGIN_FORM --> DASHBOARD
    LOGIN -->|No| GUEST_MODE[게스트 모드]
    GUEST_MODE --> LIMITED[제한된 기능]`
  },

  // 📊 시퀀스 다이어그램
  {
    id: 'payment-sequence',
    title: '결제 시스템 시퀀스',
    description: '온라인 결제 처리 과정',
    category: '시퀀스 다이어그램',
    tags: ['payment', 'sequence', 'api', 'transaction'],
    difficulty: 'intermediate',
    useCase: 'API 설계, 결제 시스템 구현',
    icon: <Activity className="w-4 h-4" />,
    code: `sequenceDiagram
    participant U as 사용자
    participant W as 웹앱
    participant A as API 서버
    participant P as 결제 게이트웨이
    participant B as 은행
    participant D as 데이터베이스
    
    U->>W: 결제 버튼 클릭
    W->>W: 주문 정보 검증
    W->>A: 결제 요청 (POST /payment)
    
    A->>D: 주문 정보 조회
    D-->>A: 주문 데이터 반환
    
    A->>A: 결제 금액 검증
    A->>P: 결제 처리 요청
    
    P->>B: 카드 승인 요청
    B-->>P: 승인 결과
    
    alt 결제 성공
        P-->>A: 결제 성공 응답
        A->>D: 결제 내역 저장
        A->>D: 주문 상태 업데이트
        A-->>W: 결제 성공 응답
        W-->>U: 결제 완료 페이지
        
        A->>A: 이메일 발송 큐 추가
        A->>A: 배송 처리 큐 추가
        
    else 결제 실패
        P-->>A: 결제 실패 응답
        A->>D: 실패 로그 저장
        A-->>W: 결제 실패 응답
        W-->>U: 오류 메시지 표시
    end
    
    Note over U,D: 모든 거래는 로그로 기록됨`
  },

  // 📅 간트 차트
  {
    id: 'project-gantt',
    title: '프로젝트 일정 관리',
    description: '소프트웨어 개발 프로젝트 타임라인',
    category: '프로젝트 관리',
    tags: ['gantt', 'project', 'timeline', 'planning'],
    difficulty: 'beginner',
    useCase: '프로젝트 계획, 일정 관리',
    icon: <Calendar className="w-4 h-4" />,
    code: `gantt
    title 웹 서비스 개발 프로젝트
    dateFormat  YYYY-MM-DD
    section 기획 단계
    요구사항 분석          :done,    req, 2024-01-01, 2024-01-15
    기술 스택 선정        :done,    tech, 2024-01-10, 2024-01-20
    UI/UX 설계           :done,    design, 2024-01-15, 2024-02-05
    
    section 개발 단계
    백엔드 API 개발       :active,  backend, 2024-02-01, 2024-03-15
    프론트엔드 개발       :frontend, 2024-02-15, 2024-03-30
    데이터베이스 설계     :done, db, 2024-02-01, 2024-02-10
    
    section 테스트 단계
    단위 테스트          :unittest, after backend, 5d
    통합 테스트          :inttest, after frontend, 7d
    사용자 테스트        :usertest, after inttest, 10d
    
    section 배포 단계
    스테이징 배포        :staging, after usertest, 3d
    성능 테스트          :perftest, after staging, 5d
    프로덕션 배포        :prod, after perftest, 2d
    
    section 운영 단계
    모니터링 설정        :monitor, after prod, 7d
    사용자 피드백 수집    :feedback, after prod, 30d`
  }
]

const CATEGORIES = [
  '전체',
  '시스템 아키텍처',
  'DevOps',
  '데이터베이스',
  '비즈니스 플로우',
  '시퀀스 다이어그램',
  '프로젝트 관리'
]

const MermaidTemplates: React.FC<MermaidTemplatesProps> = ({
  onSelectTemplate,
  onPreviewTemplate,
  className,
  searchQuery = '',
  selectedCategory = '전체',
}) => {
  const [localSearchQuery, setLocalSearchQuery] = useState(searchQuery)
  const [localCategory, setLocalCategory] = useState(selectedCategory)
  const [favorites, setFavorites] = useState<Set<string>>(new Set())

  // 템플릿 필터링
  const filteredTemplates = TEMPLATES.filter(template => {
    const matchesSearch = 
      template.title.toLowerCase().includes(localSearchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(localSearchQuery.toLowerCase()) ||
      template.tags.some(tag => tag.toLowerCase().includes(localSearchQuery.toLowerCase()))
    
    const matchesCategory = 
      localCategory === '전체' || template.category === localCategory
    
    return matchesSearch && matchesCategory
  })

  // 즐겨찾기 토글
  const toggleFavorite = (templateId: string) => {
    const newFavorites = new Set(favorites)
    if (newFavorites.has(templateId)) {
      newFavorites.delete(templateId)
    } else {
      newFavorites.add(templateId)
    }
    setFavorites(newFavorites)
  }

  // 난이도 색상
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20'
      case 'intermediate': return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20'
      case 'advanced': return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20'
      default: return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20'
    }
  }

  // 난이도 라벨
  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '초급'
      case 'intermediate': return '중급'
      case 'advanced': return '고급'
      default: return difficulty
    }
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* 헤더 및 검색 */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2 mb-3">
          <Grid3X3 className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            템플릿 라이브러리
          </h2>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {filteredTemplates.length}개 템플릿
          </span>
        </div>

        {/* 검색 바 */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="템플릿 검색..."
            value={localSearchQuery}
            onChange={(e) => setLocalSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
        </div>

        {/* 카테고리 필터 */}
        <div className="flex flex-wrap gap-1">
          {CATEGORIES.map(category => (
            <SpaceOptimizedButton
              key={category}
              variant={localCategory === category ? 'primary' : 'ghost'}
              size="xs"
              compact
              onClick={() => setLocalCategory(category)}
            >
              {category}
            </SpaceOptimizedButton>
          ))}
        </div>
      </div>

      {/* 템플릿 목록 */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-1 gap-4">
          {filteredTemplates.map(template => (
            <div
              key={template.id}
              className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-blue-300 dark:hover:border-blue-600 transition-colors cursor-pointer group"
              onClick={() => onSelectTemplate(template)}
            >
              {/* 헤더 */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-blue-600 dark:text-blue-400">
                    {template.icon}
                  </span>
                  <h3 className="font-medium text-gray-900 dark:text-gray-100">
                    {template.title}
                  </h3>
                  <span className={cn(
                    'px-2 py-0.5 text-xs rounded-full',
                    getDifficultyColor(template.difficulty)
                  )}>
                    {getDifficultyLabel(template.difficulty)}
                  </span>
                </div>

                <div className="flex items-center gap-1">
                  <SpaceOptimizedButton
                    variant="ghost"
                    size="xs"
                    icon={<Star className={cn(
                      'w-3 h-3',
                      favorites.has(template.id) ? 'text-yellow-500 fill-current' : 'text-gray-400'
                    )} />}
                    onClick={(e) => {
                      e.stopPropagation()
                      toggleFavorite(template.id)
                    }}
                  />
                  
                  {onPreviewTemplate && (
                    <SpaceOptimizedButton
                      variant="ghost"
                      size="xs"
                      icon={<Eye className="w-3 h-3" />}
                      onClick={(e) => {
                        e.stopPropagation()
                        onPreviewTemplate(template)
                      }}
                    />
                  )}
                  
                  <SpaceOptimizedButton
                    variant="ghost"
                    size="xs"
                    icon={<Copy className="w-3 h-3" />}
                    onClick={(e) => {
                      e.stopPropagation()
                      navigator.clipboard.writeText(template.code)
                    }}
                  />
                </div>
              </div>

              {/* 설명 */}
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {template.description}
              </p>

              {/* 사용 사례 */}
              <p className="text-xs text-gray-500 dark:text-gray-500 mb-3">
                💡 {template.useCase}
              </p>

              {/* 태그 */}
              <div className="flex flex-wrap gap-1 mb-3">
                {template.tags.map(tag => (
                  <span
                    key={tag}
                    className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded"
                  >
                    #{tag}
                  </span>
                ))}
              </div>

              {/* 액션 버튼 */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 dark:text-gray-500">
                  {template.category}
                </span>
                
                <SpaceOptimizedButton
                  variant="primary"
                  size="xs"
                  compact
                  icon={<ArrowRight className="w-3 h-3" />}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  사용하기
                </SpaceOptimizedButton>
              </div>
            </div>
          ))}
        </div>

        {/* 결과 없음 */}
        {filteredTemplates.length === 0 && (
          <div className="text-center py-12">
            <Grid3X3 className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              템플릿을 찾을 수 없습니다
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              다른 검색어나 카테고리를 시도해보세요
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default MermaidTemplates

// 편의를 위한 템플릿 검색 함수
export const searchTemplates = (query: string, category?: string) => {
  return TEMPLATES.filter(template => {
    const matchesSearch = 
      template.title.toLowerCase().includes(query.toLowerCase()) ||
      template.description.toLowerCase().includes(query.toLowerCase()) ||
      template.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
    
    const matchesCategory = 
      !category || category === '전체' || template.category === category
    
    return matchesSearch && matchesCategory
  })
}

export { TEMPLATES, CATEGORIES }