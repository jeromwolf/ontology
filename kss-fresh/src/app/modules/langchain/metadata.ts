import { Module } from '@/types/module'

export const langchainModule: Module = {
  id: 'langchain',
  name: 'LangChain & LangGraph',
  nameKo: 'LangChain & LangGraph 마스터',
  description: 'LLM 애플리케이션 개발의 완전한 프레임워크',
  version: '1.0.0',
  difficulty: 'intermediate',
  estimatedHours: 24,
  icon: '⛓️',
  color: '#f59e0b',

  prerequisites: ['llm-basics', 'python-programming'],

  chapters: [
    {
      id: '01-langchain-basics',
      title: 'LangChain 시작하기',
      description: 'LangChain의 핵심 개념과 설치',
      estimatedMinutes: 45,
      keywords: ['langchain', 'installation', 'architecture', 'components'],
      learningObjectives: [
        'LangChain의 필요성과 핵심 개념 이해',
        '개발 환경 설정 및 설치',
        'LangChain 아키텍처 구조 파악',
        '첫 LangChain 애플리케이션 구축'
      ]
    },
    {
      id: '02-chains-prompts',
      title: 'Chains와 Prompt Templates',
      description: '체인 구성과 프롬프트 템플릿 설계',
      estimatedMinutes: 60,
      keywords: ['chains', 'prompts', 'templates', 'sequential'],
      learningObjectives: [
        'Chain의 개념과 작동 원리',
        'Prompt Template 설계 패턴',
        'Sequential Chain과 Router Chain',
        'Custom Chain 구현 방법'
      ]
    },
    {
      id: '03-memory-context',
      title: 'Memory와 Context 관리',
      description: '대화 기록 관리와 컨텍스트 유지',
      estimatedMinutes: 50,
      keywords: ['memory', 'context', 'conversation', 'history'],
      learningObjectives: [
        '다양한 Memory 타입 이해',
        'Conversation Buffer와 Summary',
        '컨텍스트 윈도우 최적화',
        'Vector Store Memory 활용'
      ]
    },
    {
      id: '04-agents-tools',
      title: 'Agents와 Tools',
      description: 'AI 에이전트 구축과 도구 통합',
      estimatedMinutes: 70,
      keywords: ['agents', 'tools', 'reasoning', 'action'],
      learningObjectives: [
        'Agent의 작동 원리 (ReAct 패턴)',
        'Built-in Tools 활용',
        'Custom Tool 개발',
        'Agent Executor 최적화'
      ]
    },
    {
      id: '05-langgraph-intro',
      title: 'LangGraph 시작하기',
      description: 'State Graph 기반 복잡한 워크플로우',
      estimatedMinutes: 65,
      keywords: ['langgraph', 'state-graph', 'nodes', 'edges'],
      learningObjectives: [
        'LangGraph vs LangChain 차이점',
        'State Graph 설계 패턴',
        'Node와 Edge 정의',
        '첫 Graph 애플리케이션 구축'
      ]
    },
    {
      id: '06-complex-workflows',
      title: '복잡한 워크플로우와 분기',
      description: '조건부 라우팅과 멀티 에이전트 시스템',
      estimatedMinutes: 80,
      keywords: ['workflow', 'branching', 'routing', 'multi-agent'],
      learningObjectives: [
        '조건부 분기 구현',
        '멀티 에이전트 협업 패턴',
        'Human-in-the-loop 구현',
        'Error Handling과 Retry 로직'
      ]
    },
    {
      id: '07-production-deployment',
      title: '프로덕션 배포와 Best Practices',
      description: '실제 서비스 운영을 위한 필수 지식',
      estimatedMinutes: 75,
      keywords: ['production', 'deployment', 'monitoring', 'optimization'],
      learningObjectives: [
        'LangSmith를 활용한 모니터링',
        'Cache와 성능 최적화',
        'API 키 관리와 보안',
        'LangServe를 통한 배포'
      ]
    },
    {
      id: '08-real-world-projects',
      title: '실전 프로젝트와 사례 연구',
      description: '완전한 애플리케이션 구축 실습',
      estimatedMinutes: 95,
      keywords: ['projects', 'case-studies', 'integration', 'best-practices'],
      learningObjectives: [
        '문서 기반 QA 시스템 구축',
        'Code Assistant 개발',
        '리서치 에이전트 구현',
        'Multi-Modal 애플리케이션'
      ]
    }
  ],

  simulators: [
    {
      id: 'chain-builder',
      name: 'Chain Builder',
      description: '드래그앤드롭으로 체인 구성을 시각화하고 테스트',
      component: 'ChainBuilder'
    },
    {
      id: 'prompt-optimizer',
      name: 'Prompt Optimizer',
      description: '프롬프트 엔지니어링 플레이그라운드',
      component: 'PromptOptimizer'
    },
    {
      id: 'memory-manager',
      name: 'Memory Manager',
      description: '다양한 메모리 시스템 비교 및 시각화',
      component: 'MemoryManager'
    },
    {
      id: 'agent-debugger',
      name: 'Agent Debugger',
      description: 'Agent 실행 과정을 단계별로 추적',
      component: 'AgentDebugger'
    },
    {
      id: 'langgraph-designer',
      name: 'LangGraph Designer',
      description: '그래프 워크플로우 시각적 설계 도구',
      component: 'LangGraphDesigner'
    },
    {
      id: 'tool-integrator',
      name: 'Tool Integrator',
      description: 'Custom Tool 빌더와 테스트 환경',
      component: 'ToolIntegrator'
    },
    {
      id: 'rag-pipeline',
      name: 'RAG Pipeline',
      description: 'LangChain 기반 RAG 시스템 구현',
      component: 'RAGPipeline'
    },
    {
      id: 'async-executor',
      name: 'Async Executor',
      description: '비동기 체인 실행 시각화',
      component: 'AsyncExecutor'
    },
    {
      id: 'cost-calculator',
      name: 'Cost Calculator',
      description: 'Token 사용량과 API 비용 분석',
      component: 'CostCalculator'
    },
    {
      id: 'performance-profiler',
      name: 'Performance Profiler',
      description: 'LangChain 성능 프로파일링 도구',
      component: 'PerformanceProfiler'
    },
    {
      id: 'multi-agent-coordinator',
      name: 'Multi-Agent Coordinator',
      description: '여러 AI 에이전트의 협업과 조율 시뮬레이션',
      component: 'MultiAgentCoordinator'
    }
  ],

  tools: [
    {
      id: 'langchain-builder',
      name: 'LangChain App Builder',
      description: '완전한 LangChain 애플리케이션 템플릿 생성',
      url: '/modules/langchain/tools/builder'
    }
  ]
}
