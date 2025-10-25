export const vibeCodingMetadata = {
  id: 'vibe-coding',
  title: 'Vibe Coding with AI',
  description: 'AI 기반 코드 생성 도구를 활용한 차세대 개발 방법론. Cursor, GitHub Copilot, Claude Code 등 최신 AI 코딩 어시스턴트를 마스터하고 생산성을 10배 향상시키세요.',
  category: 'Programming & Development',
  difficulty: 'all-levels',
  duration: '48 hours',
  students: 1250,
  rating: 4.9,
  color: 'from-purple-500 to-pink-600',
  status: 'active' as const,

  chapters: [
    {
      id: 'ai-coding-revolution',
      title: 'Chapter 1: AI 코딩 혁명의 시작',
      description: 'AI 코딩 도구의 등장 배경과 개발 패러다임의 변화. GitHub Copilot, Cursor, Claude Code 등 주요 도구 소개.',
      duration: '3 hours',
      topics: [
        'AI 코딩 도구의 역사와 발전',
        'GitHub Copilot vs Cursor vs Claude Code 비교',
        '개발 생산성 10배 향상의 비밀',
        'AI 페어 프로그래밍의 미래'
      ],
      level: 'beginner' as const
    },
    {
      id: 'cursor-mastery',
      title: 'Chapter 2: Cursor 완벽 마스터',
      description: 'Cursor IDE의 모든 기능을 마스터. Tab, Cmd+K, Composer 등 핵심 기능과 실전 활용법.',
      duration: '4 hours',
      topics: [
        'Cursor 설치 및 환경 설정',
        'Tab (코드 자동완성) 활용법',
        'Cmd+K (인라인 편집) 마스터',
        'Composer (다중 파일 편집) 완벽 가이드',
        'Rules for AI와 .cursorrules 설정',
        '실전 프로젝트에서의 Cursor 워크플로우'
      ],
      level: 'beginner' as const
    },
    {
      id: 'github-copilot',
      title: 'Chapter 3: GitHub Copilot 전문가 되기',
      description: 'GitHub Copilot의 고급 기능과 프롬프트 엔지니어링. VSCode, JetBrains 통합 및 최적화.',
      duration: '4 hours',
      topics: [
        'GitHub Copilot 설정 및 커스터마이징',
        'Copilot Chat vs Copilot Inline',
        '효과적인 프롬프트 작성법',
        'Copilot Workspace 활용',
        '멀티 파일 코드 생성 전략',
        'Copilot Enterprise 고급 기능'
      ],
      level: 'beginner' as const
    },
    {
      id: 'claude-code-engineering',
      title: 'Chapter 4: Claude Code 엔지니어링',
      description: 'Anthropic Claude를 활용한 고급 코딩. Projects, Artifacts, API 활용 및 대규모 리팩토링.',
      duration: '5 hours',
      topics: [
        'Claude Code vs Claude.ai 차이점',
        'Projects 기능으로 컨텍스트 관리',
        'Artifacts로 인터랙티브 앱 생성',
        'Claude API 통합 및 자동화',
        '대규모 코드베이스 리팩토링',
        '200K 토큰 컨텍스트 윈도우 활용'
      ],level: 'intermediate' as const
    },
    {
      id: 'prompt-engineering',
      title: 'Chapter 5: AI 코딩을 위한 프롬프트 엔지니어링',
      description: 'AI 코딩 도구에 최적화된 프롬프트 작성법. 명확성, 컨텍스트, 제약조건 설정 마스터.',
      duration: '4 hours',
      topics: [
        'AI 코딩 프롬프트의 4가지 원칙',
        'Few-shot Learning으로 코드 스타일 학습',
        '컨텍스트 윈도우 최적화 전략',
        'Chain of Thought 프롬프팅',
        '에러 수정 프롬프트 패턴',
        '실전 프롬프트 라이브러리 구축'
      ],
      level: 'intermediate' as const
    },
    {
      id: 'ai-test-generation',
      title: 'Chapter 6: AI 기반 테스트 자동 생성',
      description: 'AI를 활용한 단위 테스트, 통합 테스트, E2E 테스트 자동 생성. TDD with AI.',
      duration: '4 hours',
      topics: [
        'AI로 단위 테스트 자동 생성',
        'Jest, Pytest, JUnit 테스트 커버리지 100%',
        'Mock 데이터 생성 자동화',
        'E2E 테스트 시나리오 생성',
        'AI TDD (Test-Driven Development)',
        '테스트 리팩토링 및 최적화'
      ],
      level: 'intermediate' as const
    },
    {
      id: 'ai-code-review',
      title: 'Chapter 7: AI 코드 리뷰 시스템 구축',
      description: 'AI를 활용한 자동 코드 리뷰. 보안 취약점, 성능 이슈, 베스트 프랙티스 자동 체크.',
      duration: '4 hours',
      topics: [
        'AI 코드 리뷰어 설정 (GitHub Actions)',
        '보안 취약점 자동 탐지',
        '성능 최적화 제안',
        '코드 스타일 일관성 체크',
        'Pull Request 자동 분석',
        'AI 리뷰어와 인간 리뷰어 협업'
      ],
      level: 'intermediate' as const
    },
    {
      id: 'ai-refactoring',
      title: 'Chapter 8: 대규모 코드베이스 AI 리팩토링',
      description: '레거시 코드를 AI로 현대화. 자동 마이그레이션, 아키텍처 개선, 기술 부채 해소.',
      duration: '5 hours',
      topics: [
        'AI 기반 코드 분석 및 이해',
        '자동 리팩토링 전략 (Extract, Inline, Rename)',
        '레거시 → 모던 프레임워크 마이그레이션',
        'Python 2 → 3, JavaScript → TypeScript',
        'Monolith → Microservices 전환',
        '대규모 리팩토링 검증 및 테스트'
      ],
      level: 'advanced' as const
    },
    {
      id: 'ai-documentation',
      title: 'Chapter 9: AI 자동 문서화 시스템',
      description: 'AI로 코드 주석, README, API 문서, 사용자 가이드 자동 생성. Documentation as Code.',
      duration: '3 hours',
      topics: [
        'Docstring 자동 생성 (JSDoc, Sphinx, Javadoc)',
        'README.md 자동 작성',
        'API 문서 자동 생성 (OpenAPI, Swagger)',
        'Architecture Decision Records (ADR)',
        '다국어 문서 자동 번역',
        'Living Documentation 시스템'
      ],
      level: 'advanced' as const
    },
    {
      id: 'ai-workflow-automation',
      title: 'Chapter 10: AI 개발 워크플로우 자동화',
      description: 'CI/CD, 배포, 모니터링까지 AI로 자동화. GitHub Actions, GitLab CI, Jenkins 통합.',
      duration: '4 hours',
      topics: [
        'AI 기반 CI/CD 파이프라인 생성',
        'Docker, Kubernetes 설정 자동 생성',
        'Infrastructure as Code (Terraform, Ansible)',
        '배포 스크립트 자동화',
        'AI 기반 에러 로그 분석',
        '성능 모니터링 및 알림 자동화'
      ],
      level: 'advanced' as const
    },
    {
      id: 'ai-security-practices',
      title: 'Chapter 11: AI 코딩 보안 및 모범 사례',
      description: 'AI 코드의 보안 검증, 라이선스 체크, 품질 보증. Responsible AI Coding.',
      duration: '4 hours',
      topics: [
        'AI 생성 코드의 보안 취약점 체크',
        'OWASP Top 10 자동 검증',
        '오픈소스 라이선스 충돌 방지',
        'AI 코드 플래지어리즘 검사',
        '민감 정보 자동 탐지 및 제거',
        'AI 코딩 윤리 가이드라인'
      ],
      level: 'advanced' as const
    },
    {
      id: 'real-world-projects',
      title: 'Chapter 12: 실전 프로젝트 - AI로 앱 처음부터 끝까지',
      description: '풀스택 앱을 AI 도구만으로 48시간 안에 개발. 기획부터 배포까지 완전 자동화.',
      duration: '8 hours',
      topics: [
        'AI로 프로젝트 기획 및 요구사항 정의',
        'Next.js + TypeScript 풀스택 앱 생성',
        'Database 스키마 및 API 자동 설계',
        'UI/UX 디자인 to 코드 변환',
        '테스트, 최적화, 보안 검증',
        'Vercel/AWS 자동 배포',
        'AI 개발 워크플로우 체크리스트'
      ],
      level: 'advanced' as const
    }
  ],

  simulators: [
    {
      id: 'ai-code-assistant',
      title: 'AI Code Assistant Playground',
      description: 'Cursor, Copilot, Claude Code 스타일의 AI 코딩 어시스턴트 시뮬레이터. 실시간 코드 생성, 자동완성, 리팩토링 체험.',
      difficulty: 'beginner',
      estimatedTime: '30 min',
      tags: ['AI Coding', 'Code Generation', 'Autocomplete', 'Refactoring']
    },
    {
      id: 'prompt-optimizer',
      title: 'Prompt Optimizer',
      description: 'AI 코딩 프롬프트를 분석하고 최적화하는 도구. 명확성, 컨텍스트, 제약조건 점수 제공.',
      difficulty: 'intermediate',
      estimatedTime: '20 min',
      tags: ['Prompt Engineering', 'Optimization', 'Best Practices']
    },
    {
      id: 'code-review-ai',
      title: 'AI Code Reviewer',
      description: 'AI 기반 자동 코드 리뷰 시뮬레이터. 보안, 성능, 스타일, 베스트 프랙티스 자동 체크.',
      difficulty: 'intermediate',
      estimatedTime: '25 min',
      tags: ['Code Review', 'Security', 'Performance', 'Quality']
    },
    {
      id: 'refactoring-engine',
      title: 'AI Refactoring Engine',
      description: '레거시 코드를 입력하면 AI가 자동으로 현대적인 코드로 리팩토링. 변경 사항 diff 제공.',
      difficulty: 'advanced',
      estimatedTime: '30 min',
      tags: ['Refactoring', 'Legacy Code', 'Modernization']
    },
    {
      id: 'test-generator',
      title: 'AI Test Generator',
      description: '코드를 분석하여 단위 테스트, 통합 테스트를 자동 생성. Jest, Pytest, JUnit 지원.',
      difficulty: 'intermediate',
      estimatedTime: '25 min',
      tags: ['Testing', 'TDD', 'Automation', 'Coverage']
    },
    {
      id: 'doc-generator',
      title: 'AI Documentation Generator',
      description: '코드에서 자동으로 README, API 문서, Docstring 생성. Markdown, HTML 출력.',
      difficulty: 'beginner',
      estimatedTime: '20 min',
      tags: ['Documentation', 'README', 'API Docs', 'Automation']
    }
  ],

  learningObjectives: [
    'AI 코딩 도구 (Cursor, Copilot, Claude) 완벽 마스터',
    '개발 생산성 10배 향상 워크플로우 구축',
    'AI 기반 테스트, 리뷰, 리팩토링 자동화',
    '풀스택 앱을 48시간 안에 개발하는 능력'
  ],

  prerequisites: [
    '기본적인 프로그래밍 경험 (Python, JavaScript, TypeScript 중 하나)',
    'Git 및 GitHub 기본 지식',
    'VSCode 또는 IDE 사용 경험',
    'Terminal/Command Line 기본 이해'
  ],

  tools: [
    'Cursor IDE',
    'GitHub Copilot',
    'Claude Code / Claude.ai',
    'VSCode',
    'Git & GitHub',
    'Docker',
    'Node.js / Python'
  ]
};
