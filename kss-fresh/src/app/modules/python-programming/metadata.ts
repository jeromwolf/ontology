export const moduleMetadata = {
  id: 'python-programming',
  title: 'Python 프로그래밍',
  description: 'Python 기초부터 고급까지 실전 중심으로 완전 정복',
  duration: '10시간',
  level: 'beginner' as const,
  category: 'programming',
  tags: ['Python', '프로그래밍', '데이터 구조', '객체지향', '비동기'],
  chapters: [
    {
      id: 1,
      slug: 'python-basics',
      title: 'Python 기초와 문법',
      duration: '45분',
      difficulty: 'beginner' as const,
      learningObjectives: [
        'Python 설치와 개발 환경 설정하기',
        '기본 문법과 데이터 타입 이해하기',
        '변수와 연산자 완전 정복하기',
        'Python REPL 활용하여 실습하기'
      ]
    },
    {
      id: 2,
      slug: 'data-types-collections',
      title: '데이터 타입과 컬렉션',
      duration: '1시간',
      difficulty: 'beginner' as const,
      learningObjectives: [
        '리스트, 튜플, 집합 완전 정복하기',
        '딕셔너리로 데이터 관리하기',
        '타입 변환의 원리와 실전 활용',
        '컬렉션 연산과 메서드 마스터하기'
      ]
    },
    {
      id: 3,
      slug: 'functions-modules',
      title: '함수와 모듈',
      duration: '1시간',
      difficulty: 'beginner' as const,
      learningObjectives: [
        '함수 정의와 호출 완벽 이해하기',
        '매개변수와 인자의 모든 것',
        '반환값을 활용한 함수 설계',
        '모듈 import로 코드 재사용하기'
      ]
    },
    {
      id: 4,
      slug: 'file-io',
      title: '파일 입출력과 데이터 처리',
      duration: '45분',
      difficulty: 'beginner' as const,
      learningObjectives: [
        '텍스트 파일 읽기와 쓰기',
        'CSV와 JSON 데이터 다루기',
        '파일 예외 처리 완벽 대응',
        '실전 데이터 파일 처리 기법'
      ]
    },
    {
      id: 5,
      slug: 'oop-basics',
      title: '객체지향 프로그래밍',
      duration: '1.5시간',
      difficulty: 'intermediate' as const,
      learningObjectives: [
        '클래스와 객체의 개념 완전 이해',
        '상속과 다형성으로 코드 확장하기',
        '캡슐화 원칙과 실전 적용',
        '클래스 메서드와 정적 메서드 활용'
      ]
    },
    {
      id: 6,
      slug: 'exception-handling',
      title: '예외 처리',
      duration: '45분',
      difficulty: 'intermediate' as const,
      learningObjectives: [
        'try-except로 안전한 코드 작성',
        '커스텀 예외 클래스 만들기',
        'finally와 else 절 완벽 활용',
        '에러 로깅과 디버깅 전략'
      ]
    },
    {
      id: 7,
      slug: 'standard-library',
      title: 'Python 표준 라이브러리',
      duration: '1시간',
      difficulty: 'intermediate' as const,
      learningObjectives: [
        'datetime으로 날짜와 시간 다루기',
        'collections 모듈의 강력한 자료구조',
        'itertools와 functools 실전 활용',
        'os와 sys 모듈로 시스템 제어하기'
      ]
    },
    {
      id: 8,
      slug: 'decorators-generators',
      title: '데코레이터와 제너레이터',
      duration: '1시간',
      difficulty: 'advanced' as const,
      learningObjectives: [
        '함수 데코레이터의 원리 이해하기',
        '커스텀 데코레이터 작성 마스터',
        '제너레이터 함수로 메모리 효율 향상',
        'yield와 이터레이터 완전 정복'
      ]
    },
    {
      id: 9,
      slug: 'async-programming',
      title: '비동기 프로그래밍',
      duration: '1.5시간',
      difficulty: 'advanced' as const,
      learningObjectives: [
        'async/await 문법 완벽 이해',
        'asyncio로 비동기 작업 다루기',
        '동시성 작업 처리와 성능 최적화',
        '실전 비동기 패턴 구현하기'
      ]
    },
    {
      id: 10,
      slug: 'best-practices',
      title: '모범 사례와 실전 배포',
      duration: '1시간',
      difficulty: 'advanced' as const,
      learningObjectives: [
        'PEP 8 스타일 가이드 준수하기',
        '가상 환경으로 프로젝트 관리',
        'pip와 패키지 관리 완전 정복',
        '프로덕션 레벨 코드 작성 기법'
      ]
    }
  ],
  simulators: [
    {
      id: 'python-repl',
      title: 'Python REPL 시뮬레이터',
      description: '실시간 코드 실행이 가능한 인터랙티브 파이썬 인터프리터',
      difficulty: 'beginner' as const
    },
    {
      id: 'data-type-converter',
      title: '데이터 타입 변환기',
      description: '타입 변환 과정을 시각적으로 이해하는 도구',
      difficulty: 'beginner' as const
    },
    {
      id: 'collection-visualizer',
      title: '컬렉션 시각화 도구',
      description: '리스트, 딕셔너리, 집합 연산을 인터랙티브하게 실습',
      difficulty: 'beginner' as const
    },
    {
      id: 'function-tracer',
      title: '함수 실행 추적기',
      description: '함수 호출과 실행 과정을 단계별로 시각화',
      difficulty: 'intermediate' as const
    },
    {
      id: 'oop-diagram-generator',
      title: 'OOP 클래스 다이어그램 생성기',
      description: 'Python 코드에서 UML 클래스 다이어그램 자동 생성',
      difficulty: 'intermediate' as const
    },
    {
      id: 'exception-simulator',
      title: '예외 처리 시뮬레이터',
      description: '다양한 예외 상황을 실습하며 처리 방법 학습',
      difficulty: 'intermediate' as const
    },
    {
      id: 'file-io-playground',
      title: '파일 입출력 플레이그라운드',
      description: '안전한 환경에서 파일 연산 실습하기',
      difficulty: 'beginner' as const
    },
    {
      id: 'coding-challenges',
      title: 'Python 코딩 챌린지',
      description: '50개 이상의 인터랙티브 코딩 문제',
      difficulty: 'advanced' as const
    }
  ],
  prerequisites: [],
  nextSteps: ['데이터 사이언스', 'AI 자동화', 'RAG 시스템'],
  instructor: 'KSS 플랫폼',
  lastUpdated: '2025-01-10'
};
