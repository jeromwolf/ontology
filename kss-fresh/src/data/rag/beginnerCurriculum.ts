export interface CurriculumItem {
  id: string
  title: string
  description: string
  topics: string[]
  resources: {
    type: 'chapter' | 'simulator' | 'external'
    title: string
    url?: string
    duration?: string
  }[]
  quiz?: {
    question: string
    options: string[]
    answer: number
  }[]
}

export const beginnerCurriculum: CurriculumItem[] = [
  {
    id: 'intro-to-rag',
    title: '1. RAG 기초 개념',
    description: 'RAG가 왜 필요한지, 어떤 문제를 해결하는지 이해합니다',
    topics: [
      'LLM의 한계점 (할루시네이션, 최신 정보 부족)',
      'RAG의 정의와 핵심 원리',
      '전통적인 검색 vs RAG',
      'RAG의 실제 활용 사례'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 1: LLM의 한계점 이해하기',
        url: '/modules/rag/beginner/chapter1',
        duration: '30분'
      },
      {
        type: 'simulator',
        title: 'RAG 플레이그라운드 - 기본 체험',
        url: '/modules/rag/simulators/rag-playground',
        duration: '20분'
      }
    ],
    quiz: [
      {
        question: 'RAG가 해결하는 LLM의 주요 문제는?',
        options: [
          '느린 응답 속도',
          '할루시네이션과 최신 정보 부족',
          '높은 비용',
          '복잡한 설정'
        ],
        answer: 1
      }
    ]
  },
  {
    id: 'document-basics',
    title: '2. 문서 처리 기초',
    description: '다양한 문서를 AI가 이해할 수 있는 형태로 변환하는 과정을 배웁니다',
    topics: [
      '지원되는 문서 형식 (PDF, Word, HTML, TXT)',
      '문서 파싱의 중요성',
      '메타데이터 추출',
      '텍스트 정제와 전처리'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 2: 문서 처리와 청킹',
        url: '/modules/rag/beginner/chapter2',
        duration: '45분'
      },
      {
        type: 'simulator',
        title: '문서 업로더 시뮬레이터',
        url: '/modules/rag/simulators/document-uploader',
        duration: '15분'
      }
    ]
  },
  {
    id: 'chunking-strategies',
    title: '3. 청킹 전략 이해',
    description: '문서를 효과적으로 분할하는 다양한 방법을 학습합니다',
    topics: [
      '고정 크기 청킹',
      '의미 단위 청킹',
      '중첩(Overlap) 청킹',
      '청크 크기 최적화'
    ],
    resources: [
      {
        type: 'simulator',
        title: '청킹 데모 - 5가지 전략 비교',
        url: '/modules/rag/simulators/chunking-demo',
        duration: '30분'
      },
      {
        type: 'external',
        title: 'LangChain 청킹 가이드',
        url: 'https://python.langchain.com/docs/modules/data_connection/document_transformers/',
        duration: '20분'
      }
    ]
  },
  {
    id: 'first-rag-system',
    title: '4. 첫 RAG 시스템 만들기',
    description: '학습한 내용을 종합하여 간단한 RAG 시스템을 구축합니다',
    topics: [
      '로컬 문서 업로드',
      '간단한 청킹 적용',
      '기본 검색 구현',
      '결과 확인 및 개선'
    ],
    resources: [
      {
        type: 'simulator',
        title: 'RAG 플레이그라운드 - 전체 파이프라인',
        url: '/modules/rag/simulators/rag-playground',
        duration: '45분'
      }
    ]
  }
]

export const beginnerChecklist = [
  'LLM의 한계점과 RAG의 필요성 이해',
  '다양한 문서 형식 처리 방법 학습',
  '기본적인 청킹 전략 이해와 적용',
  '간단한 RAG 시스템 구축 경험',
  '실제 문서로 테스트 수행'
]