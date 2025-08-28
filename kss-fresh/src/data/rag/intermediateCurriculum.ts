import { CurriculumItem } from './beginnerCurriculum'

export const intermediateCurriculum: CurriculumItem[] = [
  {
    id: 'embeddings-deep-dive',
    title: '1. 임베딩 심화 학습',
    description: '텍스트를 벡터로 변환하는 과정을 깊이 있게 이해합니다',
    topics: [
      '임베딩의 수학적 원리',
      '임베딩 모델 비교 (OpenAI, Sentence Transformers, Multilingual)',
      '벡터 차원과 성능의 관계',
      '도메인 특화 임베딩'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 3: 임베딩과 벡터화',
        url: '/modules/rag/embedding-deep-dive',
        duration: '40분'
      },
      {
        type: 'simulator',
        title: '임베딩 시각화 도구',
        url: '/modules/rag/simulators/embedding-visualizer',
        duration: '30분'
      }
    ],
    quiz: [
      {
        question: '임베딩 차원이 높을수록 항상 좋은가?',
        options: [
          '예, 차원이 높을수록 정확도가 높다',
          '아니오, 과적합과 성능 문제가 발생할 수 있다',
          '상황에 따라 다르다',
          '차원은 중요하지 않다'
        ],
        answer: 1
      }
    ]
  },
  {
    id: 'vector-databases',
    title: '2. 벡터 데이터베이스 마스터',
    description: '효율적인 벡터 저장과 검색을 위한 데이터베이스 활용법',
    topics: [
      'Pinecone, Weaviate, Chroma 비교',
      'HNSW, IVF 인덱싱 알고리즘',
      '메타데이터 필터링',
      '스케일링과 성능 최적화'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 4: 벡터 검색과 데이터베이스',
        url: '/modules/rag/vector-search',
        duration: '50분'
      },
      {
        type: 'simulator',
        title: '벡터 검색 데모',
        url: '/modules/rag/simulators/vector-search',
        duration: '25분'
      }
    ]
  },
  {
    id: 'hybrid-search',
    title: '3. 하이브리드 검색 구현',
    description: '벡터 검색과 키워드 검색을 결합한 고급 검색 기법',
    topics: [
      'BM25와 벡터 검색 결합',
      '가중치 조정 전략',
      '검색 결과 재순위화',
      'Query expansion 기법'
    ],
    resources: [
      {
        type: 'simulator',
        title: '하이브리드 검색 비교 실습',
        url: '/modules/rag/simulators/vector-search',
        duration: '35분'
      },
      {
        type: 'external',
        title: 'Elasticsearch와 벡터 검색',
        url: 'https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html',
        duration: '25분'
      }
    ]
  },
  {
    id: 'prompt-engineering',
    title: '4. RAG 프롬프트 엔지니어링',
    description: '검색 결과를 효과적으로 활용하는 프롬프트 설계',
    topics: [
      'Context window 관리',
      '소스 인용 포맷',
      'Few-shot prompting with RAG',
      'Chain-of-thought와 RAG 결합'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 5: 답변 생성과 프롬프트 엔지니어링',
        url: '/modules/rag/retrieval-methods',
        duration: '35분'
      }
    ]
  },
  {
    id: 'performance-optimization',
    title: '5. 성능 최적화',
    description: 'RAG 시스템의 속도와 정확도를 향상시키는 기법',
    topics: [
      '캐싱 전략',
      '배치 처리 최적화',
      '임베딩 사전 계산',
      '비동기 처리 패턴'
    ],
    resources: [
      {
        type: 'simulator',
        title: 'RAG 플레이그라운드 - 성능 테스트',
        url: '/modules/rag/simulators/rag-playground',
        duration: '40분'
      }
    ]
  }
]

export const intermediateChecklist = [
  '다양한 임베딩 모델의 특징과 선택 기준 이해',
  '벡터 데이터베이스 선택과 구성 능력',
  '하이브리드 검색 구현 경험',
  'RAG 특화 프롬프트 엔지니어링 숙달',
  '성능 병목 지점 파악과 최적화 능력'
]