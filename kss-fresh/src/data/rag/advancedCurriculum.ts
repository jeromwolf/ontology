import { CurriculumItem } from './beginnerCurriculum'

export const advancedCurriculum: CurriculumItem[] = [
  {
    id: 'graphrag-architecture',
    title: '1. GraphRAG 아키텍처',
    description: '지식 그래프 기반 RAG의 고급 구현',
    topics: [
      'Entity와 Relation 추출',
      'Knowledge Graph 구축',
      'Graph traversal과 reasoning',
      'Neo4j와 GraphRAG 통합'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 6: 고급 RAG 기법',
        url: '/modules/rag/advanced-architectures',
        duration: '60분'
      },
      {
        type: 'simulator',
        title: 'GraphRAG Explorer',
        url: '/modules/rag/simulators/graphrag-explorer',
        duration: '45분'
      }
    ]
  },
  {
    id: 'multi-hop-reasoning',
    title: '2. Multi-hop Reasoning',
    description: '복잡한 질문에 대한 다단계 추론 구현',
    topics: [
      'Query decomposition',
      'Iterative retrieval',
      'Context accumulation',
      'Answer synthesis'
    ],
    resources: [
      {
        type: 'simulator',
        title: 'Multi-hop RAG 실습',
        url: '/modules/rag/simulators/rag-playground',
        duration: '50분'
      },
      {
        type: 'external',
        title: 'HotpotQA Dataset 분석',
        url: 'https://hotpotqa.github.io/',
        duration: '30분'
      }
    ]
  },
  {
    id: 'reranking-strategies',
    title: '3. 고급 Reranking 전략',
    description: '검색 결과의 품질을 극대화하는 재순위화 기법',
    topics: [
      'Cross-encoder reranking',
      'Diversity-aware reranking',
      'Learning to rank',
      'Contextual reranking'
    ],
    resources: [
      {
        type: 'external',
        title: 'ColBERT v2 논문 리뷰',
        url: 'https://arxiv.org/abs/2112.01488',
        duration: '40분'
      }
    ]
  },
  {
    id: 'evaluation-metrics',
    title: '4. RAG 평가와 모니터링',
    description: '프로덕션 RAG 시스템의 품질 측정과 개선',
    topics: [
      'Relevance metrics (MRR, NDCG)',
      'Answer quality metrics',
      'A/B 테스팅 전략',
      '실시간 모니터링 대시보드'
    ],
    resources: [
      {
        type: 'external',
        title: 'RAGAS 평가 프레임워크',
        url: 'https://github.com/explodinggradients/ragas',
        duration: '35분'
      }
    ]
  },
  {
    id: 'production-deployment',
    title: '5. 프로덕션 배포와 스케일링',
    description: '대규모 RAG 시스템 구축과 운영',
    topics: [
      '마이크로서비스 아키텍처',
      'Load balancing과 sharding',
      'Streaming과 실시간 업데이트',
      'Cost optimization'
    ],
    resources: [
      {
        type: 'simulator',
        title: '전체 RAG 파이프라인 구축',
        url: '/modules/rag/simulators/rag-playground',
        duration: '60분'
      }
    ]
  },
  {
    id: 'cutting-edge-research',
    title: '6. 최신 연구 동향',
    description: 'RAG 분야의 최신 논문과 기술 트렌드',
    topics: [
      'Self-RAG와 Adaptive RAG',
      'Multimodal RAG',
      'Long-context RAG',
      'RAG with reasoning chains'
    ],
    resources: [
      {
        type: 'external',
        title: 'Self-RAG 논문',
        url: 'https://arxiv.org/abs/2310.11511',
        duration: '45분'
      },
      {
        type: 'external',
        title: 'RAPTOR 논문',
        url: 'https://arxiv.org/abs/2401.18059',
        duration: '40분'
      }
    ]
  }
]

export const advancedChecklist = [
  'GraphRAG 아키텍처 설계와 구현',
  'Multi-hop reasoning 시스템 구축',
  '고급 reranking 알고리즘 적용',
  'RAG 시스템 평가 메트릭 설계',
  '프로덕션 수준의 RAG 시스템 배포 경험',
  '최신 RAG 연구 논문 이해와 적용'
]