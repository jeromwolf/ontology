import { CurriculumItem } from './beginnerCurriculum'

export const supplementaryCurriculum: CurriculumItem[] = [
  {
    id: 'ragas-evaluation',
    title: '1. RAGAS 평가 프레임워크',
    description: 'RAG 시스템의 정확성과 품질을 측정하는 체계적 방법',
    topics: [
      'Context relevancy 측정',
      'Answer faithfulness 평가',
      'Answer relevancy 검증',
      '자동화된 평가 파이프라인'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 1: RAGAS 평가 프레임워크',
        url: '/modules/rag/supplementary/chapter1',
        duration: '45분'
      },
      {
        type: 'external',
        title: 'RAGAS GitHub',
        url: 'https://github.com/explodinggradients/ragas',
        duration: '30분'
      }
    ]
  },
  {
    id: 'security-privacy',
    title: '2. 보안과 프라이버시',
    description: 'RAG 시스템의 데이터 보안과 사용자 프라이버시 보호',
    topics: [
      'PII 감지와 마스킹',
      'Prompt injection 방어',
      'Data access control',
      'GDPR 준수 방법'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 2: 보안과 프라이버시',
        url: '/modules/rag/supplementary/chapter2',
        duration: '50분'
      },
      {
        type: 'external',
        title: 'PrivateGPT 프로젝트',
        url: 'https://github.com/imartinez/privateGPT',
        duration: '35분'
      }
    ]
  },
  {
    id: 'cost-optimization',
    title: '3. 비용 최적화',
    description: 'RAG 시스템 운영 비용을 80% 절감하는 전략',
    topics: [
      'Token usage 최적화',
      '캐싱 전략으로 API 호출 감소',
      '모델 선택과 비용 효율성',
      'Batch processing 기법'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 3: 비용 최적화',
        url: '/modules/rag/supplementary/chapter3',
        duration: '40분'
      },
      {
        type: 'external',
        title: '비용 최적화 사례연구',
        url: 'https://www.pinecone.io/learn/cost-optimization/',
        duration: '30분'
      }
    ]
  },
  {
    id: 'high-availability',
    title: '4. 고가용성과 복구 시스템',
    description: '99.9% 가동률을 위한 RAG 시스템 안정성 확보',
    topics: [
      'Failover 전략',
      '백업과 복구 절차',
      'Circuit breaker 패턴',
      '재해 복구 계획'
    ],
    resources: [
      {
        type: 'chapter',
        title: 'Chapter 4: 고가용성과 복구 시스템',
        url: '/modules/rag/supplementary/chapter4',
        duration: '45분'
      },
      {
        type: 'external',
        title: 'AWS Well-Architected Framework',
        url: 'https://aws.amazon.com/architecture/well-architected/',
        duration: '30분'
      }
    ]
  }
]

export const supplementaryResources = {
  books: [
    {
      title: 'Building LLM Apps with LangChain',
      author: 'Ben Auffarth',
      year: 2023,
      topics: ['LangChain', 'RAG', 'Agent']
    }
  ],
  papers: [
    {
      title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
      authors: 'Lewis et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2005.11401'
    },
    {
      title: 'REALM: Retrieval-Augmented Language Model Pre-Training',
      authors: 'Guu et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2002.08909'
    }
  ],
  tools: [
    {
      name: 'LangChain',
      description: 'RAG 파이프라인 구축 프레임워크',
      url: 'https://langchain.com/'
    },
    {
      name: 'LlamaIndex',
      description: 'Data framework for LLM applications',
      url: 'https://www.llamaindex.ai/'
    },
    {
      name: 'Haystack',
      description: 'Open source NLP framework',
      url: 'https://haystack.deepset.ai/'
    }
  ],
  communities: [
    {
      name: 'LangChain Discord',
      url: 'https://discord.gg/langchain',
      members: '50K+'
    },
    {
      name: 'r/LocalLLaMA',
      url: 'https://www.reddit.com/r/LocalLLaMA/',
      members: '200K+'
    }
  ]
}