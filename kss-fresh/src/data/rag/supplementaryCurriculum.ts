import { CurriculumItem } from './beginnerCurriculum'

export const supplementaryCurriculum: CurriculumItem[] = [
  {
    id: 'rag-for-korean',
    title: '한국어 특화 RAG',
    description: '한국어 문서 처리를 위한 특별한 고려사항',
    topics: [
      '한국어 토크나이저 선택',
      '형태소 분석과 청킹',
      '한영 혼용 문서 처리',
      '한국어 임베딩 모델 비교'
    ],
    resources: [
      {
        type: 'external',
        title: 'KoBART와 KoGPT 활용',
        url: 'https://github.com/SKT-AI/KoBART',
        duration: '30분'
      },
      {
        type: 'simulator',
        title: '한국어 문서 처리 실습',
        url: '/modules/rag/simulators/document-uploader',
        duration: '25분'
      }
    ]
  },
  {
    id: 'multimodal-rag',
    title: 'Multimodal RAG',
    description: '텍스트를 넘어선 이미지, 표, 차트 처리',
    topics: [
      '이미지 캡션과 OCR',
      '표 데이터 구조 보존',
      'Layout-aware 청킹',
      'Cross-modal 검색'
    ],
    resources: [
      {
        type: 'external',
        title: 'LayoutLM 모델 이해',
        url: 'https://huggingface.co/microsoft/layoutlm-base-uncased',
        duration: '35분'
      }
    ]
  },
  {
    id: 'domain-specific-rag',
    title: '도메인 특화 RAG',
    description: '특정 분야를 위한 맞춤형 RAG 구축',
    topics: [
      '의료 분야 RAG (HIPAA 준수)',
      '법률 문서 RAG',
      '금융 보고서 RAG',
      '학술 논문 RAG'
    ],
    resources: [
      {
        type: 'external',
        title: 'BioBERT와 의료 RAG',
        url: 'https://github.com/dmis-lab/biobert',
        duration: '40분'
      }
    ]
  },
  {
    id: 'privacy-security',
    title: 'RAG 보안과 프라이버시',
    description: '민감한 데이터를 다루는 RAG 시스템',
    topics: [
      'PII 감지와 마스킹',
      'Access control in RAG',
      'Differential privacy',
      'Secure enclaves'
    ],
    resources: [
      {
        type: 'external',
        title: 'PrivateGPT 프로젝트',
        url: 'https://github.com/imartinez/privateGPT',
        duration: '30분'
      }
    ]
  },
  {
    id: 'rag-alternatives',
    title: 'RAG 대안 기술',
    description: 'RAG와 함께 고려할 수 있는 다른 접근법',
    topics: [
      'Fine-tuning vs RAG',
      'Prompt tuning',
      'Adapter modules',
      'Knowledge distillation'
    ],
    resources: [
      {
        type: 'external',
        title: 'PEFT 라이브러리 튜토리얼',
        url: 'https://github.com/huggingface/peft',
        duration: '35분'
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