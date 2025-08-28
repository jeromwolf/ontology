export const intermediateCurriculum = {
  title: "RAG 중급 과정",
  description: "고급 벡터 데이터베이스, 하이브리드 검색, 멀티모달 RAG 등 실무에 필요한 심화 기술을 학습합니다",
  duration: "6주 과정",
  level: "중급",
  prerequisites: ["RAG 입문 과정 수료", "Python 중급 이상", "ML/DL 기초 지식"],
  
  modules: [
    {
      week: 1,
      title: "고급 벡터 데이터베이스",
      description: "엔터프라이즈급 벡터 DB 구축과 최적화",
      topics: [
        "Pinecone, Weaviate, Qdrant 심화",
        "벡터 인덱싱 알고리즘 (HNSW, IVF)",
        "분산 벡터 데이터베이스 아키텍처",
        "메타데이터 필터링과 하이브리드 쿼리"
      ],
      chapters: [{
        title: "Chapter 1: 고급 벡터 데이터베이스",
        url: "/modules/rag/intermediate/chapter1"
      }],
      assignments: [
        "대규모 벡터 DB 성능 벤치마킹",
        "커스텀 인덱싱 전략 구현"
      ]
    },
    {
      week: 2,
      title: "하이브리드 검색 전략",
      description: "의미 검색과 키워드 검색의 최적 조합",
      topics: [
        "BM25 + 벡터 검색 융합",
        "Re-ranking 알고리즘",
        "Query expansion 기법",
        "Cross-encoder 활용법"
      ],
      chapters: [{
        title: "Chapter 2: 하이브리드 검색 전략",
        url: "/modules/rag/intermediate/chapter2"
      }],
      simulators: ["hybrid-search"],
      assignments: [
        "하이브리드 검색 파이프라인 구축",
        "도메인별 최적 가중치 찾기"
      ]
    },
    {
      week: 3,
      title: "RAG를 위한 프롬프트 엔지니어링",
      description: "효과적인 컨텍스트 활용과 응답 생성",
      topics: [
        "Few-shot prompting for RAG",
        "Chain-of-thought in retrieval",
        "컨텍스트 윈도우 최적화",
        "프롬프트 템플릿 설계"
      ],
      chapters: [{
        title: "Chapter 3: RAG를 위한 프롬프트 엔지니어링",
        url: "/modules/rag/intermediate/chapter3"
      }],
      simulators: ["prompt-engineering"],
      assignments: [
        "도메인별 프롬프트 템플릿 개발",
        "프롬프트 A/B 테스팅"
      ]
    },
    {
      week: 4,
      title: "RAG 성능 최적화",
      description: "속도, 정확도, 비용의 균형 맞추기",
      topics: [
        "캐싱 전략과 메모리 관리",
        "배치 처리와 비동기 검색",
        "모델 양자화와 경량화",
        "엣지 디바이스 RAG"
      ],
      chapters: [{
        title: "Chapter 4: RAG 성능 최적화",
        url: "/modules/rag/intermediate/chapter4"
      }],
      simulators: ["performance-optimizer"],
      assignments: [
        "RAG 파이프라인 프로파일링",
        "최적화 전략 구현 및 비교"
      ]
    },
    {
      week: 5,
      title: "멀티모달 RAG",
      description: "텍스트를 넘어선 다양한 데이터 타입 처리",
      topics: [
        "이미지-텍스트 RAG (CLIP 활용)",
        "비디오 검색과 요약",
        "오디오/음성 데이터 처리",
        "테이블과 구조화 데이터 RAG"
      ],
      chapters: [{
        title: "Chapter 5: 멀티모달 RAG",
        url: "/modules/rag/intermediate/chapter5"
      }],
      simulators: ["multimodal-rag"],
      assignments: [
        "멀티모달 검색 시스템 구축",
        "크로스 모달 검색 구현"
      ]
    },
    {
      week: 6,
      title: "프로덕션 RAG 시스템",
      description: "실제 서비스를 위한 RAG 구축",
      topics: [
        "RAG 시스템 모니터링",
        "A/B 테스팅과 실험 관리",
        "보안과 프라이버시",
        "스케일링 전략"
      ],
      chapters: [{
        title: "Chapter 6: 프로덕션 RAG 시스템",
        url: "/modules/rag/intermediate/chapter6"
      }],
      simulators: ["rag-builder"],
      assignments: [
        "프로덕션 레디 RAG API 구축",
        "모니터링 대시보드 개발"
      ]
    }
  ],
  
  learningOutcomes: [
    "엔터프라이즈급 벡터 데이터베이스 구축 및 운영",
    "하이브리드 검색 시스템 설계 및 구현",
    "RAG 성능 최적화 및 비용 관리",
    "멀티모달 데이터를 활용한 고급 RAG 구현",
    "프로덕션 환경에서의 RAG 시스템 운영"
  ],
  
  projectIdeas: [
    {
      title: "지능형 고객 지원 시스템",
      description: "멀티모달 RAG를 활용한 실시간 고객 상담 봇",
      difficulty: "중급",
      estimatedTime: "3주"
    },
    {
      title: "기술 문서 검색 엔진",
      description: "하이브리드 검색을 활용한 개발자 문서 시스템",
      difficulty: "중급",
      estimatedTime: "2주"
    },
    {
      title: "비디오 콘텐츠 Q&A 시스템",
      description: "YouTube 강의 영상 기반 질의응답 서비스",
      difficulty: "고급",
      estimatedTime: "4주"
    }
  ],
  
  resources: [
    {
      type: "도서",
      title: "Advanced Information Retrieval",
      author: "Christopher Manning",
      link: "https://nlp.stanford.edu/IR-book/"
    },
    {
      type: "논문",
      title: "Dense Passage Retrieval for Open-Domain Question Answering",
      author: "Vladimir Karpukhin et al.",
      link: "https://arxiv.org/abs/2004.04906"
    },
    {
      type: "강의",
      title: "CS224U: Natural Language Understanding",
      author: "Stanford University",
      link: "https://web.stanford.edu/class/cs224u/"
    },
    {
      type: "블로그",
      title: "Pinecone Learning Center",
      author: "Pinecone",
      link: "https://www.pinecone.io/learn/"
    }
  ],
  
  tools: [
    {
      name: "LangChain",
      description: "고급 RAG 파이프라인 구축",
      category: "Framework"
    },
    {
      name: "Qdrant",
      description: "고성능 벡터 데이터베이스",
      category: "Vector DB"
    },
    {
      name: "Weights & Biases",
      description: "실험 추적 및 모니터링",
      category: "MLOps"
    },
    {
      name: "Ray",
      description: "분산 처리 및 스케일링",
      category: "Infrastructure"
    }
  ]
}