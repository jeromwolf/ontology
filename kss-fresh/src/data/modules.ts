export interface Module {
  id: string;
  title: string;
  description: string;
  duration: string;
  status: '학습 가능' | '개발중' | '준비중';
  icon: string;
  gradient: string;
  href: string;
}

export interface ModuleCategory {
  id: string;
  title: string;
  description: string;
  icon: string;
  modules: Module[];
}

export const moduleCategories: ModuleCategory[] = [
  {
    id: 'ai-ml',
    title: 'AI & Machine Learning',
    description: 'Core AI technologies and deep learning fundamentals',
    icon: '🧠',
    modules: [
      {
        id: 'llm',
        title: 'Large Language Models',
        description: 'Transformer, GPT, Claude 등 최신 LLM 기술 완전 정복',
        duration: '6주 과정',
        status: '학습 가능',
        icon: '🧠',
        gradient: 'from-indigo-500 to-purple-600',
        href: '/modules/llm'
      },
      {
        id: 'deep-learning',
        title: 'Deep Learning',
        description: '신경망 기초부터 CNN, Transformer, GAN까지 완전 정복',
        duration: '25시간',
        status: '학습 가능',
        icon: '🧠',
        gradient: 'from-violet-500 to-purple-600',
        href: '/modules/deep-learning'
      },
      {
        id: 'computer-vision',
        title: 'Computer Vision',
        description: '이미지 인식부터 3D 비전까지 컴퓨터 비전 전문가 과정',
        duration: '14시간',
        status: '학습 가능',
        icon: '👁️',
        gradient: 'from-teal-500 to-cyan-600',
        href: '/modules/computer-vision'
      },
      {
        id: 'rag',
        title: 'RAG Systems',
        description: 'Retrieval-Augmented Generation 시스템 설계와 구현',
        duration: '12시간',
        status: '학습 가능',
        icon: '🗃️',
        gradient: 'from-emerald-500 to-green-600',
        href: '/modules/rag'
      },
      {
        id: 'multi-agent',
        title: 'Multi-Agent Systems',
        description: '복잡한 협력 AI 시스템과 분산 지능 아키텍처',
        duration: '10시간',
        status: '학습 가능',
        icon: '🧩',
        gradient: 'from-blue-600 to-indigo-700',
        href: '/modules/multi-agent'
      },
      {
        id: 'agent-mcp',
        title: 'AI Agent & MCP',
        description: 'Model Context Protocol과 멀티 에이전트 시스템 구축',
        duration: '12시간',
        status: '학습 가능',
        icon: '🤝',
        gradient: 'from-emerald-600 to-teal-600',
        href: '/modules/agent-mcp'
      }
    ]
  },
  {
    id: 'programming',
    title: 'Programming & Development',
    description: 'Essential programming languages and development tools',
    icon: '💻',
    modules: [
      {
        id: 'python-programming',
        title: 'Python Programming',
        description: 'Python 기초부터 고급까지 - 10개 챕터 + 8개 시뮬레이터',
        duration: '10시간',
        status: '학습 가능',
        icon: '🐍',
        gradient: 'from-blue-500 to-indigo-600',
        href: '/modules/python-programming'
      },
      {
        id: 'ai-automation',
        title: 'AI Automation',
        description: 'AI 기반 업무 자동화와 워크플로우 최적화',
        duration: '10시간',
        status: '학습 가능',
        icon: '⚙️',
        gradient: 'from-green-600 to-emerald-700',
        href: '/modules/ai-automation'
      }
    ]
  },
  {
    id: 'engineering',
    title: 'Engineering & Systems',
    description: 'Industrial engineering and system architecture',
    icon: '🏗️',
    modules: [
      {
        id: 'system-design',
        title: 'System Design',
        description: '대규모 분산 시스템 설계의 핵심 원칙과 실전 패턴 학습',
        duration: '20시간',
        status: '학습 가능',
        icon: '🏗️',
        gradient: 'from-purple-500 to-indigo-600',
        href: '/modules/system-design'
      },
      {
        id: 'smart-factory',
        title: 'Smart Factory',
        description: 'Industry 4.0 기반 스마트 팩토리 구축과 운영',
        duration: '14시간',
        status: '학습 가능',
        icon: '🏭',
        gradient: 'from-orange-500 to-amber-600',
        href: '/modules/smart-factory'
      },
      {
        id: 'semiconductor',
        title: '반도체',
        description: '반도체 설계부터 제조까지 - 칩의 모든 것',
        duration: '40시간',
        status: '학습 가능',
        icon: '💎',
        gradient: 'from-blue-500 to-indigo-600',
        href: '/modules/semiconductor'
      },
      {
        id: 'autonomous-mobility',
        title: '자율주행 & 미래 모빌리티',
        description: 'AI 기반 자율주행 기술과 차세대 모빌리티 생태계 완전 정복',
        duration: '16시간',
        status: '학습 가능',
        icon: '🚗',
        gradient: 'from-cyan-500 to-blue-600',
        href: '/modules/autonomous-mobility'
      },
      {
        id: 'physical-ai',
        title: 'Physical AI & 실세계 지능',
        description: '현실 세계와 상호작용하는 AI 시스템의 설계와 구현',
        duration: '20시간',
        status: '학습 가능',
        icon: '🤖',
        gradient: 'from-slate-600 to-gray-700',
        href: '/modules/physical-ai'
      }
    ]
  },
  {
    id: 'data-analytics',
    title: 'Data & Analytics',
    description: 'Data science, statistics, and financial analysis',
    icon: '📊',
    modules: [
      {
        id: 'stock-analysis',
        title: '주식투자분석 시뮬레이터',
        description: '실전 투자 전략과 심리까지 포함한 종합 투자 마스터 과정',
        duration: '16주 과정',
        status: '학습 가능',
        icon: '📈',
        gradient: 'from-red-500 to-orange-500',
        href: '/modules/stock-analysis'
      },
      {
        id: 'probability-statistics',
        title: 'Probability & Statistics',
        description: 'AI의 수학적 기초인 확률론과 통계학 완전 정복',
        duration: '20시간',
        status: '학습 가능',
        icon: '📊',
        gradient: 'from-indigo-600 to-purple-700',
        href: '/modules/probability-statistics'
      },
      {
        id: 'linear-algebra',
        title: 'Linear Algebra',
        description: '머신러닝의 핵심 수학인 선형대수학 집중 과정',
        duration: '15시간',
        status: '학습 가능',
        icon: '📐',
        gradient: 'from-purple-600 to-pink-700',
        href: '/linear-algebra'
      }
    ]
  },
  {
    id: 'knowledge-graphs',
    title: 'Knowledge & Semantics',
    description: 'Ontology, knowledge graphs, and semantic technologies',
    icon: '🔗',
    modules: [
      {
        id: 'ontology',
        title: 'Ontology & Semantic Web',
        description: 'RDF, SPARQL, 지식 그래프를 통한 시맨틱 웹 기술 마스터',
        duration: '8주 과정',
        status: '학습 가능',
        icon: '🔗',
        gradient: 'from-purple-500 to-pink-500',
        href: '/modules/ontology'
      },
      {
        id: 'neo4j',
        title: 'Neo4j Knowledge Graph',
        description: '그래프 데이터베이스와 Cypher를 활용한 지식 그래프 구축',
        duration: '12시간',
        status: '학습 가능',
        icon: '🌐',
        gradient: 'from-blue-600 to-indigo-600',
        href: '/modules/neo4j'
      }
    ]
  },
  {
    id: 'web-security',
    title: 'Web3 & Security',
    description: 'Blockchain, cybersecurity, and AI security',
    icon: '🔐',
    modules: [
      {
        id: 'web3',
        title: 'Web3 & Blockchain',
        description: '블록체인 기술부터 DeFi, NFT까지 Web3 생태계 완전 분석',
        duration: '16시간',
        status: '학습 가능',
        icon: '⛓️',
        gradient: 'from-amber-500 to-orange-600',
        href: '/modules/web3'
      },
      {
        id: 'ai-security',
        title: 'AI Security',
        description: 'AI 시스템의 보안 위협과 방어 기법 학습',
        duration: '18시간',
        status: '학습 가능',
        icon: '🛡️',
        gradient: 'from-red-600 to-gray-700',
        href: '/modules/ai-security'
      },
      {
        id: 'cyber-security',
        title: 'Cyber Security',
        description: '해킹 시뮬레이션과 제로트러스트 보안 모델 실습',
        duration: '24시간',
        status: '개발중',
        icon: '🔒',
        gradient: 'from-red-600 to-orange-700',
        href: '/modules/cyber-security'
      }
    ]
  },
  {
    id: 'emerging-tech',
    title: 'Emerging Technologies',
    description: 'Quantum computing and next-gen technologies',
    icon: '⚛️',
    modules: [
      {
        id: 'quantum-computing',
        title: 'Quantum Computing',
        description: '양자컴퓨팅 기초부터 Qiskit 실습까지 완전 정복',
        duration: '18시간',
        status: '학습 가능',
        icon: '⚛️',
        gradient: 'from-violet-500 to-purple-600',
        href: '/modules/quantum-computing'
      },
      {
        id: 'cloud-computing',
        title: 'Cloud Computing',
        description: 'AWS, GCP, Azure 클라우드 아키텍처와 서비스 실습',
        duration: '20시간',
        status: '개발중',
        icon: '☁️',
        gradient: 'from-sky-500 to-blue-600',
        href: '/modules/cloud-computing'
      }
    ]
  },
  {
    id: 'domain-specific',
    title: 'Domain-Specific',
    description: 'Specialized domains and industry applications',
    icon: '🏥',
    modules: [
      {
        id: 'bioinformatics',
        title: 'Bioinformatics',
        description: '생물정보학과 AI를 활용한 유전체 분석',
        duration: '16시간',
        status: '학습 가능',
        icon: '🧬',
        gradient: 'from-teal-600 to-green-700',
        href: '/modules/bioinformatics'
      },
      {
        id: 'medical-ai',
        title: 'Medical AI',
        description: '의료 영상 분석, 진단 보조, 신약 개발 AI 기술',
        duration: '15시간',
        status: '학습 가능',
        icon: '🏥',
        gradient: 'from-pink-500 to-red-500',
        href: '/medical-ai'
      }
    ]
  },
  {
    id: 'foundations',
    title: 'Foundations & Soft Skills',
    description: 'Ethics, language, and professional development',
    icon: '📚',
    modules: [
      {
        id: 'english-conversation',
        title: 'English Conversation',
        description: 'AI 튜터와 함께하는 실전 영어 회화 마스터 과정',
        duration: '8시간',
        status: '학습 가능',
        icon: '🗣️',
        gradient: 'from-rose-500 to-pink-600',
        href: '/modules/english-conversation'
      },
      {
        id: 'ai-ethics',
        title: 'AI Ethics & Governance',
        description: '책임감 있는 AI 개발과 윤리적 거버넌스 체계',
        duration: '16시간',
        status: '개발중',
        icon: '🌹',
        gradient: 'from-rose-500 to-pink-600',
        href: '/modules/ai-ethics'
      }
    ]
  }
];

// Total module count
export const getTotalModuleCount = () => {
  return moduleCategories.reduce((total, category) => total + category.modules.length, 0);
};

// Get module by ID
export const getModuleById = (id: string): Module | undefined => {
  for (const category of moduleCategories) {
    const module = category.modules.find(m => m.id === id);
    if (module) return module;
  }
  return undefined;
};

// Get category by module ID
export const getCategoryByModuleId = (moduleId: string): ModuleCategory | undefined => {
  return moduleCategories.find(category =>
    category.modules.some(m => m.id === moduleId)
  );
};
