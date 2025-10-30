/**
 * Simulator Registry
 *
 * Automatically collects all simulators from module metadata files
 * and provides a unified interface for accessing simulator information.
 */

export interface Simulator {
  id: string;
  name: string;
  description: string;
  moduleId: string;
  moduleName: string;
  category: string;
  gradient: string;
  url: string;
  component?: string;
}

// Module metadata with simulators
const MODULES_WITH_SIMULATORS = [
  {
    id: 'llm',
    name: 'Large Language Models',
    category: 'AI & Machine Learning',
    gradient: 'from-indigo-500 to-purple-600',
    simulators: [
      { id: 'tokenizer-playground', name: '토크나이저 시뮬레이터', description: '다양한 토크나이저(GPT, Claude, Gemini)의 텍스트 분할 과정 비교' },
      { id: 'attention-visualizer', name: 'Attention 메커니즘 시각화', description: 'Self-Attention과 Multi-Head Attention의 실시간 동작 과정' },
      { id: 'transformer-architecture', name: 'Transformer 아키텍처 3D', description: '인코더-디코더 구조를 3D로 탐험하며 데이터 흐름 추적' },
      { id: 'training-lab', name: '모델 학습 시뮬레이터', description: '사전훈련부터 파인튜닝까지 전체 학습 과정을 단계별로 체험' },
      { id: 'prompt-playground', name: '프롬프트 플레이그라운드', description: '다양한 프롬프트 기법을 실시간으로 테스트하고 결과 비교' },
      { id: 'model-comparison', name: 'LLM 모델 비교기', description: 'Claude Opus 4, GPT-4o, Grok 4, Gemini 2.5, Llama 3.3 등 최신 모델 비교' }
    ]
  },
  {
    id: 'deep-learning',
    name: 'Deep Learning',
    category: 'AI & Machine Learning',
    gradient: 'from-violet-500 to-purple-600',
    simulators: [
      { id: 'neural-network-playground', name: 'Neural Network Playground', description: '레이어 구조 직접 설계하고 실시간 학습 결과 시각화' },
      { id: 'optimizer-comparison', name: 'Optimizer Comparison', description: 'SGD, Momentum, RMSprop, Adam 최적화 알고리즘 비교' },
      { id: 'attention-visualizer', name: 'Attention Visualizer', description: 'Multi-Head Self-Attention 메커니즘 시각화' },
      { id: 'cnn-visualizer', name: 'CNN Visualizer', description: '컨볼루션 연산과 Feature Map 실시간 생성' },
      { id: 'gan-generator', name: 'GAN Generator', description: '잠재 벡터로 이미지 생성 및 보간 실험' },
      { id: 'training-dashboard', name: 'Training Dashboard', description: 'Loss, Accuracy, Gradient Flow 실시간 모니터링' }
    ]
  },
  {
    id: 'computer-vision',
    name: 'Computer Vision',
    category: 'AI & Machine Learning',
    gradient: 'from-teal-500 to-cyan-600',
    simulators: [
      { id: 'object-detection-lab', name: 'Object Detection Lab', description: 'YOLO, R-CNN 실시간 객체 탐지 시뮬레이션' },
      { id: 'face-recognition-system', name: 'Face Recognition System', description: 'FaceNet 기반 얼굴 인식 및 임베딩 시각화' },
      { id: 'image-enhancement-studio', name: 'Image Enhancement Studio', description: 'Super-Resolution, Denoising, Colorization' },
      { id: 'pose-estimation-tracker', name: 'Pose Estimation Tracker', description: 'OpenPose, MediaPipe 실시간 포즈 추정' },
      { id: '2d-to-3d-converter', name: '2D to 3D Converter', description: '단일 이미지에서 3D 메시 재구성' }
    ]
  },
  {
    id: 'rag',
    name: 'RAG Systems',
    category: 'AI & Machine Learning',
    gradient: 'from-emerald-500 to-green-600',
    simulators: [
      { id: 'chunking-demo', name: 'Chunking Demo', description: '다양한 청킹 전략 데모 및 비교' },
      { id: 'document-uploader', name: 'Document Uploader', description: '문서 업로드 및 전처리 시뮬레이터' },
      { id: 'graphrag-explorer', name: 'GraphRAG Explorer', description: '지식 그래프 기반 RAG 시스템 시각화' },
      { id: 'rag-playground', name: 'RAG Playground', description: 'RAG 시스템 종합 실험 환경' }
    ]
  },
  {
    id: 'agent-mcp',
    name: 'AI Agent & MCP',
    category: 'AI & Machine Learning',
    gradient: 'from-emerald-600 to-teal-600',
    simulators: [
      { id: 'agent-playground', name: 'Agent Playground', description: 'AI 에이전트 실험 환경' },
      { id: 'langchain-builder', name: 'LangChain Builder', description: 'LangChain 체인 구축 도구' },
      { id: 'mcp-server', name: 'MCP Server', description: 'Model Context Protocol 서버 시뮬레이터' },
      { id: 'tool-orchestrator', name: 'Tool Orchestrator', description: '도구 오케스트레이션 시스템' }
    ]
  },
  {
    id: 'multimodal-ai',
    name: 'Multimodal AI',
    category: 'AI & Machine Learning',
    gradient: 'from-violet-600 to-purple-700',
    simulators: [
      { id: 'multimodal-architect', name: 'Multimodal Architect', description: '멀티모달 아키텍처 빌더 (드래그 앤 드롭)' },
      { id: 'clip-explorer', name: 'CLIP Explorer', description: 'CLIP 임베딩 탐색 및 유사도 분석' },
      { id: 'realtime-pipeline', name: 'Realtime Pipeline', description: '실시간 멀티모달 파이프라인 시뮬레이션' },
      { id: 'crossmodal-search', name: 'Crossmodal Search', description: 'Text→Image, Image→Text 크로스모달 검색' },
      { id: 'fusion-lab', name: 'Fusion Lab', description: '5가지 모달 퓨전 전략 비교' },
      { id: 'vqa-system', name: 'VQA System', description: 'Visual Question Answering 시스템' }
    ]
  },
  {
    id: 'python-programming',
    name: 'Python Programming',
    category: 'Programming',
    gradient: 'from-blue-500 to-indigo-600',
    simulators: [
      { id: 'python-repl', name: 'Python REPL', description: '브라우저 기반 파이썬 실행 환경' },
      { id: 'data-type-converter', name: 'Data Type Converter', description: '자료형 변환 시각화' },
      { id: 'collection-visualizer', name: 'Collection Visualizer', description: '리스트/튜플/딕셔너리 시각화' },
      { id: 'function-tracer', name: 'Function Tracer', description: '함수 실행 흐름 추적' },
      { id: 'oop-diagram-generator', name: 'OOP Diagram Generator', description: '클래스 다이어그램 자동 생성' },
      { id: 'exception-simulator', name: 'Exception Simulator', description: '예외 처리 시뮬레이션' },
      { id: 'file-io-playground', name: 'File I/O Playground', description: '파일 읽기/쓰기 실습' },
      { id: 'coding-challenges', name: 'Coding Challenges', description: '알고리즘 문제 풀이' }
    ]
  },
  {
    id: 'ai-automation',
    name: 'AI Automation',
    category: 'Programming',
    gradient: 'from-green-600 to-emerald-700',
    simulators: [
      { id: 'workflow-builder', name: 'Workflow Builder', description: 'AI 자동화 워크플로우 구축 도구' },
      { id: 'context-manager', name: 'Context Manager', description: '컨텍스트 윈도우 관리 시뮬레이터' },
      { id: 'prompt-optimizer', name: 'Prompt Optimizer', description: '프롬프트 최적화 도구' },
      { id: 'code-generator', name: 'Code Generator', description: 'AI 코드 생성 도구' }
    ]
  },
  {
    id: 'vibe-coding',
    name: 'Vibe Coding with AI',
    category: 'Programming',
    gradient: 'from-purple-500 to-pink-600',
    simulators: [
      { id: 'ai-code-assistant', name: 'AI Code Assistant', description: 'AI와 함께하는 페어 프로그래밍' },
      { id: 'prompt-optimizer', name: 'Prompt Optimizer', description: 'AI 프롬프트 최적화 도구' },
      { id: 'code-review-ai', name: 'Code Review AI', description: 'AI 코드 리뷰 시뮬레이터' },
      { id: 'refactoring-engine', name: 'Refactoring Engine', description: '코드 리팩토링 엔진' },
      { id: 'test-generator', name: 'Test Generator', description: '자동 테스트 생성기' },
      { id: 'doc-generator', name: 'Doc Generator', description: '자동 문서 생성기' }
    ]
  },
  {
    id: 'system-design',
    name: 'System Design',
    category: 'Engineering',
    gradient: 'from-purple-500 to-indigo-600',
    simulators: [
      { id: 'architecture-builder', name: 'Architecture Builder', description: '시스템 아키텍처 설계 도구' },
      { id: 'load-balancer', name: 'Load Balancer Simulator', description: '로드 밸런싱 전략 시뮬레이션' },
      { id: 'cache-simulator', name: 'Cache Simulator', description: '캐싱 전략 비교 및 성능 분석' },
      { id: 'sharding-visualizer', name: 'Sharding Visualizer', description: '데이터베이스 샤딩 시뮬레이터' },
      { id: 'cap-theorem', name: 'CAP Theorem', description: 'CAP 이론 시각화 도구' },
      { id: 'rate-limiter', name: 'Rate Limiter', description: 'API 속도 제한 시뮬레이터' },
      { id: 'mermaid-diagram-editor', name: 'Mermaid Diagram Editor', description: '전문급 다이어그램 에디터 (6개 템플릿)' }
    ]
  },
  {
    id: 'stock-analysis',
    name: '주식투자분석',
    category: 'Data & Analytics',
    gradient: 'from-red-500 to-orange-500',
    simulators: [
      { id: 'financial-calculator', name: '재무제표 분석기', description: '재무비율 자동계산, 동종업계 벤치마킹' },
      { id: 'chart-analyzer', name: 'AI 차트 분석기', description: 'AI 차트 패턴 자동 인식 및 매매 신호' },
      { id: 'portfolio-optimizer', name: '포트폴리오 최적화', description: '효율적 프론티어 계산과 최적 자산 배분' },
      { id: 'backtesting-engine', name: '백테스팅 엔진', description: '투자 전략 과거 데이터 검증' },
      { id: 'real-time-dashboard', name: '실시간 시장 대시보드', description: '실시간 가격, AI 예측, 호가창 분석' },
      { id: 'risk-management-dashboard', name: '리스크 관리 대시보드', description: 'VaR, 스트레스 테스트, 시나리오 분석' },
      { id: 'factor-investing-lab', name: '팩터 투자 연구소', description: '투자 팩터 성과 분석과 멀티팩터 백테스팅' },
      { id: 'options-strategy-analyzer', name: '옵션 전략 분석기', description: '옵션 전략 손익구조와 그릭스 시각화' },
      { id: 'ai-mentor', name: 'AI 투자 멘토', description: 'AI 기반 개인화된 투자 조언' },
      { id: 'news-impact-analyzer', name: 'AI 뉴스 분석기', description: 'AI 기반 뉴스 감정분석과 주가 영향도' },
      { id: 'dcf-valuation-model', name: 'DCF 가치평가 모델', description: '현금흐름 예측과 민감도 분석' },
      { id: 'global-market-dashboard', name: '글로벌 실시간 대시보드', description: '전 세계 주요 시장 실시간 현황' },
      { id: 'currency-impact-analyzer', name: '환율 영향 분석기', description: '환율 변동이 수익률에 미치는 영향 분석' },
      { id: 'us-stock-screener', name: '미국 주식 스크리너', description: 'NYSE, NASDAQ 전체 종목 필터링' },
      { id: 'tax-optimization-calculator', name: '세금 최적화 계산기', description: '미국/한국 주식 투자 세금 계산' },
      { id: 'etf-comparator', name: 'ETF 비교 분석기', description: 'ETF 성과, 리스크, 비용 종합 비교' },
      { id: 'sector-rotation-analyzer', name: '글로벌 섹터 로테이션 분석기', description: '경제 사이클별 섹터 성과 분석' },
      { id: 'global-macro-dashboard', name: '글로벌 매크로 대시보드', description: '전 세계 경제 지표 종합 분석' },
      { id: 'pair-trading-analyzer', name: '페어 트레이딩 분석기', description: '통계적 차익거래 기회 포착' },
      { id: 'monte-carlo-simulator', name: '몬테카를로 시뮬레이션', description: '10,000회 시뮬레이션으로 확률 분석' }
    ]
  },
  {
    id: 'data-engineering',
    name: 'Data Engineering',
    category: 'Data & Analytics',
    gradient: 'from-indigo-600 to-blue-700',
    simulators: [
      { id: 'eda-playground', name: 'EDA Playground', description: '탐색적 데이터 분석 플레이그라운드' },
      { id: 'etl-pipeline-designer', name: 'ETL Pipeline Designer', description: 'ETL/ELT 파이프라인 디자이너' },
      { id: 'stream-processing-lab', name: 'Stream Processing Lab', description: '실시간 스트림 처리 실습실' },
      { id: 'data-lakehouse-architect', name: 'Data Lakehouse Architect', description: '데이터 레이크하우스 아키텍트' },
      { id: 'airflow-dag-builder', name: 'Airflow DAG Builder', description: 'Airflow DAG 빌더' },
      { id: 'spark-optimizer', name: 'Spark Optimizer', description: 'Spark 성능 최적화 도구' },
      { id: 'data-quality-suite', name: 'Data Quality Suite', description: '데이터 품질 관리 스위트' },
      { id: 'cloud-cost-calculator', name: 'Cloud Cost Calculator', description: '클라우드 데이터 비용 계산기' },
      { id: 'data-lineage-explorer', name: 'Data Lineage Explorer', description: '데이터 계보 탐색기' },
      { id: 'sql-performance-tuner', name: 'SQL Performance Tuner', description: 'SQL 쿼리 성능 튜너' }
    ]
  },
  {
    id: 'ontology',
    name: 'Ontology & Semantic Web',
    category: 'Knowledge',
    gradient: 'from-purple-500 to-pink-500',
    simulators: [
      { id: 'inference-engine', name: 'Inference Engine', description: 'OWL 추론 엔진 시뮬레이터' }
    ]
  },
  {
    id: 'neo4j',
    name: 'Neo4j Knowledge Graph',
    category: 'Knowledge',
    gradient: 'from-blue-600 to-indigo-600',
    simulators: [
      { id: 'cypher-playground', name: 'Cypher Playground', description: 'Cypher 쿼리 실습 환경' },
      { id: 'graph-visualizer', name: 'Graph Visualizer', description: '지식 그래프 시각화 도구' },
      { id: 'algorithm-lab', name: 'Algorithm Lab', description: '그래프 알고리즘 실습실' },
      { id: 'node-editor', name: 'Node Editor', description: '노드 및 관계 편집기' },
      { id: 'import-wizard', name: 'Import Wizard', description: '데이터 임포트 마법사' }
    ]
  },
  {
    id: 'web3',
    name: 'Web3 & Blockchain',
    category: 'Web3',
    gradient: 'from-amber-500 to-orange-600',
    simulators: [
      { id: 'blockchain-explorer', name: 'Blockchain Explorer', description: '블록체인 탐색기' },
      { id: 'smart-contract-ide', name: 'Smart Contract IDE', description: 'Solidity 스마트 컨트랙트 IDE' },
      { id: 'defi-simulator', name: 'DeFi Simulator', description: 'DeFi 프로토콜 시뮬레이션' },
      { id: 'nft-minter', name: 'NFT Minter', description: 'NFT 발행 도구' },
      { id: 'gas-optimizer', name: 'Gas Optimizer', description: '가스비 최적화 도구' },
      { id: 'crypto-prediction-markets', name: 'Crypto Prediction Markets', description: '블록체인 기반 예측 시장' }
    ]
  },
  {
    id: 'quantum-computing',
    name: 'Quantum Computing',
    category: 'Emerging Tech',
    gradient: 'from-violet-500 to-purple-600',
    simulators: [
      { id: 'quantum-circuit-builder', name: 'Quantum Circuit Builder', description: '양자 회로 설계 도구' },
      { id: 'qubit-visualizer', name: 'Qubit Visualizer', description: '큐비트 상태 시각화' },
      { id: 'quantum-algorithm-lab', name: 'Quantum Algorithm Lab', description: 'Shor, Grover 알고리즘 실습' },
      { id: 'quantum-error-correction', name: 'Quantum Error Correction', description: '양자 오류 정정 코드' }
    ]
  },
  {
    id: 'ai-infrastructure',
    name: 'AI Infrastructure',
    category: 'Emerging Tech',
    gradient: 'from-slate-700 to-gray-800',
    simulators: [
      { id: 'infra-architect', name: 'Infrastructure Architect', description: 'AI 인프라 아키텍처 설계 도구' },
      { id: 'distributed-trainer', name: 'Distributed Trainer', description: '분산 학습 전략 비교 시뮬레이터' },
      { id: 'mlops-pipeline', name: 'MLOps Pipeline', description: 'MLOps 파이프라인 자동화 도구' },
      { id: 'model-monitor', name: 'Model Monitor', description: '실시간 모델 성능 모니터링' },
      { id: 'serving-optimizer', name: 'Serving Optimizer', description: '모델 서빙 최적화 도구' },
      { id: 'feature-store-sim', name: 'Feature Store', description: '피처 스토어 관리 시뮬레이터' }
    ]
  },
  {
    id: 'calculus',
    name: 'Calculus',
    category: 'Foundations',
    gradient: 'from-green-500 to-teal-600',
    simulators: [
      { id: 'limit-calculator', name: 'Limit Calculator', description: 'ε-δ definition 시각화' },
      { id: 'derivative-visualizer', name: 'Derivative Visualizer', description: '접선과 도함수 실시간 시각화' },
      { id: 'integral-calculator', name: 'Integral Calculator', description: '리만 합 4가지 방법' },
      { id: 'optimization-lab', name: 'Optimization Lab', description: 'Box, Fence, Cylinder 최적화' },
      { id: 'taylor-series-explorer', name: 'Taylor Series Explorer', description: '테일러 급수 애니메이션' },
      { id: 'gradient-field', name: 'Gradient Field', description: '2D 그래디언트 벡터장' }
    ]
  },
  {
    id: 'physics-fundamentals',
    name: 'Physics Fundamentals',
    category: 'Foundations',
    gradient: 'from-purple-500 to-pink-600',
    simulators: [
      { id: 'projectile-motion', name: 'Projectile Motion', description: '포물선 운동 애니메이션' },
      { id: 'collision-lab', name: 'Collision Lab', description: '탄성/비탄성 충돌 시뮬레이션' },
      { id: 'pendulum-simulator', name: 'Pendulum Simulator', description: '단순 조화 진동' },
      { id: 'electric-field', name: 'Electric Field', description: '다중 전하 전기장 벡터' },
      { id: 'wave-interference', name: 'Wave Interference', description: '2파원 간섭 패턴' },
      { id: 'thermodynamic-cycles', name: 'Thermodynamic Cycles', description: 'Carnot, Otto, Diesel 사이클' }
    ]
  },
  {
    id: 'linear-algebra',
    name: 'Linear Algebra',
    category: 'Foundations',
    gradient: 'from-blue-500 to-indigo-600',
    simulators: [
      { id: 'matrix-calculator', name: 'Matrix Calculator', description: '행렬 연산 계산기' },
      { id: 'linear-transformation-lab', name: 'Linear Transformation Lab', description: '선형 변환 실습실' },
      { id: 'eigenvalue-explorer', name: 'Eigenvalue Explorer', description: '고유값 탐색기' },
      { id: 'vector-visualizer', name: 'Vector Visualizer', description: '벡터 시각화 도구' },
      { id: 'svd-decomposer', name: 'SVD Decomposer', description: 'Singular Value Decomposition' },
      { id: 'gram-schmidt', name: 'Gram-Schmidt', description: 'Gram-Schmidt 직교화' }
    ]
  }
];

/**
 * Get all simulators from all modules
 */
export function getAllSimulators(): Simulator[] {
  const simulators: Simulator[] = [];

  for (const module of MODULES_WITH_SIMULATORS) {
    for (const sim of module.simulators) {
      simulators.push({
        id: sim.id,
        name: sim.name,
        description: sim.description,
        moduleId: module.id,
        moduleName: module.name,
        category: module.category,
        gradient: module.gradient,
        url: `/modules/${module.id}/simulators/${sim.id}`
      });
    }
  }

  return simulators;
}

/**
 * Get simulators by category
 */
export function getSimulatorsByCategory(category: string): Simulator[] {
  return getAllSimulators().filter(sim => sim.category === category);
}

/**
 * Get simulators by module
 */
export function getSimulatorsByModule(moduleId: string): Simulator[] {
  return getAllSimulators().filter(sim => sim.moduleId === moduleId);
}

/**
 * Get all unique categories
 */
export function getCategories(): string[] {
  const categories = new Set(MODULES_WITH_SIMULATORS.map(m => m.category));
  return Array.from(categories).sort();
}

/**
 * Get simulator statistics
 */
export function getSimulatorStats() {
  const simulators = getAllSimulators();
  const categories = getCategories();
  const modules = MODULES_WITH_SIMULATORS.length;

  return {
    total: simulators.length,
    categories: categories.length,
    modules,
    byCategory: categories.map(cat => ({
      name: cat,
      count: getSimulatorsByCategory(cat).length
    }))
  };
}
