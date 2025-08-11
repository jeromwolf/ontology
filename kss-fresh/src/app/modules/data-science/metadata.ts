export const moduleMetadata = {
  id: 'data-science',
  title: 'Data Science',
  description: '데이터에서 가치를 창출하는 과학적 접근법 - 통계부터 딥러닝까지',
  icon: '📊',
  gradient: 'from-emerald-600 to-green-700',
  category: 'Data',
  difficulty: 'Intermediate',
  estimatedHours: 40,
  students: 2340,
  rating: 4.9,
  lastUpdated: '2025-08-10',
  prerequisites: ['Python 기초', '통계학 기본', '선형대수 기초'],
  skills: [
    '통계적 사고와 가설 검정',
    '탐색적 데이터 분석 (EDA)',
    '머신러닝 알고리즘',
    '딥러닝 기초',
    'A/B 테스트',
    '시계열 분석',
    '자연어 처리',
    '비즈니스 인사이트 도출'
  ],
  chapters: [
    {
      id: 'data-science-intro',
      title: '데이터 사이언스 개요',
      description: '데이터 사이언티스트의 역할과 워크플로우',
      estimatedMinutes: 120,
    },
    {
      id: 'statistical-thinking',
      title: '통계적 사고와 추론',
      description: '가설 검정, p-value, 신뢰구간, 베이지안 추론',
      estimatedMinutes: 240,
    },
    {
      id: 'eda-visualization',
      title: 'EDA와 데이터 시각화',
      description: 'Matplotlib, Seaborn, Plotly로 하는 효과적인 시각화',
      estimatedMinutes: 180,
    },
    {
      id: 'supervised-learning',
      title: '지도학습 - 분류와 회귀',
      description: '로지스틱 회귀, SVM, 랜덤 포레스트, XGBoost',
      estimatedMinutes: 300,
    },
    {
      id: 'unsupervised-learning',
      title: '비지도학습 - 클러스터링과 차원축소',
      description: 'K-means, DBSCAN, PCA, t-SNE, UMAP',
      estimatedMinutes: 240,
    },
    {
      id: 'deep-learning-basics',
      title: '딥러닝 입문',
      description: 'TensorFlow/PyTorch로 신경망 구축하기',
      estimatedMinutes: 300,
    },
    {
      id: 'time-series-analysis',
      title: '시계열 분석과 예측',
      description: 'ARIMA, Prophet, LSTM을 활용한 시계열 예측',
      estimatedMinutes: 240,
    },
    {
      id: 'nlp-fundamentals',
      title: '자연어 처리 기초',
      description: '텍스트 전처리, Word2Vec, Transformer 기초',
      estimatedMinutes: 240,
    },
    {
      id: 'ab-testing',
      title: 'A/B 테스트와 인과추론',
      description: '실험 설계, 통계적 유의성, 인과관계 분석',
      estimatedMinutes: 180,
    },
    {
      id: 'model-deployment',
      title: '모델 배포와 모니터링',
      description: 'Flask/FastAPI, Docker, 모델 버전 관리',
      estimatedMinutes: 180,
    },
    {
      id: 'business-analytics',
      title: '비즈니스 분석과 스토리텔링',
      description: '데이터 기반 의사결정과 효과적인 커뮤니케이션',
      estimatedMinutes: 150,
    },
    {
      id: 'case-studies',
      title: '실전 프로젝트와 케이스 스터디',
      description: '추천 시스템, 이탈 예측, 고객 세분화',
      estimatedMinutes: 240,
    },
  ],
  simulators: [
    {
      id: 'ml-playground',
      title: '머신러닝 플레이그라운드',
      description: '인터랙티브 ML 알고리즘 시각화 및 실험',
      component: 'MLPlayground'
    },
    {
      id: 'statistical-lab',
      title: '통계 분석 실험실',
      description: '가설 검정, 분포 시뮬레이션, 통계적 추론',
      component: 'StatisticalLab'
    },
    {
      id: 'neural-network-builder',
      title: '신경망 빌더',
      description: '드래그 앤 드롭으로 신경망 설계 및 학습',
      component: 'NeuralNetworkBuilder'
    },
    {
      id: 'clustering-visualizer',
      title: '클러스터링 시각화 도구',
      description: '다양한 클러스터링 알고리즘 비교 분석',
      component: 'ClusteringVisualizer'
    },
    {
      id: 'time-series-forecaster',
      title: '시계열 예측기',
      description: '실시간 시계열 데이터 분석 및 예측',
      component: 'TimeSeriesForecaster'
    },
    {
      id: 'nlp-analyzer',
      title: 'NLP 분석기',
      description: '텍스트 분석, 감성 분석, 토픽 모델링',
      component: 'NLPAnalyzer'
    },
    {
      id: 'ab-test-simulator',
      title: 'A/B 테스트 시뮬레이터',
      description: '실험 설계, 샘플 크기 계산, 결과 분석',
      component: 'ABTestSimulator'
    },
    {
      id: 'feature-engineering-lab',
      title: '피처 엔지니어링 랩',
      description: '자동 피처 생성 및 중요도 분석',
      component: 'FeatureEngineeringLab'
    },
    {
      id: 'model-explainer',
      title: '모델 설명 도구',
      description: 'SHAP, LIME을 활용한 모델 해석',
      component: 'ModelExplainer'
    },
    {
      id: 'recommendation-engine',
      title: '추천 시스템 엔진',
      description: '협업 필터링, 콘텐츠 기반 추천 구현',
      component: 'RecommendationEngine'
    },
  ],
  tools: [
    {
      id: 'dataset-explorer',
      title: '데이터셋 탐색기',
      description: 'Kaggle 데이터셋 탐색 및 분석',
      icon: '🔍',
    },
    {
      id: 'notebook-runner',
      title: '노트북 실행기',
      description: 'Jupyter 노트북 온라인 실행 환경',
      icon: '📓',
    },
    {
      id: 'model-zoo',
      title: '모델 동물원',
      description: '사전 학습된 모델 라이브러리',
      icon: '🦁',
    },
    {
      id: 'metric-calculator',
      title: '평가 지표 계산기',
      description: '다양한 ML 평가 지표 자동 계산',
      icon: '📈',
    },
  ],
  learningPath: [
    {
      stage: 'Foundation',
      description: '데이터 사이언스 기초',
      chapters: ['data-science-intro', 'statistical-thinking', 'eda-visualization']
    },
    {
      stage: 'Machine Learning',
      description: '머신러닝 핵심',
      chapters: ['supervised-learning', 'unsupervised-learning', 'deep-learning-basics']
    },
    {
      stage: 'Advanced Topics',
      description: '고급 주제',
      chapters: ['time-series-analysis', 'nlp-fundamentals', 'ab-testing']
    },
    {
      stage: 'Professional',
      description: '실무 적용',
      chapters: ['model-deployment', 'business-analytics', 'case-studies']
    }
  ]
};