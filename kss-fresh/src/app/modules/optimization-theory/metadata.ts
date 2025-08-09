export const moduleMetadata = {
  id: 'optimization-theory',
  title: 'Mathematical Optimization',
  description: 'AI 최적화 이론과 메타휴리스틱 알고리즘',
  icon: '📐',
  gradient: 'from-emerald-600 to-teal-700',
  category: 'Math',
  difficulty: 'Advanced',
  estimatedHours: 30,
  chapters: [
    {
      id: 'optimization-fundamentals',
      title: '최적화 기초',
      description: '최적화 문제의 정의와 분류',
      estimatedMinutes: 120,
    },
    {
      id: 'linear-optimization',
      title: '선형 최적화',
      description: 'Simplex, Interior Point 방법',
      estimatedMinutes: 180,
    },
    {
      id: 'nonlinear-optimization',
      title: '비선형 최적화',
      description: '경사하강법, Newton 방법, Quasi-Newton',
      estimatedMinutes: 210,
    },
    {
      id: 'constrained-optimization',
      title: '제약 최적화',
      description: 'KKT 조건, Lagrange 승수법',
      estimatedMinutes: 180,
    },
    {
      id: 'convex-optimization',
      title: '볼록 최적화',
      description: '볼록 함수와 볼록 최적화 문제',
      estimatedMinutes: 150,
    },
    {
      id: 'ai-optimization',
      title: 'AI 최적화',
      description: 'Adam, RMSprop, 하이퍼파라미터 튜닝',
      estimatedMinutes: 180,
    },
    {
      id: 'metaheuristics',
      title: '메타휴리스틱',
      description: '유전 알고리즘, 시뮬레이티드 어닐링, PSO',
      estimatedMinutes: 210,
    },
    {
      id: 'multi-objective',
      title: '다목적 최적화',
      description: 'Pareto 최적해와 NSGA-II',
      estimatedMinutes: 150,
    },
    {
      id: 'dynamic-programming',
      title: '동적 계획법',
      description: 'Bellman 방정식과 최적 제어',
      estimatedMinutes: 180,
    },
    {
      id: 'optimization-applications',
      title: '최적화 응용',
      description: '산업 현장에서의 최적화 사례',
      estimatedMinutes: 120,
    },
  ],
  simulators: [
    {
      id: 'optimization-visualizer',
      title: '최적화 알고리즘 시각화',
      description: '다양한 최적화 알고리즘 비교',
    },
    {
      id: 'constraint-visualizer',
      title: '제약 조건 시각화',
      description: '제약 최적화 문제 시각화',
    },
    {
      id: 'hyperparameter-tuner',
      title: '하이퍼파라미터 튜너',
      description: 'ML 모델 하이퍼파라미터 최적화',
    },
    {
      id: 'pareto-frontier',
      title: '파레토 프론티어',
      description: '다목적 최적화 파레토 해 분석',
    },
    {
      id: 'genetic-algorithm',
      title: '유전 알고리즘 실험실',
      description: 'GA 파라미터 설정 및 진화 관찰',
    },
    {
      id: 'gradient-explorer',
      title: '경사도 탐색기',
      description: '경사하강법 경로 시각화',
    },
    {
      id: 'convex-solver',
      title: '볼록 최적화 솔버',
      description: '볼록 최적화 문제 해결',
    },
  ],
};