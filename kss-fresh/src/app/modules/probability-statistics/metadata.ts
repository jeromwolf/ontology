import { Module } from '@/types/module'

export const probabilityStatisticsModule: Module = {
  id: 'probability-statistics',
  name: 'Probability & Statistics',
  nameKo: '확률과 통계',
  description: 'AI와 머신러닝의 수학적 기초, 확률론과 통계학을 인터랙티브하게 학습',
  version: '1.0.0',
  difficulty: 'intermediate',
  estimatedHours: 20,
  icon: 'BarChart3',
  color: 'from-purple-500 to-pink-500',
  prerequisites: ['linear-algebra'],
  chapters: [
    {
      id: 'probability-basics',
      title: '확률의 기초',
      description: '확률의 개념, 조건부 확률, 베이즈 정리',
      estimatedMinutes: 45,
      keywords: ['확률', '조건부확률', '베이즈정리', '표본공간', '사건']
    },
    {
      id: 'distributions',
      title: '확률 분포',
      description: '이산/연속 확률분포, 정규분포, 이항분포, 포아송분포',
      estimatedMinutes: 60,
      keywords: ['확률분포', '정규분포', '이항분포', '포아송분포', '연속분포']
    },
    {
      id: 'descriptive-statistics',
      title: '기술 통계',
      description: '평균, 분산, 표준편차, 상관계수, 데이터 시각화',
      estimatedMinutes: 50,
      keywords: ['평균', '분산', '표준편차', '상관계수', '기술통계']
    },
    {
      id: 'inferential-statistics',
      title: '추론 통계',
      description: '가설검정, 신뢰구간, p-value, t-test, ANOVA',
      estimatedMinutes: 55,
      keywords: ['가설검정', '신뢰구간', 'p-value', 't-test', 'ANOVA']
    },
    {
      id: 'bayesian-statistics',
      title: '베이지안 통계',
      description: '베이지안 추론, 사전/사후 분포, MCMC',
      estimatedMinutes: 65,
      keywords: ['베이지안', '사전분포', '사후분포', 'MCMC', '베이즈정리']
    },
    {
      id: 'regression-analysis',
      title: '회귀 분석',
      description: '선형회귀, 다중회귀, 로지스틱 회귀, 정규화',
      estimatedMinutes: 70,
      keywords: ['선형회귀', '다중회귀', '로지스틱회귀', '정규화', '회귀분석']
    },
    {
      id: 'time-series',
      title: '시계열 분석',
      description: 'ARIMA, 계절성, 추세 분석, 예측',
      estimatedMinutes: 60,
      keywords: ['시계열', 'ARIMA', '계절성', '추세분석', '예측']
    },
    {
      id: 'ml-statistics',
      title: 'ML을 위한 통계',
      description: '교차검증, 과적합, 편향-분산 트레이드오프, A/B 테스트',
      estimatedMinutes: 75,
      keywords: ['교차검증', '과적합', '편향분산', 'A/B테스트', '머신러닝']
    }
  ],
  simulators: [
    {
      id: 'probability-playground',
      name: '확률 실험실',
      description: '동전, 주사위, 카드 등을 이용한 확률 실험',
      component: 'ProbabilityPlayground'
    },
    {
      id: 'distribution-visualizer',
      name: '분포 시각화 도구',
      description: '다양한 확률분포를 인터랙티브하게 탐색',
      component: 'DistributionVisualizer'
    },
    {
      id: 'hypothesis-tester',
      name: '가설검정 시뮬레이터',
      description: 't-test, chi-square, ANOVA 등 통계 검정 실습',
      component: 'HypothesisTester'
    },
    {
      id: 'regression-lab',
      name: '회귀분석 연구실',
      description: '데이터 입력부터 회귀모델 구축까지',
      component: 'RegressionLab'
    },
    {
      id: 'monte-carlo',
      name: '몬테카를로 시뮬레이션',
      description: '확률적 방법을 이용한 문제 해결',
      component: 'MonteCarloSimulator'
    }
  ],
  tools: [
    {
      id: 'r-studio',
      name: 'R Studio',
      description: '통계 분석 전문 도구',
      url: 'https://www.rstudio.com/'
    },
    {
      id: 'jupyter-notebook',
      name: 'Jupyter Notebook',
      description: '파이썬 통계 분석 환경',
      url: 'https://jupyter.org/'
    },
    {
      id: 'statsmodels',
      name: 'Statsmodels',
      description: '파이썬 통계 모델링 라이브러리',
      url: 'https://www.statsmodels.org/'
    }
  ]
}