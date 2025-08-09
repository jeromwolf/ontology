export const moduleMetadata = {
  id: 'ai-ethics',
  title: 'AI Ethics & Governance',
  description: '책임감 있는 AI 개발과 윤리적 거버넌스 체계',
  icon: '🌹',
  gradient: 'from-rose-500 to-pink-600',
  category: 'Ethics',
  difficulty: 'Intermediate',
  estimatedHours: 16,
  chapters: [
    {
      id: 'introduction',
      title: 'AI 윤리의 중요성',
      description: 'AI 시대의 윤리적 딜레마와 책임감 있는 개발',
      estimatedMinutes: 90,
    },
    {
      id: 'bias-fairness',
      title: '편향과 공정성',
      description: 'AI 시스템의 편향 탐지 및 완화 기법',
      estimatedMinutes: 120,
    },
    {
      id: 'transparency',
      title: '투명성과 설명가능성',
      description: 'Explainable AI와 블랙박스 문제 해결',
      estimatedMinutes: 120,
    },
    {
      id: 'privacy-security',
      title: '프라이버시와 보안',
      description: '개인정보 보호와 데이터 거버넌스',
      estimatedMinutes: 90,
    },
    {
      id: 'regulation',
      title: 'AI 규제와 법적 프레임워크',
      description: 'EU AI Act, 한국 AI 윤리 기준 분석',
      estimatedMinutes: 120,
    },
    {
      id: 'case-studies',
      title: '실제 사례 연구',
      description: 'ChatGPT, Claude 등 실제 AI 윤리 사례',
      estimatedMinutes: 120,
    },
  ],
  simulators: [
    {
      id: 'bias-detector',
      title: 'AI 편향 탐지기',
      description: '머신러닝 모델의 편향을 시각화하고 분석',
    },
    {
      id: 'fairness-analyzer',
      title: '공정성 분석 도구',
      description: '다양한 공정성 지표를 활용한 모델 평가',
    },
    {
      id: 'ethics-framework',
      title: 'AI 윤리 프레임워크 빌더',
      description: '조직별 맞춤형 AI 윤리 가이드라인 생성',
    },
    {
      id: 'impact-assessment',
      title: 'AI 영향 평가 시뮬레이터',
      description: 'AI 시스템의 사회적 영향 분석 및 예측',
    },
  ],
};