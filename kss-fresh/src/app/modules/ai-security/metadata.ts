export const aiSecurityMetadata = {
  title: 'AI 보안',
  description: 'AI 시스템의 보안 위협과 방어 기법을 학습합니다',
  simulators: [
    {
      id: 'adversarial-attack-lab',
      title: '적대적 공격 실험실',
      description: '이미지 분류 모델에 대한 적대적 공격 시뮬레이션',
      difficulty: 'intermediate'
    },
    {
      id: 'model-extraction',
      title: '모델 추출 시뮬레이터',
      description: '블랙박스 모델에서 정보를 추출하는 공격 시뮬레이션',
      difficulty: 'advanced'
    },
    {
      id: 'privacy-attack',
      title: '프라이버시 공격 시뮬레이터',
      description: '멤버십 추론과 속성 추론 공격 체험',
      difficulty: 'intermediate'
    },
    {
      id: 'defense-mechanisms',
      title: '방어 기법 테스터',
      description: '다양한 AI 보안 방어 기법의 효과 비교',
      difficulty: 'advanced'
    },
    {
      id: 'security-audit',
      title: 'AI 보안 감사 도구',
      description: 'AI 시스템의 보안 취약점 종합 평가',
      difficulty: 'expert'
    }
  ],
  chapters: [
    {
      id: 'fundamentals',
      title: 'AI 보안 기초',
      description: 'AI 시스템의 보안 위협과 취약점 이해'
    },
    {
      id: 'adversarial-attacks',
      title: '적대적 공격',
      description: '적대적 예제와 회피 공격 기법'
    },
    {
      id: 'model-security',
      title: '모델 보안',
      description: '모델 추출, 역공학, 백도어 공격'
    },
    {
      id: 'privacy-preserving',
      title: '프라이버시 보호 ML',
      description: '차분 프라이버시와 연합 학습'
    },
    {
      id: 'robustness',
      title: '견고성과 방어',
      description: '방어 기법과 견고한 학습'
    },
    {
      id: 'security-testing',
      title: '보안 테스팅',
      description: 'AI 시스템 보안 평가와 감사'
    },
    {
      id: 'deployment-security',
      title: '배포 보안',
      description: '프로덕션 환경의 AI 보안'
    },
    {
      id: 'case-studies',
      title: '사례 연구',
      description: '실제 AI 보안 사고와 대응'
    }
  ]
};