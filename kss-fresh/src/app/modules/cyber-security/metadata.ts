export const moduleMetadata = {
  id: 'cyber-security',
  title: 'Cyber Security',
  description: '해킹 시뮬레이션과 제로트러스트 보안 모델 실습',
  icon: '🔒',
  gradient: 'from-red-600 to-orange-700',
  category: 'Security',
  difficulty: 'Advanced',
  estimatedHours: 24,
  chapters: [
    {
      id: 'introduction',
      title: '사이버 보안 기초',
      description: '현대 사이버 보안의 개념과 중요성',
      estimatedMinutes: 90,
    },
    {
      id: 'network-security',
      title: '네트워크 보안',
      description: '네트워크 공격 유형과 방어 기법',
      estimatedMinutes: 150,
    },
    {
      id: 'application-security',
      title: '애플리케이션 보안',
      description: 'OWASP Top 10과 시큐어 코딩',
      estimatedMinutes: 180,
    },
    {
      id: 'cloud-security',
      title: '클라우드 보안',
      description: '클라우드 환경의 보안 아키텍처',
      estimatedMinutes: 120,
    },
    {
      id: 'zero-trust',
      title: '제로트러스트 보안',
      description: '제로트러스트 모델 설계와 구현',
      estimatedMinutes: 150,
    },
    {
      id: 'penetration-testing',
      title: '침투 테스트',
      description: '윤리적 해킹과 취약점 분석',
      estimatedMinutes: 180,
    },
    {
      id: 'incident-response',
      title: '보안 사고 대응',
      description: '보안 사고 탐지와 대응 절차',
      estimatedMinutes: 120,
    },
    {
      id: 'security-operations',
      title: '보안 운영',
      description: 'SOC 구축과 SIEM 활용',
      estimatedMinutes: 150,
    },
  ],
  simulators: [
    {
      id: 'hacking-lab',
      title: '해킹 시뮬레이션 랩',
      description: '안전한 환경에서 다양한 해킹 기법 실습',
    },
    {
      id: 'vulnerability-scanner',
      title: '취약점 스캐너',
      description: '시스템 취약점 자동 탐지 및 분석',
    },
    {
      id: 'firewall-config',
      title: '방화벽 설정 시뮬레이터',
      description: '네트워크 방화벽 규칙 설계 및 테스트',
    },
    {
      id: 'zero-trust-architect',
      title: '제로트러스트 아키텍처 빌더',
      description: '제로트러스트 보안 모델 설계 도구',
    },
    {
      id: 'incident-simulator',
      title: '보안 사고 시뮬레이터',
      description: '실시간 보안 사고 대응 훈련',
    },
    {
      id: 'crypto-analyzer',
      title: '암호화 분석기',
      description: '암호화 알고리즘 분석 및 테스트',
    },
  ],
};