export const moduleMetadata = {
  id: 'cloud-computing',
  title: 'Cloud Computing',
  description: 'AWS, Azure, GCP를 활용한 클라우드 아키텍처 설계',
  icon: '☁️',
  gradient: 'from-sky-500 to-blue-600',
  category: 'Cloud',
  difficulty: 'Intermediate',
  estimatedHours: 30,
  chapters: [
    {
      id: 'cloud-fundamentals',
      title: '클라우드 컴퓨팅 기초',
      description: 'IaaS, PaaS, SaaS 개념과 클라우드 서비스 모델',
      estimatedMinutes: 120,
    },
    {
      id: 'aws-essentials',
      title: 'AWS 핵심 서비스',
      description: 'EC2, S3, RDS, Lambda 등 AWS 핵심 서비스',
      estimatedMinutes: 240,
    },
    {
      id: 'azure-fundamentals',
      title: 'Azure 기초',
      description: 'Azure VM, Storage, Azure Functions 활용',
      estimatedMinutes: 180,
    },
    {
      id: 'gcp-overview',
      title: 'Google Cloud Platform',
      description: 'Compute Engine, Cloud Storage, BigQuery 실습',
      estimatedMinutes: 180,
    },
    {
      id: 'cloud-architecture',
      title: '클라우드 아키텍처 패턴',
      description: '확장 가능한 클라우드 아키텍처 설계',
      estimatedMinutes: 150,
    },
    {
      id: 'serverless',
      title: '서버리스 아키텍처',
      description: 'Lambda, Functions, Cloud Run 활용 서버리스 구현',
      estimatedMinutes: 180,
    },
    {
      id: 'containerization',
      title: '컨테이너와 오케스트레이션',
      description: 'Docker, Kubernetes on Cloud 활용',
      estimatedMinutes: 210,
    },
    {
      id: 'cloud-security',
      title: '클라우드 보안',
      description: 'IAM, VPC, 암호화, 컴플라이언스',
      estimatedMinutes: 150,
    },
    {
      id: 'cost-optimization',
      title: '비용 최적화',
      description: '클라우드 비용 분석 및 최적화 전략',
      estimatedMinutes: 120,
    },
    {
      id: 'multi-cloud',
      title: '멀티 클라우드 전략',
      description: '하이브리드 및 멀티 클라우드 아키텍처',
      estimatedMinutes: 120,
    },
  ],
  simulators: [
    {
      id: 'cloud-architect',
      title: '클라우드 아키텍처 디자이너',
      description: '드래그 앤 드롭으로 클라우드 아키텍처 설계',
    },
    {
      id: 'cost-calculator',
      title: '클라우드 비용 계산기',
      description: '실시간 클라우드 비용 예측 및 분석',
    },
    {
      id: 'serverless-lab',
      title: '서버리스 실습 환경',
      description: '서버리스 함수 개발 및 배포 실습',
    },
    {
      id: 'container-orchestrator',
      title: '컨테이너 오케스트레이터',
      description: 'Kubernetes 클러스터 관리 시뮬레이터',
    },
    {
      id: 'cloud-migration',
      title: '클라우드 마이그레이션 플래너',
      description: '온프레미스에서 클라우드로 마이그레이션 계획',
    },
    {
      id: 'multi-cloud-manager',
      title: '멀티 클라우드 매니저',
      description: '여러 클라우드 플랫폼 통합 관리',
    },
    {
      id: 'cloud-security-lab',
      title: '클라우드 보안 랩',
      description: 'IAM 정책 설정 및 VPC 보안 구성',
    },
    {
      id: 'infrastructure-as-code',
      title: 'IaC 실습 환경',
      description: 'Terraform, CloudFormation으로 인프라 관리',
    },
  ],
};