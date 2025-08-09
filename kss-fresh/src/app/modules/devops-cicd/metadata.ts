export const devopsMetadata = {
  id: 'devops-cicd',
  title: 'DevOps & CI/CD',
  description: 'Docker, Kubernetes, GitOps로 구축하는 현대적 개발 운영',
  category: 'DevOps',
  difficulty: 'intermediate' as const,
  duration: '16시간',
  moduleColor: 'from-gray-500 to-slate-600',
  chapters: [
    {
      id: 'devops-culture',
      title: 'DevOps 문화와 철학',
      description: 'DevOps의 핵심 개념, 문화 변화, 도구체인 이해',
      duration: '90분',
      learningObjectives: [
        'DevOps의 정의와 핵심 원칙 이해',
        '전통적인 개발 방식과의 차이점 파악',
        'DevOps 문화 구축 방법',
        'DevOps 도구체인 생태계 개요'
      ]
    },
    {
      id: 'docker-fundamentals',
      title: 'Docker 기초와 컨테이너화',
      description: '컨테이너 개념부터 Docker 실습까지',
      duration: '120분',
      learningObjectives: [
        'VM vs 컨테이너의 차이점 이해',
        'Docker 아키텍처와 핵심 개념',
        'Docker 명령어와 기본 조작',
        'Dockerfile 작성 및 이미지 빌드'
      ]
    },
    {
      id: 'docker-advanced',
      title: 'Docker 고급 기법',
      description: 'Docker Compose, 네트워킹, 볼륨, 최적화',
      duration: '150분',
      learningObjectives: [
        'Docker Compose로 멀티컨테이너 관리',
        'Docker 네트워킹과 볼륨 관리',
        '이미지 최적화 및 멀티스테이지 빌드',
        '컨테이너 레지스트리 활용'
      ]
    },
    {
      id: 'kubernetes-basics',
      title: 'Kubernetes 기초',
      description: 'K8s 아키텍처와 핵심 오브젝트 이해',
      duration: '120분',
      learningObjectives: [
        'Kubernetes 아키텍처 이해',
        'Pod, Service, Deployment 개념',
        'kubectl 기본 명령어',
        'YAML 매니페스트 작성'
      ]
    },
    {
      id: 'kubernetes-advanced',
      title: 'Kubernetes 운영',
      description: 'Ingress, ConfigMap, Secret, 스케일링',
      duration: '150분',
      learningObjectives: [
        'Ingress Controller로 외부 노출',
        'ConfigMap과 Secret으로 설정 관리',
        'HPA와 VPA를 통한 자동 스케일링',
        'Helm을 통한 패키지 관리'
      ]
    },
    {
      id: 'cicd-pipelines',
      title: 'CI/CD 파이프라인 구축',
      description: 'GitHub Actions, Jenkins로 자동화 파이프라인',
      duration: '120분',
      learningObjectives: [
        'CI/CD의 개념과 이점',
        'GitHub Actions 워크플로우 작성',
        'Jenkins 파이프라인 구성',
        '테스트 자동화 통합'
      ]
    },
    {
      id: 'gitops-deployment',
      title: 'GitOps와 배포 전략',
      description: '선언적 배포, Blue-Green, Canary, Rolling Update',
      duration: '120분',
      learningObjectives: [
        'GitOps 개념과 ArgoCD 사용',
        'Blue-Green 배포 전략',
        'Canary 배포와 점진적 롤아웃',
        'Rolling Update와 롤백'
      ]
    },
    {
      id: 'monitoring-security',
      title: '모니터링, 로깅, 보안',
      description: 'Prometheus, Grafana, ELK Stack, 컨테이너 보안',
      duration: '150분',
      learningObjectives: [
        'Prometheus로 메트릭 수집',
        'Grafana 대시보드 구성',
        'ELK Stack으로 로그 분석',
        '컨테이너 보안 모범 사례'
      ]
    }
  ],
  simulators: [
    {
      id: 'docker-builder',
      title: 'Docker 컨테이너 빌더',
      description: 'Dockerfile 최적화와 이미지 빌드 실습',
      difficulty: 'intermediate',
      estimatedTime: '30분'
    },
    {
      id: 'k8s-cluster-sim',
      title: 'Kubernetes 클러스터 시뮬레이터',
      description: 'Pod, Service, Deployment 관리 체험',
      difficulty: 'intermediate',
      estimatedTime: '40분'
    },
    {
      id: 'cicd-pipeline-builder',
      title: 'CI/CD 파이프라인 설계기',
      description: 'GitHub Actions 워크플로우 구성',
      difficulty: 'intermediate',
      estimatedTime: '35분'
    },
    {
      id: 'deployment-strategies',
      title: '배포 전략 비교 실습',
      description: 'Blue-Green vs Canary vs Rolling Update',
      difficulty: 'advanced',
      estimatedTime: '45분'
    },
    {
      id: 'monitoring-dashboard',
      title: '모니터링 대시보드',
      description: 'Prometheus 메트릭을 Grafana로 시각화',
      difficulty: 'intermediate',
      estimatedTime: '25분'
    },
    {
      id: 'security-scanner',
      title: '컨테이너 보안 스캐너',
      description: '이미지 취약점 분석과 보안 정책',
      difficulty: 'advanced',
      estimatedTime: '30분'
    }
  ],
  prerequisites: [
    'Linux 기본 명령어',
    'Git 사용법',
    '기본적인 웹 애플리케이션 이해'
  ],
  outcomes: [
    'Docker를 활용한 애플리케이션 컨테이너화',
    'Kubernetes 클러스터에서 애플리케이션 배포 및 관리',
    'CI/CD 파이프라인 설계 및 구현',
    'GitOps 기반 배포 자동화',
    '프로덕션 환경의 모니터링 및 보안'
  ],
  tools: [
    'Docker & Docker Compose',
    'Kubernetes & kubectl',
    'GitHub Actions',
    'Jenkins',
    'ArgoCD',
    'Prometheus & Grafana',
    'ELK Stack',
    'Helm'
  ]
}