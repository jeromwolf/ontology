export const moduleMetadata = {
  id: 'hpc-computing',
  title: 'High-Performance Computing',
  description: 'CUDA 프로그래밍과 분산 컴퓨팅 최적화',
  icon: '⚡',
  gradient: 'from-yellow-500 to-orange-600',
  category: 'HPC',
  difficulty: 'Advanced',
  estimatedHours: 30,
  chapters: [
    {
      id: 'hpc-fundamentals',
      title: 'HPC 기초',
      description: '고성능 컴퓨팅의 개념과 중요성',
      estimatedMinutes: 120,
    },
    {
      id: 'parallel-computing',
      title: '병렬 컴퓨팅',
      description: 'OpenMP, MPI를 활용한 병렬 프로그래밍',
      estimatedMinutes: 240,
    },
    {
      id: 'cuda-programming',
      title: 'CUDA 프로그래밍',
      description: 'GPU 병렬처리의 기초와 최적화',
      estimatedMinutes: 300,
    },
    {
      id: 'gpu-architecture',
      title: 'GPU 아키텍처',
      description: 'NVIDIA GPU 아키텍처와 메모리 계층',
      estimatedMinutes: 180,
    },
    {
      id: 'cluster-computing',
      title: '클러스터 컴퓨팅',
      description: 'HPC 클러스터 구축과 관리',
      estimatedMinutes: 210,
    },
    {
      id: 'performance-optimization',
      title: '성능 최적화',
      description: '병렬 알고리즘 최적화 기법',
      estimatedMinutes: 240,
    },
    {
      id: 'distributed-systems',
      title: '분산 시스템',
      description: '대규모 분산 컴퓨팅 시스템 설계',
      estimatedMinutes: 180,
    },
    {
      id: 'scientific-computing',
      title: '과학 컴퓨팅 응용',
      description: '시뮬레이션, 수치해석, 빅데이터 분석',
      estimatedMinutes: 150,
    },
    {
      id: 'cloud-hpc',
      title: '클라우드 HPC',
      description: 'AWS, Azure에서의 HPC 환경 구축',
      estimatedMinutes: 120,
    },
    {
      id: 'ai-acceleration',
      title: 'AI 가속화',
      description: '딥러닝 모델 학습 최적화',
      estimatedMinutes: 180,
    },
  ],
  simulators: [
    {
      id: 'cuda-kernel-analyzer',
      title: 'CUDA 커널 분석기',
      description: 'GPU 커널 성능 분석 및 최적화',
    },
    {
      id: 'parallel-algorithm-viz',
      title: '병렬 알고리즘 시각화',
      description: '분산 알고리즘 실행 흐름 시각화',
    },
    {
      id: 'gpu-memory-optimizer',
      title: 'GPU 메모리 최적화기',
      description: '메모리 사용 패턴 분석 및 개선',
    },
    {
      id: 'cluster-scheduler',
      title: '클러스터 스케줄러',
      description: 'HPC 작업 스케줄링 시뮬레이터',
    },
    {
      id: 'mpi-debugger',
      title: 'MPI 디버거',
      description: '분산 프로그램 디버깅 도구',
    },
    {
      id: 'performance-profiler',
      title: '성능 프로파일러',
      description: 'HPC 애플리케이션 성능 분석',
    },
    {
      id: 'load-balancer',
      title: '부하 균형 시뮬레이터',
      description: '작업 분배 최적화 도구',
    },
    {
      id: 'hpc-benchmark',
      title: 'HPC 벤치마크 도구',
      description: '시스템 성능 측정 및 비교',
    },
  ],
};