import { Metadata } from 'next'

export const metadata: Metadata = {
  title: '반도체 - KSS',
  description: '반도체 설계부터 제조까지 - 칩의 모든 것',
}

export const moduleInfo = {
  id: 'semiconductor',
  title: '반도체',
  description: '반도체 설계부터 제조까지 - 칩의 모든 것',
  icon: '💎',
  color: 'from-blue-500 to-indigo-600',
  category: 'Hardware',
  level: 'intermediate',
  duration: '44시간',
  students: 280,
  rating: 4.8,
  chapters: [
    {
      id: 'basics',
      title: '반도체 기초',
      description: '실리콘부터 PN 접합까지',
      duration: '4시간'
    },
    {
      id: 'design',
      title: '회로 설계',
      description: 'CMOS 회로 설계 원리',
      duration: '5시간'
    },
    {
      id: 'lithography',
      title: '포토리소그래피',
      description: 'EUV와 패터닝 기술',
      duration: '6시간'
    },
    {
      id: 'fabrication',
      title: '제조 공정',
      description: '웨이퍼부터 패키징까지',
      duration: '6시간'
    },
    {
      id: 'advanced',
      title: '첨단 기술',
      description: 'GAA, 3D NAND, Chiplet',
      duration: '5시간'
    },
    {
      id: 'ai-chips',
      title: 'AI 반도체',
      description: 'NPU, TPU, GPU 아키텍처',
      duration: '6시간'
    },
    {
      id: 'memory',
      title: '메모리 반도체',
      description: 'DRAM, NAND, HBM',
      duration: '5시간'
    },
    {
      id: 'future',
      title: '미래 반도체',
      description: '양자 칩, 뉴로모픽, 광자 칩',
      duration: '3시간'
    },
    {
      id: 'image-display',
      title: '이미지센서 & 디스플레이',
      description: 'CMOS 이미지센서, DDI',
      duration: '4시간'
    }
  ],
  simulators: [
    {
      id: 'pn-junction',
      title: 'PN 접합 시뮬레이터',
      description: '다이오드 동작 원리 시각화'
    },
    {
      id: 'lithography-process',
      title: '리소그래피 공정 시뮬레이터',
      description: 'EUV 패터닝 과정 체험'
    },
    {
      id: 'wafer-fab',
      title: '웨이퍼 제조 시뮬레이터',
      description: '반도체 제조 전 과정'
    },
    {
      id: 'transistor-designer',
      title: '트랜지스터 설계 도구',
      description: 'FinFET, GAA 설계'
    },
    {
      id: 'yield-analyzer',
      title: '수율 분석기',
      description: '불량 분석과 수율 최적화'
    },
    {
      id: 'chip-architecture',
      title: '칩 아키텍처 시각화',
      description: 'CPU, GPU, NPU 구조'
    }
  ]
}
