export const metadata = {
  title: '기초 물리학 (Physics Fundamentals)',
  description: '뉴턴 역학, 전자기학, 열역학의 기본 원리',
  icon: '⚛️',
  category: 'foundations',
  difficulty: 'beginner',
  duration: '20시간',
  students: 756,
  rating: 4.7,
  color: 'from-purple-500 to-pink-600',

  chapters: [
    {
      id: 'mechanics-basics',
      title: 'Chapter 1: 역학의 기초',
      description: '뉴턴의 운동 법칙, 힘과 운동',
      duration: '120분',
      objectives: ['뉴턴 법칙 이해', '자유 물체 다이어그램', '운동 방정식 풀이']
    },
    {
      id: 'kinematics',
      title: 'Chapter 2: 운동학',
      description: '직선 운동, 포물선 운동, 원운동',
      duration: '110분',
      objectives: ['등가속도 운동', '포물선 궤적 계산', '원운동 분석']
    },
    {
      id: 'energy-work',
      title: 'Chapter 3: 일과 에너지',
      description: '일-에너지 정리, 위치에너지, 운동에너지',
      duration: '130분',
      objectives: ['일 계산', '에너지 보존 법칙', '역학적 에너지']
    },
    {
      id: 'momentum',
      title: 'Chapter 4: 운동량과 충돌',
      description: '운동량 보존, 탄성/비탄성 충돌',
      duration: '120분',
      objectives: ['운동량 보존', '충돌 분석', '충격량']
    },
    {
      id: 'rotation',
      title: 'Chapter 5: 회전 운동',
      description: '각속도, 관성모멘트, 토크',
      duration: '140분',
      objectives: ['회전 운동 방정식', '관성모멘트 계산', '각운동량 보존']
    },
    {
      id: 'oscillations',
      title: 'Chapter 6: 진동과 파동',
      description: '단순 조화 운동, 파동 방정식',
      duration: '130분',
      objectives: ['단순 조화 진동', '파동의 전파', '공명 현상']
    },
    {
      id: 'electromagnetism',
      title: 'Chapter 7: 전자기학 입문',
      description: '전기장, 자기장, 맥스웰 방정식',
      duration: '150분',
      objectives: ['쿨롱 법칙', '패러데이 법칙', '전자기 유도']
    },
    {
      id: 'thermodynamics',
      title: 'Chapter 8: 열역학',
      description: '열역학 법칙, 엔트로피, 열기관',
      duration: '140분',
      objectives: ['열역학 제1법칙', '제2법칙과 엔트로피', '카르노 사이클']
    }
  ],

  simulators: [
    {
      id: 'projectile-motion',
      title: '포물선 운동 시뮬레이터',
      description: '발사 각도와 속도에 따른 궤적 시뮬레이션',
      difficulty: 'beginner'
    },
    {
      id: 'collision-lab',
      title: '충돌 실험실',
      description: '탄성/비탄성 충돌 시뮬레이션',
      difficulty: 'beginner'
    },
    {
      id: 'pendulum-simulator',
      title: '진자 시뮬레이터',
      description: '단순 진자와 복잡한 진동 시스템',
      difficulty: 'intermediate'
    },
    {
      id: 'electric-field',
      title: '전기장 시각화',
      description: '전하 배치에 따른 전기장 시각화',
      difficulty: 'intermediate'
    },
    {
      id: 'wave-interference',
      title: '파동 간섭 시뮬레이터',
      description: '파동의 중첩과 간섭 패턴',
      difficulty: 'intermediate'
    },
    {
      id: 'thermodynamic-cycles',
      title: '열역학 사이클',
      description: '카르노, 오토, 디젤 사이클 시뮬레이션',
      difficulty: 'advanced'
    }
  ],

  prerequisites: ['calculus'],
  nextModules: ['physical-ai', 'robotics-manipulation'],

  resources: [
    { title: 'MIT OCW - Physics I', url: 'https://ocw.mit.edu/courses/physics/8-01sc-classical-mechanics-fall-2016/' },
    { title: 'Khan Academy - Physics', url: 'https://www.khanacademy.org/science/physics' },
    { title: 'Feynman Lectures on Physics', url: 'https://www.feynmanlectures.caltech.edu/' }
  ]
}
