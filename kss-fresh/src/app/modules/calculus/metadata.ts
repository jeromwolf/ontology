export const metadata = {
  title: '미적분학 (Calculus)',
  description: '극한, 미분, 적분의 기초부터 다변수 미적분까지',
  icon: '∫',
  category: 'foundations',
  difficulty: 'beginner',
  duration: '18시간',
  students: 1024,
  rating: 4.9,
  color: 'from-green-500 to-teal-600',

  chapters: [
    {
      id: 'limits',
      title: 'Chapter 1: 극한과 연속',
      description: '극한의 정의, 연속함수, 중간값 정리',
      duration: '100분',
      objectives: ['극한 계산', '연속성 판별', '무한대의 극한']
    },
    {
      id: 'derivatives',
      title: 'Chapter 2: 미분법',
      description: '도함수, 미분 규칙, 연쇄법칙',
      duration: '120분',
      objectives: ['기본 미분 공식', '연쇄법칙 적용', '음함수 미분']
    },
    {
      id: 'applications-derivatives',
      title: 'Chapter 3: 미분의 응용',
      description: '최적화, 평균값 정리, 곡선 스케칭',
      duration: '130분',
      objectives: ['극값 찾기', '로피탈 법칙', '곡선 분석']
    },
    {
      id: 'integration',
      title: 'Chapter 4: 적분법',
      description: '부정적분, 정적분, 미적분학의 기본 정리',
      duration: '140분',
      objectives: ['기본 적분 공식', '치환적분', '부분적분']
    },
    {
      id: 'applications-integration',
      title: 'Chapter 5: 적분의 응용',
      description: '넓이, 부피, 평균값',
      duration: '120분',
      objectives: ['회전체 부피', '곡선 길이', '물리적 응용']
    },
    {
      id: 'sequences-series',
      title: 'Chapter 6: 급수와 수열',
      description: '수열, 무한급수, 테일러 급수',
      duration: '150분',
      objectives: ['수렴성 판정', '테일러 전개', '매클로린 급수']
    },
    {
      id: 'multivariable',
      title: 'Chapter 7: 다변수 미적분',
      description: '편미분, 중적분, 그래디언트',
      duration: '160분',
      objectives: ['편미분 계산', '이중적분', '그래디언트 벡터']
    },
    {
      id: 'vector-calculus',
      title: 'Chapter 8: 벡터 미적분',
      description: '선적분, 면적분, 발산과 회전',
      duration: '140분',
      objectives: ['선적분 계산', '그린 정리', '스토크스 정리']
    }
  ],

  simulators: [
    {
      id: 'limit-calculator',
      title: '극한 계산기',
      description: '극한을 시각적으로 이해하고 계산',
      difficulty: 'beginner'
    },
    {
      id: 'derivative-visualizer',
      title: '미분 시각화 도구',
      description: '접선, 도함수를 실시간으로 시각화',
      difficulty: 'beginner'
    },
    {
      id: 'optimization-lab',
      title: '최적화 실험실',
      description: '극값 문제와 최적화 시뮬레이션',
      difficulty: 'intermediate'
    },
    {
      id: 'integral-calculator',
      title: '적분 계산기',
      description: '정적분을 리만 합으로 시각화',
      difficulty: 'intermediate'
    },
    {
      id: 'taylor-series-explorer',
      title: '테일러 급수 탐색기',
      description: '테일러 전개를 애니메이션으로 이해',
      difficulty: 'advanced'
    },
    {
      id: 'gradient-field',
      title: '그래디언트 필드 시각화',
      description: '다변수 함수의 그래디언트 벡터장',
      difficulty: 'advanced'
    }
  ],

  prerequisites: [],
  nextModules: ['linear-algebra', 'physics-fundamentals'],

  resources: [
    { title: '3Blue1Brown - Essence of Calculus', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr' },
    { title: 'MIT OpenCourseWare - Single Variable Calculus', url: 'https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/' },
    { title: 'Khan Academy - Calculus', url: 'https://www.khanacademy.org/math/calculus-1' }
  ]
}
