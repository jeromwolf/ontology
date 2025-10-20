export const metadata = {
  title: '선형대수학 (Linear Algebra)',
  description: '벡터, 행렬, 선형변환의 기초부터 고급 응용까지',
  icon: '📐',
  category: 'foundations',
  difficulty: 'beginner',
  duration: '16시간',
  students: 892,
  rating: 4.8,
  color: 'from-blue-500 to-indigo-600',

  chapters: [
    {
      id: 'vectors-basics',
      title: 'Chapter 1: 벡터의 기초',
      description: '벡터의 정의, 연산, 내적과 외적',
      duration: '90분',
      objectives: [
        '벡터의 기하학적 의미 이해',
        '벡터 덧셈, 스칼라곱 계산',
        '내적과 외적의 응용'
      ]
    },
    {
      id: 'matrices',
      title: 'Chapter 2: 행렬과 행렬 연산',
      description: '행렬의 정의, 연산, 전치, 역행렬',
      duration: '100분',
      objectives: [
        '행렬 곱셈 이해',
        '역행렬 계산',
        '전치행렬의 성질'
      ]
    },
    {
      id: 'linear-systems',
      title: 'Chapter 3: 선형 시스템',
      description: '연립방정식, 가우스 소거법, LU 분해',
      duration: '120분',
      objectives: [
        '가우스 소거법으로 연립방정식 풀이',
        'LU 분해 이해',
        '행렬의 계수(Rank) 계산'
      ]
    },
    {
      id: 'vector-spaces',
      title: 'Chapter 4: 벡터 공간',
      description: '벡터 공간, 부분공간, 기저와 차원',
      duration: '110분',
      objectives: [
        '벡터 공간의 정의 이해',
        '기저와 차원 계산',
        '부분공간 판별'
      ]
    },
    {
      id: 'eigenvalues',
      title: 'Chapter 5: 고유값과 고유벡터',
      description: '고유값, 고유벡터, 대각화',
      duration: '130분',
      objectives: [
        '고유값과 고유벡터 계산',
        '행렬의 대각화',
        '고유값의 응용'
      ]
    },
    {
      id: 'orthogonality',
      title: 'Chapter 6: 직교성',
      description: '직교 벡터, 그람-슈미트, 직교 사영',
      duration: '100분',
      objectives: [
        '그람-슈미트 정규직교화',
        '직교 사영 계산',
        'QR 분해 이해'
      ]
    },
    {
      id: 'linear-transformations',
      title: 'Chapter 7: 선형변환',
      description: '선형변환의 정의, 행렬 표현, 커널과 상',
      duration: '120분',
      objectives: [
        '선형변환의 행렬 표현',
        '커널과 상 계산',
        '차원 정리 이해'
      ]
    },
    {
      id: 'svd',
      title: 'Chapter 8: 특이값 분해 (SVD)',
      description: 'SVD의 이론과 응용, 차원 축소',
      duration: '140분',
      objectives: [
        'SVD 계산',
        '주성분 분석(PCA)',
        '이미지 압축 응용'
      ]
    }
  ],

  simulators: [
    {
      id: 'vector-visualizer',
      title: '벡터 시각화 도구',
      description: '2D/3D 벡터 연산을 실시간으로 시각화',
      difficulty: 'beginner'
    },
    {
      id: 'matrix-calculator',
      title: '행렬 계산기',
      description: '행렬 연산, 역행렬, 고유값 계산',
      difficulty: 'intermediate'
    },
    {
      id: 'eigenvalue-explorer',
      title: '고유값 탐색기',
      description: '고유값과 고유벡터를 시각적으로 탐색',
      difficulty: 'intermediate'
    },
    {
      id: 'svd-decomposer',
      title: 'SVD 분해 도구',
      description: '특이값 분해를 통한 이미지 압축 실습',
      difficulty: 'advanced'
    },
    {
      id: 'gram-schmidt',
      title: '그람-슈미트 시뮬레이터',
      description: '정규직교 기저 생성 과정 시각화',
      difficulty: 'intermediate'
    },
    {
      id: 'linear-transformation-lab',
      title: '선형변환 실험실',
      description: '다양한 선형변환의 기하학적 효과 시각화',
      difficulty: 'intermediate'
    }
  ],

  prerequisites: [],
  nextModules: ['calculus', 'probability-statistics'],

  resources: [
    { title: 'Gilbert Strang - MIT Linear Algebra', url: 'https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/' },
    { title: '3Blue1Brown - Essence of Linear Algebra', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab' },
    { title: 'Khan Academy - Linear Algebra', url: 'https://www.khanacademy.org/math/linear-algebra' }
  ]
}
