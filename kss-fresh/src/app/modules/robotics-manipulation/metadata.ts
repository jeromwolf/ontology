export const moduleMetadata = {
  id: 'robotics-manipulation',
  title: 'Robotics & Manipulation',
  description: '산업용 로봇부터 협동 로봇까지 - 로봇 매니퓰레이션의 모든 것',
  category: '로보틱스',
  difficulty: 'advanced' as const,
  duration: '30시간',
  chapters: [
    {
      id: 1,
      title: '로봇 공학 기초',
      description: '로봇의 구조, 좌표계, 자유도 이해',
      duration: '3시간',
      learningObjectives: [
        '로봇의 정의와 분류 (산업용, 서비스, 협동로봇)',
        '로봇 좌표계와 변환 행렬 (DH Parameters)',
        '자유도(DOF)와 작업 공간(Workspace) 개념',
        '로봇 관절 종류: Revolute, Prismatic, Spherical',
        '로봇 매니퓰레이터 아키텍처 (직교, SCARA, 다관절)'
      ]
    },
    {
      id: 2,
      title: '순기구학 (Forward Kinematics)',
      description: '관절 각도로부터 엔드이펙터 위치 계산',
      duration: '4시간',
      learningObjectives: [
        'DH (Denavit-Hartenberg) 파라미터 설정',
        '동차 변환 행렬 (Homogeneous Transformation)',
        '순기구학 방정식 유도와 계산',
        '다관절 로봇의 순기구학 구현',
        '작업 공간 시각화와 특이점(Singularity) 분석'
      ]
    },
    {
      id: 3,
      title: '역기구학 (Inverse Kinematics)',
      description: '목표 위치로부터 관절 각도 계산',
      duration: '4시간',
      learningObjectives: [
        '역기구학 문제의 해석적 해법 (Analytical Solution)',
        '기하학적 접근법과 대수적 접근법',
        '다중 해(Multiple Solutions)와 선택 전략',
        '수치 해법: Jacobian 기반 반복법',
        '특이점 회피와 안전 영역 설정'
      ]
    },
    {
      id: 4,
      title: '경로 계획 (Path Planning)',
      description: '충돌 없는 안전한 경로 생성',
      duration: '4시간',
      learningObjectives: [
        'Configuration Space (C-Space) 개념',
        'RRT (Rapidly-exploring Random Tree) 알고리즘',
        'RRT* 최적화와 경로 스무딩',
        'A* 기반 그리드 경로 계획',
        'Dynamic Window Approach (DWA) 실시간 회피'
      ]
    },
    {
      id: 5,
      title: '궤적 생성 (Trajectory Generation)',
      description: '부드러운 로봇 동작을 위한 궤적 설계',
      duration: '3시간',
      learningObjectives: [
        '점대점 궤적: 3차/5차 다항식 보간',
        'Trapezoidal Velocity Profile 설계',
        'S-Curve 프로파일과 저크(Jerk) 제한',
        '다관절 동기화와 시간 최적 궤적',
        'Spline 기반 부드러운 경로 생성'
      ]
    },
    {
      id: 6,
      title: '그리퍼와 조작 (Grasping)',
      description: '물체 파지와 조작 기술',
      duration: '4시간',
      learningObjectives: [
        '그리퍼 종류: Parallel, Three-finger, Suction',
        '파지 안정성과 Force Closure 개념',
        '접촉 모델링과 마찰력 계산',
        'Visual Servoing: 비전 기반 물체 인식',
        'Pick-and-Place 작업 시퀀스 설계'
      ]
    },
    {
      id: 7,
      title: 'ROS2 실전 프로그래밍',
      description: 'Robot Operating System 2 마스터',
      duration: '5시간',
      learningObjectives: [
        'ROS2 아키텍처: Node, Topic, Service, Action',
        'MoveIt2로 모션 계획 구현',
        'Gazebo 시뮬레이션 환경 구축',
        'URDF/XACRO로 로봇 모델링',
        'RViz2 시각화와 디버깅 도구'
      ]
    },
    {
      id: 8,
      title: '협동 로봇과 산업 응용',
      description: '인간-로봇 협업과 실전 응용',
      duration: '3시간',
      learningObjectives: [
        '협동 로봇(Cobot)의 안전 기준과 설계',
        '충돌 감지와 힘/토크 제어',
        '조립 작업 자동화 (Peg-in-Hole)',
        '빈 피킹(Bin Picking)과 3D 비전',
        '산업 현장 사례: 자동차, 전자, 물류'
      ]
    }
  ],
  simulators: [
    {
      id: 'forward-kinematics-lab',
      title: '순기구학 실험실',
      description: '로봇 관절 각도 조작과 엔드이펙터 위치 시각화',
      difficulty: 'beginner'
    },
    {
      id: 'inverse-kinematics-solver',
      title: '역기구학 솔버',
      description: '목표 위치 설정과 관절 각도 자동 계산',
      difficulty: 'intermediate'
    },
    {
      id: 'path-planning-visualizer',
      title: '경로 계획 시각화',
      description: 'RRT/A* 알고리즘으로 장애물 회피 경로 생성',
      difficulty: 'intermediate'
    },
    {
      id: 'trajectory-generator',
      title: '궤적 생성기',
      description: '다항식/스플라인 기반 부드러운 궤적 설계',
      difficulty: 'intermediate'
    },
    {
      id: 'gripper-force-simulator',
      title: '그리퍼 힘 시뮬레이터',
      description: '파지 안정성과 접촉력 계산',
      difficulty: 'advanced'
    },
    {
      id: 'pick-and-place-lab',
      title: 'Pick-and-Place 실습',
      description: '완전한 픽앤플레이스 작업 시퀀스 구현',
      difficulty: 'advanced'
    }
  ],
  prerequisites: ['선형대수', '미적분학', 'Python 프로그래밍', '기초 물리학'],
  outcomes: [
    '산업용 로봇 프로그래밍 능력 습득',
    'ROS2 기반 로봇 시스템 개발 경험',
    '로봇 매니퓰레이션 알고리즘 구현 능력',
    '실전 로봇 응용 프로젝트 완성'
  ],
  tools: [
    'ROS2', 'MoveIt2', 'Gazebo', 'RViz2',
    'PyBullet', 'NumPy', 'SciPy',
    'OpenCV', 'PCL (Point Cloud Library)'
  ]
}
