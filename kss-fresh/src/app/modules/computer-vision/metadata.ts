export const moduleMetadata = {
  title: "Computer Vision",
  description: "컴퓨터 비전의 핵심 개념과 최신 기술을 학습하고 실습합니다",
  icon: "Eye",
  color: "teal",
  chapters: [
    {
      id: "cv-basics",
      title: "컴퓨터 비전 기초",
      description: "디지털 이미지의 표현과 기본 처리 기법",
      topics: [
        "디지털 이미지의 이해",
        "픽셀과 색상 공간",
        "이미지 변환과 필터링",
        "히스토그램과 통계"
      ]
    },
    {
      id: "image-processing",
      title: "이미지 처리",
      description: "고급 이미지 처리 기법과 향상 알고리즘",
      topics: [
        "공간 도메인 필터링",
        "주파수 도메인 처리",
        "모폴로지 연산",
        "노이즈 제거와 향상"
      ]
    },
    {
      id: "feature-detection",
      title: "특징점 검출",
      description: "이미지에서 중요한 특징을 찾고 매칭하는 기법",
      topics: [
        "코너와 엣지 검출",
        "SIFT, SURF, ORB",
        "특징점 매칭",
        "이미지 정합"
      ]
    },
    {
      id: "deep-learning-vision",
      title: "딥러닝 비전",
      description: "CNN과 최신 딥러닝 모델을 활용한 비전 처리",
      topics: [
        "CNN 아키텍처",
        "전이 학습",
        "Vision Transformer",
        "생성 모델 (GAN, Diffusion)"
      ]
    },
    {
      id: "2d-to-3d",
      title: "2D to 3D 변환",
      description: "2D 이미지에서 3D 정보를 추출하는 기술",
      topics: [
        "스테레오 비전",
        "깊이 추정",
        "3D 재구성",
        "포인트 클라우드"
      ]
    },
    {
      id: "object-detection-tracking",
      title: "객체 탐지와 추적",
      description: "실시간 객체 검출과 추적 알고리즘",
      topics: [
        "YOLO, R-CNN 계열",
        "실시간 객체 추적",
        "다중 객체 추적",
        "행동 인식"
      ]
    },
    {
      id: "face-recognition",
      title: "얼굴 인식",
      description: "얼굴 검출, 인식 및 분석 기술",
      topics: [
        "얼굴 검출 알고리즘",
        "얼굴 특징 추출",
        "감정 인식",
        "나이/성별 추정"
      ]
    },
    {
      id: "real-time-applications",
      title: "실시간 비전 응용",
      description: "실제 환경에서의 컴퓨터 비전 응용",
      topics: [
        "증강 현실 (AR)",
        "자율주행 비전",
        "의료 영상 분석",
        "산업용 비전 검사"
      ]
    }
  ],
  simulators: [
    {
      id: "2d-to-3d-converter",
      title: "2D to 3D Converter",
      description: "2D 이미지에서 3D 모델을 생성하는 시뮬레이터",
      icon: "Box"
    },
    {
      id: "object-detection-lab",
      title: "Object Detection Lab",
      description: "실시간 객체 검출과 추적을 실습하는 환경",
      icon: "Scan"
    },
    {
      id: "face-recognition-system",
      title: "Face Recognition System",
      description: "얼굴 인식과 분석 시스템 구현",
      icon: "UserCheck"
    },
    {
      id: "image-enhancement-studio",
      title: "Image Enhancement Studio",
      description: "이미지 향상과 필터링 도구 모음",
      icon: "ImagePlus"
    },
    {
      id: "pose-estimation-tracker",
      title: "Pose Estimation Tracker",
      description: "인체 포즈 추정과 동작 분석 시스템",
      icon: "User"
    }
  ]
};