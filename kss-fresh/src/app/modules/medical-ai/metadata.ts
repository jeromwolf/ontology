export const moduleMetadata = {
  id: 'medical-ai',
  title: 'Medical AI',
  description: '의료 영상 분석, 진단 보조, 신약 개발 AI 기술',
  icon: '🏥',
  gradient: 'from-pink-500 to-red-500',
  category: 'Healthcare & AI',
  difficulty: 'Advanced',
  estimatedHours: 15,
  chapters: [
    {
      id: 'introduction',
      title: 'Medical AI 기초',
      description: '의료 AI의 역사, 현황, 그리고 미래',
      estimatedMinutes: 90,
    },
    {
      id: 'medical-imaging',
      title: '의료 영상 분석',
      description: 'X-ray, CT, MRI 영상 AI 분석 기술',
      estimatedMinutes: 120,
    },
    {
      id: 'diagnosis-support',
      title: 'AI 진단 보조 시스템',
      description: '질병 진단과 예후 예측 AI 모델',
      estimatedMinutes: 120,
    },
    {
      id: 'drug-discovery',
      title: 'AI 신약 개발',
      description: '분자 설계부터 임상시험까지',
      estimatedMinutes: 150,
    },
    {
      id: 'clinical-nlp',
      title: '임상 NLP',
      description: '전자의무기록 분석과 의학 문헌 마이닝',
      estimatedMinutes: 100,
    },
    {
      id: 'personalized-medicine',
      title: '정밀 의료',
      description: '유전체 데이터 기반 개인화 치료',
      estimatedMinutes: 120,
    },
    {
      id: 'regulation-ethics',
      title: '규제와 윤리',
      description: 'FDA 승인, HIPAA, 의료 AI 윤리',
      estimatedMinutes: 90,
    },
    {
      id: 'real-world-applications',
      title: '실전 프로젝트',
      description: '실제 의료 AI 시스템 구축 사례',
      estimatedMinutes: 150,
    },
  ],
  simulators: [
    {
      id: 'chest-xray-classifier',
      title: '흉부 X-ray 분류기',
      description: 'CNN 기반 폐렴/결핵 진단 모델',
    },
    {
      id: 'tumor-segmentation',
      title: '종양 영역 분할',
      description: 'U-Net으로 종양 영역 정밀 추출',
    },
    {
      id: 'ecg-anomaly-detector',
      title: 'ECG 이상 탐지',
      description: '심전도 신호 분석 및 부정맥 탐지',
    },
    {
      id: 'molecule-generator',
      title: '약물 분자 생성기',
      description: 'GAN/Transformer 기반 신약 후보 생성',
    },
    {
      id: 'clinical-ner',
      title: '임상 개체명 인식',
      description: '의무기록에서 질병/약물/증상 추출',
    },
    {
      id: 'survival-predictor',
      title: '생존율 예측 모델',
      description: 'Cox 모델 + ML로 환자 예후 예측',
    },
  ],
};
