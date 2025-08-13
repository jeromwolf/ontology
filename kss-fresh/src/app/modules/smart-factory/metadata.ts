import { Module } from '@/types/module'

export const smartFactoryModule: Module = {
  id: 'smart-factory',
  name: 'Smart Factory & Industry 4.0 Complete Guide',
  nameKo: '스마트팩토리 & Industry 4.0 완전정복',
  description: '컨셉맵 기반 체계적 학습: Why → What → How 3단계로 마스터하는 차세대 스마트 제조업',
  version: '3.0.0',
  difficulty: 'intermediate',
  estimatedHours: 0, // 학습자별 맞춤 속도로 진행
  icon: '🏭',
  color: '#f59e0b', // amber-500
  
  prerequisites: ['기본적인 제조업 지식', 'IT 기초 이해', '데이터 분석 기초'],
  
  chapters: [
    // === PART 1: WHY - 스마트팩토리가 필요한 이유 (4개 챕터) ===
    {
      id: 'why-smart-factory',
      title: '스마트팩토리, 왜 필요한가?',
      description: '글로벌 제조업 패러다임 변화와 스마트팩토리 도입 필요성을 명확히 이해합니다',
      estimatedMinutes: 60,
      keywords: ['Industry4.0', '제조업위기', '디지털전환', '경쟁력', '생산성혁신'],
      learningObjectives: [
        '전 세계 제조업이 직면한 5가지 핵심 위기 (인건비 상승, 품질 요구 증가, 맞춤화 수요, 환경 규제, 인력 부족)',
        '독일 Industry 4.0, 미국 Advanced Manufacturing, 중국 Made in China 2025 비교 분석',
        '스마트팩토리 도입으로 달성 가능한 구체적 효과 (생산성 30% 향상, 불량률 50% 감소, 에너지 20% 절약)',
        '국내 제조업 현황과 스마트팩토리 도입 시급성 (삼성, LG, 현대차, 포스코 사례)',
        'ROI 관점에서 본 스마트팩토리 투자 타당성 분석'
      ]
    },
    {
      id: 'global-trends-cases',
      title: '글로벌 트렌드와 성공 사례',
      description: '세계 각국의 스마트팩토리 정책과 선도기업들의 성공 사례를 심층 분석합니다',
      estimatedMinutes: 200,
      keywords: ['벤치마킹', '글로벌사례', '정책분석', 'ROI성과', '디지털전환전략'],
      learningObjectives: [
        '독일 지멘스 암베르크 공장: 75% 자동화, 99.99% 품질 달성 비결',
        '미국 GE 브릴리언트 팩토리: 디지털 트윈 기반 생산성 30% 향상',
        '일본 도요타 TPS 2.0: 린 제조 + AI의 결합',
        '중국 하이얼 렌단헤이: 개인화 대량생산 혁신',
        '국내 성공 사례: 삼성전자 화성캠퍼스, LG디스플레이 파주공장, 현대차 울산공장 스마트화',
        '실패 사례 분석과 교훈: 디지털 전환 실패 요인 7가지'
      ]
    },
    {
      id: 'digital-transformation-roadmap',
      title: '제조업 디지털 전환 로드맵',
      description: '전통 제조업에서 스마트팩토리로의 단계적 전환 전략과 실행 계획을 수립합니다',
      estimatedMinutes: 170,
      keywords: ['디지털전환', '로드맵', '단계별전략', '변화관리', '투자계획'],
      learningObjectives: [
        '디지털 성숙도 평가 모델: Level 0(수작업) → Level 5(완전자율) 진단',
        '3단계 전환 로드맵: 1단계(가시화), 2단계(최적화), 3단계(자율화)',
        'Quick Win 전략: 즉시 효과를 볼 수 있는 8가지 디지털화 프로젝트',
        '투자 우선순위 결정 방법론: ROI vs 리스크 매트릭스 활용',
        '조직 변화 관리: 직원 저항 최소화와 디지털 리터러시 향상 전략',
        '정부 지원 정책 활용: 스마트공장 보급확산사업, 뿌리기업 스마트화 지원'
      ]
    },
    {
      id: 'business-case-roi',
      title: '비즈니스 케이스와 ROI 분석',
      description: '스마트팩토리 투자의 재무적 타당성을 분석하고 설득력 있는 비즈니스 케이스를 만듭니다',
      estimatedMinutes: 150,
      keywords: ['ROI분석', '비즈니스케이스', '투자타당성', '비용편익', '재무분석'],
      learningObjectives: [
        'TCO(Total Cost of Ownership) 계산: 초기 투자비 + 운영비 + 유지보수비',
        'ROI 측정 지표 15가지: OEE 향상, 품질비용 절감, 인건비 절약, 에너지 효율',
        '정량적 효과 vs 정성적 효과 구분과 가치 평가 방법',
        '투자 회수 기간 계산과 NPV, IRR 분석',
        '리스크 평가: 기술적 리스크, 조직적 리스크, 시장 리스크',
        'CFO를 설득하는 비즈니스 케이스 작성법과 프레젠테이션 기법'
      ]
    },

    // === PART 2: WHAT - 스마트팩토리의 구성 요소 (8개 챕터) ===
    {
      id: 'iot-sensor-networks',
      title: 'IoT & 센서 네트워크 시스템',
      description: '스마트팩토리의 신경망인 IoT 센서 네트워크 구축과 실시간 데이터 수집 시스템을 마스터합니다',
      estimatedMinutes: 220,
      keywords: ['IoT', '센서네트워크', '실시간데이터', '통신프로토콜', '엣지컴퓨팅'],
      learningObjectives: [
        '산업용 IoT 센서 30가지: 온도, 압력, 진동, 유량, 위치, 가스, 영상 센서 특성',
        '산업 통신 프로토콜 완전 가이드: MQTT, OPC-UA, Modbus, EtherCAT, PROFINET',
        '엣지 컴퓨팅 vs 클라우드 컴퓨팅: 지연시간, 대역폭, 보안 고려사항',
        '센서 데이터 품질 관리: 노이즈 필터링, 이상값 처리, 데이터 검증',
        '무선 vs 유선 네트워크 설계: Wi-Fi 6, 5G, LoRaWAN, Zigbee 선택 가이드',
        '실습: 아두이노/라즈베리파이로 IoT 센서 네트워크 구축'
      ]
    },
    {
      id: 'ai-data-analytics',
      title: 'AI & 빅데이터 분석 플랫폼',
      description: '제조 데이터를 활용한 AI 모델 개발과 예측 분석 시스템을 구축합니다',
      estimatedMinutes: 240,
      keywords: ['인공지능', '머신러닝', '빅데이터', '예측분석', '최적화알고리즘'],
      learningObjectives: [
        '제조 AI 5대 영역: 예측 유지보수, 품질 예측, 수요 예측, 공정 최적화, 에너지 관리',
        '시계열 분석과 이상 탐지: ARIMA, LSTM, Isolation Forest, One-Class SVM',
        '딥러닝 기반 이미지 분석: CNN을 활용한 제품 결함 분류 및 검출',
        '강화학습 기반 생산 스케줄링: Q-Learning으로 최적 작업 순서 결정',
        'MLOps 파이프라인: 모델 학습 → 검증 → 배포 → 모니터링 자동화',
        '실습: TensorFlow/PyTorch로 제조업 AI 모델 개발'
      ]
    },
    {
      id: 'robotics-automation',
      title: '로봇공학 & 생산 자동화',
      description: '산업용 로봇과 협동로봇을 활용한 유연하고 효율적인 생산 자동화 시스템을 구축합니다',
      estimatedMinutes: 210,
      keywords: ['산업용로봇', '협동로봇', 'ROS', 'AGV', '자동화시스템'],
      learningObjectives: [
        '로봇 종류별 특성: 관절형, 직교좌표형, 원통좌표형, SCARA, 델타로봇',
        '협동로봇(Cobot) 안전 설계: ISO 10218, ISO 15066 안전 표준',
        'ROS(Robot Operating System) 기초: 노드, 토픽, 서비스, 액션 프로그래밍',
        '로봇 비전 시스템: 2D/3D 비전 가이드, 피킹, 검사 애플리케이션',
        'AGV/AMR 시스템: 경로 계획, SLAM, 플릿 관리',
        '실습: ROS 시뮬레이터(Gazebo)로 로봇 제어 프로그래밍'
      ]
    },
    {
      id: 'digital-twin-simulation',
      title: '디지털 트윈 & 시뮬레이션',
      description: '실물 공장의 정확한 디지털 복제본을 만들고 다양한 시나리오를 시뮬레이션합니다',
      estimatedMinutes: 250,
      keywords: ['디지털트윈', '3D모델링', '시뮬레이션', 'Unity3D', '물리엔진'],
      learningObjectives: [
        '디지털 트윈 5단계 구축 프로세스: 설계 → 모델링 → 연결 → 분석 → 최적화',
        '3D 모델링 도구 활용: AutoCAD, SolidWorks, Blender를 활용한 공장 모델링',
        'Unity 3D/Unreal Engine 기반 실시간 시각화 시스템 개발',
        '물리 엔진 활용: 중력, 마찰, 충돌 등 실제 물리 현상 시뮬레이션',
        'What-if 시나리오 분석: 설비 고장, 수요 변화, 레이아웃 변경 시뮬레이션',
        '실습: Unity로 간단한 생산라인 디지털 트윈 제작'
      ]
    },
    {
      id: 'predictive-maintenance',
      title: '예측 유지보수 시스템',
      description: 'AI 기반 장비 상태 모니터링과 고장 예측을 통한 예방적 유지보수 체계를 구축합니다',
      estimatedMinutes: 200,
      keywords: ['예측유지보수', 'CBM', 'RUL예측', '진동분석', '열화상분석'],
      learningObjectives: [
        '유지보수 전략 비교: 사후보전 vs 예방보전 vs 예측보전 vs CBM',
        '센서 기반 상태 모니터링: 진동, 온도, 소음, 전류, 오일 분석',
        'RUL(Remaining Useful Life) 예측 모델: Weibull 분포, PHM, 딥러닝 기반',
        '고장 모드 분석: FMEA, FTA를 활용한 잠재 고장 원인 식별',
        '정비 일정 최적화: 생산 계획과 연동한 최적 정비 시점 결정',
        '실습: Python으로 베어링 고장 예측 모델 개발'
      ]
    },
    {
      id: 'quality-management-ai',
      title: 'AI 품질 관리 시스템',
      description: '컴퓨터 비전과 머신러닝을 활용한 자동 품질 검사와 SPC 기반 품질 관리를 구현합니다',
      estimatedMinutes: 190,
      keywords: ['품질관리', '컴퓨터비전', 'SPC', '자동검사', '6시그마'],
      learningObjectives: [
        '머신 비전 시스템 구성: 조명, 렌즈, 카메라, 이미지 처리 알고리즘',
        'AI 기반 결함 검출: YOLO, Faster R-CNN을 활용한 실시간 불량 탐지',
        'SPC(Statistical Process Control) 차트: Xbar-R, p-chart, c-chart 활용법',
        '6시그마 DMAIC 방법론을 활용한 품질 개선 프로세스',
        'OCR과 바코드/QR 인식: 제품 추적성(Traceability) 시스템 구축',
        '실습: OpenCV와 YOLO로 제품 결함 자동 분류 시스템 개발'
      ]
    },
    {
      id: 'mes-erp-integration',
      title: 'MES & ERP 시스템 통합',
      description: '제조실행시스템(MES)과 전사자원관리(ERP)의 완전한 통합을 통한 지능형 생산관리를 구현합니다',
      estimatedMinutes: 180,
      keywords: ['MES', 'ERP', '시스템통합', '생산계획', '실시간모니터링'],
      learningObjectives: [
        'MES 11대 기능 완전 가이드: 작업지시, 자원관리, 일정계획, 추적관리 등',
        'ERP-MES 인터페이스 설계: 마스터 데이터 동기화, 실시간 연동 아키텍처',
        'SAP, Oracle, Microsoft Dynamics 365 주요 ERP와 MES 연동 사례',
        '생산 스케줄링 최적화: APS(Advanced Planning & Scheduling) 알고리즘',
        '실시간 대시보드 설계: KPI 모니터링, 알람 시스템, 모바일 앱 연동',
        '실습: 오픈소스 MES(OpenMES) 설치 및 기본 설정'
      ]
    },
    {
      id: 'cybersecurity-standards',
      title: 'OT 보안 & 국제 표준',
      description: '스마트팩토리의 운영기술(OT) 보안과 국제 표준 준수를 통한 안전한 제조 환경을 구축합니다',
      estimatedMinutes: 160,
      keywords: ['OT보안', '사이버보안', 'IEC62443', 'ISO27001', '보안표준'],
      learningObjectives: [
        'OT vs IT 보안의 차이점: 가용성 우선 vs 기밀성 우선',
        '산업 제어 시스템 위협: Stuxnet, TRITON, 랜섬웨어 공격 사례 분석',
        'IEC 62443 보안 표준: 4개 시리즈, 7가지 보안 레벨 이해',
        '네트워크 분할(Network Segmentation): DMZ, 방화벽, VPN 설계',
        '보안 모니터링: SIEM, 이상 행위 탐지, 침입 차단 시스템',
        '실습: 가상환경에서 PLC 보안 설정 및 모니터링'
      ]
    },

    // === PART 3: HOW - 스마트팩토리 구현 방법론 (4개 챕터) ===
    {
      id: 'implementation-methodology',
      title: '스마트팩토리 구현 방법론',
      description: '성공적인 스마트팩토리 구축을 위한 체계적 방법론과 프로젝트 관리 기법을 마스터합니다',
      estimatedMinutes: 200,
      keywords: ['구현방법론', '프로젝트관리', '위험관리', '성공요인', '실행전략'],
      learningObjectives: [
        '스마트팩토리 구축 7단계 방법론: 현황분석 → 전략수립 → 설계 → 구축 → 테스트 → 운영 → 개선',
        '애자일 vs 워터폴: 스마트팩토리 프로젝트에 적합한 관리 방식',
        'PoC(Proof of Concept) 기획과 실행: 파일럿 프로젝트 성공 전략',
        '변화관리와 조직 역량 강화: 교육 계획, 저항 관리, 문화 변화',
        '위험 관리: 기술적 위험, 일정 위험, 비용 위험 식별과 대응',
        '실습: 가상 기업의 스마트팩토리 구축 프로젝트 기획서 작성'
      ]
    },
    {
      id: 'system-architecture-design',
      title: '시스템 아키텍처 설계',
      description: '확장 가능하고 안정적인 스마트팩토리 IT/OT 통합 아키텍처를 설계합니다',
      estimatedMinutes: 220,
      keywords: ['시스템아키텍처', 'IT/OT통합', '클라우드', '엣지컴퓨팅', '플랫폼설계'],
      learningObjectives: [
        'IT/OT 융합 아키텍처: Purdue Model, RAMI 4.0 참조 모델',
        '클라우드 vs 온프레미스 vs 하이브리드: 제조업 특성에 따른 선택 가이드',
        '엣지-클라우드 연동 설계: 실시간 처리 vs 배치 처리 최적 분배',
        '데이터 레이크 vs 데이터 웨어하우스: 제조 빅데이터 저장 전략',
        'API 설계와 마이크로서비스 아키텍처: 확장성과 유지보수성 확보',
        '실습: AWS/Azure를 활용한 스마트팩토리 클라우드 아키텍처 설계'
      ]
    },
    {
      id: 'change-management-training',
      title: '조직 변화관리 & 인력 양성',
      description: '스마트팩토리 전환 과정에서 발생하는 조직 변화를 성공적으로 관리하고 디지털 인재를 양성합니다',
      estimatedMinutes: 170,
      keywords: ['변화관리', '인력양성', '디지털리터러시', '조직문화', '교육훈련'],
      learningObjectives: [
        'Kotter의 8단계 변화관리 모델을 스마트팩토리에 적용',
        '디지털 역량 진단과 개인별 맞춤 교육 계획 수립',
        '세대별 차별화 교육: 베이비부머, X세대, 밀레니얼, Z세대',
        '저항 요인 분석과 대응 전략: 일자리 불안, 기술 두려움, 업무 변화',
        '디지털 리더십과 문화 혁신: 실패를 허용하는 학습 조직 구축',
        '실습: 조직 변화관리 계획서 작성 및 교육 프로그램 설계'
      ]
    },
    {
      id: 'future-outlook-strategy',
      title: '미래 전망과 고도화 전략',
      description: '5G, 메타버스, 탄소중립 등 미래 기술 트렌드와 스마트팩토리의 진화 방향을 전망합니다',
      estimatedMinutes: 150,
      keywords: ['미래전망', '5G', '메타버스', '탄소중립', '지속가능제조'],
      learningObjectives: [
        '5G 기반 초연결 스마트팩토리: 네트워크 슬라이싱, URLLC 활용',
        '메타버스 팩토리: VR/AR 기반 원격 작업, 가상 교육, 디지털 협업',
        '탄소중립과 그린 팩토리: 에너지 효율화, 순환경제, ESG 경영',
        'AI의 진화: GPT, 생성형 AI의 제조업 적용 가능성',
        '완전 자율 팩토리(Lights-Out Factory): 무인화 기술과 한계',
        '실습: 2030년 스마트팩토리 비전 제시 및 로드맵 수립'
      ]
    }
  ],
  
  simulators: [
    // === 실제 구현된 시뮬레이터 ===
    {
      id: 'digital-twin-factory',
      name: '디지털 트윈 팩토리',
      description: '3D 가상 공장과 실시간 시뮬레이션 환경 - 4가지 운영 시나리오 체험',
      component: 'DigitalTwinFactory'
    },
    {
      id: 'predictive-maintenance-lab',
      name: '예측 정비 실험실',
      description: '실시간 센서 데이터로 장비 고장을 예측하고 RUL을 계산하는 AI 시뮬레이터',
      component: 'PredictiveMaintenanceLab'
    },
    {
      id: 'production-line-monitor',
      name: '생산 라인 모니터',
      description: '실시간 생산 라인 모니터링과 OEE(Overall Equipment Effectiveness) 분석',
      component: 'ProductionLineMonitor'
    },
    {
      id: 'quality-control-vision',
      name: '품질 관리 비전',
      description: 'AI 컴퓨터 비전으로 제품 결함을 실시간 감지하고 분류하는 시뮬레이터',
      component: 'QualityControlVision'
    },
    {
      id: 'physics-simulation',
      name: '물리 시뮬레이션 실험실',
      description: '제조 현장의 물리 현상(중력, 마찰, 충돌)을 실시간으로 시뮬레이션',
      component: 'PhysicsSimulation'
    },
    {
      id: 'bearing-failure-prediction',
      name: '베어링 고장 예측 AI Lab',
      description: '진동 데이터 기반 RUL 예측과 주파수 분석을 통한 고장 진단',
      component: 'BearingFailurePrediction'
    },
    {
      id: 'spc-control-system',
      name: 'SPC 통계 공정 관리',
      description: 'X-bar R 관리도와 공정능력(Cp, Cpk) 분석으로 품질 모니터링',
      component: 'SPCControlSystem'
    },
    {
      id: 'mes-erp-dashboard',
      name: 'MES/ERP 통합 대시보드',
      description: '생산, 재고, 품질, 재무 정보를 실시간으로 통합 모니터링',
      component: 'MESERPDashboard'
    },
    {
      id: 'smart-factory-ecosystem',
      name: '스마트팩토리 생태계 맵',
      description: '전체 시스템 구성요소와 데이터 흐름을 인터랙티브 관계도로 시각화',
      component: 'SmartFactoryEcosystem'
    }
  ],

  // === 실무 도구 및 템플릿 ===
  tools: [
    {
      id: 'maturity-assessment',
      name: '디지털 성숙도 진단도구',
      description: '5단계 성숙도 모델로 현재 수준 진단 및 개선 방향 제시',
      url: '/modules/smart-factory/tools/maturity-assessment'
    },
    {
      id: 'vendor-selection-guide',
      name: '솔루션 업체 선정 가이드',
      description: '100개 국내외 스마트팩토리 솔루션 업체 비교 분석',
      url: '/modules/smart-factory/tools/vendor-guide'
    },
    {
      id: 'project-templates',
      name: '프로젝트 템플릿 라이브러리',
      description: '제안서, 설계서, 테스트 계획서 등 30개 실무 템플릿',
      url: '/modules/smart-factory/tools/templates'
    },
    {
      id: 'case-study-database',
      name: '사례 연구 데이터베이스',
      description: '국내외 200개 스마트팩토리 구축 사례와 ROI 분석',
      url: '/modules/smart-factory/tools/case-studies'
    }
  ],

  // === 학습 경로 === (타입 에러로 인해 일시 주석처리)
  /*learningPaths: [
    {
      id: 'executive-path',
      name: '경영진 의사결정 코스',
      description: 'Why 중심의 전략적 이해 (1주)',
      chapters: ['why-smart-factory', 'global-trends-cases', 'business-case-roi'],
      estimatedWeeks: 1
    },
    {
      id: 'engineer-path',
      name: '엔지니어 기술 마스터 코스',
      description: 'What 중심의 기술적 깊이 (3주)',
      chapters: ['iot-sensor-networks', 'ai-data-analytics', 'robotics-automation', 'digital-twin-simulation', 'predictive-maintenance'],
      estimatedWeeks: 3
    },
    {
      id: 'pm-path',
      name: '프로젝트 매니저 실무 코스',
      description: 'How 중심의 구현 방법론 (2주)',
      chapters: ['implementation-methodology', 'system-architecture-design', 'change-management-training'],
      estimatedWeeks: 2
    },
    {
      id: 'complete-path',
      name: '스마트팩토리 전문가 마스터 코스',
      description: 'Why → What → How 완전 정복 (4주)',
      chapters: [], // 전체 16개 챕터
      estimatedWeeks: 4
    }
  ],*/

  // === 평가 시스템 === (타입 에러로 인해 일시 주석처리)
  /*assessments: [
    {
      id: 'strategic-understanding',
      name: '전략적 이해 평가',
      description: 'Why 파트 이해도 평가 (50문항)',
      passingScore: 75,
      timeLimit: 60
    },
    {
      id: 'technical-competency',
      name: '기술 역량 평가',
      description: 'What 파트 기술 지식 평가 (100문항)',
      passingScore: 80,
      timeLimit: 120
    },
    {
      id: 'implementation-skills',
      name: '구현 실무 평가',
      description: 'How 파트 실무 역량 평가 (실습 프로젝트)',
      passingScore: 85,
      timeLimit: 240
    }
  ],*/

  // === 실습 프로젝트 === (타입 에러로 인해 일시 주석처리)
  /*projects: [
    {
      id: 'factory-digital-transformation',
      name: '기업 스마트팩토리 전환 프로젝트',
      description: '실제 기업 사례를 바탕으로 한 완전한 디지털 전환 계획 수립',
      deliverables: ['현황분석보고서', '전환전략수립', 'ROI분석', '구현로드맵', '위험관리계획']
    },
    {
      id: 'predictive-maintenance-system',
      name: '예측 유지보수 시스템 개발',
      description: '실제 장비 데이터를 활용한 AI 기반 고장 예측 시스템 구축',
      deliverables: ['데이터분석', '예측모델개발', '시스템구현', '성능평가', '운영가이드']
    },
    {
      id: 'digital-twin-prototype',
      name: '디지털 트윈 프로토타입 개발',
      description: 'Unity 3D를 활용한 생산라인 디지털 트윈 제작',
      deliverables: ['3D모델링', '시뮬레이션로직', 'UI/UX설계', '실시간연동', '데모영상']
    }
  ]*/
}

export const getChapter = (chapterId: string) => {
  return smartFactoryModule.chapters.find(chapter => chapter.id === chapterId)
}

export const getNextChapter = (currentChapterId: string) => {
  const currentIndex = smartFactoryModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex < smartFactoryModule.chapters.length - 1 ? smartFactoryModule.chapters[currentIndex + 1] : undefined
}

export const getPrevChapter = (currentChapterId: string) => {
  const currentIndex = smartFactoryModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex > 0 ? smartFactoryModule.chapters[currentIndex - 1] : undefined
}