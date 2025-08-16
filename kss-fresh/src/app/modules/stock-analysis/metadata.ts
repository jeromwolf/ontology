import { Module } from '@/types/module'

export const stockAnalysisModule: Module = {
  id: 'stock-analysis',
  name: 'Professional Investment Analysis',
  nameKo: '전문 투자 분석',
  description: '금융시장의 구조적 이해부터 퀀트 전략까지, 실무 중심의 체계적인 투자 분석 전문가 양성 프로그램',
  version: '3.0.0',
  difficulty: 'advanced',
  estimatedHours: 240, // 240시간 (6개월 전문 과정)
  icon: '📊',
  color: '#1e293b',
  
  prerequisites: ['경제학/재무관리 기초', '통계학 및 확률론 이해', 'Excel/Python 데이터 분석 능력', '금융시장 기본 지식'],
  
  chapters: [
    // === PART 1: 투자 기초 및 시장 이해 (4개 챕터) ===
    {
      id: 'market-fundamentals',
      title: '글로벌 금융시장의 이해',
      description: '전 세계 주식시장의 구조, 참여자, 거래 시스템을 완전히 파악하고 투자의 기초를 확립합니다',
      estimatedMinutes: 300,
      keywords: ['주식시장', '증권거래소', '시장참여자', '거래시스템', 'IPO', '공매도'],
      learningObjectives: [
        '글로벌 주요 증권거래소(NYSE, NASDAQ, 런던, 도쿄, 홍콩) 특징과 차이점',
        '시장 참여자별 역할과 영향력 (기관투자자, 개인투자자, HFT)',
        '주문 유형과 체결 시스템 완전 마스터 (시장가, 지정가, 조건부 주문)',
        'IPO, 증자, 액면분할 등 기업 행위가 주가에 미치는 영향',
        '공매도와 대차거래의 메커니즘과 리스크'
      ]
    },
    {
      id: 'investment-psychology',
      title: '투자 심리학 및 행동재무학',
      description: '투자자의 심리적 편향과 군중 심리를 이해하고 합리적 투자 의사결정을 위한 프레임워크를 구축합니다',
      estimatedMinutes: 240,
      keywords: ['행동재무학', '인지편향', '손실회피', '군중심리', '투자심리'],
      learningObjectives: [
        '주요 인지 편향 15가지와 투자에 미치는 영향 (확증편향, 손실회피, 과신편향 등)',
        '버블과 패닉의 심리적 메커니즘과 역사적 사례 분석',
        '감정적 투자 결정을 방지하는 체계적 접근법',
        '투자 일지 작성과 자기성찰을 통한 심리적 훈련',
        '스트레스 관리와 장기 투자 마인드셋 구축'
      ]
    },
    {
      id: 'economic-indicators',
      title: '거시경제 지표와 시장 사이클',
      description: '거시경제 지표를 통해 시장 사이클을 이해하고 경제 상황에 따른 투자 전략을 수립합니다',
      estimatedMinutes: 280,
      keywords: ['거시경제', '금리정책', 'GDP', '인플레이션', '시장사이클'],
      learningObjectives: [
        '핵심 경제지표 25개와 주식시장에 미치는 영향 분석',
        '중앙은행 정책(금리, QE)이 자산가격에 미치는 메커니즘',
        '경기 사이클별 섹터 로테이션 전략',
        '인플레이션과 디플레이션 환경에서의 투자 전략',
        '지정학적 리스크와 글로벌 경제 상황 분석'
      ]
    },
    {
      id: 'investment-vehicles',
      title: '투자 수단의 이해',
      description: '주식, 채권, ETF, 파생상품 등 다양한 투자 수단의 특징과 활용법을 마스터합니다',
      estimatedMinutes: 320,
      keywords: ['주식', '채권', 'ETF', '파생상품', 'REIT', '원자재'],
      learningObjectives: [
        '개별 주식 vs ETF vs 뮤추얼펀드의 장단점과 선택 기준',
        '채권의 종류와 금리 리스크, 신용 리스크 분석',
        '옵션과 선물의 기본 원리와 헤징 전략',
        'REIT, 원자재, 통화 투자의 포트폴리오 내 역할',
        '대체투자(사모펀드, 헤지펀드, 크라우드펀딩) 이해'
      ]
    },
    {
      id: 'global-brokerage-accounts',
      title: '해외 증권사 계좌 개설',
      description: '미국, 일본, 중국, 유럽 등 주요 시장의 증권사 계좌 개설 방법과 실무 노하우를 완벽 가이드합니다',
      estimatedMinutes: 240,
      keywords: ['해외증권사', '계좌개설', 'IB', 'Charles Schwab', 'W-8BEN', '환전'],
      learningObjectives: [
        '미국 주요 증권사(IB, Charles Schwab, TD Ameritrade) 계좌 개설 절차',
        '세금 서류(W-8BEN) 작성과 세금 협약 적용 방법',
        '효율적인 환전과 송금 방법 (전신송금 vs 증권사 환전)',
        '일본, 중국, 유럽 시장 접근을 위한 증권사 선택',
        '해외 증권 계좌 관리와 세금 신고 요령'
      ]
    },
    {
      id: 'global-sectors-understanding',
      title: '글로벌 섹터 이해',
      description: '미국 GICS 11개 섹터를 중심으로 글로벌 산업 분류와 섹터별 특성, 대표 기업을 심층 분석합니다',
      estimatedMinutes: 320,
      keywords: ['GICS', '섹터분석', 'Technology', 'Healthcare', 'Financials', '산업분류'],
      learningObjectives: [
        'GICS(Global Industry Classification Standard) 11개 섹터 완벽 이해',
        '섹터별 경기 민감도와 사이클 특성 분석',
        '미국, 유럽, 아시아 주요 시장의 섹터별 대표 기업',
        '섹터 로테이션 전략과 경제 사이클별 투자 포인트',
        '신흥 섹터(클린에너지, 우주항공, 사이버보안) 트렌드 분석'
      ]
    },
    {
      id: 'gaap-vs-ifrs',
      title: 'GAAP vs IFRS 회계기준',
      description: '미국 회계기준(GAAP)과 국제회계기준(IFRS)의 차이를 이해하고 해외기업 재무제표를 정확히 분석합니다',
      estimatedMinutes: 280,
      keywords: ['GAAP', 'IFRS', '회계기준', '재무제표', '손익인식', '자산평가'],
      learningObjectives: [
        'GAAP과 IFRS의 주요 차이점 10가지 완벽 정리',
        '매출 인식, 재고자산 평가, 무형자산 처리의 차이',
        '리스 회계, 연구개발비, 금융상품 회계 처리 비교',
        '실제 기업 사례로 보는 회계기준 차이의 영향',
        '해외기업 재무제표 읽기와 비교 분석 실무'
      ]
    },

    // === PART 2: 기본적 분석 마스터 (5개 챕터) ===
    {
      id: 'financial-statements-deep',
      title: '재무제표 완전 분석',
      description: '3대 재무제표를 전문가 수준으로 분석하고 숨겨진 리스크와 기회를 발견하는 능력을 기릅니다',
      estimatedMinutes: 360,
      keywords: ['손익계산서', '대차대조표', '현금흐름표', '주석', '감사의견'],
      learningObjectives: [
        '손익계산서 심화 분석: 매출 인식 정책, 비용 구조, 영업외 손익',
        '대차대조표 완전 해부: 자산의 질, 부채 구조, 우발채무 분석',
        '현금흐름표 마스터: 영업, 투자, 재무 활동별 현금흐름 해석',
        '재무제표 주석과 감사 의견서 읽기',
        '회계 조작 탐지와 적색 경고신호 15가지'
      ]
    },
    {
      id: 'valuation-methods',
      title: '기업가치 평가 방법론',
      description: 'DCF, 상대가치, 자산가치 등 다양한 밸류에이션 기법을 실무에 적용하는 전문성을 기릅니다',
      estimatedMinutes: 380,
      keywords: ['DCF', '상대가치', 'PER', 'PBR', 'EV/EBITDA', '잔여이익모델'],
      learningObjectives: [
        'DCF 모델 구축: 현금흐름 예측, 할인율 산정, 터미널 밸류 계산',
        '상대가치 평가: 멀티플 선택, 비교기업 분석, 조정 방법',
        '섹터별 특화 지표: REIT(FFO), 은행(PBR), 통신(EV/EBITDA)',
        '성장기업 vs 가치주 vs 배당주 평가 방법론',
        '시나리오 분석과 몬테카를로 시뮬레이션 활용'
      ]
    },
    {
      id: 'industry-analysis',
      title: '산업 분석과 경쟁 구조',
      description: '포터의 5가지 경쟁요인을 활용한 산업 분석과 기업의 경쟁 우위 평가 방법을 학습합니다',
      estimatedMinutes: 320,
      keywords: ['산업분석', '경쟁우위', '포터5Forces', '밸류체인', '해자'],
      learningObjectives: [
        '포터의 5가지 힘을 이용한 산업 매력도 분석',
        '기업의 경쟁 우위 요소와 지속가능성 평가',
        '산업 생명주기별 투자 전략 (도입기, 성장기, 성숙기, 쇠퇴기)',
        '디지털 전환이 전통 산업에 미치는 영향 분석',
        'ESG 요소가 산업과 기업가치에 미치는 영향'
      ]
    },
    {
      id: 'financial-ratios-advanced',
      title: '고급 재무비율 분석',
      description: '70개 이상의 재무비율을 마스터하고 기업의 수익성, 효율성, 안전성을 종합 평가합니다',
      estimatedMinutes: 300,
      keywords: ['재무비율', '수익성', '효율성', '안전성', '성장성', '벤치마킹'],
      learningObjectives: [
        '수익성 지표 15개: ROE, ROA, ROI, 영업이익률, 순이익률 등',
        '효율성 지표 15개: 자산회전율, 재고회전율, 매출채권회전율 등',
        '안전성 지표 15개: 부채비율, 유동비율, 이자보상배수 등',
        '성장성 지표 10개: 매출성장률, EPS성장률, 배당성장률 등',
        '동종업계 벤치마킹과 시계열 분석을 통한 종합 평가'
      ]
    },
    {
      id: 'quality-investing',
      title: '퀄리티 투자와 ESG 분석',
      description: '지속가능한 경쟁우위를 가진 우량기업을 발굴하고 ESG 요소를 투자에 통합하는 방법을 학습합니다',
      estimatedMinutes: 280,
      keywords: ['퀄리티투자', 'ESG', '지속가능성', '거버넌스', '해자'],
      learningObjectives: [
        '워렌 버핏의 퀄리티 투자 철학과 실제 적용 사례',
        '경제적 해자(Economic Moat) 5가지 유형과 평가 방법',
        'ESG 평가 기준과 ESG 등급이 주가에 미치는 영향',
        '기업 지배구조 분석: 이사회 구성, 경영진 보상, 주주권익',
        '지속가능 투자 전략과 임팩트 투자 접근법'
      ]
    },

    // === PART 3: 기술적 분석 전문가 과정 (4개 챕터) ===
    {
      id: 'technical-analysis-foundation',
      title: '기술적 분석 기초와 차트 패턴',
      description: '다우 이론부터 최신 차트 패턴까지 기술적 분석의 핵심 원리를 마스터합니다',
      estimatedMinutes: 350,
      keywords: ['다우이론', '차트패턴', '지지저항', '추세선', '캔들스틱'],
      learningObjectives: [
        '다우 이론의 6가지 원칙과 현대적 해석',
        '30가지 캔들스틱 패턴과 신뢰도별 분류',
        '추세선, 채널, 삼각형 등 기하학적 패턴 분석',
        '헤드앤숄더, 더블탑/바텀 등 반전 패턴 인식',
        '플래그, 페넌트 등 지속 패턴과 목표주가 계산'
      ]
    },
    {
      id: 'technical-indicators-mastery',
      title: '기술적 지표 완전 마스터',
      description: '50개 이상의 기술적 지표를 상황별로 활용하고 조합하여 매매 신호의 정확도를 높입니다',
      estimatedMinutes: 380,
      keywords: ['이동평균', 'RSI', 'MACD', '스토캐스틱', '볼린저밴드', 'ATR'],
      learningObjectives: [
        '트렌드 지표 12개: 이동평균, MACD, ADX, 파라볼릭 SAR',
        '모멘텀 지표 12개: RSI, 스토캐스틱, CCI, Williams %R',
        '변동성 지표 8개: 볼린저밴드, ATR, 표준편차, VIX',
        '거래량 지표 10개: OBV, 축적/분산 라인, 차이킨 오실레이터',
        '복합 지표 조합 전략과 거짓 신호 필터링 기법'
      ]
    },
    {
      id: 'advanced-charting',
      title: '고급 차트 분석 기법',
      description: '엘리어트 파동, 피보나치, 갠 이론 등 고급 차트 분석 기법을 마스터합니다',
      estimatedMinutes: 320,
      keywords: ['엘리어트파동', '피보나치', '갠이론', '하모닉패턴', '마켓프로파일'],
      learningObjectives: [
        '엘리어트 파동 이론: 5파동 상승, 3파동 하락 패턴 인식',
        '피보나치 되돌림과 확장을 이용한 목표가 설정',
        '갠 팬과 갠 그리드를 활용한 시간과 가격 분석',
        '하모닉 패턴 4가지: 가틀리, 배트, 크랩, 버터플라이',
        '마켓 프로파일과 POC(Point of Control) 분석'
      ]
    },
    {
      id: 'trading-systems',
      title: '체계적 트레이딩 시스템',
      description: '수익성 있는 트레이딩 시스템을 구축하고 리스크를 관리하는 전문적 방법론을 학습합니다',
      estimatedMinutes: 360,
      keywords: ['트레이딩시스템', '백테스팅', '포지션사이징', '손절매', '리스크관리'],
      learningObjectives: [
        '트레이딩 시스템 설계: 진입, 청산, 리스크 관리 규칙',
        '백테스팅 방법론과 결과 해석 (승률, 손익비, 최대낙폭)',
        '포지션 사이징: 고정비율, 켈리공식, 변동성 기반 방법',
        '다양한 손절매 기법: 고정%, ATR 기반, 트레일링 스톱',
        '심리적 함정과 트레이딩 일지를 통한 개선'
      ]
    },

    // === PART 4: 포트폴리오 관리 및 리스크 관리 (4개 챕터) ===
    {
      id: 'modern-portfolio-theory',
      title: '현대 포트폴리오 이론 (MPT)',
      description: '해리 마코위츠의 현대 포트폴리오 이론을 실무에 적용하고 효율적 프론티어를 구축합니다',
      estimatedMinutes: 340,
      keywords: ['MPT', '효율적프론티어', '상관계수', '분산투자', '샤프비율'],
      learningObjectives: [
        '평균-분산 모델과 효율적 프론티어 계산',
        '상관계수와 공분산을 이용한 최적 포트폴리오 구성',
        '자본자산가격모델(CAPM)과 베타의 실무 활용',
        '샤프 비율, 트레이너 비율, 젠센 알파로 성과 평가',
        'Black-Litterman 모델을 이용한 기대수익률 조정'
      ]
    },
    {
      id: 'risk-management-advanced',
      title: '고급 리스크 관리',
      description: 'VaR, CVaR 등 정량적 리스크 측정 기법과 헤징 전략을 마스터합니다',
      estimatedMinutes: 380,
      keywords: ['VaR', 'CVaR', '스트레스테스트', '헤징', '파생상품'],
      learningObjectives: [
        'VaR(Value at Risk) 3가지 계산법: 역사적 시뮬레이션, 파라메트릭, 몬테카를로',
        'CVaR(Conditional VaR)과 Expected Shortfall 계산',
        '시나리오 분석과 스트레스 테스트 설계',
        '옵션을 이용한 포트폴리오 헤징 전략',
        '통화 리스크와 금리 리스크 관리 기법'
      ]
    },
    {
      id: 'asset-allocation-strategies',
      title: '전략적 자산 배분',
      description: '생애주기별, 목표별 자산 배분 전략과 동적 리밸런싱 기법을 학습합니다',
      estimatedMinutes: 300,
      keywords: ['자산배분', '리밸런싱', '생애주기', '목표기반투자', '전술적배분'],
      learningObjectives: [
        '연령별 자산 배분 가이드라인과 개인화 방법',
        '목표 기반 투자(Goal-Based Investing) 접근법',
        '전략적 vs 전술적 자산 배분의 차이와 활용',
        '리밸런싱 타이밍과 방법 (시간 기반, 편차 기준, 변동성 기반)',
        '세금 효율적 투자와 자산 위치 전략'
      ]
    },
    {
      id: 'alternative-investments',
      title: '대체투자 및 글로벌 분산투자',
      description: '부동산, 원자재, 사모펀드 등 대체투자와 글로벌 분산투자 전략을 학습합니다',
      estimatedMinutes: 320,
      keywords: ['대체투자', 'REIT', '원자재', '사모펀드', '헤지펀드', '글로벌투자'],
      learningObjectives: [
        'REIT 투자: 종류별 특징, 평가 지표, 포트폴리오 내 역할',
        '원자재 투자: 금, 은, 원유, 농산물 투자 방법과 인플레이션 헤지',
        '사모펀드와 헤지펀드: 구조, 수수료, 성과 평가',
        '글로벌 분산투자: 신흥국 vs 선진국, 통화 리스크 관리',
        '크립토 자산의 포트폴리오 내 역할과 리스크'
      ]
    },

    // === PART 5: AI 및 퀀트 투자 (3개 챕터) ===
    {
      id: 'quantitative-analysis',
      title: '퀀트 투자 기초',
      description: '통계적 방법론을 활용한 퀀트 투자의 기본 개념과 팩터 투자를 마스터합니다',
      estimatedMinutes: 360,
      keywords: ['퀀트투자', '팩터투자', '통계적거래', '페어트레이딩', '평균회귀'],
      learningObjectives: [
        '퀀트 투자의 철학과 전통적 투자와의 차이점',
        '팩터 투자: 가치, 모멘텀, 퀄리티, 저변동성, 수익성 팩터',
        '통계적 차익거래: 페어 트레이딩, 평균 회귀 전략',
        '멀티 팩터 모델 구축와 팩터 익스포저 관리',
        '스마트 베타 ETF 활용과 팩터 타이밍'
      ]
    },
    {
      id: 'machine-learning-investing',
      title: '머신러닝 투자 전략',
      description: '머신러닝과 딥러닝을 활용한 주가 예측 모델과 알고리즘 트레이딩을 구현합니다',
      estimatedMinutes: 420,
      keywords: ['머신러닝', '딥러닝', 'LSTM', '강화학습', '알고리즘트레이딩'],
      learningObjectives: [
        '머신러닝 기본 개념: 지도학습, 비지도학습, 강화학습',
        '주가 예측을 위한 특성 엔지니어링과 데이터 전처리',
        'LSTM과 GRU를 이용한 시계열 예측 모델',
        '강화학습을 이용한 포트폴리오 최적화',
        '백테스팅과 교차검증을 통한 모델 평가'
      ]
    },
    {
      id: 'algo-trading-systems',
      title: '알고리즘 트레이딩 시스템',
      description: '전문가급 알고리즘 트레이딩 시스템을 구축하고 실전에 적용하는 방법을 학습합니다',
      estimatedMinutes: 400,
      keywords: ['알고리즘트레이딩', 'API트레이딩', '고빈도거래', '마켓메이킹', '실행알고리즘'],
      learningObjectives: [
        '알고리즘 트레이딩 인프라 구축: 데이터 피드, 주문 시스템',
        '실행 알고리즘: TWAP, VWAP, Implementation Shortfall',
        '마켓 메이킹과 아비트리지 전략',
        '지연시간 최적화와 고빈도 트레이딩 기법',
        '규제 요구사항과 리스크 관리 시스템'
      ]
    },

    // === PART 6: 실전 투자 및 사례 연구 (3개 챕터) ===
    {
      id: 'investment-case-studies',
      title: '전설적 투자자 사례 연구',
      description: '워렌 버핏, 피터 린치, 조지 소로스 등 전설적 투자자들의 투자 철학과 실제 사례를 분석합니다',
      estimatedMinutes: 360,
      keywords: ['워렌버핏', '피터린치', '조지소로스', '투자철학', '사례연구'],
      learningObjectives: [
        '워렌 버핏의 가치 투자 철학과 버크셔 해서웨이 포트폴리오 분석',
        '피터 린치의 성장주 투자와 텐배거 발굴 전략',
        '조지 소로스의 재귀성 이론과 통화 위기 베팅',
        '레이 달리오의 올웨더 포트폴리오와 원칙 기반 투자',
        '벤저민 그레이엄의 가치 투자 원리와 현대적 적용'
      ]
    },
    {
      id: 'market-crisis-analysis',
      title: '시장 위기 분석 및 대응 전략',
      description: '역사적 시장 위기를 분석하고 위기 상황에서의 투자 전략과 심리적 대응 방법을 학습합니다',
      estimatedMinutes: 340,
      keywords: ['시장위기', '금융위기', '버블', '패닉', '위기대응전략'],
      learningObjectives: [
        '역사적 시장 위기 분석: 1929 대공황, 1987 블랙먼데이, 2008 금융위기',
        '버블의 형성과 붕괴 과정: 닷컴 버블, 부동산 버블',
        '위기 시 투자자 행동과 시장 메커니즘',
        '위기 상황에서의 포트폴리오 보호 전략',
        '위기를 기회로 전환하는 역발상 투자법'
      ]
    },
    {
      id: 'professional-investment-practice',
      title: '전문가 투자 실무',
      description: '실제 투자 현장에서 요구되는 실무 역량과 윤리적 투자 원칙을 마스터합니다',
      estimatedMinutes: 320,
      keywords: ['투자실무', '윤리', '규제', '보고서작성', '고객관리'],
      learningObjectives: [
        '투자 보고서 작성과 프레젠테이션 기법',
        '고객 맞춤형 포트폴리오 관리와 상담 기법',
        '투자 윤리와 이해상충 관리',
        '금융투자업법과 규제 준수 사항',
        '지속적인 자기개발과 전문성 향상 방법'
      ]
    },

    // === PART 7: 글로벌 투자 전문가 과정 (3개 챕터) ===
    {
      id: 'currency-hedging-strategies',
      title: '통화 헤지 전략',
      description: '해외 투자 시 환율 변동 리스크를 관리하고 효과적인 헤징 전략을 수립하는 전문 기법을 학습합니다',
      estimatedMinutes: 360,
      keywords: ['통화헤지', '환율리스크', '선물환', '통화스왑', 'NDF', '크로스헤지'],
      learningObjectives: [
        '환율 변동이 해외 투자 수익률에 미치는 영향 정량 분석',
        '선물환, 통화 옵션, 통화 스왑 등 헤징 수단의 특성과 활용',
        'Natural Hedge와 Financial Hedge의 차이와 적용 방법',
        '부분 헤지 vs 완전 헤지의 비용-효익 분석',
        '신흥국 통화 헤지의 특수성과 NDF(Non-Deliverable Forward) 활용'
      ]
    },
    {
      id: 'global-macro-investing',
      title: '글로벌 매크로 투자',
      description: '전 세계 경제 동향과 정책 변화를 분석하여 자산 배분과 국가별 투자 전략을 수립합니다',
      estimatedMinutes: 400,
      keywords: ['글로벌매크로', '자산배분', '국가분석', '통화정책', '재정정책', '지정학'],
      learningObjectives: [
        '글로벌 경제 사이클과 국가별 비동조화 현상 분석',
        '주요국 중앙은행 정책(Fed, ECB, BOJ, PBOC)의 글로벌 영향',
        'Top-down 접근법을 통한 국가/자산군 선택 프로세스',
        '신흥국 vs 선진국 투자 기회와 리스크 평가 프레임워크',
        '지정학적 이벤트(무역전쟁, 전쟁, 제재)의 투자 영향 분석'
      ]
    },
    {
      id: 'international-diversification',
      title: '국제 분산투자',
      description: '효과적인 글로벌 포트폴리오 구축을 위한 국제 분산투자 이론과 실무를 마스터합니다',
      estimatedMinutes: 340,
      keywords: ['국제분산투자', '상관관계', '홈바이어스', 'ADR', 'GDR', '국가리스크'],
      learningObjectives: [
        '국제 분산투자의 이론적 근거와 실증적 효과 분석',
        '국가 간 상관계수 변화와 위기 시 상관관계 증가 현상',
        'ADR, GDR, ETF를 활용한 해외 시장 접근 방법',
        '홈 바이어스 극복과 최적 해외 투자 비중 결정',
        '국가 리스크(정치, 규제, 환율) 평가와 관리 방법'
      ]
    }
  ],
  
  simulators: [
    // === 핵심 분석 도구 ===
    {
      id: 'financial-calculator',
      name: '재무제표 분석기',
      description: '재무비율 자동계산, 동종업계 벤치마킹, 5년 트렌드 분석을 지원하는 전문가급 도구',
      component: 'FinancialCalculator'
    },
    {
      id: 'chart-analyzer',
      name: 'AI 차트 분석기',
      description: 'AI가 차트 패턴을 자동 인식하고 매매 신호를 생성하는 기술적 분석 도구',
      component: 'ChartAnalyzer'
    },
    {
      id: 'portfolio-optimizer',
      name: '포트폴리오 최적화',
      description: '현대 포트폴리오 이론 기반 효율적 프론티어 계산과 최적 자산 배분',
      component: 'PortfolioOptimizer'
    },
    {
      id: 'backtesting-engine',
      name: '백테스팅 엔진',
      description: '투자 전략을 과거 데이터로 검증하는 전문가급 백테스팅 시뮬레이터',
      component: 'BacktestingEngine'
    },

    // === 고급 분석 도구 ===
    {
      id: 'real-time-dashboard',
      name: '실시간 시장 대시보드',
      description: '실시간 가격, AI 예측, 호가창 분석을 통합한 트레이딩 대시보드',
      component: 'RealTimeStockDashboard'
    },
    {
      id: 'risk-management-dashboard',
      name: '리스크 관리 대시보드',
      description: 'VaR, 스트레스 테스트, 시나리오 분석을 통한 포트폴리오 리스크 모니터링',
      component: 'RiskManagementDashboard'
    },
    {
      id: 'factor-investing-lab',
      name: '팩터 투자 연구소',
      description: '가치, 모멘텀, 퀄리티 등 투자 팩터의 성과 분석과 멀티팩터 백테스팅',
      component: 'FactorInvestingLab'
    },
    {
      id: 'options-strategy-analyzer',
      name: '옵션 전략 분석기',
      description: '옵션 전략의 손익구조와 그릭스를 시각화하는 전문 도구',
      component: 'OptionsStrategyAnalyzer'
    },

    // === AI 도구 ===
    {
      id: 'ai-mentor',
      name: 'AI 투자 멘토',
      description: 'AI 기반 개인화된 투자 조언과 포트폴리오 진단',
      component: 'AIMentor'
    },
    {
      id: 'news-impact-analyzer',
      name: 'AI 뉴스 분석기',
      description: 'AI 기반 뉴스 감정분석과 주가 영향도 예측',
      component: 'NewsImpactAnalyzer'
    },
    {
      id: 'news-ontology-analyzer',
      name: 'AI 뉴스 온톨로지 분석기',
      description: '고급 뉴스 검색, 엔티티 추출, 인터랙티브 온톨로지 그래프 시각화',
      component: 'NewsOntologyAnalyzer'
    },
    {
      id: 'news-cache-dashboard',
      name: '뉴스 API 캐시 대시보드',
      description: '실시간 뉴스 데이터 캐시 관리, API 사용량 통계 및 성능 모니터링',
      component: 'NewsCacheDashboard'
    },

    // === 기타 유용한 도구 ===
    {
      id: 'dcf-valuation-model',
      name: 'DCF 가치평가 모델',
      description: '현금흐름 예측과 민감도 분석이 포함된 기업가치 평가 도구',
      component: 'DCFValuationModel'
    },
    
    // === 글로벌 투자 도구 ===
    {
      id: 'global-market-dashboard',
      name: '글로벌 실시간 대시보드',
      description: '전 세계 주요 시장의 실시간 현황, 환율, 거래시간을 한눈에 모니터링',
      component: 'GlobalMarketDashboard'
    },
    {
      id: 'currency-impact-analyzer',
      name: '환율 영향 분석기',
      description: '해외 주식 투자 시 환율 변동이 수익률에 미치는 영향을 분석하고 헤지 전략 시뮬레이션',
      component: 'CurrencyImpactAnalyzer'
    },
    {
      id: 'us-stock-screener',
      name: '미국 주식 스크리너',
      description: '100개 이상의 조건으로 NYSE, NASDAQ 전체 종목을 필터링하여 투자 기회 발굴',
      component: 'USStockScreener'
    },
    {
      id: 'tax-optimization-calculator',
      name: '세금 최적화 계산기',
      description: '미국과 한국 주식 투자의 세금을 계산하고 절세 전략을 수립',
      component: 'TaxOptimizationCalculator'
    },
    {
      id: 'etf-comparator',
      name: 'ETF 비교 분석기',
      description: '다양한 ETF의 성과, 리스크, 비용을 종합 비교하고 최적 포트폴리오 구성',
      component: 'ETFComparator'
    },
    {
      id: 'sector-rotation-analyzer',
      name: '글로벌 섹터 로테이션 분석기',
      description: '경제 사이클에 따른 섹터별 성과 분석과 최적의 섹터 로테이션 전략 수립',
      component: 'SectorRotationAnalyzer'
    },
    {
      id: 'global-macro-dashboard',
      name: '글로벌 매크로 대시보드',
      description: '전 세계 경제 지표와 시장 동향을 종합 분석하여 매크로 투자 전략 제시',
      component: 'GlobalMacroDashboard'
    },
    {
      id: 'options-strategy-simulator',
      name: '옵션 전략 시뮬레이터',
      description: '다양한 옵션 전략의 손익 구조를 시각화하고 Greeks를 분석하여 최적 전략 수립',
      component: 'OptionsStrategySimulator'
    },
    {
      id: 'risk-parity-portfolio',
      name: '리스크 패리티 포트폴리오',
      description: '각 자산의 리스크 기여도를 균등 배분하여 안정적이고 효율적인 포트폴리오 구성',
      component: 'RiskParityPortfolio'
    },
    {
      id: 'pair-trading-analyzer',
      name: '페어 트레이딩 분석기',
      description: '상관관계가 높은 주식 쌍을 찾아 통계적 차익거래 기회를 포착하고 백테스트',
      component: 'PairTradingAnalyzer'
    },
    {
      id: 'dividend-optimizer',
      name: '배당 수익률 최적화',
      description: '다양한 배당 전략으로 안정적인 현금흐름을 창출하고 장기적인 배당 성장 추구',
      component: 'DividendOptimizer'
    },
    {
      id: 'momentum-backtester',
      name: '모멘텀 전략 백테스터',
      description: '다양한 모멘텀 지표를 활용하여 추세 추종 전략을 백테스트하고 최적화',
      component: 'MomentumBacktester'
    },
    
    // === 전문가용 고급 도구 ===
    {
      id: 'monte-carlo-simulator',
      name: '몬테카를로 시뮬레이션',
      description: '10,000회 이상 시뮬레이션으로 포트폴리오 미래 가치 확률 분석, VaR/CVaR 계산',
      component: 'MonteCarloSimulator'
    },
    {
      id: 'stress-test-scenarios',
      name: '스트레스 테스트 시나리오',
      description: '2008년 금융위기, 코로나 등 역사적 시나리오로 포트폴리오 취약점 분석',
      component: 'StressTestScenarios'
    },
    {
      id: 'real-time-risk-dashboard',
      name: '실시간 리스크 대시보드',
      description: 'P&L 추적, 리스크 한도 모니터링, 포지션 집중도 알림 등 종합 리스크 관리',
      component: 'RealTimeRiskDashboard'
    },
    {
      id: 'trading-cost-calculator',
      name: '거래 비용 상세 모델링',
      description: '슬리피지, 시장충격, 세금, 수수료 등 모든 거래 비용 계산 및 최적화',
      component: 'TradingCostCalculator'
    }
  ],
  
  // === 전문가 도구 확장 ===
  tools: [
    {
      id: 'advanced-stock-screener',
      name: '고급 종목 스크리너',
      description: '100개 조건으로 전 세계 20,000개 종목에서 투자 기회 발굴',
      url: '/modules/stock-analysis/tools/advanced-screener'
    },
    {
      id: 'financial-dictionary',
      name: '금융 전문용어 사전',
      description: '500개+ 금융 용어와 실무 사례를 담은 종합 금융 사전',
      url: '/stock-dictionary'
    },
    {
      id: 'virtual-trading-platform',
      name: '가상 트레이딩 플랫폼',
      description: '실제 시장과 동일한 환경에서 무제한 모의투자 연습',
      url: '/modules/stock-analysis/tools/virtual-trading'
    },
    {
      id: 'earnings-calendar',
      name: '실적발표 캘린더',
      description: '전 세계 주요 기업 실적발표 일정과 컨센서스 전망',
      url: '/modules/stock-analysis/tools/earnings-calendar'
    },
    {
      id: 'dividend-calendar',
      name: '배당 캘린더',
      description: '국내외 배당주 배당락일, 지급일 종합 관리',
      url: '/modules/stock-analysis/tools/dividend-calendar'
    },
    {
      id: 'ipo-tracker',
      name: 'IPO 추적기',
      description: '신규상장 예정 기업 정보와 공모주 투자 가이드',
      url: '/modules/stock-analysis/tools/ipo-tracker'
    },
    {
      id: 'investment-journal',
      name: '투자 일지',
      description: '매매 기록, 수익률 분석, 투자 성찰을 위한 디지털 일지',
      url: '/modules/stock-analysis/tools/investment-journal'
    },
    {
      id: 'research-reports',
      name: '투자 리포트 라이브러리',
      description: '국내외 증권사 리포트와 AI 분석 결과 통합 제공',
      url: '/modules/stock-analysis/tools/research-reports'
    }
  ]
}

export const getChapter = (chapterId: string) => {
  return stockAnalysisModule.chapters.find(chapter => chapter.id === chapterId)
}

export const getNextChapter = (currentChapterId: string) => {
  const currentIndex = stockAnalysisModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex < stockAnalysisModule.chapters.length - 1 ? stockAnalysisModule.chapters[currentIndex + 1] : undefined
}

export const getPrevChapter = (currentChapterId: string) => {
  const currentIndex = stockAnalysisModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex > 0 ? stockAnalysisModule.chapters[currentIndex - 1] : undefined
}
