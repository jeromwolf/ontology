export const englishConversationMetadata = {
  id: 'english-conversation',
  name: 'English Conversation Master',
  description: 'AI와 함께하는 실전 영어회화, 상황별 대화 연습과 발음 교정',
  version: '1.0.0',
  category: '언어/교육',
  difficulty: 'beginner' as const,
  duration: '12주',
  students: 1247,
  rating: 4.9,
  chapters: [
    {
      id: 'conversation-basics',
      title: 'Chapter 1: 기초 회화 패턴',
      description: '일상생활에서 가장 많이 사용하는 기본 대화 패턴과 표현',
      duration: '1.5시간',
      objectives: [
        '인사와 자기소개 표현 익히기',
        '감정과 상태 표현하는 방법',
        '기본 질문과 대답 패턴',
        '예의를 갖춘 대화 방식'
      ]
    },
    {
      id: 'daily-situations',
      title: 'Chapter 2: 일상 상황 대화',
      description: '쇼핑, 식당, 교통 등 일상에서 마주치는 상황별 대화',
      duration: '2시간',
      objectives: [
        '쇼핑과 가격 협상 대화',
        '식당에서 주문과 요청',
        '대중교통 이용 시 대화',
        '길 묻기와 설명하기'
      ]
    },
    {
      id: 'business-english',
      title: 'Chapter 3: 비즈니스 영어',
      description: '회의, 프레젠테이션, 이메일 등 업무 상황에서의 영어',
      duration: '2시간',
      objectives: [
        '회의에서 의견 표현하기',
        '프레젠테이션 기법',
        '비즈니스 이메일 작성',
        '협상과 토론 스킬'
      ]
    },
    {
      id: 'travel-english',
      title: 'Chapter 4: 여행 영어',
      description: '공항, 호텔, 관광지에서 필요한 실전 여행 영어',
      duration: '1.5시간',
      objectives: [
        '공항 체크인과 출입국',
        '호텔 예약과 체크인',
        '관광지에서 정보 문의',
        '응급상황 대처 방법'
      ]
    },
    {
      id: 'pronunciation-intonation',
      title: 'Chapter 5: 발음과 억양',
      description: 'AI 음성 인식을 활용한 발음 교정과 억양 연습',
      duration: '2시간',
      objectives: [
        '영어 음성학 기초',
        '발음 교정 기법',
        '자연스러운 억양 연습',
        '리듬과 강세 패턴'
      ]
    },
    {
      id: 'listening-comprehension',
      title: 'Chapter 6: 듣기와 이해',
      description: '다양한 액센트와 속도의 영어 듣기 훈련',
      duration: '1.5시간',
      objectives: [
        '다양한 영어 액센트 이해',
        '빠른 말하기 듣기 연습',
        '맥락 파악 기법',
        '듣기 전략과 노트 테이킹'
      ]
    },
    {
      id: 'cultural-context',
      title: 'Chapter 7: 문화와 맥락',
      description: '영어권 문화 이해와 상황에 맞는 적절한 표현',
      duration: '1.5시간',
      objectives: [
        '영어권 문화 특징 이해',
        '상황별 적절한 표현',
        '유머와 농담 이해하기',
        '예의와 매너 표현'
      ]
    },
    {
      id: 'advanced-conversation',
      title: 'Chapter 8: 고급 회화 기법',
      description: '토론, 설득, 복잡한 주제에 대한 고급 회화',
      duration: '2시간',
      objectives: [
        '복잡한 주제 토론하기',
        '논리적 설득 기법',
        '감정적 뉘앙스 표현',
        '고급 어휘와 관용구'
      ]
    }
  ],
  simulators: [
    {
      id: 'ai-conversation-partner',
      name: 'AI 대화 파트너',
      description: '실시간 AI 대화로 자연스러운 영어 회화 연습'
    },
    {
      id: 'pronunciation-trainer',
      name: '발음 트레이너',
      description: 'AI 음성 인식으로 발음 교정과 피드백'
    },
    {
      id: 'dialogue-practice',
      name: '상황별 대화 연습',
      description: '다양한 실생활 상황을 모사한 대화 연습'
    },
    {
      id: 'scenario-practice',
      name: '실전 시나리오 연습',
      description: '선택형 대답으로 진행하는 시나리오별 대화 시뮬레이션'
    },
    {
      id: 'listening-lab',
      name: '듣기 실험실',
      description: '다양한 액센트와 속도의 듣기 연습'
    }
  ],
  prerequisites: ['기초 영어 문법', '중학교 수준 영어 어휘'],
  learningPath: {
    next: ['advanced-english', 'business-communication'],
    previous: ['basic-english-grammar']
  }
}