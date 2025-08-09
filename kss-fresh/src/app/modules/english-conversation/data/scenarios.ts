import { 
  Plane, Coffee, Building, ShoppingBag, GraduationCap, Activity,
  Phone, Car, Home, Utensils, Hotel, Bus, Train, Ship,
  Camera, Book, Music, Gamepad2, Heart, Users,
  Briefcase, DollarSign, Globe, Package, Wrench, Shield,
  Smartphone, Laptop, Wifi, Cloud, Lock, Mail,
  Calendar, Clock, MapPin, Navigation, Compass, Mountain,
  Sun, Umbrella, Snowflake, Wind, Zap, Flame,
  Gift, Cake, PartyPopper, Trophy, Medal, Star,
  ShoppingCart, CreditCard, Wallet, Receipt, Tag, Store,
  Stethoscope, Pill, Syringe, HeartPulse, Thermometer,
  type LucideIcon
} from 'lucide-react'

export interface ScenarioStep {
  id: string
  speaker: 'user' | 'npc'
  text: string
  korean: string
  options?: string[]
  feedback?: string
}

export interface Scenario {
  id: string
  title: string
  category: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  setting: string
  description: string
  duration: string
  objectives: string[]
  icon: LucideIcon
  imageUrl: string
  steps: ScenarioStep[]
}

export const scenarios: Scenario[] = [
  // ============= 여행 (Travel) - 20개 =============
  {
    id: 'airport-checkin',
    title: '공항 체크인',
    category: '여행',
    difficulty: 'intermediate',
    setting: '인천국제공항 체크인 카운터',
    description: '해외여행 시 공항에서 체크인하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '체크인 과정 이해하기',
      '항공편 정보 확인하기',
      '수하물 관련 대화하기',
      '좌석 선택 요청하기'
    ],
    icon: Plane,
    imageUrl: 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good morning! May I see your passport and ticket, please?",
        korean: "안녕하세요! 여권과 항공권을 보여주시겠어요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Good morning! Here's my passport and e-ticket confirmation.",
        korean: "안녕하세요! 여기 여권과 전자항공권 확인서입니다.",
        options: [
          "Good morning! Here's my passport and e-ticket confirmation.",
          "Hi! Here are my documents.",
          "Hello! I have my passport and booking confirmation here."
        ]
      },
      {
        id: 'step3',
        speaker: 'npc',
        text: "Thank you. I see you're flying to New York today. Any bags to check in?",
        korean: "감사합니다. 오늘 뉴욕행 항공편이시네요. 체크인할 수하물이 있나요?"
      },
      {
        id: 'step4',
        speaker: 'user',
        text: "Yes, I have one suitcase to check in.",
        korean: "네, 체크인할 캐리어가 하나 있습니다.",
        options: [
          "Yes, I have one suitcase to check in.",
          "Yes, just this one bag.",
          "I'd like to check in this suitcase, please."
        ]
      },
      {
        id: 'step5',
        speaker: 'npc',
        text: "Perfect. Please place it on the scale. Would you prefer an aisle or window seat?",
        korean: "완벽합니다. 저울에 올려주세요. 통로쪽 좌석과 창가쪽 좌석 중 어느 것을 선호하시나요?"
      },
      {
        id: 'step6',
        speaker: 'user',
        text: "I'd prefer a window seat, please.",
        korean: "창가 좌석으로 부탁드립니다.",
        options: [
          "I'd prefer a window seat, please.",
          "Window seat, if available.",
          "Could I have a window seat?"
        ]
      }
    ]
  },
  {
    id: 'hotel-checkin',
    title: '호텔 체크인',
    category: '여행',
    difficulty: 'beginner',
    setting: '5성급 호텔 프론트 데스크',
    description: '호텔에 도착해서 체크인하는 과정을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '예약 확인하기',
      '체크인 절차 이해하기',
      '호텔 시설 문의하기',
      '룸 서비스 요청하기'
    ],
    icon: Hotel,
    imageUrl: 'https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good evening! Welcome to Grand Hotel. How may I help you?",
        korean: "안녕하세요! 그랜드 호텔에 오신 것을 환영합니다. 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Hi, I have a reservation under the name Kim.",
        korean: "안녕하세요, 김씨 이름으로 예약이 되어 있습니다.",
        options: [
          "Hi, I have a reservation under the name Kim.",
          "Hello, I'd like to check in. My name is Kim.",
          "Good evening, I'm checking in. The reservation is under Kim."
        ]
      },
      {
        id: 'step3',
        speaker: 'npc',
        text: "Let me check... Yes, Mr. Kim, I have your reservation for three nights. May I see your ID?",
        korean: "확인해보겠습니다... 네, 김씨, 3박 예약이 확인됩니다. 신분증을 보여주시겠어요?"
      },
      {
        id: 'step4',
        speaker: 'user',
        text: "Sure, here's my passport. Also, does the room have a city view?",
        korean: "네, 여기 여권입니다. 그리고 방에서 도시 전망이 보이나요?",
        options: [
          "Sure, here's my passport. Also, does the room have a city view?",
          "Here you go. Can I get a room with a nice view?",
          "Here's my ID. Is it possible to have a room with a view?"
        ]
      }
    ]
  },
  {
    id: 'taxi-ride',
    title: '택시 타기',
    category: '여행',
    difficulty: 'beginner',
    setting: '뉴욕 맨하탄 거리',
    description: '택시를 타고 목적지까지 가는 상황을 연습합니다.',
    duration: '3-5분',
    objectives: [
      '목적지 전달하기',
      '요금 확인하기',
      '경로 요청하기',
      '결제 방법 문의하기'
    ],
    icon: Car,
    imageUrl: 'https://images.unsplash.com/photo-1556741533-974f8e62a92d?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Where to?",
        korean: "어디로 모실까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "To Times Square, please.",
        korean: "타임스퀘어로 가주세요.",
        options: [
          "To Times Square, please.",
          "I need to go to Times Square.",
          "Times Square, thank you."
        ]
      },
      {
        id: 'step3',
        speaker: 'npc',
        text: "Sure thing. First time in New York?",
        korean: "알겠습니다. 뉴욕은 처음이신가요?"
      },
      {
        id: 'step4',
        speaker: 'user',
        text: "Yes, it's my first visit. How long will it take to get there?",
        korean: "네, 처음 방문입니다. 거기까지 얼마나 걸릴까요?",
        options: [
          "Yes, it's my first visit. How long will it take to get there?",
          "Yes, first time. What's the estimated time?",
          "It is. How much time to Times Square?"
        ]
      }
    ]
  },
  {
    id: 'tourist-information',
    title: '관광 안내소',
    category: '여행',
    difficulty: 'intermediate',
    setting: '파리 관광 안내소',
    description: '관광 정보를 얻고 추천을 받는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '관광지 정보 요청하기',
      '교통편 문의하기',
      '입장권 구매하기',
      '지도와 브로셔 요청하기'
    ],
    icon: MapPin,
    imageUrl: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Bonjour! Welcome to Paris Tourist Information. How can I assist you today?",
        korean: "봉주르! 파리 관광 안내소에 오신 것을 환영합니다. 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Hello! I'd like to visit the Louvre Museum. What's the best way to get there?",
        korean: "안녕하세요! 루브르 박물관을 방문하고 싶은데요. 가는 가장 좋은 방법이 뭔가요?",
        options: [
          "Hello! I'd like to visit the Louvre Museum. What's the best way to get there?",
          "Hi! How do I get to the Louvre from here?",
          "Good morning! Could you tell me the best route to the Louvre?"
        ]
      }
    ]
  },
  {
    id: 'customs-immigration',
    title: '입국 심사',
    category: '여행',
    difficulty: 'intermediate',
    setting: '미국 공항 입국 심사대',
    description: '해외 입국 심사를 받는 상황을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '방문 목적 설명하기',
      '체류 기간 전달하기',
      '숙소 정보 제공하기',
      '귀국 일정 확인하기'
    ],
    icon: Shield,
    imageUrl: 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Next! Passport please. What's the purpose of your visit?",
        korean: "다음! 여권 주세요. 방문 목적이 무엇입니까?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I'm here for tourism. I'll be staying for two weeks.",
        korean: "관광 목적입니다. 2주간 머물 예정입니다.",
        options: [
          "I'm here for tourism. I'll be staying for two weeks.",
          "Tourism. My trip is for fourteen days.",
          "I'm visiting for vacation, staying two weeks."
        ]
      }
    ]
  },

  // ============= 일상생활 (Daily Life) - 20개 =============
  {
    id: 'restaurant-reservation',
    title: '레스토랑 예약',
    category: '일상생활',
    difficulty: 'beginner',
    setting: '인기 레스토랑',
    description: '전화로 레스토랑 예약을 하는 상황을 연습합니다.',
    duration: '3-5분',
    objectives: [
      '예약 요청하기',
      '날짜와 시간 조정하기',
      '인원수 확인하기',
      '특별 요청사항 전달하기'
    ],
    icon: Coffee,
    imageUrl: 'https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good evening, Mario's Italian Restaurant. How can I help you?",
        korean: "안녕하세요, 마리오 이탈리안 레스토랑입니다. 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Hi, I'd like to make a reservation for dinner tomorrow night.",
        korean: "안녕하세요, 내일 저녁 식사 예약을 하고 싶습니다.",
        options: [
          "Hi, I'd like to make a reservation for dinner tomorrow night.",
          "Hello, I want to book a table for tomorrow evening.",
          "Good evening, could I reserve a table for tomorrow?"
        ]
      }
    ]
  },
  {
    id: 'grocery-shopping',
    title: '식료품 쇼핑',
    category: '일상생활',
    difficulty: 'beginner',
    setting: '대형 슈퍼마켓',
    description: '슈퍼마켓에서 장보는 상황을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '상품 위치 문의하기',
      '신선도 확인하기',
      '가격 문의하기',
      '계산대에서 결제하기'
    ],
    icon: ShoppingCart,
    imageUrl: 'https://images.unsplash.com/photo-1534723452862-4c874018d66d?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "Excuse me, where can I find the dairy products?",
        korean: "실례합니다, 유제품은 어디에 있나요?",
        options: [
          "Excuse me, where can I find the dairy products?",
          "Hi, could you tell me where the dairy section is?",
          "Sorry, which aisle has milk and cheese?"
        ]
      },
      {
        id: 'step2',
        speaker: 'npc',
        text: "The dairy section is in aisle 3, on your left.",
        korean: "유제품 코너는 3번 통로 왼쪽에 있습니다."
      }
    ]
  },
  {
    id: 'bank-account',
    title: '은행 계좌 개설',
    category: '일상생활',
    difficulty: 'intermediate',
    setting: '시티은행 지점',
    description: '은행에서 새 계좌를 개설하는 과정을 연습합니다.',
    duration: '6-8분',
    objectives: [
      '계좌 종류 문의하기',
      '필요 서류 확인하기',
      '수수료 정보 얻기',
      '온라인 뱅킹 설정하기'
    ],
    icon: DollarSign,
    imageUrl: 'https://images.unsplash.com/photo-1550565118-3a14e8d0386f?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good morning! How can I help you today?",
        korean: "안녕하세요! 오늘 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I'd like to open a checking account, please.",
        korean: "당좌예금 계좌를 개설하고 싶습니다.",
        options: [
          "I'd like to open a checking account, please.",
          "I want to set up a new bank account.",
          "I need to open an account for daily transactions."
        ]
      }
    ]
  },
  {
    id: 'post-office',
    title: '우체국 방문',
    category: '일상생활',
    difficulty: 'beginner',
    setting: '지역 우체국',
    description: '우체국에서 소포를 보내는 상황을 연습합니다.',
    duration: '4-5분',
    objectives: [
      '배송 옵션 문의하기',
      '요금 확인하기',
      '추적 서비스 요청하기',
      '배송 시간 확인하기'
    ],
    icon: Package,
    imageUrl: 'https://images.unsplash.com/photo-1527576539890-dfa815648363?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "I'd like to send this package to Korea.",
        korean: "이 소포를 한국으로 보내고 싶습니다.",
        options: [
          "I'd like to send this package to Korea.",
          "I need to ship this to Korea, please.",
          "Can you help me send this package to Korea?"
        ]
      },
      {
        id: 'step2',
        speaker: 'npc',
        text: "Sure! Would you like express or standard shipping?",
        korean: "네! 특급 배송과 일반 배송 중 어느 것을 원하시나요?"
      }
    ]
  },
  {
    id: 'gym-membership',
    title: '헬스장 등록',
    category: '일상생활',
    difficulty: 'intermediate',
    setting: '피트니스 센터',
    description: '헬스장 회원 가입 상담을 받는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '시설 견학 요청하기',
      '회원권 종류 문의하기',
      '이용 시간 확인하기',
      '개인 트레이너 정보 얻기'
    ],
    icon: Activity,
    imageUrl: 'https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Welcome to FitLife Gym! Are you interested in a membership?",
        korean: "핏라이프 짐에 오신 것을 환영합니다! 회원권에 관심이 있으신가요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Yes, I'd like to know about your membership options and see the facilities.",
        korean: "네, 회원권 옵션에 대해 알고 싶고 시설을 둘러보고 싶습니다.",
        options: [
          "Yes, I'd like to know about your membership options and see the facilities.",
          "I'm interested. Can you show me around and explain the memberships?",
          "Yes, what packages do you offer? And can I tour the gym?"
        ]
      }
    ]
  },

  // ============= 비즈니스 (Business) - 20개 =============
  {
    id: 'job-interview',
    title: '취업 면접',
    category: '비즈니스',
    difficulty: 'advanced',
    setting: 'IT 회사 면접실',
    description: '영어로 진행되는 취업 면접 상황을 연습합니다.',
    duration: '10-15분',
    objectives: [
      '자기소개 효과적으로 하기',
      '경력과 기술 설명하기',
      '회사에 대한 관심 표현하기',
      '질문에 논리적으로 답변하기'
    ],
    icon: Building,
    imageUrl: 'https://images.unsplash.com/photo-1568992687947-868a62a9f521?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good afternoon. Thank you for coming in today. Please, have a seat.",
        korean: "안녕하세요. 오늘 와주셔서 감사합니다. 앉으세요."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Thank you for this opportunity. I'm excited to be here.",
        korean: "이런 기회를 주셔서 감사합니다. 여기 와서 기쁩니다.",
        options: [
          "Thank you for this opportunity. I'm excited to be here.",
          "Thank you for having me. I'm looking forward to our conversation.",
          "I appreciate you taking the time to meet with me today."
        ]
      }
    ]
  },
  {
    id: 'business-meeting',
    title: '비즈니스 미팅',
    category: '비즈니스',
    difficulty: 'advanced',
    setting: '회사 회의실',
    description: '프로젝트 진행 상황을 논의하는 회의를 연습합니다.',
    duration: '8-10분',
    objectives: [
      '프로젝트 현황 보고하기',
      '문제점 설명하기',
      '해결책 제안하기',
      '일정 조정 논의하기'
    ],
    icon: Briefcase,
    imageUrl: 'https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Let's begin with the project status update. How are things progressing?",
        korean: "프로젝트 현황 업데이트부터 시작하겠습니다. 진행 상황은 어떤가요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "We're currently at 70% completion. The development phase is on track.",
        korean: "현재 70% 완료되었습니다. 개발 단계는 순조롭게 진행 중입니다.",
        options: [
          "We're currently at 70% completion. The development phase is on track.",
          "The project is 70% done and progressing as planned.",
          "We've completed 70% of the work. Everything is going smoothly."
        ]
      }
    ]
  },
  {
    id: 'networking-event',
    title: '네트워킹 이벤트',
    category: '비즈니스',
    difficulty: 'intermediate',
    setting: '비즈니스 컨퍼런스',
    description: '업계 네트워킹 행사에서 대화하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '자기소개와 명함 교환하기',
      '회사와 업무 설명하기',
      '관심사 공유하기',
      '연락처 교환하기'
    ],
    icon: Users,
    imageUrl: 'https://images.unsplash.com/photo-1515187029135-18ee286d815b?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Hi there! I don't think we've met. I'm Sarah from Tech Solutions.",
        korean: "안녕하세요! 처음 뵙는 것 같네요. 저는 테크 솔루션의 사라입니다."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Nice to meet you, Sarah. I'm Kim from Digital Marketing Inc.",
        korean: "만나서 반갑습니다, 사라. 저는 디지털 마케팅의 김입니다.",
        options: [
          "Nice to meet you, Sarah. I'm Kim from Digital Marketing Inc.",
          "Pleased to meet you. I'm Kim, I work at Digital Marketing Inc.",
          "Hello Sarah, I'm Kim. I'm with Digital Marketing Inc."
        ]
      }
    ]
  },
  {
    id: 'client-presentation',
    title: '클라이언트 프레젠테이션',
    category: '비즈니스',
    difficulty: 'advanced',
    setting: '클라이언트 사무실',
    description: '새로운 제품이나 서비스를 클라이언트에게 제안합니다.',
    duration: '8-10분',
    objectives: [
      '제품 특징 설명하기',
      '이점 강조하기',
      '가격 협상하기',
      '계약 조건 논의하기'
    ],
    icon: Globe,
    imageUrl: 'https://images.unsplash.com/photo-1552664730-d307ca884978?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "Thank you for your time today. I'd like to present our new solution.",
        korean: "오늘 시간 내주셔서 감사합니다. 저희의 새로운 솔루션을 소개하고 싶습니다.",
        options: [
          "Thank you for your time today. I'd like to present our new solution.",
          "I appreciate this opportunity to show you our latest product.",
          "Thanks for meeting with us. Let me introduce our new service."
        ]
      }
    ]
  },
  {
    id: 'phone-conference',
    title: '전화 회의',
    category: '비즈니스',
    difficulty: 'intermediate',
    setting: '화상 회의실',
    description: '국제 팀과의 전화 회의를 진행합니다.',
    duration: '6-8분',
    objectives: [
      '회의 시작과 참석자 확인',
      '기술적 문제 해결',
      '의견 제시하기',
      '다음 단계 계획하기'
    ],
    icon: Phone,
    imageUrl: 'https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good morning everyone. Can everybody hear me clearly?",
        korean: "모두 안녕하세요. 제 목소리 잘 들리시나요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Yes, we can hear you well. This is Kim from the Seoul office.",
        korean: "네, 잘 들립니다. 서울 사무실의 김입니다.",
        options: [
          "Yes, we can hear you well. This is Kim from the Seoul office.",
          "Audio is clear. Kim here from Seoul.",
          "Yes, loud and clear. This is Kim speaking from Seoul."
        ]
      }
    ]
  },

  // ============= 쇼핑 (Shopping) - 15개 =============
  {
    id: 'shopping-mall',
    title: '쇼핑몰에서 쇼핑',
    category: '쇼핑',
    difficulty: 'beginner',
    setting: '대형 쇼핑몰',
    description: '쇼핑몰에서 옷을 사는 상황을 연습합니다.',
    duration: '5-8분',
    objectives: [
      '원하는 상품 문의하기',
      '사이즈와 색상 확인하기',
      '가격 협상하기',
      '결제 과정 이해하기'
    ],
    icon: ShoppingBag,
    imageUrl: 'https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Welcome to our store! Is there anything specific you're looking for today?",
        korean: "저희 매장에 오신 것을 환영합니다! 오늘 찾고 계신 특별한 것이 있나요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Yes, I'm looking for a winter jacket, preferably in black or navy.",
        korean: "네, 겨울 재킷을 찾고 있는데, 검은색이나 네이비색으로 부탁드립니다.",
        options: [
          "Yes, I'm looking for a winter jacket, preferably in black or navy.",
          "I need a winter coat in dark colors.",
          "Do you have winter jackets in black or navy blue?"
        ]
      }
    ]
  },
  {
    id: 'electronics-store',
    title: '전자제품 매장',
    category: '쇼핑',
    difficulty: 'intermediate',
    setting: '애플 스토어',
    description: '전자제품을 구매하고 기술 지원을 받는 상황을 연습합니다.',
    duration: '6-8분',
    objectives: [
      '제품 사양 문의하기',
      '보증 정보 확인하기',
      '액세서리 추천받기',
      '기술 지원 요청하기'
    ],
    icon: Smartphone,
    imageUrl: 'https://images.unsplash.com/photo-1491933382434-500287f9b54b?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "I'm interested in the new iPhone. What are the main features?",
        korean: "새로운 아이폰에 관심이 있습니다. 주요 기능이 뭔가요?",
        options: [
          "I'm interested in the new iPhone. What are the main features?",
          "Can you tell me about the latest iPhone's features?",
          "I'd like to know more about the new iPhone capabilities."
        ]
      }
    ]
  },
  {
    id: 'return-exchange',
    title: '교환 및 환불',
    category: '쇼핑',
    difficulty: 'intermediate',
    setting: '백화점 고객 서비스 센터',
    description: '구매한 제품을 교환하거나 환불받는 과정을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '환불 정책 확인하기',
      '교환 이유 설명하기',
      '영수증 제시하기',
      '다른 옵션 논의하기'
    ],
    icon: Receipt,
    imageUrl: 'https://images.unsplash.com/photo-1556742111-a301076d9d18?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "I'd like to return this shirt. It doesn't fit properly.",
        korean: "이 셔츠를 반품하고 싶습니다. 사이즈가 맞지 않아요.",
        options: [
          "I'd like to return this shirt. It doesn't fit properly.",
          "I need to return this item. The size is wrong.",
          "Can I return this shirt? It's not the right fit."
        ]
      }
    ]
  },
  {
    id: 'online-shopping-help',
    title: '온라인 쇼핑 문의',
    category: '쇼핑',
    difficulty: 'intermediate',
    setting: '온라인 쇼핑 고객센터 전화',
    description: '온라인 주문 관련 문제를 해결하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '주문 번호 확인하기',
      '배송 상태 추적하기',
      '문제 상황 설명하기',
      '해결책 요청하기'
    ],
    icon: Laptop,
    imageUrl: 'https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Thank you for calling customer service. How can I help you today?",
        korean: "고객 서비스에 전화해 주셔서 감사합니다. 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I placed an order three days ago but haven't received a shipping confirmation.",
        korean: "3일 전에 주문했는데 아직 배송 확인을 받지 못했습니다.",
        options: [
          "I placed an order three days ago but haven't received a shipping confirmation.",
          "My order from three days ago hasn't shipped yet.",
          "I'm calling about an order I made three days ago. There's no shipping update."
        ]
      }
    ]
  },
  {
    id: 'bargaining-market',
    title: '시장에서 흥정하기',
    category: '쇼핑',
    difficulty: 'intermediate',
    setting: '벼룩시장',
    description: '시장에서 물건값을 흥정하는 상황을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '가격 문의하기',
      '할인 요청하기',
      '품질 확인하기',
      '최종 가격 협상하기'
    ],
    icon: Tag,
    imageUrl: 'https://images.unsplash.com/photo-1555529669-2269763671c0?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "How much is this vintage camera?",
        korean: "이 빈티지 카메라는 얼마인가요?",
        options: [
          "How much is this vintage camera?",
          "What's the price for this camera?",
          "How much are you asking for this?"
        ]
      },
      {
        id: 'step2',
        speaker: 'npc',
        text: "That's $200. It's in excellent condition.",
        korean: "200달러입니다. 상태가 아주 좋아요."
      },
      {
        id: 'step3',
        speaker: 'user',
        text: "Would you take $150 for it?",
        korean: "150달러에 주실 수 있나요?",
        options: [
          "Would you take $150 for it?",
          "Can you do $150?",
          "How about $150?"
        ]
      }
    ]
  },

  // ============= 교육 (Education) - 15개 =============
  {
    id: 'university-registration',
    title: '대학교 수강신청',
    category: '교육',
    difficulty: 'intermediate',
    setting: '대학교 학사과',
    description: '대학교에서 수강신청과 관련된 업무를 처리하는 상황을 연습합니다.',
    duration: '7-10분',
    objectives: [
      '수강신청 절차 문의하기',
      '강의 정보 확인하기',
      '선수과목 요건 이해하기',
      '학점 관련 상담하기'
    ],
    icon: GraduationCap,
    imageUrl: 'https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good morning! How can I assist you with your course registration today?",
        korean: "안녕하세요! 오늘 수강신청과 관련해서 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Hi, I'd like to register for the Advanced Statistics course, but I'm not sure if I meet the prerequisites.",
        korean: "안녕하세요, 고급 통계학 강의를 수강신청하고 싶은데, 선수과목 요건을 충족하는지 확실하지 않습니다.",
        options: [
          "Hi, I'd like to register for the Advanced Statistics course, but I'm not sure if I meet the prerequisites.",
          "I want to take Advanced Statistics, but I need to check the requirements.",
          "Could you help me with registering for Advanced Statistics? I'm unsure about the prerequisites."
        ]
      }
    ]
  },
  {
    id: 'library-assistance',
    title: '도서관 이용',
    category: '교육',
    difficulty: 'beginner',
    setting: '대학 도서관',
    description: '도서관에서 자료를 찾고 대출하는 상황을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '도서 위치 문의하기',
      '대출 절차 이해하기',
      '연구 자료 요청하기',
      '스터디룸 예약하기'
    ],
    icon: Book,
    imageUrl: 'https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "Excuse me, I'm looking for books on machine learning. Where can I find them?",
        korean: "실례합니다, 머신러닝 관련 책을 찾고 있는데요. 어디에서 찾을 수 있나요?",
        options: [
          "Excuse me, I'm looking for books on machine learning. Where can I find them?",
          "Hi, could you help me find machine learning books?",
          "Where's the section for computer science and AI books?"
        ]
      }
    ]
  },
  {
    id: 'professor-office-hours',
    title: '교수님 면담',
    category: '교육',
    difficulty: 'intermediate',
    setting: '교수 연구실',
    description: '교수님과 학업 상담을 하는 상황을 연습합니다.',
    duration: '6-8분',
    objectives: [
      '과제 관련 질문하기',
      '성적 상담하기',
      '연구 기회 문의하기',
      '추천서 요청하기'
    ],
    icon: Users,
    imageUrl: 'https://images.unsplash.com/photo-1577896851231-70ef18881754?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "Professor Smith, thank you for seeing me. I have some questions about the research paper.",
        korean: "스미스 교수님, 만나주셔서 감사합니다. 연구 논문에 대해 몇 가지 질문이 있습니다.",
        options: [
          "Professor Smith, thank you for seeing me. I have some questions about the research paper.",
          "Thanks for your time, Professor. I'd like to discuss the research assignment.",
          "Professor, I appreciate your time. Could we talk about the paper requirements?"
        ]
      }
    ]
  },
  {
    id: 'study-group',
    title: '스터디 그룹',
    category: '교육',
    difficulty: 'intermediate',
    setting: '도서관 스터디룸',
    description: '동료들과 함께 공부하며 토론하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '의견 제시하기',
      '설명 요청하기',
      '역할 분담하기',
      '일정 조율하기'
    ],
    icon: Users,
    imageUrl: 'https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Okay everyone, let's divide up the chapters for our presentation.",
        korean: "자, 모두들, 발표를 위해 챕터를 나눠봅시다."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I can take chapters 3 and 4 on data analysis.",
        korean: "저는 데이터 분석에 관한 3장과 4장을 맡을 수 있습니다.",
        options: [
          "I can take chapters 3 and 4 on data analysis.",
          "I'll handle the data analysis sections.",
          "Let me do chapters 3 and 4 about data analysis."
        ]
      }
    ]
  },
  {
    id: 'campus-tour',
    title: '캠퍼스 투어',
    category: '교육',
    difficulty: 'beginner',
    setting: '대학 캠퍼스',
    description: '신입생으로서 캠퍼스 투어에 참여하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '시설 위치 문의하기',
      '캠퍼스 서비스 이해하기',
      '학생 활동 정보 얻기',
      '기숙사 정보 확인하기'
    ],
    icon: MapPin,
    imageUrl: 'https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Welcome to our campus tour! I'm Jake, your guide today.",
        korean: "캠퍼스 투어에 오신 것을 환영합니다! 저는 오늘 가이드를 맡은 제이크입니다."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "Nice to meet you, Jake. Where is the main library located?",
        korean: "만나서 반갑습니다, 제이크. 중앙 도서관은 어디에 있나요?",
        options: [
          "Nice to meet you, Jake. Where is the main library located?",
          "Hi Jake! Can you show us where the library is?",
          "Thanks Jake. I'd like to know where to find the library."
        ]
      }
    ]
  },

  // ============= 의료 (Medical) - 15개 =============
  {
    id: 'medical-appointment',
    title: '병원 진료 예약',
    category: '의료',
    difficulty: 'intermediate',
    setting: '종합병원 접수처',
    description: '병원에서 진료 예약을 잡고 증상을 설명하는 상황을 연습합니다.',
    duration: '6-9분',
    objectives: [
      '진료 예약 요청하기',
      '증상 구체적으로 설명하기',
      '의료진과 효과적으로 소통하기',
      '후속 조치 이해하기'
    ],
    icon: Activity,
    imageUrl: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Good afternoon. How can I help you today?",
        korean: "안녕하세요. 오늘 어떻게 도와드릴까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I'd like to schedule an appointment with a doctor. I've been having persistent headaches.",
        korean: "의사선생님과 진료 예약을 잡고 싶습니다. 계속 두통이 있어서요.",
        options: [
          "I'd like to schedule an appointment with a doctor. I've been having persistent headaches.",
          "I need to see a doctor about my headaches.",
          "Could I book an appointment? I'm experiencing ongoing headaches."
        ]
      }
    ]
  },
  {
    id: 'pharmacy-prescription',
    title: '약국에서 처방전',
    category: '의료',
    difficulty: 'beginner',
    setting: '동네 약국',
    description: '처방전을 가지고 약국에서 약을 받는 상황을 연습합니다.',
    duration: '4-6분',
    objectives: [
      '처방전 제출하기',
      '복용법 확인하기',
      '부작용 문의하기',
      '보험 처리하기'
    ],
    icon: Pill,
    imageUrl: 'https://images.unsplash.com/photo-1587854692152-cbe660dbde88?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "Hi, I have a prescription to fill.",
        korean: "안녕하세요, 처방전 조제를 부탁드립니다.",
        options: [
          "Hi, I have a prescription to fill.",
          "I need to get this prescription filled, please.",
          "Could you fill this prescription for me?"
        ]
      },
      {
        id: 'step2',
        speaker: 'npc',
        text: "Sure, let me see the prescription. Do you have insurance?",
        korean: "네, 처방전을 보여주세요. 보험이 있으신가요?"
      }
    ]
  },
  {
    id: 'emergency-room',
    title: '응급실 방문',
    category: '의료',
    difficulty: 'advanced',
    setting: '병원 응급실',
    description: '응급 상황에서 증상을 설명하고 치료받는 과정을 연습합니다.',
    duration: '7-10분',
    objectives: [
      '긴급 상황 설명하기',
      '통증 정도 표현하기',
      '의료 이력 제공하기',
      '치료 과정 이해하기'
    ],
    icon: Activity,
    imageUrl: 'https://images.unsplash.com/photo-1516549655169-df83a0774514?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "What brings you to the emergency room today?",
        korean: "오늘 응급실에 오신 이유가 무엇인가요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I've been having severe chest pain for the last hour.",
        korean: "지난 한 시간 동안 심한 가슴 통증이 있었습니다.",
        options: [
          "I've been having severe chest pain for the last hour.",
          "My chest hurts badly. It started an hour ago.",
          "I have intense chest pain that began about an hour ago."
        ]
      }
    ]
  },
  {
    id: 'dental-checkup',
    title: '치과 검진',
    category: '의료',
    difficulty: 'intermediate',
    setting: '치과 클리닉',
    description: '정기 치과 검진을 받는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '치아 문제 설명하기',
      '치료 옵션 논의하기',
      '예약 일정 잡기',
      '치료 비용 확인하기'
    ],
    icon: Heart,
    imageUrl: 'https://images.unsplash.com/photo-1606811841689-23dfddce3e95?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "How long has it been since your last dental checkup?",
        korean: "마지막 치과 검진이 언제였나요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "It's been about six months. I'm here for my regular cleaning.",
        korean: "약 6개월 전이었습니다. 정기 스케일링을 받으러 왔습니다.",
        options: [
          "It's been about six months. I'm here for my regular cleaning.",
          "Six months ago. I need my routine dental cleaning.",
          "About half a year. I'm due for my regular checkup and cleaning."
        ]
      }
    ]
  },
  {
    id: 'health-insurance',
    title: '건강보험 상담',
    category: '의료',
    difficulty: 'advanced',
    setting: '보험회사 사무실',
    description: '건강보험 가입과 보장 내용을 상담받는 상황을 연습합니다.',
    duration: '8-10분',
    objectives: [
      '보험 플랜 비교하기',
      '보장 범위 확인하기',
      '보험료 협상하기',
      '청구 절차 이해하기'
    ],
    icon: Shield,
    imageUrl: 'https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Welcome! Are you looking for individual or family health insurance?",
        korean: "환영합니다! 개인 건강보험을 찾으시나요, 가족 보험을 찾으시나요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I need family coverage for myself, my spouse, and two children.",
        korean: "저와 배우자, 그리고 두 자녀를 위한 가족 보험이 필요합니다.",
        options: [
          "I need family coverage for myself, my spouse, and two children.",
          "Family insurance for four people - two adults and two kids.",
          "I'm looking for a family plan that covers four members."
        ]
      }
    ]
  },

  // ============= 기술/IT (Technology) - 10개 =============
  {
    id: 'tech-support',
    title: '기술 지원 요청',
    category: '기술/IT',
    difficulty: 'intermediate',
    setting: 'IT 헬프데스크',
    description: '컴퓨터 문제를 해결하기 위해 IT 지원을 받는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '문제 상황 설명하기',
      '이미 시도한 해결책 전달하기',
      '기술 용어 이해하기',
      '단계별 지침 따르기'
    ],
    icon: Wrench,
    imageUrl: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "IT Support, this is Tom. What seems to be the problem?",
        korean: "IT 지원팀 톰입니다. 무엇이 문제인가요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "My computer won't connect to the WiFi network since this morning.",
        korean: "오늘 아침부터 컴퓨터가 와이파이 네트워크에 연결되지 않습니다.",
        options: [
          "My computer won't connect to the WiFi network since this morning.",
          "I can't get WiFi to work on my laptop.",
          "The WiFi connection stopped working this morning."
        ]
      }
    ]
  },
  {
    id: 'software-training',
    title: '소프트웨어 교육',
    category: '기술/IT',
    difficulty: 'intermediate',
    setting: '회사 교육실',
    description: '새로운 소프트웨어 사용법을 배우는 교육 상황을 연습합니다.',
    duration: '6-8분',
    objectives: [
      '기능 관련 질문하기',
      '단축키 확인하기',
      '실습 도움 요청하기',
      '추가 자료 요청하기'
    ],
    icon: Laptop,
    imageUrl: 'https://images.unsplash.com/photo-1531482615713-2afd69097998?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Today we'll be learning the basics of our new project management software.",
        korean: "오늘은 새로운 프로젝트 관리 소프트웨어의 기본 사항을 배우겠습니다."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "How do I create a new project and assign team members?",
        korean: "새 프로젝트를 만들고 팀원을 할당하는 방법은 무엇인가요?",
        options: [
          "How do I create a new project and assign team members?",
          "Can you show me how to set up a project with team assignments?",
          "What's the process for creating projects and adding people?"
        ]
      }
    ]
  },
  {
    id: 'internet-setup',
    title: '인터넷 설치',
    category: '기술/IT',
    difficulty: 'beginner',
    setting: '가정집',
    description: '인터넷 설치 기사와 소통하며 설치를 진행하는 상황을 연습합니다.',
    duration: '5-7분',
    objectives: [
      '설치 위치 지정하기',
      '속도 플랜 확인하기',
      '장비 설명 듣기',
      '문제 해결 방법 배우기'
    ],
    icon: Wifi,
    imageUrl: 'https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Hi, I'm here to install your internet service. Where would you like the router?",
        korean: "안녕하세요, 인터넷 서비스 설치하러 왔습니다. 라우터를 어디에 설치하면 좋을까요?"
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "I'd like it in the living room, near the TV if possible.",
        korean: "거실에, 가능하면 TV 근처에 설치했으면 좋겠습니다.",
        options: [
          "I'd like it in the living room, near the TV if possible.",
          "Can you put it in the living room by the television?",
          "The living room would be best, close to the TV area."
        ]
      }
    ]
  },
  {
    id: 'cloud-storage',
    title: '클라우드 스토리지',
    category: '기술/IT',
    difficulty: 'intermediate',
    setting: '온라인 고객 지원',
    description: '클라우드 스토리지 서비스를 설정하고 사용하는 방법을 배웁니다.',
    duration: '5-7분',
    objectives: [
      '계정 설정하기',
      '파일 업로드 방법 배우기',
      '공유 설정 이해하기',
      '보안 옵션 확인하기'
    ],
    icon: Cloud,
    imageUrl: 'https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'user',
        text: "I need help setting up cloud storage for my business files.",
        korean: "비즈니스 파일을 위한 클라우드 스토리지 설정에 도움이 필요합니다.",
        options: [
          "I need help setting up cloud storage for my business files.",
          "Can you assist me with cloud storage setup for work documents?",
          "I'd like to configure cloud storage for my company data."
        ]
      }
    ]
  },
  {
    id: 'cybersecurity-awareness',
    title: '사이버보안 교육',
    category: '기술/IT',
    difficulty: 'advanced',
    setting: '기업 보안 교육',
    description: '사이버보안 위협과 대응 방법에 대해 학습하는 상황을 연습합니다.',
    duration: '7-9분',
    objectives: [
      '보안 위협 이해하기',
      '비밀번호 정책 확인하기',
      '피싱 이메일 식별하기',
      '사고 대응 절차 배우기'
    ],
    icon: Lock,
    imageUrl: 'https://images.unsplash.com/photo-1563986768609-322da13575f3?w=800&q=80',
    steps: [
      {
        id: 'step1',
        speaker: 'npc',
        text: "Today's training covers identifying and preventing phishing attacks.",
        korean: "오늘 교육은 피싱 공격을 식별하고 예방하는 방법을 다룹니다."
      },
      {
        id: 'step2',
        speaker: 'user',
        text: "What are the most common signs of a phishing email?",
        korean: "피싱 이메일의 가장 일반적인 징후는 무엇인가요?",
        options: [
          "What are the most common signs of a phishing email?",
          "How can I identify phishing attempts in emails?",
          "What should I look for to spot phishing messages?"
        ]
      }
    ]
  }
]