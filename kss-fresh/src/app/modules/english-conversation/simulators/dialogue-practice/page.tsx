'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  MessageCircle, Volume2, Users, Home, RotateCcw, 
  CheckCircle, Star, Clock, Target, Play, Pause, Shuffle
} from 'lucide-react'

interface DialogueScenario {
  id: string
  title: string
  category: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  setting: string
  roleA: string
  roleB: string
  objective: string
  dialogue: Array<{
    speaker: 'A' | 'B'
    text: string
    korean: string
    tips?: string[]
  }>
}

const dialogueScenarios: DialogueScenario[] = [
  {
    id: 'coffee-shop',
    title: '커피숍에서 주문하기',
    category: '일상생활',
    difficulty: 'beginner',
    setting: '스타벅스 같은 커피숍',
    roleA: '고객 (Customer)',
    roleB: '바리스타 (Barista)',
    objective: '원하는 음료를 주문하고 결제하기',
    dialogue: [
      {
        speaker: 'B',
        text: "Good morning! Welcome to StarCafe. What can I get started for you today?",
        korean: "안녕하세요! 스타카페에 오신 것을 환영합니다. 오늘 뭘 드려드릴까요?"
      },
      {
        speaker: 'A',
        text: "Hi! I'd like a large iced americano, please.",
        korean: "안녕하세요! 아이스 아메리카노 라지 사이즈로 주세요.",
        tips: ["Would you like...로 시작하면 더 정중해요", "Could I have... 도 좋은 표현입니다"]
      },
      {
        speaker: 'B',
        text: "Sure! Would you like to add any extra shots or flavoring?",
        korean: "네! 추가 샷이나 시럽을 넣으시겠어요?"
      },
      {
        speaker: 'A',
        text: "No thanks, just regular is fine. How much is that?",
        korean: "아니요 괜찮습니다, 그냥 기본으로 주세요. 얼마인가요?"
      },
      {
        speaker: 'B',
        text: "That'll be $4.50. Will you be paying with cash or card?",
        korean: "4달러 50센트입니다. 현금으로 드실 건가요 카드로 드실 건가요?"
      },
      {
        speaker: 'A',
        text: "Card, please. Here you go.",
        korean: "카드로 주세요. 여기 있습니다."
      },
      {
        speaker: 'B',
        text: "Perfect! Your drink will be ready in just a few minutes. Have a great day!",
        korean: "완벽합니다! 음료는 몇 분 후에 준비될 예정입니다. 좋은 하루 되세요!"
      }
    ]
  },
  {
    id: 'job-interview',
    title: '취업 면접',
    category: '비즈니스',
    difficulty: 'advanced',
    setting: 'IT 회사 면접실',
    roleA: '면접관 (Interviewer)',
    roleB: '지원자 (Candidate)',
    objective: '전문적이고 자신감 있게 면접 응답하기',
    dialogue: [
      {
        speaker: 'A',
        text: "Good afternoon. Please, have a seat. Could you start by telling me a little about yourself?",
        korean: "안녕하세요. 앉으세요. 먼저 자기소개를 해주시겠어요?",
        tips: ["첫인상이 중요합니다", "간결하면서도 핵심적인 내용으로"]
      },
      {
        speaker: 'B',
        text: "Thank you for this opportunity. I'm a software developer with 5 years of experience in full-stack development, particularly in React and Node.js. I'm passionate about creating user-friendly applications and solving complex problems.",
        korean: "이런 기회를 주셔서 감사합니다. 저는 5년간 풀스택 개발 경험이 있는 소프트웨어 개발자이며, 특히 React와 Node.js에 특화되어 있습니다. 사용자 친화적인 애플리케이션을 만들고 복잡한 문제를 해결하는 것에 열정을 가지고 있습니다."
      },
      {
        speaker: 'A',
        text: "That sounds impressive. What drew you to apply for this position at our company?",
        korean: "인상적이네요. 저희 회사의 이 포지션에 지원하게 된 동기가 무엇인가요?"
      },
      {
        speaker: 'B',
        text: "I've been following your company's innovative work in AI-powered solutions. Your recent project on automated customer service really caught my attention. I believe my background in both frontend and backend development, combined with my interest in AI, would allow me to contribute meaningfully to your team.",
        korean: "저는 귀하 회사의 AI 기반 솔루션에서의 혁신적인 업무를 관심 있게 지켜봤습니다. 특히 자동화된 고객 서비스 프로젝트가 제 관심을 끌었습니다. 프론트엔드와 백엔드 개발 경험과 AI에 대한 관심을 결합해 팀에 의미 있는 기여를 할 수 있을 것이라고 생각합니다."
      },
      {
        speaker: 'A',
        text: "What would you say is your greatest strength as a developer?",
        korean: "개발자로서 가장 큰 강점이 무엇이라고 생각하시나요?"
      },
      {
        speaker: 'B',
        text: "I'd say my greatest strength is my ability to break down complex problems into manageable pieces. For example, in my previous role, I led the refactoring of a legacy system that was affecting performance. I approached it systematically, identifying bottlenecks and implementing solutions one by one, which resulted in a 40% improvement in load times.",
        korean: "제 가장 큰 강점은 복잡한 문제를 관리 가능한 단위로 분해하는 능력이라고 생각합니다. 예를 들어, 이전 직장에서 성능에 영향을 주는 레거시 시스템 리팩토링을 주도했습니다. 체계적으로 접근해서 병목 지점을 파악하고 하나씩 솔루션을 구현한 결과, 로드 타임을 40% 개선했습니다."
      }
    ]
  },
  {
    id: 'hotel-checkin',
    title: '호텔 체크인',
    category: '여행',
    difficulty: 'intermediate',
    setting: '5성급 호텔 프론트 데스크',
    roleA: '투숙객 (Guest)',
    roleB: '호텔 직원 (Front Desk)',
    objective: '원활하게 체크인하고 호텔 정보 얻기',
    dialogue: [
      {
        speaker: 'B',
        text: "Good evening and welcome to the Grand Palace Hotel. How may I assist you tonight?",
        korean: "안녕하세요, 그랜드 팰리스 호텔에 오신 것을 환영합니다. 오늘 밤 어떻게 도와드릴까요?"
      },
      {
        speaker: 'A',
        text: "Hi, I have a reservation under the name Johnson for tonight.",
        korean: "안녕하세요, 오늘 밤 Johnson 이름으로 예약이 되어 있습니다."
      },
      {
        speaker: 'B',
        text: "Let me check that for you. Yes, I see a reservation for two nights in a deluxe king room. May I have your ID and credit card for incidentals?",
        korean: "확인해보겠습니다. 네, 디럭스 킹룸으로 2박 예약이 보입니다. 신분증과 부대비용용 신용카드를 주시겠어요?"
      },
      {
        speaker: 'A',
        text: "Of course, here they are. Is breakfast included in my reservation?",
        korean: "물론이죠, 여기 있습니다. 제 예약에 조식이 포함되어 있나요?"
      },
      {
        speaker: 'B',
        text: "Yes, you have complimentary breakfast included. It's served from 6:30 AM to 10:30 AM in our Garden Restaurant on the second floor.",
        korean: "네, 무료 조식이 포함되어 있습니다. 2층 가든 레스토랑에서 오전 6시 30분부터 10시 30분까지 제공됩니다."
      },
      {
        speaker: 'A',
        text: "Perfect! What time is checkout, and do you have a gym or pool?",
        korean: "완벽하네요! 체크아웃은 몇 시이고, 헬스장이나 수영장이 있나요?"
      },
      {
        speaker: 'B',
        text: "Checkout is at 12:00 PM, but we can arrange a late checkout until 2:00 PM if needed. We have a fully equipped gym on the 3rd floor and a rooftop pool that's open until 10:00 PM. Here's your key card for room 815.",
        korean: "체크아웃은 오후 12시이지만, 필요하시면 오후 2시까지 늦은 체크아웃을 안내해 드릴 수 있습니다. 3층에 완전히 갖춰진 헬스장이 있고, 옥상 수영장은 오후 10시까지 운영합니다. 815호실 키카드입니다."
      }
    ]
  },
  {
    id: 'doctor-appointment',
    title: '병원 진료 예약',
    category: '건강',
    difficulty: 'intermediate',
    setting: '종합병원 접수처',
    roleA: '환자 (Patient)',
    roleB: '접수 직원 (Receptionist)',
    objective: '증상 설명하고 진료 예약 잡기',
    dialogue: [
      {
        speaker: 'B',
        text: "Good morning, how can I help you today?",
        korean: "안녕하세요, 오늘 어떻게 도와드릴까요?"
      },
      {
        speaker: 'A',
        text: "Hi, I'd like to schedule an appointment with a doctor. I've been having some persistent headaches for the past week.",
        korean: "안녕하세요, 의사선생님과 진료 예약을 잡고 싶습니다. 지난 주부터 계속 두통이 있어서요."
      },
      {
        speaker: 'B',
        text: "I'm sorry to hear that. For headaches, I'd recommend seeing Dr. Martinez in our neurology department. Are you experiencing any other symptoms?",
        korean: "안타깝네요. 두통의 경우 신경과의 마르티네즈 박사님을 추천드립니다. 다른 증상은 없으신가요?"
      },
      {
        speaker: 'A',
        text: "Yes, I've also been feeling dizzy sometimes, especially when I stand up quickly. And I've been more tired than usual.",
        korean: "네, 가끔 어지럽기도 하고, 특히 빨리 일어날 때 그렇습니다. 그리고 평소보다 더 피곤해요."
      },
      {
        speaker: 'B',
        text: "I see. Those symptoms together should definitely be checked. Dr. Martinez has an opening this Thursday at 2:30 PM or Friday at 10:00 AM. Which works better for you?",
        korean: "그렇군요. 그런 증상들은 확실히 검사를 받아보셔야 합니다. 마르티네즈 박사님은 이번 주 목요일 오후 2시 30분이나 금요일 오전 10시에 시간이 있습니다. 언제가 더 좋으신가요?"
      },
      {
        speaker: 'A',
        text: "Thursday at 2:30 PM would be perfect. Do I need to bring anything specific?",
        korean: "목요일 오후 2시 30분이 완벽합니다. 특별히 가져와야 할 것이 있나요?"
      },
      {
        speaker: 'B',
        text: "Please bring your insurance card, a valid ID, and a list of any medications you're currently taking. Also, try to avoid caffeine on the morning of your appointment. We'll see you Thursday!",
        korean: "보험카드, 유효한 신분증, 그리고 현재 복용 중인 약물 목록을 가져오세요. 그리고 진료 당일 아침에는 카페인을 피해주세요. 목요일에 뵙겠습니다!"
      }
    ]
  },
  {
    id: 'apartment-viewing',
    title: '아파트 보러가기',
    category: '부동산',
    difficulty: 'intermediate',
    setting: '임대용 아파트',
    roleA: '임차인 (Tenant)',
    roleB: '부동산 중개인 (Realtor)',
    objective: '아파트 조건 확인하고 임대 조건 협상하기',
    dialogue: [
      {
        speaker: 'B',
        text: "Welcome! This is the two-bedroom apartment I mentioned over the phone. Let me show you around.",
        korean: "어서오세요! 전화로 말씀드렸던 투룸 아파트입니다. 둘러보여드릴게요."
      },
      {
        speaker: 'A',
        text: "Great! It looks nice from the outside. How much is the monthly rent, and what's included?",
        korean: "좋네요! 밖에서 봤을 때도 좋아보입니다. 월세는 얼마이고, 뭐가 포함되어 있나요?"
      },
      {
        speaker: 'B',
        text: "The rent is $1,800 per month. Water and heating are included, but electricity and internet are separate. There's also a $200 monthly parking fee if you need a space.",
        korean: "월세는 1,800달러입니다. 상하수도와 난방비는 포함되어 있지만, 전기와 인터넷은 따로입니다. 주차 공간이 필요하시면 월 200달러 추가입니다."
      },
      {
        speaker: 'A',
        text: "I see. What about the security deposit and lease terms?",
        korean: "그렇군요. 보증금과 임대 조건은 어떻게 되나요?"
      },
      {
        speaker: 'B',
        text: "We require first month's rent, last month's rent, and one month security deposit upfront. That's $5,400 total to move in. The lease is typically 12 months, but we can discuss shorter terms.",
        korean: "첫 달 월세, 마지막 달 월세, 그리고 한 달 보증금을 미리 내셔야 합니다. 입주하는 데 총 5,400달러입니다. 임대 기간은 보통 12개월이지만, 더 짧은 기간도 논의할 수 있습니다."
      },
      {
        speaker: 'A',
        text: "The apartment is nice, but the upfront cost is quite high. Is there any flexibility on the move-in costs?",
        korean: "아파트는 좋은데, 초기 비용이 꽤 높네요. 입주 비용에 융통성이 있을까요?"
      },
      {
        speaker: 'B',
        text: "I understand it's a significant amount. If you can commit to an 18-month lease, we might be able to waive the last month's rent requirement. That would bring it down to $3,600 upfront.",
        korean: "상당한 금액이라는 걸 이해합니다. 18개월 임대를 약속하신다면, 마지막 달 월세는 면제해드릴 수 있을 것 같습니다. 그러면 초기 비용이 3,600달러로 내려갑니다."
      }
    ]
  }
]

export default function DialoguePractice() {
  const [selectedScenario, setSelectedScenario] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [userRole, setUserRole] = useState<'A' | 'B'>('A')
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)
  const [practiceMode, setPracticeMode] = useState(false)
  const [showTips, setShowTips] = useState(true)

  // 음성 리스트 로딩
  useEffect(() => {
    if ('speechSynthesis' in window) {
      const loadVoices = () => {
        const voices = speechSynthesis.getVoices()
        if (voices.length === 0) {
          speechSynthesis.addEventListener('voiceschanged', loadVoices)
        }
      }
      loadVoices()
    }
  }, [])

  const playAudio = (text: string, index: number) => {
    if (isPlaying) return
    
    setIsPlaying(true)
    setPlayingIndex(index)
    
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel()
      
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'en-US'
      utterance.rate = 0.8
      utterance.pitch = 1.0
      utterance.volume = 1.0
      
      utterance.onend = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      utterance.onerror = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      // 고품질 영어 음성 선택
      const voices = speechSynthesis.getVoices()
      const preferredVoices = [
        'Microsoft Zira - English (United States)',
        'Google US English',
        'Alex',
        'Samantha'
      ]
      
      let selectedVoice = voices.find(voice => 
        preferredVoices.some(preferred => voice.name.includes(preferred))
      ) || voices.find(voice => voice.lang === 'en-US') ||
         voices.find(voice => voice.lang.startsWith('en-'))
      
      if (selectedVoice) {
        utterance.voice = selectedVoice
      }
      
      speechSynthesis.speak(utterance)
    }
  }

  const nextStep = () => {
    if (currentStep < dialogueScenarios[selectedScenario].dialogue.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const resetDialogue = () => {
    setCurrentStep(0)
    setPracticeMode(false)
  }

  const shuffleScenario = () => {
    const randomIndex = Math.floor(Math.random() * dialogueScenarios.length)
    setSelectedScenario(randomIndex)
    setCurrentStep(0)
    setPracticeMode(false)
  }

  const currentDialogue = dialogueScenarios[selectedScenario]
  const currentLine = currentDialogue.dialogue[currentStep]

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300'
      case 'intermediate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300'
      case 'advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-300'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-blue-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/english-conversation"
              className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
                상황별 대화 연습
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                실제 상황별 시나리오로 영어 회화 실력 향상하기
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={shuffleScenario}
              className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <Shuffle className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            </button>
            <button
              onClick={resetDialogue}
              className="p-2 bg-blue-500 text-white rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Scenario Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                시나리오 선택
              </h3>
              <div className="space-y-2">
                {dialogueScenarios.map((scenario, idx) => (
                  <button
                    key={scenario.id}
                    onClick={() => {
                      setSelectedScenario(idx)
                      setCurrentStep(0)
                      setPracticeMode(false)
                    }}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedScenario === idx
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-blue-900/50'
                    }`}
                  >
                    <div className="font-medium">{scenario.title}</div>
                    <div className="text-xs opacity-80">{scenario.category}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-2 py-1 rounded text-xs ${getDifficultyColor(scenario.difficulty)}`}>
                        {scenario.difficulty}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Practice Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mt-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                연습 설정
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    내 역할
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setUserRole('A')}
                      className={`px-3 py-2 rounded-lg text-sm ${
                        userRole === 'A'
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {currentDialogue.roleA}
                    </button>
                    <button
                      onClick={() => setUserRole('B')}
                      className={`px-3 py-2 rounded-lg text-sm ${
                        userRole === 'B'
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {currentDialogue.roleB}
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="practice-mode"
                    checked={practiceMode}
                    onChange={(e) => setPracticeMode(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="practice-mode" className="text-sm text-gray-700 dark:text-gray-300">
                    연습 모드 (내 대사 숨기기)
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="show-tips"
                    checked={showTips}
                    onChange={(e) => setShowTips(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="show-tips" className="text-sm text-gray-700 dark:text-gray-300">
                    팁 표시
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Main Dialogue Area */}
          <div className="lg:col-span-3">
            {/* Scenario Info */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
                    {currentDialogue.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">
                    {currentDialogue.setting}
                  </p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(currentDialogue.difficulty)}`}>
                  {currentDialogue.difficulty.toUpperCase()}
                </span>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4">
                <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🎯 목표</h3>
                <p className="text-gray-700 dark:text-gray-300">{currentDialogue.objective}</p>
              </div>
            </div>

            {/* Dialogue Display */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                  대화 진행 ({currentStep + 1}/{currentDialogue.dialogue.length})
                </h3>
                <div className="flex gap-2">
                  <button
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="px-3 py-1 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded disabled:opacity-50"
                  >
                    이전
                  </button>
                  <button
                    onClick={nextStep}
                    disabled={currentStep === currentDialogue.dialogue.length - 1}
                    className="px-3 py-1 bg-blue-500 text-white rounded disabled:opacity-50"
                  >
                    다음
                  </button>
                </div>
              </div>

              {/* Current Line */}
              <div className="space-y-4">
                {currentDialogue.dialogue.slice(0, currentStep + 1).map((line, idx) => {
                  const isUserLine = practiceMode && line.speaker === userRole
                  const isCurrentLine = idx === currentStep
                  
                  return (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border-l-4 ${
                        line.speaker === 'A'
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20'
                          : 'border-pink-500 bg-pink-50 dark:bg-pink-950/20'
                      } ${isCurrentLine ? 'ring-2 ring-blue-300 dark:ring-blue-600' : ''}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="font-semibold text-gray-800 dark:text-gray-200">
                              {line.speaker === 'A' ? currentDialogue.roleA : currentDialogue.roleB}
                            </span>
                            {line.speaker === userRole && (
                              <span className="px-2 py-1 bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200 text-xs rounded">
                                내 대사
                              </span>
                            )}
                          </div>
                          
                          {!isUserLine && (
                            <>
                              <p className="text-gray-800 dark:text-gray-200 font-medium mb-2">
                                {line.text}
                              </p>
                              <p className="text-gray-600 dark:text-gray-400 text-sm">
                                {line.korean}
                              </p>
                            </>
                          )}
                          
                          {isUserLine && (
                            <div className="bg-yellow-100 dark:bg-yellow-900/20 p-3 rounded">
                              <p className="text-gray-700 dark:text-gray-300 font-medium">
                                여기서 당신이 말할 차례입니다!
                              </p>
                              <button
                                onClick={() => {
                                  const element = document.getElementById(`reveal-${idx}`)
                                  if (element) {
                                    element.style.display = element.style.display === 'none' ? 'block' : 'none'
                                  }
                                }}
                                className="mt-2 text-blue-600 dark:text-blue-400 text-sm underline"
                              >
                                정답 보기
                              </button>
                              <div id={`reveal-${idx}`} style={{ display: 'none' }} className="mt-2">
                                <p className="text-gray-800 dark:text-gray-200 font-medium">
                                  {line.text}
                                </p>
                                <p className="text-gray-600 dark:text-gray-400 text-sm">
                                  {line.korean}
                                </p>
                              </div>
                            </div>
                          )}

                          {line.tips && showTips && (
                            <div className="mt-3 p-3 bg-green-50 dark:bg-green-950/20 rounded-lg">
                              <h4 className="font-medium text-green-800 dark:text-green-300 mb-1">💡 팁</h4>
                              <ul className="text-green-700 dark:text-green-400 text-sm space-y-1">
                                {line.tips.map((tip, tipIdx) => (
                                  <li key={tipIdx}>• {tip}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>

                        {!isUserLine && (
                          <button
                            onClick={() => playAudio(line.text, idx)}
                            disabled={isPlaying}
                            className="ml-4 p-2 hover:bg-white dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
                          >
                            {isPlaying && playingIndex === idx ? (
                              <Pause className="w-5 h-5 text-blue-500" />
                            ) : (
                              <Volume2 className="w-5 h-5 text-blue-500" />
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Progress Bar */}
              <div className="mt-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">진행률</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {Math.round(((currentStep + 1) / currentDialogue.dialogue.length) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${((currentStep + 1) / currentDialogue.dialogue.length) * 100}%`
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}