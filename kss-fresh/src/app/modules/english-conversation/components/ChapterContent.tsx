'use client'

import { useState, useEffect } from 'react'
import { MessageCircle, Volume2, Users, Globe, Copy, CheckCircle, Play, Pause } from 'lucide-react'

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const renderContent = () => {
    switch (chapterId) {
      case 'conversation-basics':
        return <ConversationBasicsContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'daily-situations':
        return <DailySituationsContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'business-english':
        return <BusinessEnglishContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'travel-english':
        return <TravelEnglishContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'pronunciation-intonation':
        return <PronunciationContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'listening-comprehension':
        return <ListeningContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'cultural-context':
        return <CulturalContextContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'advanced-conversation':
        return <AdvancedConversationContent copyCode={copyCode} copiedCode={copiedCode} />
      default:
        return <div>챕터 콘텐츠를 불러올 수 없습니다.</div>
    }
  }

  return <div className="prose prose-lg dark:prose-invert max-w-none">{renderContent()}</div>
}

// Chapter 1: Conversation Basics
function ConversationBasicsContent({ copyCode, copiedCode }: any) {
  const [activeDialogue, setActiveDialogue] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)

  // 음성 리스트 로딩 보장
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
      // 이전 음성이 재생 중이면 중지
      speechSynthesis.cancel()
      
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'en-US'
      utterance.rate = 0.7  // 조금 더 천천히
      utterance.pitch = 1.1  // 약간 높은 톤
      utterance.volume = 1.0
      
      utterance.onend = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      utterance.onerror = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      // 더 나은 영어 음성 선택
      const voices = speechSynthesis.getVoices()
      const preferredVoices = [
        'Microsoft Zira - English (United States)',
        'Google US English',
        'Alex',
        'Samantha'
      ]
      
      let selectedVoice = voices.find(voice => 
        preferredVoices.some(preferred => voice.name.includes(preferred))
      ) || voices.find(voice => voice.lang === 'en-US' && voice.name.toLowerCase().includes('natural')) ||
         voices.find(voice => voice.lang === 'en-US' && voice.name.toLowerCase().includes('neural')) ||
         voices.find(voice => voice.lang === 'en-US') ||
         voices.find(voice => voice.lang.startsWith('en-'))
      
      if (selectedVoice) {
        utterance.voice = selectedVoice
        console.log('Using voice:', selectedVoice.name)
      }
      
      speechSynthesis.speak(utterance)
    } else {
      // SpeechSynthesis가 지원되지 않는 경우
      setTimeout(() => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }, 2000)
    }
  }
  
  const basicDialogues = [
    {
      title: "인사와 자기소개",
      english: [
        "A: Hi! I'm Sarah. Nice to meet you!",
        "B: Hello, Sarah! I'm Mike. Nice to meet you too!",
        "A: Where are you from, Mike?",
        "B: I'm from Canada. How about you?",
        "A: I'm from Korea. What do you do?",
        "B: I'm a software engineer. What about you?"
      ],
      korean: [
        "A: 안녕하세요! 저는 사라예요. 만나서 반가워요!",
        "B: 안녕하세요, 사라! 저는 마이크예요. 저도 만나서 반가워요!",
        "A: 마이크는 어디 출신이에요?",
        "B: 저는 캐나다 출신이에요. 사라는 어떠세요?",
        "A: 저는 한국 출신이에요. 무슨 일을 하세요?",
        "B: 저는 소프트웨어 엔지니어예요. 사라는 어떤 일을 하세요?"
      ]
    },
    {
      title: "감정과 상태 표현",
      english: [
        "A: How are you feeling today?",
        "B: I'm feeling great! I had a good night's sleep.",
        "A: That's wonderful! I'm a bit tired myself.",
        "B: Oh, why is that?",
        "A: I stayed up late watching a movie last night.",
        "B: Which movie? Was it good?"
      ],
      korean: [
        "A: 오늘 기분이 어떠세요?",
        "B: 기분이 아주 좋아요! 밤에 잠을 잘 잤거든요.",
        "A: 정말 좋네요! 저는 좀 피곤해요.",
        "B: 아, 왜 그러세요?",
        "A: 어젯밤에 영화를 보느라 늦게 잤어요.",
        "B: 무슨 영화요? 재밌었어요?"
      ]
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          기초 회화 패턴 마스터하기
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          영어 회화의 첫걸음은 기본적인 대화 패턴을 익히는 것입니다. 
          일상생활에서 가장 자주 사용되는 표현들을 통해 자연스러운 대화의 기초를 다져보겠습니다.
        </p>
      </div>

      <div className="bg-rose-50 dark:bg-rose-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🗣️ 실전 대화 연습
        </h3>
        
        <div className="flex gap-2 mb-4">
          {basicDialogues.map((dialogue, idx) => (
            <button
              key={idx}
              onClick={() => setActiveDialogue(idx)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeDialogue === idx
                  ? 'bg-rose-500 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-rose-100 dark:hover:bg-rose-900/50'
              }`}
            >
              {dialogue.title}
            </button>
          ))}
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 space-y-3">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200">
            {basicDialogues[activeDialogue].title}
          </h4>
          {basicDialogues[activeDialogue].english.map((line, idx) => (
            <div key={idx} className="space-y-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => playAudio(line, idx)}
                  disabled={isPlaying}
                  className="p-1 hover:bg-rose-100 dark:hover:bg-rose-900/50 rounded transition-colors disabled:opacity-50"
                >
                  {isPlaying && playingIndex === idx ? (
                    <Pause className="w-4 h-4 text-rose-500" />
                  ) : (
                    <Volume2 className="w-4 h-4 text-rose-500" />
                  )}
                </button>
                <p className="text-gray-800 dark:text-gray-200 font-medium">{line}</p>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 ml-6">
                {basicDialogues[activeDialogue].korean[idx]}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            💡 핵심 패턴
          </h3>
          <div className="space-y-3">
            <div>
              <h4 className="font-medium text-gray-700 dark:text-gray-300">인사 패턴</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 ml-4">
                <li>• Hi/Hello + 이름 + Nice to meet you</li>
                <li>• How are you? / How's it going?</li>
                <li>• What's up? (비격식)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 dark:text-gray-300">질문 패턴</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 ml-4">
                <li>• Where are you from?</li>
                <li>• What do you do?</li>
                <li>• How long have you been...?</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🎯 연습 포인트
          </h3>
          <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-start gap-2">
              <div className="w-2 h-2 bg-rose-500 rounded-full mt-2"></div>
              <div>
                <span className="font-medium">자연스러운 억양:</span> 문장 끝을 살짝 올려서 친근함을 표현
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-2 h-2 bg-rose-500 rounded-full mt-2"></div>
              <div>
                <span className="font-medium">아이컨택:</span> 말할 때 상대방의 눈을 보며 소통
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-2 h-2 bg-rose-500 rounded-full mt-2"></div>
              <div>
                <span className="font-medium">적절한 속도:</span> 너무 빠르지 않게, 명확하게 발음
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-rose-500 to-pink-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🚀 실습 미션</h3>
        <div className="space-y-2 text-rose-100">
          <p>1. 거울을 보며 자기소개를 5번 연습해보세요</p>
          <p>2. 가족이나 친구와 역할극으로 대화 연습을 해보세요</p>
          <p>3. AI 대화 파트너와 실제 대화를 시도해보세요</p>
        </div>
      </div>
    </div>
  )
}

// Chapter 2: Daily Situations
function DailySituationsContent({ copyCode, copiedCode }: any) {
  const [activeScenario, setActiveScenario] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)

  // 음성 리스트 로딩 보장
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
      utterance.rate = 0.7  // 조금 더 천천히
      utterance.pitch = 1.1  // 약간 높은 톤
      utterance.volume = 1.0
      
      utterance.onend = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      utterance.onerror = () => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }
      
      // 더 나은 영어 음성 선택
      const voices = speechSynthesis.getVoices()
      const preferredVoices = [
        'Microsoft Zira - English (United States)',
        'Google US English',
        'Alex',
        'Samantha'
      ]
      
      let selectedVoice = voices.find(voice => 
        preferredVoices.some(preferred => voice.name.includes(preferred))
      ) || voices.find(voice => voice.lang === 'en-US' && voice.name.toLowerCase().includes('natural')) ||
         voices.find(voice => voice.lang === 'en-US' && voice.name.toLowerCase().includes('neural')) ||
         voices.find(voice => voice.lang === 'en-US') ||
         voices.find(voice => voice.lang.startsWith('en-'))
      
      if (selectedVoice) {
        utterance.voice = selectedVoice
        console.log('Using voice:', selectedVoice.name)
      }
      
      speechSynthesis.speak(utterance)
    } else {
      setTimeout(() => {
        setIsPlaying(false)
        setPlayingIndex(null)
      }, 2000)
    }
  }
  
  const scenarios = [
    {
      title: "쇼핑",
      situation: "의류 매장에서 쇼핑하기",
      dialogue: [
        "Staff: Can I help you find anything?",
        "Customer: Yes, I'm looking for a winter jacket.",
        "Staff: What size are you looking for?",
        "Customer: Medium, please. Do you have this in blue?",
        "Staff: Let me check for you. Yes, here's a medium in navy blue.",
        "Customer: Perfect! How much is it?",
        "Staff: It's $89. Would you like to try it on?"
      ]
    },
    {
      title: "식당",
      situation: "레스토랑에서 주문하기",
      dialogue: [
        "Waiter: Good evening! Table for how many?",
        "Customer: Table for two, please.",
        "Waiter: Right this way. Here's your menu.",
        "Customer: Thank you. Could I have a few minutes to decide?",
        "Waiter: Of course. Can I start you with something to drink?",
        "Customer: I'll have water and a coffee, please."
      ]
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          일상 상황별 실전 영어
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          실제 생활에서 마주치는 다양한 상황에서 자신감 있게 영어로 소통하는 방법을 배워보겠습니다.
        </p>
      </div>

      <div className="bg-pink-50 dark:bg-pink-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎬 상황별 시나리오
        </h3>
        
        <div className="flex gap-2 mb-4">
          {scenarios.map((scenario, idx) => (
            <button
              key={idx}
              onClick={() => setActiveScenario(idx)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeScenario === idx
                  ? 'bg-pink-500 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-pink-100 dark:hover:bg-pink-900/50'
              }`}
            >
              {scenario.title}
            </button>
          ))}
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
            {scenarios[activeScenario].situation}
          </h4>
          <div className="space-y-2">
            {scenarios[activeScenario].dialogue.map((line, idx) => (
              <div key={idx} className="flex items-start gap-3">
                <div className="w-8 h-8 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center text-xs font-bold text-pink-600 dark:text-pink-400">
                  {idx + 1}
                </div>
                <p className="text-gray-800 dark:text-gray-200 flex-1">{line}</p>
                <button
                  onClick={() => playAudio(line, idx)}
                  disabled={isPlaying}
                  className="p-1 hover:bg-pink-100 dark:hover:bg-pink-900/50 rounded transition-colors disabled:opacity-50 mt-1"
                >
                  {isPlaying && playingIndex === idx ? (
                    <Pause className="w-4 h-4 text-pink-500" />
                  ) : (
                    <Volume2 className="w-4 h-4 text-pink-500" />
                  )}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🛍️ 쇼핑 필수 표현
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Can I help you?</span>
              <span className="text-gray-500 dark:text-gray-400">도와드릴까요?</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">How much is it?</span>
              <span className="text-gray-500 dark:text-gray-400">얼마예요?</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Can I try it on?</span>
              <span className="text-gray-500 dark:text-gray-400">입어볼 수 있나요?</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Do you have this in...?</span>
              <span className="text-gray-500 dark:text-gray-400">이것을 ...로 있나요?</span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🍽️ 식당 필수 표현
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Table for two</span>
              <span className="text-gray-500 dark:text-gray-400">2명 테이블</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Can I see the menu?</span>
              <span className="text-gray-500 dark:text-gray-400">메뉴 보여주세요</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">I'll have...</span>
              <span className="text-gray-500 dark:text-gray-400">저는 ...로 할게요</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-700 dark:text-gray-300">Check, please</span>
              <span className="text-gray-500 dark:text-gray-400">계산해주세요</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 3: Business English
function BusinessEnglishContent({ copyCode, copiedCode }: any) {
  const [activeTab, setActiveTab] = useState('meetings')
  const [expandedSection, setExpandedSection] = useState<string | null>(null)

  const businessTopics = [
    { id: 'meetings', name: '회의', icon: '📊' },
    { id: 'presentations', name: '프레젠테이션', icon: '📈' },
    { id: 'emails', name: '이메일', icon: '✉️' },
    { id: 'negotiations', name: '협상', icon: '🤝' },
    { id: 'networking', name: '네트워킹', icon: '🌐' },
    { id: 'phone-calls', name: '전화 통화', icon: '📞' }
  ]

  const meetingExpressions = {
    opening: [
      { expression: "Let's call this meeting to order.", korean: "회의를 시작하겠습니다." },
      { expression: "Thank you all for coming today.", korean: "오늘 참석해 주셔서 감사합니다." },
      { expression: "I'd like to welcome everyone to today's meeting.", korean: "오늘 회의에 참석하신 모든 분들을 환영합니다." },
      { expression: "Let's go around the table and introduce ourselves.", korean: "돌아가면서 자기소개를 해보겠습니다." },
      { expression: "The purpose of today's meeting is to discuss...", korean: "오늘 회의의 목적은 ...에 대해 논의하는 것입니다." }
    ],
    agenda: [
      { expression: "Let's review the agenda.", korean: "의제를 검토해보겠습니다." },
      { expression: "We have three main items on the agenda today.", korean: "오늘 의제에는 세 가지 주요 안건이 있습니다." },
      { expression: "Let's move on to the next item.", korean: "다음 안건으로 넘어가겠습니다." },
      { expression: "Are there any questions about this agenda item?", korean: "이 안건에 대해 질문이 있으신가요?" },
      { expression: "Let's table this discussion for now.", korean: "이 논의는 잠시 보류하겠습니다." }
    ],
    opinions: [
      { expression: "I think we should consider all options.", korean: "모든 선택지를 고려해야 한다고 생각합니다." },
      { expression: "From my perspective, this is the best approach.", korean: "제 관점에서는 이것이 최선의 접근법입니다." },
      { expression: "I'd like to suggest an alternative solution.", korean: "대안을 제시하고 싶습니다." },
      { expression: "I have some concerns about this proposal.", korean: "이 제안에 대해 우려되는 점이 있습니다." },
      { expression: "I couldn't agree more with that point.", korean: "그 점에 전적으로 동의합니다." }
    ],
    disagreeing: [
      { expression: "I respectfully disagree with that assessment.", korean: "그 평가에 정중히 반대합니다." },
      { expression: "I see your point, but I have a different view.", korean: "당신의 요점은 이해하지만, 저는 다른 견해를 가지고 있습니다." },
      { expression: "That's an interesting perspective, however...", korean: "흥미로운 관점이지만, 그러나..." },
      { expression: "I'm not entirely convinced by that argument.", korean: "그 논거에 완전히 설득되지는 않습니다." },
      { expression: "May I offer a counterpoint?", korean: "반박 의견을 제시해도 될까요?" }
    ],
    closing: [
      { expression: "Let's wrap up today's meeting.", korean: "오늘 회의를 마무리하겠습니다." },
      { expression: "To summarize what we've discussed...", korean: "우리가 논의한 내용을 요약하면..." },
      { expression: "What are our next steps?", korean: "다음 단계는 무엇입니까?" },
      { expression: "Who will be responsible for this action item?", korean: "이 실행 항목을 누가 담당할 것입니까?" },
      { expression: "Thank you for your time and participation.", korean: "시간을 내어 참여해 주셔서 감사합니다." }
    ]
  }

  const presentationStructure = [
    {
      section: "Opening",
      expressions: [
        { expression: "Good morning, everyone. Thank you for being here.", korean: "안녕하세요, 여러분. 참석해 주셔서 감사합니다." },
        { expression: "Today I'm going to talk about...", korean: "오늘 저는 ...에 대해 말씀드리겠습니다." },
        { expression: "My presentation will take approximately 20 minutes.", korean: "제 발표는 약 20분 정도 소요될 예정입니다." },
        { expression: "Please feel free to interrupt if you have any questions.", korean: "질문이 있으시면 언제든지 말씀해 주세요." }
      ]
    },
    {
      section: "Main Content",
      expressions: [
        { expression: "Let me start by giving you some background information.", korean: "배경 정보부터 말씀드리겠습니다." },
        { expression: "This slide shows our quarterly results.", korean: "이 슬라이드는 분기별 결과를 보여줍니다." },
        { expression: "As you can see from this chart...", korean: "이 차트에서 보시는 바와 같이..." },
        { expression: "Moving on to the next point...", korean: "다음 요점으로 넘어가서..." },
        { expression: "This brings me to my next slide.", korean: "이제 다음 슬라이드로 넘어가겠습니다." }
      ]
    },
    {
      section: "Closing",
      expressions: [
        { expression: "To sum up, our main points are...", korean: "요약하면, 주요 요점들은..." },
        { expression: "In conclusion, I'd like to emphasize...", korean: "결론적으로, 강조하고 싶은 것은..." },
        { expression: "Thank you for your attention.", korean: "경청해 주셔서 감사합니다." },
        { expression: "Are there any questions?", korean: "질문이 있으신가요?" }
      ]
    }
  ]

  const emailTemplates = [
    {
      type: "Meeting Request",
      subject: "Meeting Request - Q4 Budget Review",
      body: `Dear [Name],

I hope this email finds you well.

I would like to schedule a meeting to discuss our Q4 budget review. The meeting would cover:
• Budget allocations for each department
• Cost optimization opportunities
• Planning for Q1 next year

Would you be available next Tuesday, October 15th, at 2:00 PM? The meeting is expected to last about 60 minutes and will be held in Conference Room A.

Please let me know if this time works for you, or suggest an alternative that fits your schedule.

Best regards,
[Your Name]`,
      korean: "회의 요청 - 4분기 예산 검토"
    },
    {
      type: "Follow-up Email",
      subject: "Follow-up: Action Items from Today's Meeting",
      body: `Dear Team,

Thank you for your participation in today's project review meeting.

As discussed, here are the key action items and deadlines:

1. Market research report - Due: October 20th (John)
2. Technical specifications - Due: October 22nd (Sarah)
3. Budget proposal revision - Due: October 25th (Mike)

Please confirm receipt of this email and let me know if you have any questions about your assigned tasks.

Our next meeting is scheduled for October 30th at 10:00 AM.

Best regards,
[Your Name]`,
      korean: "후속 조치 - 오늘 회의의 실행 항목들"
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          비즈니스 영어 마스터 과정
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          국제 비즈니스 환경에서 성공하기 위한 전문적인 영어 커뮤니케이션 스킬을 체계적으로 학습합니다. 
          회의, 프레젠테이션, 이메일, 협상 등 실무에서 바로 활용할 수 있는 실전 표현들을 마스터하세요.
        </p>
      </div>

      {/* Business Topics Navigation */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🏢 비즈니스 영어 핵심 영역
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {businessTopics.map(topic => (
            <button
              key={topic.id}
              onClick={() => setActiveTab(topic.id)}
              className={`p-3 rounded-lg text-center transition-colors ${
                activeTab === topic.id
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-blue-900/50'
              }`}
            >
              <div className="text-xl mb-1">{topic.icon}</div>
              <div className="text-xs font-medium">{topic.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Meeting Content */}
      {activeTab === 'meetings' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📊 효과적인 회의 진행을 위한 필수 표현
            </h3>
            
            {Object.entries(meetingExpressions).map(([category, expressions]) => (
              <div key={category} className="mb-6">
                <button
                  onClick={() => setExpandedSection(expandedSection === category ? null : category)}
                  className="w-full text-left p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                >
                  <h4 className="font-medium text-gray-800 dark:text-gray-200 capitalize">
                    {category === 'opening' && '🚀 회의 시작'}
                    {category === 'agenda' && '📋 의제 관리'}
                    {category === 'opinions' && '💭 의견 표현'}
                    {category === 'disagreeing' && '🤔 정중한 반대'}
                    {category === 'closing' && '🏁 회의 마무리'}
                  </h4>
                </button>
                
                {expandedSection === category && (
                  <div className="mt-3 space-y-3 pl-4">
                    {expressions.map((item, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                          "{item.expression}"
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {item.korean}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Meeting Role Play Scenarios */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🎭 회의 역할극 시나리오
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">팀 리더 역할</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 회의 시작과 마무리 진행</li>
                  <li>• 의제 관리와 시간 조절</li>
                  <li>• 팀원들의 참여 유도</li>
                  <li>• 결정사항 정리와 후속 조치 배정</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">팀원 역할</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 적극적인 의견 표현</li>
                  <li>• 건설적인 질문하기</li>
                  <li>• 정중한 반대 의견 제시</li>
                  <li>• 실행 가능한 제안하기</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Presentation Content */}
      {activeTab === 'presentations' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📈 임팩트 있는 프레젠테이션 구성법
            </h3>
            
            {presentationStructure.map((section, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {idx + 1}. {section.section}
                </h4>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  {section.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 text-sm mb-1">
                        "{expr.expression}"
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {expr.korean}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Presentation Tips */}
          <div className="bg-amber-50 dark:bg-amber-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              💡 프레젠테이션 성공 비법
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">💬 언어적 요소</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 명확하고 간단한 문장 사용</li>
                  <li>• 핵심 키워드 반복 강조</li>
                  <li>• 논리적 순서로 내용 전개</li>
                  <li>• 청중과의 상호작용 유도</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">👥 청중 관리</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 아이컨택으로 집중도 유지</li>
                  <li>• 적절한 제스처 활용</li>
                  <li>• 질문으로 참여 유도</li>
                  <li>• 피드백에 열린 자세</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">📊 시각 자료</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 한 슬라이드 하나의 메시지</li>
                  <li>• 글보다는 시각적 요소 활용</li>
                  <li>• 일관된 디자인 유지</li>
                  <li>• 데이터는 그래프로 표현</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Email Content */}
      {activeTab === 'emails' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              ✉️ 프로페셔널 이메일 작성법
            </h3>
            
            {emailTemplates.map((template, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  {template.type} - {template.korean}
                </h4>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 font-mono text-sm">
                  <div className="border-b border-gray-200 dark:border-gray-600 pb-2 mb-3">
                    <strong>Subject:</strong> {template.subject}
                  </div>
                  <pre className="whitespace-pre-wrap text-gray-600 dark:text-gray-400">
                    {template.body}
                  </pre>
                </div>
              </div>
            ))}
          </div>

          {/* Email Writing Guidelines */}
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-950/20 dark:to-blue-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📝 이메일 작성 가이드라인
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">DO's ✅</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 명확하고 구체적인 제목 작성</li>
                  <li>• 정중하고 전문적인 톤 유지</li>
                  <li>• 핵심 내용을 먼저 제시</li>
                  <li>• 액션 아이템을 명확히 명시</li>
                  <li>• 마감일과 책임자 지정</li>
                  <li>• 감사 인사로 마무리</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">DON'Ts ❌</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 너무 길거나 복잡한 문장</li>
                  <li>• 모호하거나 불명확한 표현</li>
                  <li>• 감정적이거나 비판적인 톤</li>
                  <li>• 중요한 정보 누락</li>
                  <li>• 맞춤법이나 문법 오류</li>
                  <li>• 불필요한 전체 답장</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Negotiation Content */}
      {activeTab === 'negotiations' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🤝 Win-Win 협상 전략과 표현법
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">협상 시작</h4>
                <ul className="space-y-2 text-sm">
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"Let's find a solution that works for both parties."</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">양측 모두에게 효과적인 해결책을 찾아봅시다.</span>
                  </li>
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"What are your main concerns about this proposal?"</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">이 제안에 대한 주요 우려사항은 무엇입니까?</span>
                  </li>
                </ul>
              </div>
              
              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">조건 제시</h4>
                <ul className="space-y-2 text-sm">
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"We could consider that if you're willing to..."</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">당신이 ...을 기꺼이 한다면 그것을 고려할 수 있습니다.</span>
                  </li>
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"How about we meet in the middle?"</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">중간에서 만나는 것은 어떨까요?</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Final Success Tips */}
      <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🚀 비즈니스 영어 성공을 위한 최종 팁</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-emerald-100">
          <div>
            <h4 className="font-semibold mb-2">💪 자신감 키우기</h4>
            <p className="text-sm">지속적인 연습과 실전 적용을 통해 자연스러운 비즈니스 영어 구사능력을 기르세요.</p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🌐 문화적 맥락 이해</h4>
            <p className="text-sm">언어뿐만 아니라 비즈니스 문화와 에티켓을 함께 학습하여 효과적인 소통을 하세요.</p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">📈 지속적 발전</h4>
            <p className="text-sm">피드백을 적극 수용하고 새로운 표현을 계속 학습하여 전문성을 향상시키세요.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 4: Travel English
function TravelEnglishContent({ copyCode, copiedCode }: any) {
  const [activeTab, setActiveTab] = useState('airport')
  const [expandedTip, setExpandedTip] = useState<string | null>(null)

  const travelSections = [
    { id: 'airport', name: '공항', icon: '✈️' },
    { id: 'hotel', name: '호텔', icon: '🏨' },
    { id: 'restaurant', name: '레스토랑', icon: '🍽️' },
    { id: 'transportation', name: '교통', icon: '🚗' },
    { id: 'shopping', name: '쇼핑', icon: '🛍️' },
    { id: 'emergency', name: '응급상황', icon: '🆘' }
  ]

  const airportSituations = [
    {
      title: "체크인 카운터",
      expressions: [
        { eng: "I have a reservation under the name Smith.", kor: "스미스 이름으로 예약했습니다." },
        { eng: "I'd like a window seat, please.", kor: "창가 좌석으로 부탁드립니다." },
        { eng: "How many bags can I check in?", kor: "몇 개의 가방을 체크인할 수 있나요?" },
        { eng: "Is there an extra charge for overweight luggage?", kor: "수하물 초과 중량에 대한 추가 요금이 있나요?" },
        { eng: "Could I get an aisle seat instead?", kor: "대신 통로쪽 좌석으로 바꿀 수 있을까요?" }
      ]
    },
    {
      title: "보안검색대",
      expressions: [
        { eng: "Do I need to take off my shoes?", kor: "신발을 벗어야 하나요?" },
        { eng: "Can I keep my laptop in the bag?", kor: "노트북을 가방에 넣어둘 수 있나요?" },
        { eng: "Is this the line for international flights?", kor: "이것이 국제선 줄인가요?" },
        { eng: "Where should I put my liquids?", kor: "액체류는 어디에 두어야 하나요?" }
      ]
    },
    {
      title: "출입국 심사",
      expressions: [
        { eng: "I'm here for tourism/business.", kor: "관광/출장으로 왔습니다." },
        { eng: "I'll be staying for two weeks.", kor: "2주 동안 머물 예정입니다." },
        { eng: "This is my first time visiting your country.", kor: "귀하의 나라를 처음 방문합니다." },
        { eng: "I'm staying at the Hilton Hotel.", kor: "힐튼 호텔에 머물 예정입니다." }
      ]
    }
  ]

  const hotelSituations = [
    {
      title: "체크인",
      expressions: [
        { eng: "I have a reservation under Johnson.", kor: "존슨 이름으로 예약이 있습니다." },
        { eng: "Is breakfast included in the rate?", kor: "요금에 조식이 포함되어 있나요?" },
        { eng: "What time is checkout?", kor: "체크아웃 시간이 언제인가요?" },
        { eng: "Could I have a room on a higher floor?", kor: "더 높은 층의 방으로 가능할까요?" },
        { eng: "Is Wi-Fi available in the rooms?", kor: "객실에서 와이파이를 사용할 수 있나요?" }
      ]
    },
    {
      title: "호텔 서비스",
      expressions: [
        { eng: "Could you call me a taxi?", kor: "택시를 불러주실 수 있나요?" },
        { eng: "I'd like to extend my stay for one more night.", kor: "하루 더 연장하고 싶습니다." },
        { eng: "The air conditioning in my room isn't working.", kor: "제 방의 에어컨이 작동하지 않습니다." },
        { eng: "Could I get some extra towels?", kor: "수건을 더 받을 수 있을까요?" },
        { eng: "Is there a gym/pool in the hotel?", kor: "호텔에 헬스장/수영장이 있나요?" }
      ]
    }
  ]

  const emergencySituations = [
    {
      title: "의료 응급상황",
      expressions: [
        { eng: "I need to see a doctor immediately.", kor: "즉시 의사를 만나야 합니다." },
        { eng: "I'm having chest pain.", kor: "가슴이 아픕니다." },
        { eng: "I think I broke my arm.", kor: "팔이 부러진 것 같습니다." },
        { eng: "I'm allergic to penicillin.", kor: "저는 페니실린에 알레르기가 있습니다." },
        { eng: "Where is the nearest hospital?", kor: "가장 가까운 병원이 어디인가요?" }
      ]
    },
    {
      title: "경찰서/분실신고",
      expressions: [
        { eng: "I'd like to report a theft.", kor: "절도를 신고하고 싶습니다." },
        { eng: "My passport has been stolen.", kor: "여권을 도난당했습니다." },
        { eng: "I lost my wallet.", kor: "지갑을 잃어버렸습니다." },
        { eng: "Could you help me find the embassy?", kor: "대사관을 찾는 것을 도와주실 수 있나요?" },
        { eng: "I need to file a police report.", kor: "경찰서에 신고서를 작성해야 합니다." }
      ]
    }
  ]

  const culturalTips = [
    {
      country: "미국",
      tips: [
        "팁 문화: 레스토랑에서 15-20%, 택시에서 15-18% 팁이 관례입니다.",
        "개인공간: 대화할 때 팔 길이 정도의 거리를 유지하세요.",
        "인사: 악수가 일반적이며, 눈을 맞추는 것이 중요합니다.",
        "시간 관념: 약속 시간을 정확히 지키는 것이 매우 중요합니다."
      ]
    },
    {
      country: "영국",
      tips: [
        "줄서기: 영국인들은 줄서기를 매우 중요하게 생각합니다.",
        "예의: 'Please', 'Thank you', 'Sorry' 등의 표현을 자주 사용하세요.",
        "날씨 대화: 날씨에 대한 대화는 좋은 아이스브레이커입니다.",
        "펍 문화: 펍에서는 바에서 직접 주문하고 팁은 필수가 아닙니다."
      ]
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          완벽한 여행 영어 가이드
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          해외여행의 모든 순간을 자신감 있게! 공항부터 호텔, 레스토랑, 쇼핑까지 
          여행의 전 과정에서 필요한 실전 영어 표현을 마스터하세요.
        </p>
      </div>

      {/* Travel Sections Navigation */}
      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🌍 여행 상황별 가이드
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {travelSections.map(section => (
            <button
              key={section.id}
              onClick={() => setActiveTab(section.id)}
              className={`p-3 rounded-lg text-center transition-colors ${
                activeTab === section.id
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-blue-900/50'
              }`}
            >
              <div className="text-xl mb-1">{section.icon}</div>
              <div className="text-xs font-medium">{section.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Airport Content */}
      {activeTab === 'airport' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              ✈️ 공항에서 필요한 모든 표현
            </h3>
            
            {airportSituations.map((situation, idx) => (
              <div key={idx} className="mb-6">
                <button
                  onClick={() => setExpandedTip(expandedTip === situation.title ? null : situation.title)}
                  className="w-full text-left p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                >
                  <h4 className="font-medium text-gray-800 dark:text-gray-200">
                    {idx + 1}. {situation.title}
                  </h4>
                </button>
                
                {expandedTip === situation.title && (
                  <div className="mt-3 space-y-3 pl-4">
                    {situation.expressions.map((expr, exprIdx) => (
                      <div key={exprIdx} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                          "{expr.eng}"
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {expr.kor}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Airport Survival Tips */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/20 dark:to-orange-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              💡 공항 서바이벌 팁
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">📝 체크인 전 준비사항</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 여권과 e-ticket 준비</li>
                  <li>• 수하물 중량 제한 확인</li>
                  <li>• 좌석 선호도 미리 결정</li>
                  <li>• 특별식 요청사항 확인</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🔍 보안검색 통과 요령</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 액체류는 100ml 이하로 준비</li>
                  <li>• 전자기기는 별도 트레이에</li>
                  <li>• 금속 액세서리 미리 제거</li>
                  <li>• 신발 벗기 준비</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hotel Content */}
      {activeTab === 'hotel' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🏨 호텔에서의 완벽한 소통
            </h3>
            
            {hotelSituations.map((situation, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {situation.title}
                </h4>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  {situation.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 text-sm mb-1">
                        "{expr.eng}"
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {expr.kor}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Emergency Content */}
      {activeTab === 'emergency' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🆘 응급상황 대처 영어
            </h3>
            
            {emergencySituations.map((situation, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-950/20 dark:to-pink-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {situation.title}
                </h4>
                <div className="space-y-2">
                  {situation.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                        "{expr.eng}"
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {expr.kor}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Emergency Numbers */}
          <div className="bg-red-100 dark:bg-red-950/20 rounded-xl p-6 border border-red-200 dark:border-red-800">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📞 국가별 응급 전화번호
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇺🇸 미국</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 911</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇬🇧 영국</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 999</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇪🇺 유럽연합</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 112</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cultural Tips */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🌏 여행지별 문화 팁
        </h3>
        <div className="space-y-4">
          {culturalTips.map((country, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                {country.country} 여행 시 알아두면 좋은 문화
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {country.tips.map((tip, tipIdx) => (
                  <div key={tipIdx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                    <span>{tip}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Travel Checklist */}
      <div className="bg-gradient-to-r from-teal-500 to-cyan-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🎒 여행 영어 준비 체크리스트</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-teal-100">
          <div>
            <h4 className="font-semibold mb-2">📚 출발 전 준비</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 기본 인사말 숙지</li>
              <li>✓ 숫자와 날짜 표현</li>
              <li>✓ 응급상황 표현</li>
              <li>✓ 방향과 교통 관련 표현</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🗣️ 실전 연습</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 호텔 체크인 역할극</li>
              <li>✓ 레스토랑 주문 연습</li>
              <li>✓ 길 묻기 시뮬레이션</li>
              <li>✓ 쇼핑 대화 연습</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">📱 유용한 앱</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 번역 앱 다운로드</li>
              <li>✓ 지도 앱 오프라인 설정</li>
              <li>✓ 통화 변환 앱</li>
              <li>✓ 현지 교통 앱</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 5: Pronunciation & Intonation
function PronunciationContent({ copyCode, copiedCode }: any) {
  const [activeCategory, setActiveCategory] = useState('vowels')
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)

  const pronunciationCategories = [
    { id: 'vowels', name: '모음', icon: '🅰️' },
    { id: 'consonants', name: '자음', icon: '🇧' },
    { id: 'stress', name: '강세', icon: '💪' },
    { id: 'intonation', name: '억양', icon: '🎵' },
    { id: 'linking', name: '연음', icon: '🔗' },
    { id: 'rhythm', name: '리듬', icon: '🥁' }
  ]

  const vowelSounds = [
    {
      sound: '/iː/',
      examples: ['beat', 'meat', 'seat', 'feet'],
      description: '긴 이 소리',
      tips: '입술을 옆으로 길게 늘이고 혀를 앞쪽 높은 위치에'
    },
    {
      sound: '/ɪ/',
      examples: ['bit', 'hit', 'sit', 'fit'],
      description: '짧은 이 소리',
      tips: '한국어 "이"보다 약간 더 느슨하게'
    },
    {
      sound: '/æ/',
      examples: ['cat', 'bat', 'hat', 'mat'],
      description: '애 소리',
      tips: '입을 크게 벌리고 혀를 낮게 위치'
    },
    {
      sound: '/ʌ/',
      examples: ['but', 'cut', 'shut', 'nut'],
      description: '어 소리',
      tips: '한국어 "어"와 "아"의 중간 소리'
    },
    {
      sound: '/uː/',
      examples: ['boot', 'fruit', 'suit', 'cute'],
      description: '긴 우 소리',
      tips: '입술을 동그랗게 모으고 혀를 뒤쪽 높은 위치에'
    },
    {
      sound: '/ʊ/',
      examples: ['book', 'look', 'good', 'foot'],
      description: '짧은 우 소리',
      tips: '한국어 "우"보다 약간 더 느슨하게'
    }
  ]

  const consonantSounds = [
    {
      sound: '/θ/',
      examples: ['think', 'three', 'math', 'birth'],
      description: '무성 th 소리',
      tips: '혀끝을 윗니와 아랫니 사이에 살짝 내밀고 공기를 내보내기'
    },
    {
      sound: '/ð/',
      examples: ['this', 'that', 'brother', 'weather'],
      description: '유성 th 소리',
      tips: '혀끝을 윗니와 아랫니 사이에 살짝 내밀고 성대를 울리며'
    },
    {
      sound: '/r/',
      examples: ['red', 'right', 'very', 'every'],
      description: 'R 소리',
      tips: '혀끝을 입천장에 닿지 않게 하고 둥글게 말기'
    },
    {
      sound: '/l/',
      examples: ['light', 'love', 'fall', 'will'],
      description: 'L 소리',
      tips: '혀끝을 윗니 뒤 잇몸에 대고 양옆으로 공기 보내기'
    },
    {
      sound: '/v/',
      examples: ['very', 'voice', 'love', 'give'],
      description: 'V 소리',
      tips: '윗니를 아랫입술에 살짝 대고 성대를 울리며'
    },
    {
      sound: '/w/',
      examples: ['water', 'will', 'way', 'work'],
      description: 'W 소리',
      tips: '입술을 동그랗게 모았다가 빠르게 펴기'
    }
  ]

  const stressPatterns = [
    {
      word: 'photograph',
      pattern: '●○○',
      stressed: 'PHO-to-graph',
      meaning: '사진',
      rule: '3음절 단어의 첫 번째 음절 강세'
    },
    {
      word: 'photography',
      pattern: '○●○○',
      stressed: 'pho-TOG-ra-phy',
      meaning: '사진술',
      rule: '-graphy로 끝나는 단어는 뒤에서 3번째 음절 강세'
    },
    {
      word: 'photographer',
      pattern: '○●○○',
      stressed: 'pho-TOG-ra-pher',
      meaning: '사진작가',
      rule: '-er로 끝나는 명사는 기본형과 같은 강세'
    },
    {
      word: 'understand',
      pattern: '○○●',
      stressed: 'un-der-STAND',
      meaning: '이해하다',
      rule: '동사는 보통 마지막 음절에 강세'
    }
  ]

  const intonationPatterns = [
    {
      type: 'Rising Intonation ↗',
      use: 'Yes/No 질문, 확인, 놀람',
      examples: [
        { text: 'Are you coming?', pattern: '↗' },
        { text: 'Is this your bag?', pattern: '↗' },
        { text: 'Really?', pattern: '↗' },
        { text: 'You did what?', pattern: '↗' }
      ]
    },
    {
      type: 'Falling Intonation ↘',
      use: '진술문, WH 질문, 명령문',
      examples: [
        { text: 'I love pizza.', pattern: '↘' },
        { text: 'Where are you going?', pattern: '↘' },
        { text: 'Close the door.', pattern: '↘' },
        { text: 'Nice to meet you.', pattern: '↘' }
      ]
    },
    {
      type: 'Rise-Fall Intonation ↗↘',
      use: '강조, 놀람, 선택',
      examples: [
        { text: 'That was AMAZING!', pattern: '↗↘' },
        { text: 'Coffee or tea?', pattern: '↗↘' },
        { text: 'I TOLD you so!', pattern: '↗↘' },
        { text: 'What a beautiful day!', pattern: '↗↘' }
      ]
    }
  ]

  const linkingRules = [
    {
      rule: 'Consonant + Vowel',
      description: '자음으로 끝나는 단어 + 모음으로 시작하는 단어',
      examples: [
        { written: 'an apple', linked: 'a-napple' },
        { written: 'pick it up', linked: 'pi-cki-tup' },
        { written: 'turn on', linked: 'tur-non' },
        { written: 'look at', linked: 'loo-kat' }
      ]
    },
    {
      rule: 'Vowel + Vowel',
      description: '모음으로 끝나는 단어 + 모음으로 시작하는 단어',
      examples: [
        { written: 'go away', linked: 'go-waway' },
        { written: 'see it', linked: 'see-yit' },
        { written: 'try again', linked: 'try-yagain' },
        { written: 'blue eyes', linked: 'blue-weyes' }
      ]
    },
    {
      rule: 'Same Consonant',
      description: '같은 자음이 만날 때',
      examples: [
        { written: 'good day', linked: 'goo-day' },
        { written: 'black cat', linked: 'bla-cat' },
        { written: 'big game', linked: 'bi-game' },
        { written: 'stop playing', linked: 'sto-playing' }
      ]
    }
  ]

  const playAudio = (text: string, index: number) => {
    if (isPlaying) return
    
    setIsPlaying(true)
    setPlayingIndex(index)
    
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel()
      
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'en-US'
      utterance.rate = 0.7
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
      
      const voices = speechSynthesis.getVoices()
      const englishVoice = voices.find(voice => voice.lang === 'en-US')
      
      if (englishVoice) {
        utterance.voice = englishVoice
      }
      
      speechSynthesis.speak(utterance)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          영어 발음과 억양 완전 정복
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          정확한 발음과 자연스러운 억양으로 네이티브와 같은 영어 실력을 갖춰보세요. 
          체계적인 훈련을 통해 듣기 좋은 영어 발음을 마스터할 수 있습니다.
        </p>
      </div>

      {/* Pronunciation Categories */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 발음 연습 카테고리
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {pronunciationCategories.map(category => (
            <button
              key={category.id}
              onClick={() => setActiveCategory(category.id)}
              className={`p-3 rounded-lg text-center transition-colors ${
                activeCategory === category.id
                  ? 'bg-purple-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-purple-100 dark:hover:bg-purple-900/50'
              }`}
            >
              <div className="text-xl mb-1">{category.icon}</div>
              <div className="text-xs font-medium">{category.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Vowels Content */}
      {activeCategory === 'vowels' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🅰️ 영어 모음 정확한 발음법
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {vowelSounds.map((vowel, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-r from-pink-50 to-purple-50 dark:from-pink-950/20 dark:to-purple-950/20 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xl font-bold text-gray-800 dark:text-gray-200">
                      {vowel.sound}
                    </h4>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {vowel.description}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    💡 {vowel.tips}
                  </p>
                  <div className="space-y-2">
                    <h5 className="font-medium text-gray-700 dark:text-gray-300">예시 단어:</h5>
                    <div className="flex flex-wrap gap-2">
                      {vowel.examples.map((example, exampleIdx) => (
                        <button
                          key={exampleIdx}
                          onClick={() => playAudio(example, idx * 10 + exampleIdx)}
                          className="px-3 py-1 bg-white dark:bg-gray-800 rounded-lg text-sm hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors flex items-center gap-1"
                        >
                          <Volume2 className="w-3 h-3" />
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Consonants Content */}
      {activeCategory === 'consonants' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🇧 어려운 자음 정복하기
            </h3>
            <div className="space-y-4">
              {consonantSounds.map((consonant, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xl font-bold text-gray-800 dark:text-gray-200">
                      {consonant.sound}
                    </h4>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {consonant.description}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    💡 {consonant.tips}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {consonant.examples.map((example, exampleIdx) => (
                      <button
                        key={exampleIdx}
                        onClick={() => playAudio(example, idx * 10 + exampleIdx + 100)}
                        className="px-3 py-1 bg-white dark:bg-gray-800 rounded-lg text-sm hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors flex items-center gap-1"
                      >
                        <Volume2 className="w-3 h-3" />
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Stress Content */}
      {activeCategory === 'stress' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              💪 단어 강세 패턴 마스터
            </h3>
            <div className="space-y-4">
              {stressPatterns.map((stress, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xl font-bold text-gray-800 dark:text-gray-200">
                      {stress.word}
                    </h4>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        패턴: {stress.pattern}
                      </span>
                      <button
                        onClick={() => playAudio(stress.word, idx + 200)}
                        className="p-1 hover:bg-green-100 dark:hover:bg-green-900/50 rounded transition-colors"
                      >
                        <Volume2 className="w-4 h-4 text-green-600" />
                      </button>
                    </div>
                  </div>
                  <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                    {stress.stressed}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                    뜻: {stress.meaning}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    규칙: {stress.rule}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Stress Rules */}
          <div className="bg-amber-50 dark:bg-amber-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📏 강세 규칙 가이드
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">명사 강세 규칙</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 2음절 명사: 첫 번째 음절 (TABLE, WATER)</li>
                  <li>• -tion, -sion: 뒤에서 2번째 (inforMAtion)</li>
                  <li>• -ic: 뒤에서 2번째 (ecoNOmic)</li>
                  <li>• -ity: 뒤에서 3번째 (uniVERsity)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">동사 강세 규칙</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 2음절 동사: 두 번째 음절 (beGIN, forGET)</li>
                  <li>• 접두사가 있는 동사: 두 번째 부분 (unDERstand)</li>
                  <li>• -ate로 끝나는 동사: 뒤에서 2번째 (CREate)</li>
                  <li>• -fy로 끝나는 동사: 뒤에서 2번째 (CLArify)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Intonation Content */}
      {activeCategory === 'intonation' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🎵 억양 패턴으로 자연스러운 영어
            </h3>
            <div className="space-y-6">
              {intonationPatterns.map((pattern, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/20 dark:to-purple-950/20 rounded-lg">
                  <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                    {pattern.type}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    사용: {pattern.use}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {pattern.examples.map((example, exampleIdx) => (
                      <div key={exampleIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                        <span className="text-gray-800 dark:text-gray-200">
                          {example.text} <span className="text-purple-500">{example.pattern}</span>
                        </span>
                        <button
                          onClick={() => playAudio(example.text, idx * 10 + exampleIdx + 300)}
                          className="p-1 hover:bg-purple-100 dark:hover:bg-purple-900/50 rounded transition-colors"
                        >
                          <Volume2 className="w-4 h-4 text-purple-600" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Linking Content */}
      {activeCategory === 'linking' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🔗 연음으로 자연스러운 대화
            </h3>
            <div className="space-y-6">
              {linkingRules.map((rule, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-950/20 dark:to-blue-950/20 rounded-lg">
                  <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                    {rule.rule}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {rule.description}
                  </p>
                  <div className="space-y-2">
                    {rule.examples.map((example, exampleIdx) => (
                      <div key={exampleIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                        <div>
                          <span className="text-gray-600 dark:text-gray-400 text-sm">Written: </span>
                          <span className="text-gray-800 dark:text-gray-200">{example.written}</span>
                          <span className="text-gray-600 dark:text-gray-400 text-sm ml-4">Linked: </span>
                          <span className="text-cyan-600 dark:text-cyan-400 font-medium">{example.linked}</span>
                        </div>
                        <button
                          onClick={() => playAudio(example.written, idx * 10 + exampleIdx + 400)}
                          className="p-1 hover:bg-cyan-100 dark:hover:bg-cyan-900/50 rounded transition-colors"
                        >
                          <Volume2 className="w-4 h-4 text-cyan-600" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Pronunciation Success Tips */}
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🏆 발음 마스터 비법</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-purple-100">
          <div>
            <h4 className="font-semibold mb-2">👁️ 시각적 학습</h4>
            <ul className="text-sm space-y-1">
              <li>• 거울보며 입모양 연습</li>
              <li>• 발음 기호 익히기</li>
              <li>• 혀의 위치 의식하기</li>
              <li>• 입술 모양 주의 깊게 관찰</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🎧 청각적 학습</h4>
            <ul className="text-sm space-y-1">
              <li>• 네이티브 발음 모방</li>
              <li>• 녹음해서 비교 분석</li>
              <li>• 셰도잉 연습</li>
              <li>• 음성학 앱 활용</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🔄 반복 연습</h4>
            <ul className="text-sm space-y-1">
              <li>• 매일 10분씩 꾸준히</li>
              <li>• 어려운 소리 집중 연습</li>
              <li>• 문장 단위로 연습</li>
              <li>• 점진적 속도 증가</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 6: Listening Comprehension
function ListeningContent({ copyCode, copiedCode }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          듣기 능력 향상 전략
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          다양한 영어 액센트와 말하기 속도에 적응하여 듣기 실력을 체계적으로 향상시키는 방법을 학습합니다.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🌍 액센트 종류
          </h3>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="text-gray-700 dark:text-gray-300">American English</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="text-gray-700 dark:text-gray-300">British English</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-gray-700 dark:text-gray-300">Australian English</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span className="text-gray-700 dark:text-gray-300">Canadian English</span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            📝 듣기 전략
          </h3>
          <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <p>• 전체적인 맥락 파악하기</p>
            <p>• 키워드에 집중하기</p>
            <p>• 예측하며 듣기</p>
            <p>• 모르는 단어는 넘어가기</p>
            <p>• 반복해서 들어보기</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 7: Cultural Context
function CulturalContextContent({ copyCode, copiedCode }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          영어권 문화와 소통 에티켓
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          언어는 문화와 밀접한 관련이 있습니다. 영어권 문화를 이해하고 상황에 맞는 적절한 표현을 사용하는 방법을 배워보겠습니다.
        </p>
      </div>

      <div className="bg-amber-50 dark:bg-amber-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🤝 소통 스타일 차이
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">직접적 표현 (Direct)</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• "I disagree with that"</li>
              <li>• "That won't work"</li>
              <li>• "I need this by Friday"</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">간접적 표현 (Indirect)</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• "I see what you mean, but..."</li>
              <li>• "That might be challenging"</li>
              <li>• "If possible, could you...?"</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 8: Advanced Conversation
function AdvancedConversationContent({ copyCode, copiedCode }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          고급 회화 기법과 설득력 있는 소통
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          복잡한 주제에 대한 토론, 논리적 설득, 감정적 뉘앙스 표현 등 고급 수준의 영어 회화 기법을 마스터합니다.
        </p>
      </div>

      <div className="bg-indigo-50 dark:bg-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 논리적 설득 구조
        </h3>
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">1. 주장 제시 (Claim)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "I believe that remote work should be the default option for our company."
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">2. 근거 제시 (Evidence)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "Studies show that remote workers are 13% more productive, and our team's performance has improved by 25% since going remote."
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">3. 결론 강화 (Warrant)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "Therefore, implementing a remote-first policy would benefit both the company and employees."
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            💡 고급 표현법
          </h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">가정법:</span>
              <p className="text-gray-600 dark:text-gray-400">"If I were in your position..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">강조법:</span>
              <p className="text-gray-600 dark:text-gray-400">"What really matters is..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">대조법:</span>
              <p className="text-gray-600 dark:text-gray-400">"On the one hand... On the other hand..."</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🔥 토론 기법
          </h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">의견 제시:</span>
              <p className="text-gray-600 dark:text-gray-400">"From my perspective..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">반박:</span>
              <p className="text-gray-600 dark:text-gray-400">"I see your point, however..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">타협:</span>
              <p className="text-gray-600 dark:text-gray-400">"Perhaps we could find a middle ground..."</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}