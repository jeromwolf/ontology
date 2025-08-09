'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  Volume2, VolumeX, Home, RotateCcw, Play, Pause,
  CheckCircle, X, Clock, Globe, Users, Target,
  SkipForward, SkipBack, RotateCw, Award, BookOpen
} from 'lucide-react'

interface ListeningExercise {
  id: string
  title: string
  category: string
  accent: 'american' | 'british' | 'australian' | 'canadian'
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  duration: string
  transcript: string
  questions: Question[]
  vocabulary: VocabularyItem[]
  audioUrl?: string // 실제 구현에서는 오디오 파일
}

interface Question {
  id: string
  type: 'multiple-choice' | 'true-false' | 'fill-blank' | 'short-answer'
  question: string
  options?: string[]
  correctAnswer: string | string[]
  explanation: string
}

interface VocabularyItem {
  word: string
  pronunciation: string
  meaning: string
  example: string
}

const listeningExercises: ListeningExercise[] = [
  {
    id: 'business-meeting',
    title: 'Quarterly Business Review Meeting',
    category: '비즈니스',
    accent: 'american',
    difficulty: 'advanced',
    duration: '4:30',
    transcript: `Sarah: Good morning, everyone. Thank you for joining today's quarterly review meeting. I'm pleased to report that we've exceeded our Q3 targets by 15%. Our revenue growth has been particularly strong in the Asian markets, where we've seen a 28% increase compared to last quarter.

Mike: That's fantastic news, Sarah. Could you break down the numbers by department? I'm especially interested in how our marketing initiatives performed.

Sarah: Absolutely. Marketing contributed significantly with their digital campaign, which resulted in a 35% increase in qualified leads. Sales converted these leads at an impressive 22% rate, which is up from 18% last quarter.

Jennifer: What about our customer satisfaction scores? I know we implemented new support protocols in August.

Sarah: Great question, Jennifer. Customer satisfaction improved from 4.2 to 4.6 out of 5, and our response time decreased by 40%. The new protocols are clearly working well.`,
    questions: [
      {
        id: 'q1',
        type: 'multiple-choice',
        question: 'By how much did the company exceed their Q3 targets?',
        options: ['10%', '15%', '20%', '25%'],
        correctAnswer: '15%',
        explanation: 'Sarah explicitly states that they exceeded Q3 targets by 15%.'
      },
      {
        id: 'q2',
        type: 'multiple-choice',
        question: 'What was the revenue growth percentage in Asian markets?',
        options: ['15%', '22%', '28%', '35%'],
        correctAnswer: '28%',
        explanation: 'The Asian markets showed a 28% increase compared to last quarter.'
      },
      {
        id: 'q3',
        type: 'true-false',
        question: 'The marketing campaign resulted in a 35% increase in qualified leads.',
        correctAnswer: 'true',
        explanation: 'Sarah confirms that the digital campaign resulted in a 35% increase in qualified leads.'
      },
      {
        id: 'q4',
        type: 'fill-blank',
        question: 'Customer satisfaction improved from __ to __ out of 5.',
        correctAnswer: ['4.2', '4.6'],
        explanation: 'Jennifer asked about satisfaction scores, and Sarah replied they improved from 4.2 to 4.6.'
      }
    ],
    vocabulary: [
      {
        word: 'exceed',
        pronunciation: '/ɪkˈsiːd/',
        meaning: '초과하다, 넘어서다',
        example: 'We exceeded our sales targets this quarter.'
      },
      {
        word: 'qualified leads',
        pronunciation: '/ˈkwɒlɪfaɪd liːdz/',
        meaning: '검증된 잠재 고객',
        example: 'Our marketing team generated 500 qualified leads last month.'
      },
      {
        word: 'protocols',
        pronunciation: '/ˈprəʊtəkɒlz/',
        meaning: '규정, 절차',
        example: 'We need to follow safety protocols in the laboratory.'
      }
    ]
  },
  {
    id: 'weather-forecast',
    title: 'Weekly Weather Forecast',
    category: '일상생활',
    accent: 'british',
    difficulty: 'intermediate',
    duration: '2:45',
    transcript: `Good evening, I'm Tom Richardson with your weekly weather forecast. This week we're expecting quite changeable conditions across the UK.

Monday will start off rather cloudy with some light showers expected in the morning, particularly across Scotland and Northern England. Temperatures will reach a modest 16 degrees Celsius in the south, dropping to around 12 degrees in the north.

Tuesday brings much brighter conditions with sunny spells developing throughout the day. We might see the occasional cloud, but generally, it'll be a pleasant day with temperatures climbing to 19 degrees in London and 15 degrees in Edinburgh.

Wednesday sees the return of more unsettled weather. A weather front moving in from the Atlantic will bring heavy rain and strong winds, particularly along the western coast. Temperatures will drop slightly to 14-17 degrees.

The weekend looks more promising with Friday and Saturday showing partly cloudy skies and lighter winds. Perfect weather for outdoor activities, with temperatures reaching a comfortable 20 degrees in the south.`,
    questions: [
      {
        id: 'q1',
        type: 'multiple-choice',
        question: 'What type of weather is expected on Monday morning?',
        options: ['Heavy rain', 'Light showers', 'Snow', 'Sunny spells'],
        correctAnswer: 'Light showers',
        explanation: 'The forecast mentions "light showers expected in the morning" for Monday.'
      },
      {
        id: 'q2',
        type: 'true-false',
        question: 'Tuesday will be mostly sunny throughout the day.',
        correctAnswer: 'true',
        explanation: 'Tuesday is described as having "much brighter conditions with sunny spells developing throughout the day."'
      },
      {
        id: 'q3',
        type: 'fill-blank',
        question: 'On Wednesday, temperatures will drop to __-__ degrees.',
        correctAnswer: ['14', '17'],
        explanation: 'The forecast states temperatures will drop slightly to 14-17 degrees on Wednesday.'
      }
    ],
    vocabulary: [
      {
        word: 'changeable',
        pronunciation: '/ˈtʃeɪndʒəbəl/',
        meaning: '변화하기 쉬운, 변덕스러운',
        example: 'The weather in spring is quite changeable.'
      },
      {
        word: 'unsettled',
        pronunciation: '/ʌnˈsetəld/',
        meaning: '불안정한 (날씨)',
        example: 'Unsettled weather conditions are expected this week.'
      },
      {
        word: 'weather front',
        pronunciation: '/ˈweðə frʌnt/',
        meaning: '기상 전선',
        example: 'A cold front is moving across the region.'
      }
    ]
  },
  {
    id: 'university-lecture',
    title: 'Introduction to Psychology',
    category: '교육',
    accent: 'canadian',
    difficulty: 'advanced',
    duration: '6:15',
    transcript: `Welcome to Psychology 101. I'm Professor Martinez, and today we'll be exploring the fascinating world of human behavior and mental processes.

Psychology, as a scientific discipline, seeks to understand how we think, feel, and behave. It's a field that bridges the gap between biology and social sciences, offering insights into everything from memory formation to social interactions.

One of the fundamental questions in psychology is the nature versus nurture debate. This concerns whether our behaviors and traits are primarily determined by our genetic makeup or by our environment and experiences. Modern research suggests it's actually a complex interaction between both factors.

For instance, consider language acquisition in children. While humans are born with an innate capacity for language – what Noam Chomsky called the "Language Acquisition Device" – the specific language a child learns depends entirely on their environment. A child born in Tokyo will naturally learn Japanese, while one born in Paris will learn French.

Today's homework assignment involves observing and documenting three different behavioral patterns in your daily life. Try to identify what might influence these behaviors – is it biological, environmental, or a combination of both? We'll discuss your findings in our next seminar.`,
    questions: [
      {
        id: 'q1',
        type: 'multiple-choice',
        question: 'According to the lecture, psychology bridges the gap between which fields?',
        options: ['Biology and chemistry', 'Biology and social sciences', 'Medicine and sociology', 'Neuroscience and philosophy'],
        correctAnswer: 'Biology and social sciences',
        explanation: 'Professor Martinez states that psychology "bridges the gap between biology and social sciences."'
      },
      {
        id: 'q2',
        type: 'true-false',
        question: 'The nature versus nurture debate suggests that behavior is determined only by genetics.',
        correctAnswer: 'false',
        explanation: 'The lecture explains it\'s "a complex interaction between both factors" - genetics and environment.'
      },
      {
        id: 'q3',
        type: 'fill-blank',
        question: 'Noam Chomsky called the innate capacity for language the "____".',
        correctAnswer: ['Language Acquisition Device'],
        explanation: 'The professor mentions Chomsky\'s term "Language Acquisition Device" for our innate language capacity.'
      }
    ],
    vocabulary: [
      {
        word: 'innate',
        pronunciation: '/ɪˈneɪt/',
        meaning: '타고난, 선천적인',
        example: 'Humans have an innate ability to learn language.'
      },
      {
        word: 'acquisition',
        pronunciation: '/ˌækwɪˈzɪʃən/',
        meaning: '습득, 획득',
        example: 'Language acquisition happens naturally in childhood.'
      },
      {
        word: 'behavioral patterns',
        pronunciation: '/bɪˈheɪvjərəl ˈpætərnz/',
        meaning: '행동 패턴',
        example: 'Scientists study behavioral patterns in animals.'
      }
    ]
  },
  {
    id: 'travel-announcement',
    title: 'Airport Gate Announcement',
    category: '여행',
    accent: 'australian',
    difficulty: 'beginner',
    duration: '1:30',
    transcript: `Good afternoon, passengers. This is a boarding announcement for Jetstar flight JQ507 to Melbourne. We are now ready to begin boarding at gate 23.

We will be boarding in the following order: first, passengers requiring special assistance and those traveling with small children. Next, business class passengers and frequent flyers. Finally, we'll board economy class passengers starting with rows 30 to 45, followed by rows 15 to 29, and lastly rows 1 to 14.

Please have your boarding pass and photo identification ready for inspection. All carry-on baggage must fit in the overhead compartments or under the seat in front of you.

We expect to depart on time at 3:45 PM local time, arriving in Melbourne at approximately 7:20 PM. The weather in Melbourne is currently 18 degrees and partly cloudy.

Thank you for choosing Jetstar, and we look forward to welcoming you aboard flight JQ507.`,
    questions: [
      {
        id: 'q1',
        type: 'multiple-choice',
        question: 'What is the flight number mentioned in the announcement?',
        options: ['JQ507', 'JQ705', 'JS507', 'JL507'],
        correctAnswer: 'JQ507',
        explanation: 'The announcement clearly states "Jetstar flight JQ507 to Melbourne."'
      },
      {
        id: 'q2',
        type: 'true-false',
        question: 'Business class passengers board before economy class passengers.',
        correctAnswer: 'true',
        explanation: 'The boarding order lists business class passengers before economy class passengers.'
      },
      {
        id: 'q3',
        type: 'fill-blank',
        question: 'The flight is expected to arrive in Melbourne at approximately ____.',
        correctAnswer: ['7:20 PM'],
        explanation: 'The announcement states arrival time as "approximately 7:20 PM."'
      }
    ],
    vocabulary: [
      {
        word: 'boarding',
        pronunciation: '/ˈbɔːdɪŋ/',
        meaning: '탑승',
        example: 'Boarding will begin in 15 minutes.'
      },
      {
        word: 'carry-on baggage',
        pronunciation: '/ˈkæri ɒn ˈbæɡɪdʒ/',
        meaning: '기내 반입 수하물',
        example: 'Your carry-on baggage must meet size requirements.'
      },
      {
        word: 'overhead compartments',
        pronunciation: '/ˈəʊvəhed kəmˈpɑːtmənts/',
        meaning: '머리 위 수하물 보관함',
        example: 'Please store your bag in the overhead compartments.'
      }
    ]
  }
]

export default function ListeningLab() {
  const [selectedExercise, setSelectedExercise] = useState(listeningExercises[0])
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [showTranscript, setShowTranscript] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [userAnswers, setUserAnswers] = useState<{ [key: string]: string | string[] }>({})
  const [showResults, setShowResults] = useState(false)
  const [score, setScore] = useState(0)
  const [isLooping, setIsLooping] = useState(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const synthRef = useRef<SpeechSynthesisUtterance | null>(null)

  // 음성 합성으로 오디오 생성 (실제 구현에서는 실제 오디오 파일 사용)
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

  const playAudio = () => {
    if (isPlaying) {
      stopAudio()
      return
    }

    if ('speechSynthesis' in window) {
      speechSynthesis.cancel()
      
      const utterance = new SpeechSynthesisUtterance(selectedExercise.transcript)
      utterance.lang = getAccentLang(selectedExercise.accent)
      utterance.rate = playbackSpeed * 0.8 // 조금 더 천천히
      utterance.pitch = selectedExercise.accent === 'british' ? 1.1 : 1.0
      utterance.volume = 1.0
      
      utterance.onstart = () => {
        setIsPlaying(true)
        setCurrentTime(0)
      }
      
      utterance.onend = () => {
        setIsPlaying(false)
        if (isLooping) {
          setTimeout(() => playAudio(), 1000)
        }
      }
      
      utterance.onerror = () => {
        setIsPlaying(false)
      }
      
      // 액센트에 맞는 음성 선택
      const voices = speechSynthesis.getVoices()
      const selectedVoice = selectVoiceByAccent(voices, selectedExercise.accent)
      
      if (selectedVoice) {
        utterance.voice = selectedVoice
      }
      
      synthRef.current = utterance
      speechSynthesis.speak(utterance)
      
      // 현재 시간 업데이트 시뮬레이션
      const updateTime = () => {
        if (speechSynthesis.speaking) {
          setCurrentTime(prev => prev + 1)
          setTimeout(updateTime, 1000)
        }
      }
      updateTime()
    }
  }

  const stopAudio = () => {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel()
    }
    setIsPlaying(false)
  }

  const getAccentLang = (accent: string) => {
    switch (accent) {
      case 'american': return 'en-US'
      case 'british': return 'en-GB'
      case 'australian': return 'en-AU'
      case 'canadian': return 'en-CA'
      default: return 'en-US'
    }
  }

  const selectVoiceByAccent = (voices: SpeechSynthesisVoice[], accent: string) => {
    const lang = getAccentLang(accent)
    
    // 선호하는 음성들
    const preferredVoices = {
      'american': ['Microsoft Zira', 'Google US English', 'Samantha', 'Alex'],
      'british': ['Google UK English Female', 'Daniel', 'Microsoft Hazel'],
      'australian': ['Microsoft Catherine', 'Karen'],
      'canadian': ['Microsoft Linda', 'Tessa']
    }
    
    const preferred = preferredVoices[accent as keyof typeof preferredVoices] || preferredVoices.american
    
    return voices.find(voice => 
      preferred.some(p => voice.name.includes(p))
    ) || voices.find(voice => voice.lang === lang) ||
       voices.find(voice => voice.lang.startsWith('en-'))
  }

  const handleAnswerChange = (questionId: string, answer: string | string[]) => {
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }))
  }

  const checkAnswers = () => {
    let correctCount = 0
    selectedExercise.questions.forEach(question => {
      const userAnswer = userAnswers[question.id]
      if (Array.isArray(question.correctAnswer)) {
        // Fill-in-the-blank questions
        const userArr = Array.isArray(userAnswer) ? userAnswer : [userAnswer]
        if (question.correctAnswer.every((correct, idx) => 
          userArr[idx]?.toLowerCase().trim() === correct.toLowerCase().trim()
        )) {
          correctCount++
        }
      } else {
        // Multiple choice and true/false questions
        if (userAnswer === question.correctAnswer) {
          correctCount++
        }
      }
    })
    
    setScore(correctCount)
    setShowResults(true)
  }

  const resetQuiz = () => {
    setUserAnswers({})
    setShowResults(false)
    setScore(0)
    setCurrentQuestion(0)
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300'
      case 'intermediate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300'
      case 'advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-300'
    }
  }

  const getAccentFlag = (accent: string) => {
    switch (accent) {
      case 'american': return '🇺🇸'
      case 'british': return '🇬🇧'
      case 'australian': return '🇦🇺'
      case 'canadian': return '🇨🇦'
      default: return '🌐'
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-900 dark:to-indigo-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/english-conversation"
              className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
                듣기 실험실
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                다양한 액센트와 상황별 영어 듣기 연습
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowTranscript(!showTranscript)}
              className="px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all text-sm"
            >
              {showTranscript ? '스크립트 숨기기' : '스크립트 보기'}
            </button>
            <button
              onClick={resetQuiz}
              className="p-2 bg-indigo-500 text-white rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Exercise Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                듣기 연습 선택
              </h3>
              <div className="space-y-3">
                {listeningExercises.map((exercise) => (
                  <button
                    key={exercise.id}
                    onClick={() => {
                      setSelectedExercise(exercise)
                      stopAudio()
                      resetQuiz()
                    }}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedExercise.id === exercise.id
                        ? 'bg-indigo-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-indigo-100 dark:hover:bg-indigo-900/50'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="font-medium text-sm">{exercise.title}</div>
                      <span className="text-lg">{getAccentFlag(exercise.accent)}</span>
                    </div>
                    <div className="text-xs opacity-80 mb-2">{exercise.category}</div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded text-xs ${getDifficultyColor(exercise.difficulty)}`}>
                        {exercise.difficulty}
                      </span>
                      <span className="text-xs opacity-60">{exercise.duration}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Audio Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mt-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                오디오 설정
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    재생 속도
                  </label>
                  <select
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                    className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                  >
                    <option value={0.5}>0.5x (매우 느림)</option>
                    <option value={0.75}>0.75x (느림)</option>
                    <option value={1.0}>1.0x (정상)</option>
                    <option value={1.25}>1.25x (빠름)</option>
                    <option value={1.5}>1.5x (매우 빠름)</option>
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="loop-audio"
                    checked={isLooping}
                    onChange={(e) => setIsLooping(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="loop-audio" className="text-sm text-gray-700 dark:text-gray-300">
                    반복 재생
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Exercise Info */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">
                    {selectedExercise.title}
                  </h2>
                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                    <span className="flex items-center gap-1">
                      <Globe className="w-4 h-4" />
                      {getAccentFlag(selectedExercise.accent)} {selectedExercise.accent.charAt(0).toUpperCase() + selectedExercise.accent.slice(1)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {selectedExercise.duration}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${getDifficultyColor(selectedExercise.difficulty)}`}>
                      {selectedExercise.difficulty}
                    </span>
                  </div>
                </div>
              </div>

              {/* Audio Player */}
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/20 dark:to-purple-950/20 rounded-xl p-6">
                <div className="flex items-center justify-center gap-4 mb-4">
                  <button
                    onClick={playAudio}
                    className={`w-16 h-16 rounded-full flex items-center justify-center transition-all shadow-lg hover:shadow-xl ${
                      isPlaying
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                    }`}
                  >
                    {isPlaying ? (
                      <Pause className="w-8 h-8" />
                    ) : (
                      <Play className="w-8 h-8 ml-1" />
                    )}
                  </button>
                </div>

                <div className="text-center">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {isPlaying ? '재생 중...' : '재생하려면 플레이 버튼을 누르세요'}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500">
                    속도: {playbackSpeed}x {isLooping && '• 반복 재생 중'}
                  </div>
                </div>
              </div>
            </div>

            {/* Transcript */}
            {showTranscript && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                  📝 스크립트
                </h3>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <p className="text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-line">
                    {selectedExercise.transcript}
                  </p>
                </div>
              </div>
            )}

            {/* Questions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                📋 이해도 확인 문제
              </h3>

              {!showResults ? (
                <div className="space-y-6">
                  {selectedExercise.questions.map((question, idx) => (
                    <div key={question.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                        {idx + 1}. {question.question}
                      </h4>

                      {question.type === 'multiple-choice' && question.options && (
                        <div className="space-y-2">
                          {question.options.map((option, optionIdx) => (
                            <label key={optionIdx} className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="radio"
                                name={question.id}
                                value={option}
                                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                                className="text-indigo-600"
                              />
                              <span className="text-gray-700 dark:text-gray-300">{option}</span>
                            </label>
                          ))}
                        </div>
                      )}

                      {question.type === 'true-false' && (
                        <div className="space-y-2">
                          <label className="flex items-center gap-2 cursor-pointer">
                            <input
                              type="radio"
                              name={question.id}
                              value="true"
                              onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                              className="text-indigo-600"
                            />
                            <span className="text-gray-700 dark:text-gray-300">True</span>
                          </label>
                          <label className="flex items-center gap-2 cursor-pointer">
                            <input
                              type="radio"
                              name={question.id}
                              value="false"
                              onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                              className="text-indigo-600"
                            />
                            <span className="text-gray-700 dark:text-gray-300">False</span>
                          </label>
                        </div>
                      )}

                      {question.type === 'fill-blank' && (
                        <div className="space-y-2">
                          {Array.isArray(question.correctAnswer) ? (
                            <div className="flex gap-2">
                              {question.correctAnswer.map((_, blankIdx) => (
                                <input
                                  key={blankIdx}
                                  type="text"
                                  placeholder={`답 ${blankIdx + 1}`}
                                  onChange={(e) => {
                                    const currentAnswers = Array.isArray(userAnswers[question.id]) 
                                      ? [...(userAnswers[question.id] as string[])]
                                      : []
                                    currentAnswers[blankIdx] = e.target.value
                                    handleAnswerChange(question.id, currentAnswers)
                                  }}
                                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200"
                                />
                              ))}
                            </div>
                          ) : (
                            <input
                              type="text"
                              placeholder="답을 입력하세요"
                              onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200"
                            />
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  <button
                    onClick={checkAnswers}
                    className="w-full py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors font-semibold"
                  >
                    답안 확인하기
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Results Summary */}
                  <div className="text-center p-6 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl text-white">
                    <Award className="w-12 h-12 mx-auto mb-3" />
                    <h3 className="text-2xl font-bold mb-2">결과 발표</h3>
                    <p className="text-indigo-100">
                      {selectedExercise.questions.length}문제 중 {score}문제 정답
                    </p>
                    <div className="text-3xl font-bold mt-2">
                      {Math.round((score / selectedExercise.questions.length) * 100)}%
                    </div>
                  </div>

                  {/* Detailed Results */}
                  {selectedExercise.questions.map((question, idx) => {
                    const userAnswer = userAnswers[question.id]
                    const isCorrect = Array.isArray(question.correctAnswer)
                      ? question.correctAnswer.every((correct, i) => 
                          (Array.isArray(userAnswer) ? userAnswer[i] : userAnswer)?.toLowerCase().trim() === correct.toLowerCase().trim()
                        )
                      : userAnswer === question.correctAnswer

                    return (
                      <div key={question.id} className={`p-4 rounded-lg border-l-4 ${
                        isCorrect 
                          ? 'border-green-500 bg-green-50 dark:bg-green-950/20' 
                          : 'border-red-500 bg-red-50 dark:bg-red-950/20'
                      }`}>
                        <div className="flex items-start gap-3">
                          {isCorrect ? (
                            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-1" />
                          ) : (
                            <X className="w-5 h-5 text-red-600 dark:text-red-400 mt-1" />
                          )}
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                              {idx + 1}. {question.question}
                            </h4>
                            <div className="space-y-1 text-sm">
                              <p className="text-gray-700 dark:text-gray-300">
                                <strong>당신의 답:</strong> {Array.isArray(userAnswer) ? userAnswer.join(', ') : userAnswer || '미응답'}
                              </p>
                              <p className="text-gray-700 dark:text-gray-300">
                                <strong>정답:</strong> {Array.isArray(question.correctAnswer) ? question.correctAnswer.join(', ') : question.correctAnswer}
                              </p>
                              <p className="text-gray-600 dark:text-gray-400">
                                <strong>해설:</strong> {question.explanation}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}

                  <button
                    onClick={resetQuiz}
                    className="w-full py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors font-semibold"
                  >
                    다시 시도하기
                  </button>
                </div>
              )}
            </div>

            {/* Vocabulary */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                📚 핵심 어휘
              </h3>
              <div className="space-y-4">
                {selectedExercise.vocabulary.map((vocab, idx) => (
                  <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-gray-800 dark:text-gray-200">
                        {vocab.word}
                      </h4>
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                        {vocab.pronunciation}
                      </span>
                    </div>
                    <p className="text-gray-700 dark:text-gray-300 mb-2">
                      {vocab.meaning}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">
                      예문: {vocab.example}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}