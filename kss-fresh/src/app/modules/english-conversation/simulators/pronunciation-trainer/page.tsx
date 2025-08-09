'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { 
  Mic, MicOff, Volume2, Home, RotateCcw, 
  CheckCircle, AlertCircle, TrendingUp, Target,
  Play, Pause, Award, BookOpen
} from 'lucide-react'

interface PronunciationScore {
  overall: number
  accuracy: number
  fluency: number
  pronunciation: number
  phonemes: PhonemeScore[]
}

interface PhonemeScore {
  phoneme: string
  score: number
  feedback: string
}

interface PracticeWord {
  word: string
  phonetic: string
  audio: string
  difficulty: 'easy' | 'medium' | 'hard'
  category: string
}

export default function PronunciationTrainer() {
  const [selectedCategory, setSelectedCategory] = useState('vowels')
  const [currentWord, setCurrentWord] = useState<PracticeWord | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentScore, setCurrentScore] = useState<PronunciationScore | null>(null)
  const [practiceHistory, setPracticeHistory] = useState<PronunciationScore[]>([])
  const [showPhonetics, setShowPhonetics] = useState(true)
  const recognitionRef = useRef<any>(null)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isRecordingRef = useRef(false)

  const categories = [
    { id: 'vowels', name: '모음 (Vowels)', description: '영어 모음 발음 연습' },
    { id: 'consonants', name: '자음 (Consonants)', description: '영어 자음 발음 연습' },
    { id: 'diphthongs', name: '이중모음 (Diphthongs)', description: '복합 모음 발음 연습' },
    { id: 'common-words', name: '일반 단어', description: '자주 사용되는 단어들' },
    { id: 'difficult-words', name: '어려운 단어', description: '발음하기 어려운 단어들' },
    { id: 'minimal-pairs', name: '최소대립쌍', description: '유사한 발음의 단어 구별하기' }
  ]

  const practiceWords: { [key: string]: PracticeWord[] } = {
    vowels: [
      { word: 'beat', phonetic: '/biːt/', audio: '', difficulty: 'easy', category: 'vowels' },
      { word: 'bit', phonetic: '/bɪt/', audio: '', difficulty: 'easy', category: 'vowels' },
      { word: 'bat', phonetic: '/bæt/', audio: '', difficulty: 'easy', category: 'vowels' },
      { word: 'but', phonetic: '/bʌt/', audio: '', difficulty: 'medium', category: 'vowels' },
      { word: 'boot', phonetic: '/buːt/', audio: '', difficulty: 'easy', category: 'vowels' },
      { word: 'boat', phonetic: '/boʊt/', audio: '', difficulty: 'medium', category: 'vowels' }
    ],
    consonants: [
      { word: 'think', phonetic: '/θɪŋk/', audio: '', difficulty: 'hard', category: 'consonants' },
      { word: 'that', phonetic: '/ðæt/', audio: '', difficulty: 'hard', category: 'consonants' },
      { word: 'ship', phonetic: '/ʃɪp/', audio: '', difficulty: 'medium', category: 'consonants' },
      { word: 'chip', phonetic: '/tʃɪp/', audio: '', difficulty: 'medium', category: 'consonants' },
      { word: 'measure', phonetic: '/ˈmeʒər/', audio: '', difficulty: 'hard', category: 'consonants' },
      { word: 'vision', phonetic: '/ˈvɪʒən/', audio: '', difficulty: 'hard', category: 'consonants' }
    ],
    'difficult-words': [
      { word: 'thoroughly', phonetic: '/ˈθɜːroʊli/', audio: '', difficulty: 'hard', category: 'difficult-words' },
      { word: 'comfortable', phonetic: '/ˈkʌmftərbəl/', audio: '', difficulty: 'hard', category: 'difficult-words' },
      { word: 'necessary', phonetic: '/ˈnesəseri/', audio: '', difficulty: 'hard', category: 'difficult-words' },
      { word: 'pronunciation', phonetic: '/prəˌnʌnsiˈeɪʃən/', audio: '', difficulty: 'hard', category: 'difficult-words' }
    ]
  }

  useEffect(() => {
    if (practiceWords[selectedCategory] && practiceWords[selectedCategory].length > 0) {
      setCurrentWord(practiceWords[selectedCategory][0])
    }
  }, [selectedCategory])

  // 음성 리스트 로딩 보장
  useEffect(() => {
    if ('speechSynthesis' in window) {
      // 음성 리스트가 비어있는 경우 강제로 로딩
      const loadVoices = () => {
        const voices = speechSynthesis.getVoices()
        if (voices.length === 0) {
          speechSynthesis.addEventListener('voiceschanged', loadVoices)
        }
      }
      loadVoices()
    }
  }, [])

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      cleanupRecording()
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop()
        } catch (error) {
          console.error('Error stopping recognition on unmount:', error)
        }
      }
    }
  }, [recognitionRef, timeoutRef])

  const cleanupRecording = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    recognitionRef.current = null
  }, [])

  const setRecordingState = useCallback((recording: boolean) => {
    setIsRecording(recording)
    isRecordingRef.current = recording
  }, [])

  const startRecording = useCallback(() => {
    if (!currentWord || isRecordingRef.current) return
    
    // 마이크 권한 확인
    if (!confirm('이 기능은 마이크 권한이 필요합니다. 발음 연습을 위해 마이크를 사용하시겠습니까?')) {
      return
    }
    
    console.log('🎤 Starting recording process...')
    
    // 이전 녹음 정리
    cleanupRecording()
    
    setRecordingState(true)
    setCurrentScore(null) // 이전 점수 지우기
    
    // Web Speech API 음성 인식 시작
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      const recognition = new SpeechRecognition()
      
      recognition.lang = 'en-US'
      recognition.interimResults = false
      recognition.maxAlternatives = 1
      recognition.continuous = false
      
      let hasResult = false
      
      recognition.onstart = () => {
        console.log('✅ Speech recognition started successfully')
        setRecordingState(true)
      }
      
      recognition.onresult = (event: any) => {
        console.log('🎯 Speech recognition result received')
        hasResult = true
        const transcript = event.results[0][0].transcript.toLowerCase().trim()
        const confidence = event.results[0][0].confidence
        
        // 타이머 정리
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
          timeoutRef.current = null
        }
        
        setRecordingState(false)
        
        if (transcript && currentWord) {
          const analysisScore = analyzePronunciation(transcript, currentWord.word, confidence || 0)
          setCurrentScore(analysisScore)
          setPracticeHistory(prev => [...prev, analysisScore])
        }
      }
      
      recognition.onerror = (event: any) => {
        console.error('❌ Speech recognition error:', event.error)
        hasResult = true
        
        // 타이머 정리
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
          timeoutRef.current = null
        }
        
        setRecordingState(false)
        handleRecordingError(event.error)
      }
      
      recognition.onend = () => {
        console.log('🔚 Speech recognition ended')
        
        // 타이머가 아직 있고 결과가 없었다면 no-speech 처리
        if (!hasResult && isRecordingRef.current) {
          if (timeoutRef.current) {
            clearTimeout(timeoutRef.current)
            timeoutRef.current = null
          }
          setRecordingState(false)
          handleNoSpeech()
        }
      }
      
      recognitionRef.current = recognition
      
      try {
        recognition.start()
        console.log('🚀 Speech recognition API called')
        
        // 5초 후 강제 종료 타이머
        timeoutRef.current = setTimeout(() => {
          console.log('⏰ 5초 타임아웃! 강제 종료 중...')
          
          if (recognitionRef.current && isRecordingRef.current) {
            try {
              recognitionRef.current.stop()
              console.log('🛑 Recognition stopped by timeout')
            } catch (error) {
              console.error('Error stopping recognition:', error)
            }
          }
          
          // 타임아웃 상태 설정
          setRecordingState(false)
          setCurrentScore({
            overall: 0,
            accuracy: 0,
            fluency: 0,
            pronunciation: 0,
            phonemes: [
              { phoneme: '⏰', score: 0, feedback: '5초 안에 음성이 인식되지 않았습니다.' },
              { phoneme: '🎯', score: 0, feedback: '마이크에 더 가까이 말해보세요.' }
            ]
          })
          
          // 정리
          timeoutRef.current = null
          recognitionRef.current = null
        }, 5000)
        
        console.log('⏰ 5초 타이머 설정 완료')
        
      } catch (error) {
        console.error('Error starting recognition:', error)
        setRecordingState(false)
        handleRecordingError('start-error')
      }
    } else {
      // Speech Recognition이 지원되지 않는 경우 시뮬레이션
      console.log('Speech Recognition not supported, using simulation')
      timeoutRef.current = setTimeout(() => {
        setRecordingState(false)
        const simulatedScore = generateSimulatedScore()
        setCurrentScore(simulatedScore)
        setPracticeHistory(prev => [...prev, simulatedScore])
      }, 3000)
    }
  }, [currentWord, cleanupRecording, setRecordingState])

  const handleRecordingError = (error: string) => {
    setRecordingState(false)
    
    let errorMessage = '음성 인식 오류가 발생했습니다.'
    if (error === 'no-speech') {
      errorMessage = '음성이 감지되지 않았습니다. 마이크에 더 가까이 말해주세요.'
    } else if (error === 'audio-capture') {
      errorMessage = '마이크에 접근할 수 없습니다. 권한을 확인해주세요.'
    } else if (error === 'not-allowed') {
      errorMessage = '마이크 권한이 거부되었습니다. 브라우저 설정을 확인해주세요.'
    }
    
    const errorScore: PronunciationScore = {
      overall: 0,
      accuracy: 0,
      fluency: 0,
      pronunciation: 0,
      phonemes: [
        { phoneme: '⚠️', score: 0, feedback: errorMessage },
        { phoneme: '🎤', score: 0, feedback: '마이크 권한을 허용하고 다시 시도해주세요.' }
      ]
    }
    setCurrentScore(errorScore)
  }

  const handleNoSpeech = () => {
    setRecordingState(false)
    const noSpeechScore: PronunciationScore = {
      overall: 0,
      accuracy: 0,
      fluency: 0,
      pronunciation: 0,
      phonemes: [
        { phoneme: '🔇', score: 0, feedback: '음성이 감지되지 않았습니다.' },
        { phoneme: '💡', score: 0, feedback: '목표 발음을 듣고 다시 시도해보세요.' }
      ]
    }
    setCurrentScore(noSpeechScore)
  }

  const handleTimeout = () => {
    setRecordingState(false)
    const timeoutScore: PronunciationScore = {
      overall: 0,
      accuracy: 0,
      fluency: 0,
      pronunciation: 0,
      phonemes: [
        { phoneme: '⏰', score: 0, feedback: '5초 안에 음성이 인식되지 않았습니다.' },
        { phoneme: '🎯', score: 0, feedback: '더 명확하고 빠르게 발음해보세요.' }
      ]
    }
    setCurrentScore(timeoutScore)
  }

  const stopRecording = (transcript?: string, confidence?: number) => {
    cleanupRecording()
    setRecordingState(false)
    
    if (transcript && currentWord) {
      // 실제 발음 분석
      const analysisScore = analyzePronunciation(transcript, currentWord.word, confidence || 0)
      setCurrentScore(analysisScore)
      setPracticeHistory(prev => [...prev, analysisScore])
    } else {
      // 시뮬레이션된 발음 점수 생성
      const simulatedScore = generateSimulatedScore()
      setCurrentScore(simulatedScore)
      setPracticeHistory(prev => [...prev, simulatedScore])
    }
  }

  const forceStopRecording = () => {
    if (recognitionRef) {
      try {
        recognitionRef.current.stop()
      } catch (error) {
        console.error('Error manually stopping recognition:', error)
      }
    }
    cleanupRecording()
    setIsRecording(false)
    
    const manualStopScore: PronunciationScore = {
      overall: 0,
      accuracy: 0,
      fluency: 0,
      pronunciation: 0,
      phonemes: [
        { phoneme: '✋', score: 0, feedback: '녹음이 수동으로 중지되었습니다.' },
        { phoneme: '🔄', score: 0, feedback: '다시 시도해보세요.' }
      ]
    }
    setCurrentScore(manualStopScore)
  }

  const analyzePronunciation = (transcript: string, targetWord: string, confidence: number): PronunciationScore => {
    console.log(`Analyzing: "${transcript}" vs "${targetWord}" (confidence: ${confidence})`)
    
    // 기본 정확도는 음성 인식 신뢰도 기반
    let accuracy = Math.floor(confidence * 100)
    
    // 단어 유사도 분석
    const similarity = calculateWordSimilarity(transcript, targetWord)
    
    // 정확한 단어를 말한 경우 보너스
    if (transcript === targetWord.toLowerCase()) {
      accuracy = Math.min(95 + Math.floor(Math.random() * 5), 100)
    } else if (similarity > 0.7) {
      accuracy = Math.floor(similarity * 80) + Math.floor(Math.random() * 15)
    } else {
      accuracy = Math.max(accuracy, 30) // 최소 30점 보장
    }
    
    // 유창성: 음성 인식 신뢰도와 단어 길이 기반
    const fluency = Math.min(
      Math.floor(confidence * 90) + Math.floor(Math.random() * 15),
      95
    )
    
    // 발음: 정확도와 유창성의 조합
    const pronunciation = Math.floor((accuracy + fluency) / 2) + Math.floor(Math.random() * 10)
    
    const overall = Math.floor((accuracy + fluency + pronunciation) / 3)
    
    // 실제 발음된 단어에 따른 피드백
    const feedback = generateFeedback(transcript, targetWord, similarity)
    
    return {
      overall: Math.min(overall, 100),
      accuracy: Math.min(accuracy, 100),
      fluency: Math.min(fluency, 100),
      pronunciation: Math.min(pronunciation, 100),
      phonemes: feedback
    }
  }

  const calculateWordSimilarity = (word1: string, word2: string): number => {
    // Levenshtein distance 기반 유사도 계산
    const len1 = word1.length
    const len2 = word2.length
    const matrix = Array(len2 + 1).fill(null).map(() => Array(len1 + 1).fill(null))
    
    for (let i = 0; i <= len1; i++) matrix[0][i] = i
    for (let j = 0; j <= len2; j++) matrix[j][0] = j
    
    for (let j = 1; j <= len2; j++) {
      for (let i = 1; i <= len1; i++) {
        if (word1[i - 1] === word2[j - 1]) {
          matrix[j][i] = matrix[j - 1][i - 1]
        } else {
          matrix[j][i] = Math.min(
            matrix[j - 1][i - 1] + 1,
            matrix[j][i - 1] + 1,
            matrix[j - 1][i] + 1
          )
        }
      }
    }
    
    const distance = matrix[len2][len1]
    return 1 - distance / Math.max(len1, len2)
  }

  const generateFeedback = (transcript: string, targetWord: string, similarity: number) => {
    const phonemes = []
    
    if (transcript === targetWord.toLowerCase()) {
      phonemes.push({ 
        phoneme: '🎯', 
        score: 95, 
        feedback: 'Perfect! Excellent pronunciation!' 
      })
    } else if (similarity > 0.8) {
      phonemes.push({ 
        phoneme: '👍', 
        score: 85, 
        feedback: 'Very good! Almost perfect pronunciation.' 
      })
    } else if (similarity > 0.5) {
      phonemes.push({ 
        phoneme: '📝', 
        score: 70, 
        feedback: 'Good attempt. Try to speak more clearly.' 
      })
    } else {
      phonemes.push({ 
        phoneme: '🔄', 
        score: 50, 
        feedback: 'Try again. Listen to the target pronunciation first.' 
      })
    }
    
    // 인식된 단어 표시
    if (transcript && transcript !== targetWord.toLowerCase()) {
      phonemes.push({ 
        phoneme: '🎤', 
        score: Math.floor(similarity * 100), 
        feedback: `Heard: "${transcript}" | Target: "${targetWord}"` 
      })
    }
    
    return phonemes
  }

  const generateSimulatedScore = (): PronunciationScore => {
    const accuracy = Math.floor(Math.random() * 30) + 70 // 70-100
    const fluency = Math.floor(Math.random() * 25) + 75 // 75-100
    const pronunciation = Math.floor(Math.random() * 20) + 80 // 80-100
    const overall = Math.floor((accuracy + fluency + pronunciation) / 3)

    return {
      overall,
      accuracy,
      fluency,
      pronunciation,
      phonemes: [
        { phoneme: '🔊', score: 80, feedback: 'Voice recognition not available - using simulation' },
        { phoneme: '💡', score: 85, feedback: 'Try using Chrome or Safari for real voice analysis' }
      ]
    }
  }

  const playTargetAudio = () => {
    if (!currentWord || isPlaying) return
    
    // 이전 음성이 재생 중이면 중지
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel()
    }
    
    setIsPlaying(true)
    
    // Web Speech API의 SpeechSynthesis 사용
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(currentWord.word)
      
      // 영어 음성으로 설정
      utterance.lang = 'en-US'
      utterance.rate = 0.8 // 조금 느리게
      utterance.pitch = 1.0
      utterance.volume = 1.0
      
      // 재생 완료 시 상태 업데이트
      utterance.onend = () => {
        setIsPlaying(false)
      }
      
      utterance.onerror = () => {
        setIsPlaying(false)
        console.error('Speech synthesis error')
      }
      
      // 사용 가능한 영어 음성 찾기 (우선순위: 미국 영어 > 영국 영어 > 기타 영어)
      const voices = speechSynthesis.getVoices()
      const englishVoice = 
        voices.find(voice => voice.lang === 'en-US' && voice.name.toLowerCase().includes('female')) ||
        voices.find(voice => voice.lang === 'en-US') ||
        voices.find(voice => voice.lang === 'en-GB') ||
        voices.find(voice => voice.lang.startsWith('en-')) ||
        voices.find(voice => voice.name.toLowerCase().includes('english'))
      
      if (englishVoice) {
        utterance.voice = englishVoice
        console.log('Using voice:', englishVoice.name, englishVoice.lang)
      } else {
        console.log('No English voice found, using default')
      }
      
      speechSynthesis.speak(utterance)
    } else {
      // SpeechSynthesis가 지원되지 않는 경우 시뮬레이션
      console.warn('SpeechSynthesis not supported')
      setTimeout(() => setIsPlaying(false), 1500)
    }
  }

  const nextWord = () => {
    const currentWords = practiceWords[selectedCategory] || []
    const currentIndex = currentWords.findIndex(w => w.word === currentWord?.word)
    const nextIndex = (currentIndex + 1) % currentWords.length
    setCurrentWord(currentWords[nextIndex])
    setCurrentScore(null)
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600 dark:text-green-400'
    if (score >= 80) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getScoreIcon = (score: number) => {
    if (score >= 90) return <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
    if (score >= 80) return <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
    return <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300'
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300'
      case 'hard': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-300'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-gray-900 dark:to-purple-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/english-conversation"
              className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
                AI 발음 트레이너
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                정확한 영어 발음을 위한 AI 기반 훈련 시스템
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowPhonetics(!showPhonetics)}
              className="px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all text-sm"
            >
              {showPhonetics ? '발음기호 숨기기' : '발음기호 보기'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Category Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                연습 카테고리
              </h3>
              <div className="space-y-2">
                {categories.map(category => (
                  <button
                    key={category.id}
                    onClick={() => setSelectedCategory(category.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedCategory === category.id
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-purple-100 dark:hover:bg-purple-900/50'
                    }`}
                  >
                    <div className="font-medium">{category.name}</div>
                    <div className="text-xs opacity-80">{category.description}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Progress Summary */}
            {practiceHistory.length > 0 && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mt-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                  연습 현황
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">평균 점수</span>
                    <span className="font-bold text-gray-800 dark:text-gray-200">
                      {Math.floor(practiceHistory.reduce((sum, score) => sum + score.overall, 0) / practiceHistory.length)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">연습 횟수</span>
                    <span className="font-bold text-gray-800 dark:text-gray-200">
                      {practiceHistory.length}회
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">최고 점수</span>
                    <span className="font-bold text-gray-800 dark:text-gray-200">
                      {Math.max(...practiceHistory.map(s => s.overall))}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Main Practice Area */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
              {currentWord && (
                <>
                  {/* Current Word Display */}
                  <div className="text-center mb-8">
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyColor(currentWord.difficulty)}`}>
                        {currentWord.difficulty.toUpperCase()}
                      </span>
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300">
                        {selectedCategory.toUpperCase()}
                      </span>
                    </div>
                    
                    <h2 className="text-6xl font-bold text-gray-800 dark:text-gray-200 mb-4">
                      {currentWord.word}
                    </h2>
                    
                    {showPhonetics && (
                      <p className="text-2xl text-purple-600 dark:text-purple-400 mb-6 font-mono">
                        {currentWord.phonetic}
                      </p>
                    )}

                    <div className="flex items-center justify-center gap-4">
                      <button
                        onClick={playTargetAudio}
                        disabled={isPlaying}
                        className="flex items-center gap-2 px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {isPlaying ? (
                          <>
                            <Pause className="w-5 h-5" />
                            재생 중...
                          </>
                        ) : (
                          <>
                            <Play className="w-5 h-5" />
                            목표 발음 듣기
                          </>
                        )}
                      </button>

                      <button
                        onClick={nextWord}
                        className="px-6 py-3 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors"
                      >
                        다음 단어
                      </button>
                    </div>
                  </div>

                  {/* Recording Interface */}
                  <div className="text-center mb-8">
                    <button
                      onClick={isRecording ? forceStopRecording : startRecording}
                      disabled={isPlaying}
                      className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                        isRecording
                          ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                          : 'bg-purple-500 hover:bg-purple-600'
                      } text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl`}
                    >
                      {isRecording ? (
                        <MicOff className="w-8 h-8" />
                      ) : (
                        <Mic className="w-8 h-8" />
                      )}
                    </button>
                    
                    <p className="mt-4 text-gray-600 dark:text-gray-400">
                      {isRecording ? `녹음 중... "${currentWord.word}"를 또렷하게 발음해주세요` : '마이크 버튼을 눌러 실제 발음을 분석하세요'}
                    </p>
                    
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      {isRecording ? '5초 후 자동 종료됩니다' : '실제 음성 인식으로 정확한 발음 점수를 받아보세요'}
                    </div>
                  </div>

                  {/* Pronunciation Score */}
                  {currentScore && (
                    <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                        <Target className="w-5 h-5" />
                        발음 분석 결과
                      </h3>

                      {/* Overall Score */}
                      <div className="text-center mb-6">
                        <div className="flex items-center justify-center gap-2 mb-2">
                          {getScoreIcon(currentScore.overall)}
                          <span className={`text-3xl font-bold ${getScoreColor(currentScore.overall)}`}>
                            {currentScore.overall}%
                          </span>
                        </div>
                        <p className="text-gray-600 dark:text-gray-400">전체 점수</p>
                      </div>

                      {/* Detailed Scores */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div className="text-center">
                          <div className={`text-2xl font-bold ${getScoreColor(currentScore.accuracy)}`}>
                            {currentScore.accuracy}%
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">정확도</div>
                        </div>
                        <div className="text-center">
                          <div className={`text-2xl font-bold ${getScoreColor(currentScore.fluency)}`}>
                            {currentScore.fluency}%
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">유창성</div>
                        </div>
                        <div className="text-center">
                          <div className={`text-2xl font-bold ${getScoreColor(currentScore.pronunciation)}`}>
                            {currentScore.pronunciation}%
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">발음</div>
                        </div>
                      </div>

                      {/* Phoneme Analysis */}
                      <div>
                        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                          음소별 분석
                        </h4>
                        <div className="space-y-3">
                          {currentScore.phonemes.map((phoneme, idx) => (
                            <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-mono text-lg">{phoneme.phoneme}</span>
                                <span className={`font-bold ${getScoreColor(phoneme.score)}`}>
                                  {phoneme.score}%
                                </span>
                              </div>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                {phoneme.feedback}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}