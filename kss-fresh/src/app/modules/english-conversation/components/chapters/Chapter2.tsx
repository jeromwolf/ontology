'use client'

import { useState, useEffect } from 'react'
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react'

export default function Chapter2() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

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

