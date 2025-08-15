'use client';

import { useState, useEffect } from 'react';
import { Volume2, Pause } from 'lucide-react';

export default function Chapter1() {
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