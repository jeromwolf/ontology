'use client'

import { useState, useEffect } from 'react'
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react'

export default function Chapter5() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

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

