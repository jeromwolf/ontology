'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  Volume2, Home, RotateCcw, Clock, Users, MapPin,
  Target, Star, ChevronRight, Play, Pause, Award,
  CheckCircle, AlertCircle, Filter, Search
} from 'lucide-react'
import { scenarios, type Scenario, type ScenarioStep } from '../../data/scenarios'

// Scenarios are now imported from a separate file (line 10)

export default function ScenarioPractice() {
  const [selectedScenario, setSelectedScenario] = useState<Scenario | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [userResponses, setUserResponses] = useState<string[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)
  const [showOptions, setShowOptions] = useState(true)
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set())
  const [score, setScore] = useState(0)
  const [selectedCategory, setSelectedCategory] = useState<string>('전체')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('전체')
  const [scenarioVoice, setScenarioVoice] = useState<SpeechSynthesisVoice | null>(null)

  // Get unique categories
  const categories = ['전체', ...Array.from(new Set(scenarios.map(s => s.category)))]
  const difficulties = ['전체', 'beginner', 'intermediate', 'advanced']

  // Filter scenarios
  const filteredScenarios = scenarios.filter(scenario => {
    const matchesCategory = selectedCategory === '전체' || scenario.category === selectedCategory
    const matchesDifficulty = selectedDifficulty === '전체' || scenario.difficulty === selectedDifficulty
    const matchesSearch = searchTerm === '' || 
      scenario.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      scenario.description.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesCategory && matchesDifficulty && matchesSearch
  })

  // Initialize scenario after mount (removed auto-selection)
  useEffect(() => {
    // Don't auto-select a scenario, let user browse and choose
    selectVoiceForScenario()
  }, [])
  
  // Select a voice when scenario changes
  useEffect(() => {
    if (selectedScenario) {
      selectVoiceForScenario()
    }
  }, [selectedScenario?.id])
  
  const selectVoiceForScenario = () => {
    if ('speechSynthesis' in window) {
      const voices = speechSynthesis.getVoices()
      if (voices.length === 0) return
      
      // High-quality voice options categorized by type
      const voiceProfiles = [
        // Professional female voices
        ['Microsoft Zira - English (United States)', 'Google US English Female', 'Samantha', 'Victoria'],
        // Professional male voices
        ['Microsoft David - English (United States)', 'Google US English Male', 'Alex', 'Daniel'],
        // British accent voices
        ['Microsoft Hazel - English (United Kingdom)', 'Google UK English Female', 'Daniel (United Kingdom)'],
        // Friendly female voices
        ['Karen', 'Moira', 'Tessa', 'Fiona'],
        // Friendly male voices
        ['Oliver', 'Tom', 'James', 'Gordon'],
        // Australian voices
        ['Karen (Australia)', 'Lee (Australia)', 'Microsoft Catherine - English (Australia)']
      ]
      
      // Select a random voice profile for this scenario
      const selectedProfile = voiceProfiles[Math.floor(Math.random() * voiceProfiles.length)]
      
      // Find the first available voice from the selected profile
      let selectedVoice = null
      for (const voiceName of selectedProfile) {
        selectedVoice = voices.find(voice => 
          voice.name.includes(voiceName) || voice.name === voiceName
        )
        if (selectedVoice) break
      }
      
      // Fallback to any English voice
      if (!selectedVoice) {
        selectedVoice = voices.find(voice => voice.lang === 'en-US') ||
                       voices.find(voice => voice.lang === 'en-GB') ||
                       voices.find(voice => voice.lang.startsWith('en-'))
      }
      
      setScenarioVoice(selectedVoice || null)
    }
  }

  // Auto-advance to next step if current step is NPC
  useEffect(() => {
    if (selectedScenario && currentStep < selectedScenario.steps.length) {
      const currentStepData = selectedScenario.steps[currentStep]
      // If current step is NPC and there's a next step that is user, auto-advance
      if (currentStepData.speaker === 'npc' && 
          currentStep < selectedScenario.steps.length - 1 &&
          selectedScenario.steps[currentStep + 1].speaker === 'user' &&
          !userResponses[currentStep + 1]) {
        // Small delay to show NPC message before showing user options
        const timer = setTimeout(() => {
          setCurrentStep(currentStep + 1)
        }, 100)
        return () => clearTimeout(timer)
      }
    }
  }, [currentStep, selectedScenario, userResponses])

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
      utterance.rate = 0.9  // Optimal rate for clarity
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
      
      // Use the pre-selected voice for this scenario for consistency
      if (scenarioVoice) {
        utterance.voice = scenarioVoice
        
        // Fine-tune voice parameters based on voice characteristics
        const voiceName = scenarioVoice.name.toLowerCase()
        
        // Gender-based adjustments
        if (voiceName.includes('female') || voiceName.includes('zira') || 
            voiceName.includes('samantha') || voiceName.includes('victoria') ||
            voiceName.includes('karen') || voiceName.includes('moira')) {
          utterance.pitch = 1.1  // Slightly higher pitch for female voices
          utterance.rate = 0.92   // Natural speaking rate
        } else if (voiceName.includes('male') || voiceName.includes('david') ||
                   voiceName.includes('alex') || voiceName.includes('daniel') ||
                   voiceName.includes('james') || voiceName.includes('oliver')) {
          utterance.pitch = 0.9   // Lower pitch for male voices
          utterance.rate = 0.88   // Slightly slower for clarity
        }
        
        // Accent-based adjustments
        if (voiceName.includes('british') || voiceName.includes('uk') || 
            voiceName.includes('hazel') || voiceName.includes('george')) {
          utterance.rate = 0.85   // British accent - more deliberate
          utterance.pitch = 1.05  // Slightly higher for clarity
        } else if (voiceName.includes('australian') || voiceName.includes('catherine')) {
          utterance.rate = 0.95   // Australian accent - natural flow
          utterance.pitch = 1.02
        } else if (voiceName.includes('google')) {
          // Google voices tend to be clearer at slightly faster rates
          utterance.rate = 0.95
        } else if (voiceName.includes('microsoft')) {
          // Microsoft voices work well with these settings
          utterance.rate = 0.9
          utterance.pitch = 1.0
        }
        
        // Quality enhancement for specific high-quality voices
        if (voiceName.includes('aria') || voiceName.includes('zira') || 
            voiceName.includes('david') || voiceName.includes('mark')) {
          // These are premium Microsoft voices
          utterance.volume = 1.0
          utterance.rate = 0.88  // Optimal for these voices
        }
      } else {
        // Fallback: Select a voice if none was pre-selected
        const voices = speechSynthesis.getVoices()
        const englishVoice = voices.find(voice => voice.lang === 'en-US') ||
                           voices.find(voice => voice.lang === 'en-GB') ||
                           voices.find(voice => voice.lang.startsWith('en-'))
        if (englishVoice) {
          utterance.voice = englishVoice
        }
      }
      
      speechSynthesis.speak(utterance)
    }
  }

  const selectResponse = (response: string) => {
    if (!selectedScenario) return
    
    const newResponses = [...userResponses]
    newResponses[currentStep] = response
    setUserResponses(newResponses)
    
    // 점수 계산 (선택한 옵션이 첫 번째일 때 더 높은 점수)
    const currentStepData = selectedScenario.steps[currentStep]
    if (currentStepData.options) {
      const responseIndex = currentStepData.options.indexOf(response)
      const stepScore = responseIndex === 0 ? 100 : responseIndex === 1 ? 85 : 70
      setScore(prevScore => prevScore + stepScore)
    }
    
    setCompletedSteps(prev => new Set([...Array.from(prev), currentStepData.id]))
    
    // 다음 단계로 이동
    if (currentStep < selectedScenario.steps.length - 1) {
      setTimeout(() => {
        setCurrentStep(currentStep + 1)
      }, 500)
    }
  }

  const resetScenario = () => {
    setCurrentStep(0)
    setUserResponses([])
    setCompletedSteps(new Set())
    setScore(0)
  }

  const changeScenario = (scenario: Scenario) => {
    setSelectedScenario(scenario)
    resetScenario()
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300'
      case 'intermediate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300'
      case 'advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-300'
    }
  }

  const getCurrentStepData = () => selectedScenario?.steps[currentStep]
  const isCompleted = selectedScenario && currentStep >= selectedScenario.steps.length - 1 && completedSteps.has(selectedScenario.steps[selectedScenario.steps.length - 1]?.id)
  const progress = selectedScenario ? (completedSteps.size / selectedScenario.steps.length) * 100 : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-green-50 dark:from-gray-900 dark:to-emerald-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/english-conversation"
              className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
                실전 시나리오 연습
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                다양한 실생활 상황에서의 영어 대화 시뮬레이션
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowOptions(!showOptions)}
              className="px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all text-sm"
            >
              {showOptions ? '선택지 숨기기' : '선택지 보기'}
            </button>
            <button
              onClick={resetScenario}
              className="p-2 bg-emerald-500 text-white rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Filters Bar */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg mb-6">
          <div className="flex flex-wrap items-center gap-4">
            {/* Search */}
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="시나리오 검색..."
                  className="w-full pl-9 pr-3 py-2 bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 text-sm"
                />
              </div>
            </div>

            {/* Category Pills */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">카테고리:</span>
              <div className="flex flex-wrap gap-2">
                {categories.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-3 py-1 rounded-full text-sm transition-colors ${
                      selectedCategory === cat
                        ? 'bg-emerald-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Difficulty Pills */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">난이도:</span>
              <div className="flex gap-2">
                {difficulties.map(diff => {
                  const isSelected = selectedDifficulty === diff
                  let className = 'px-3 py-1 rounded-full text-sm transition-colors '
                  if (isSelected) {
                    if (diff === '전체') {
                      className += 'bg-gray-500 text-white'
                    } else if (diff === 'beginner') {
                      className += 'bg-green-500 text-white'
                    } else if (diff === 'intermediate') {
                      className += 'bg-yellow-500 text-white'
                    } else if (diff === 'advanced') {
                      className += 'bg-red-500 text-white'
                    }
                  } else {
                    className += 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }
                  
                  return (
                    <button
                      key={diff}
                      onClick={() => setSelectedDifficulty(diff)}
                      className={className}
                    >
                      {diff === '전체' ? '전체' : diff === 'beginner' ? '초급' : diff === 'intermediate' ? '중급' : '고급'}
                    </button>
                  )
                })}
              </div>
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              총 {filteredScenarios.length}개
            </div>
          </div>
        </div>

        {/* Scenario Cards Grid */}
        {!selectedScenario ? (
          filteredScenarios.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
              {filteredScenarios.map((scenario) => (
              <button
                key={scenario.id}
                onClick={() => setSelectedScenario(scenario)}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-all hover:scale-105 text-left"
              >
                {/* Card Image */}
                <div className="relative h-32 bg-gradient-to-br from-emerald-100 to-green-100 dark:from-emerald-900/20 dark:to-green-900/20">
                  <img 
                    src={scenario.imageUrl} 
                    alt={scenario.title}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent" />
                  <div className="absolute top-2 right-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(scenario.difficulty)}`}>
                      {scenario.difficulty === 'beginner' ? '초급' : scenario.difficulty === 'intermediate' ? '중급' : '고급'}
                    </span>
                  </div>
                  <div className="absolute bottom-2 left-2">
                    {scenario.icon && React.createElement(scenario.icon, {
                      className: "w-6 h-6 text-white drop-shadow-lg"
                    })}
                  </div>
                </div>
                
                {/* Card Content */}
                <div className="p-4">
                  <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">
                    {scenario.title}
                  </h3>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                    {scenario.setting}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-emerald-600 dark:text-emerald-400">
                      {scenario.category}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-500">
                      {scenario.duration}
                    </span>
                  </div>
                </div>
              </button>
            ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
              <div className="text-gray-400 dark:text-gray-500 mb-4">
                <Search className="w-16 h-16" />
              </div>
              <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
                검색 결과가 없습니다
              </h3>
              <p className="text-gray-500 dark:text-gray-400 text-center max-w-md">
                선택하신 카테고리와 난이도에 맞는 시나리오가 없습니다.<br />
                다른 필터를 선택해보세요.
              </p>
              <button
                onClick={() => {
                  setSelectedCategory('전체')
                  setSelectedDifficulty('전체')
                  setSearchTerm('')
                }}
                className="mt-4 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors"
              >
                필터 초기화
              </button>
            </div>
          )
        ) : (
          <>
            {/* Back Button */}
            <button
              onClick={() => {
                setSelectedScenario(null)
                resetScenario()
              }}
              className="mb-4 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
            >
              <ChevronRight className="w-4 h-4 rotate-180" />
              시나리오 목록으로 돌아가기
            </button>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Selected Scenario Detail */}
              <div className="lg:col-span-2">
              {/* Scenario Info */}
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg mb-6 overflow-hidden">
              {/* Scenario Image */}
              <div className="relative h-48 bg-gradient-to-br from-emerald-100 to-green-100 dark:from-emerald-900/20 dark:to-green-900/20">
                <img 
                  src={selectedScenario.imageUrl} 
                  alt={selectedScenario.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                <div className="absolute bottom-4 left-6 right-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      {selectedScenario.icon && React.createElement(selectedScenario.icon, {
                        className: "w-8 h-8 text-white drop-shadow-lg"
                      })}
                      <div>
                        <h2 className="text-2xl font-bold text-white drop-shadow-lg">
                          {selectedScenario.title}
                        </h2>
                        <p className="text-white/90 drop-shadow">
                          {selectedScenario.setting}
                        </p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(selectedScenario.difficulty)}`}>
                      {selectedScenario.difficulty.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  {selectedScenario.description}
                </p>

                <div className="bg-emerald-50 dark:bg-emerald-950/20 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🎯 학습 목표</h3>
                  <ul className="space-y-1">
                    {selectedScenario.objectives.map((objective, idx) => (
                      <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                        {objective}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Conversation Flow */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                  대화 진행 ({Math.min(currentStep + 1, selectedScenario.steps.length)}/{selectedScenario.steps.length})
                </h3>
                {isCompleted && (
                  <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                    <Award className="w-5 h-5" />
                    <span className="font-semibold">시나리오 완료!</span>
                  </div>
                )}
              </div>

              {/* Conversation Steps */}
              <div className="space-y-4 mb-6">
                {selectedScenario.steps.slice(0, currentStep + 1).map((step, idx) => (
                  <div
                    key={step.id}
                    className={`p-4 rounded-lg border-l-4 ${
                      step.speaker === 'npc'
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20'
                        : 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20'
                    } ${idx === currentStep ? 'ring-2 ring-emerald-300 dark:ring-emerald-600' : ''}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="font-semibold text-gray-800 dark:text-gray-200">
                            {step.speaker === 'npc' ? '상대방' : '나'}
                          </span>
                          {completedSteps.has(step.id) && (
                            <CheckCircle className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                          )}
                        </div>
                        
                        {step.speaker === 'npc' && (
                          <>
                            <p className="text-gray-800 dark:text-gray-200 font-medium mb-2">
                              {step.text}
                            </p>
                            <p className="text-gray-600 dark:text-gray-400 text-sm">
                              {step.korean}
                            </p>
                          </>
                        )}
                        
                        {step.speaker === 'user' && userResponses[idx] && (
                          <>
                            <p className="text-gray-800 dark:text-gray-200 font-medium mb-2">
                              {userResponses[idx]}
                            </p>
                            <p className="text-gray-600 dark:text-gray-400 text-sm">
                              {step.korean}
                            </p>
                          </>
                        )}

                        {step.speaker === 'user' && !userResponses[idx] && idx === currentStep && (
                          <div className="space-y-3">
                            <p className="text-gray-700 dark:text-gray-300 font-medium">
                              당신의 응답을 선택하세요:
                            </p>
                            {showOptions && step.options && (
                              <div className="space-y-2">
                                {step.options.map((option, optionIdx) => (
                                  <button
                                    key={optionIdx}
                                    onClick={() => selectResponse(option)}
                                    className="w-full text-left p-3 bg-white dark:bg-gray-700 border-2 border-gray-200 dark:border-gray-600 rounded-lg hover:border-emerald-500 dark:hover:border-emerald-400 transition-colors"
                                  >
                                    <div className="flex items-center gap-2">
                                      <span className="w-6 h-6 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center text-sm font-bold">
                                        {optionIdx + 1}
                                      </span>
                                      <span className="text-gray-800 dark:text-gray-200">{option}</span>
                                    </div>
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {step.speaker === 'npc' && (
                        <button
                          onClick={() => playAudio(step.text, idx)}
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
                ))}
              </div>

              {/* Progress Bar */}
              <div className="mt-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">시나리오 진행률</span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {Math.round(progress)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {isCompleted && (
                <div className="mt-6 p-4 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl text-white">
                  <h3 className="font-semibold mb-2">🎉 시나리오 완료!</h3>
                  <p className="text-emerald-100 text-sm">
                    평균 점수: {Math.round(score / Math.max(completedSteps.size, 1))}점 
                    • 다른 시나리오도 도전해보세요!
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar - Progress & Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Progress Summary */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                진행 상황
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">완료율</span>
                    <span className="text-sm font-bold text-gray-800 dark:text-gray-200">
                      {Math.round(progress)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
                
                {score > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">현재 점수</span>
                    <span className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                      {Math.round(score / Math.max(completedSteps.size, 1))}
                    </span>
                  </div>
                )}

                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <button
                    onClick={() => setShowOptions(!showOptions)}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-sm"
                  >
                    {showOptions ? '선택지 숨기기' : '선택지 보기'}
                  </button>
                  <button
                    onClick={resetScenario}
                    className="w-full mt-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors text-sm flex items-center justify-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    다시 시작
                  </button>
                </div>
              </div>
            </div>

            {/* Learning Objectives */}
            <div className="bg-emerald-50 dark:bg-emerald-950/20 rounded-xl p-6 shadow-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                🎯 학습 목표
              </h3>
              <ul className="space-y-2">
                {selectedScenario.objectives.map((objective, idx) => (
                  <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                    <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full mt-1.5 flex-shrink-0" />
                    {objective}
                  </li>
                ))}
              </ul>
            </div>

            {/* Quick Stats */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                시나리오 정보
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">카테고리</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{selectedScenario.category}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">난이도</span>
                  <span className={`px-2 py-1 rounded text-xs ${getDifficultyColor(selectedScenario.difficulty)}`}>
                    {selectedScenario.difficulty === 'beginner' ? '초급' : selectedScenario.difficulty === 'intermediate' ? '중급' : '고급'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">예상 시간</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{selectedScenario.duration}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">총 대화</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{selectedScenario.steps.length}턴</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        </>
        )}
      </div>
    </div>
  )
}