'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  MessageCircle, Mic, MicOff, Volume2, VolumeX, 
  Send, RotateCcw, Settings, Home, Bot, User,
  Play, Pause, CheckCircle, AlertCircle
} from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'ai'
  content: string
  timestamp: Date
  feedback?: {
    grammar: number
    pronunciation: number
    fluency: number
    suggestions: string[]
  }
}

interface ConversationSettings {
  topic: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  aiPersonality: 'friendly' | 'professional' | 'casual'
  voiceEnabled: boolean
}

export default function AIConversationPartner() {
  const [messages, setMessages] = useState<Message[]>([])
  const [currentMessage, setCurrentMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [settings, setSettings] = useState<ConversationSettings>({
    topic: 'daily-life',
    difficulty: 'intermediate',
    aiPersonality: 'friendly',
    voiceEnabled: true
  })
  const [showSettings, setShowSettings] = useState(false)
  const [isThinking, setIsThinking] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const conversationTopics = [
    { id: 'daily-life', name: '일상 생활', description: '일상적인 대화와 경험 공유' },
    { id: 'work', name: '직장 생활', description: '업무 관련 대화와 비즈니스 영어' },
    { id: 'travel', name: '여행', description: '여행 경험과 계획에 대한 대화' },
    { id: 'hobbies', name: '취미', description: '관심사와 여가 활동에 대한 대화' },
    { id: 'current-events', name: '시사', description: '뉴스와 현재 이슈에 대한 토론' },
    { id: 'food', name: '음식', description: '요리와 음식 문화에 대한 대화' }
  ]

  const difficulties = [
    { id: 'beginner', name: '초급', description: '기본 단어와 간단한 문장' },
    { id: 'intermediate', name: '중급', description: '일반적인 대화와 복잡한 문장' },
    { id: 'advanced', name: '고급', description: '전문적인 주제와 고급 어휘' }
  ]

  const personalities = [
    { id: 'friendly', name: '친근한', description: '따뜻하고 격려하는 스타일' },
    { id: 'professional', name: '전문적인', description: '격식있고 비즈니스 중심' },
    { id: 'casual', name: '캐주얼', description: '편안하고 자연스러운 스타일' }
  ]

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (messages.length === 0) {
      startConversation()
    }
  }, [settings.topic, settings.difficulty, settings.aiPersonality])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const startConversation = async () => {
    const topic = conversationTopics.find(t => t.id === settings.topic)
    const greeting = generateAIGreeting(topic?.name || '일상 생활')
    
    const aiMessage: Message = {
      id: Date.now().toString(),
      type: 'ai',
      content: greeting,
      timestamp: new Date()
    }
    
    setMessages([aiMessage])
  }

  const generateAIGreeting = (topicName: string): string => {
    const greetings = {
      'daily-life': "Hi there! I'm excited to chat with you today. How has your day been so far?",
      'work': "Hello! I'd love to hear about your work experiences. What do you do for a living?",
      'travel': "Hi! I'm passionate about travel. Have you been anywhere interesting recently?",
      'hobbies': "Hello! I'd love to learn about your hobbies. What do you enjoy doing in your free time?",
      'current-events': "Hi there! What's your take on the latest news? Anything particular catching your attention?",
      'food': "Hello! I'm a food enthusiast. What's your favorite cuisine or dish?"
    }
    
    return greetings[settings.topic as keyof typeof greetings] || 
           "Hi! I'm here to help you practice English. What would you like to talk about?"
  }

  const generateAIResponse = async (userMessage: string): Promise<string> => {
    // 실제 구현에서는 OpenAI API나 다른 AI 서비스를 사용
    // 여기서는 시뮬레이션을 위한 간단한 응답 생성
    
    const responses = {
      beginner: [
        "That's interesting! Can you tell me more?",
        "I see. How do you feel about that?",
        "That sounds nice. What else do you like?",
        "Really? That's cool! Why do you think so?"
      ],
      intermediate: [
        "That's a fascinating perspective! I'd love to hear more about your experience with that.",
        "I can understand why you might feel that way. What led you to that conclusion?",
        "That's quite interesting! Have you always been interested in this topic?",
        "I appreciate you sharing that. Could you elaborate on what makes it special to you?"
      ],
      advanced: [
        "That's a remarkably nuanced viewpoint. I'm curious about the underlying factors that shaped your opinion.",
        "Your perspective reveals a deep understanding of the subject matter. How did you develop such insight?",
        "That's a thought-provoking observation. What implications do you think this might have for the broader context?",
        "I find your analysis quite compelling. Could you walk me through the reasoning that brought you to this conclusion?"
      ]
    }

    const levelResponses = responses[settings.difficulty]
    return levelResponses[Math.floor(Math.random() * levelResponses.length)]
  }

  const sendMessage = async () => {
    if (!currentMessage.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage.trim(),
      timestamp: new Date(),
      feedback: generateFeedback(currentMessage.trim())
    }

    setMessages(prev => [...prev, userMessage])
    setCurrentMessage('')
    setIsThinking(true)

    // AI 응답 생성 (실제로는 API 호출)
    setTimeout(async () => {
      const aiResponse = await generateAIResponse(currentMessage.trim())
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: aiResponse,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
      setIsThinking(false)
    }, 1500)
  }

  const generateFeedback = (message: string) => {
    // 실제 구현에서는 음성 인식과 자연어 처리를 통한 피드백
    return {
      grammar: Math.floor(Math.random() * 30) + 70, // 70-100
      pronunciation: Math.floor(Math.random() * 25) + 75, // 75-100
      fluency: Math.floor(Math.random() * 20) + 80, // 80-100
      suggestions: [
        "Try using more varied vocabulary",
        "Great use of past tense!",
        "Consider adding more details"
      ]
    }
  }

  const startRecording = () => {
    setIsRecording(true)
    // 실제 구현에서는 Web Speech API 사용
  }

  const stopRecording = () => {
    setIsRecording(false)
    // 음성을 텍스트로 변환하여 currentMessage에 설정
    setCurrentMessage("This would be the transcribed text from speech recognition.")
  }

  const playMessage = (content: string) => {
    if (settings.voiceEnabled) {
      setIsPlaying(true)
      // 실제 구현에서는 Text-to-Speech API 사용
      setTimeout(() => setIsPlaying(false), 2000)
    }
  }

  const resetConversation = () => {
    setMessages([])
    setCurrentMessage('')
    startConversation()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/english-conversation"
              className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-rose-600 dark:text-rose-400" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
                AI 대화 파트너
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                실시간 AI와 자연스러운 영어 회화 연습
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            </button>
            <button
              onClick={resetConversation}
              className="p-2 bg-rose-500 text-white rounded-lg shadow-md hover:shadow-lg transition-all"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Settings Panel */}
          {showSettings && (
            <div className="lg:col-span-1">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                  대화 설정
                </h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      대화 주제
                    </label>
                    <select
                      value={settings.topic}
                      onChange={(e) => setSettings(prev => ({ ...prev, topic: e.target.value }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                    >
                      {conversationTopics.map(topic => (
                        <option key={topic.id} value={topic.id}>
                          {topic.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      난이도
                    </label>
                    <select
                      value={settings.difficulty}
                      onChange={(e) => setSettings(prev => ({ ...prev, difficulty: e.target.value as any }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                    >
                      {difficulties.map(diff => (
                        <option key={diff.id} value={diff.id}>
                          {diff.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      AI 성격
                    </label>
                    <select
                      value={settings.aiPersonality}
                      onChange={(e) => setSettings(prev => ({ ...prev, aiPersonality: e.target.value as any }))}
                      className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                    >
                      {personalities.map(personality => (
                        <option key={personality.id} value={personality.id}>
                          {personality.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="voice-enabled"
                      checked={settings.voiceEnabled}
                      onChange={(e) => setSettings(prev => ({ ...prev, voiceEnabled: e.target.checked }))}
                      className="rounded"
                    />
                    <label htmlFor="voice-enabled" className="text-sm text-gray-700 dark:text-gray-300">
                      음성 재생 활성화
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Chat Interface */}
          <div className={showSettings ? 'lg:col-span-3' : 'lg:col-span-4'}>
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg flex flex-col h-[70vh]">
              {/* Chat Header */}
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-rose-500 to-pink-600 rounded-full flex items-center justify-center">
                    <Bot className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200">
                      AI Assistant
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {conversationTopics.find(t => t.id === settings.topic)?.description}
                    </p>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-xl p-4 ${
                        message.type === 'user'
                          ? 'bg-rose-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        {message.type === 'ai' && (
                          <Bot className="w-5 h-5 text-gray-600 dark:text-gray-400 mt-1" />
                        )}
                        {message.type === 'user' && (
                          <User className="w-5 h-5 text-white mt-1" />
                        )}
                        <div className="flex-1">
                          <p className="mb-2">{message.content}</p>
                          {message.type === 'ai' && settings.voiceEnabled && (
                            <button
                              onClick={() => playMessage(message.content)}
                              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
                            >
                              {isPlaying ? (
                                <VolumeX className="w-4 h-4" />
                              ) : (
                                <Volume2 className="w-4 h-4" />
                              )}
                            </button>
                          )}
                          {message.feedback && (
                            <div className="mt-3 p-3 bg-white/10 rounded-lg text-xs">
                              <div className="grid grid-cols-3 gap-2 mb-2">
                                <div className="text-center">
                                  <div className="font-semibold">문법</div>
                                  <div className={`${message.feedback.grammar >= 80 ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {message.feedback.grammar}%
                                  </div>
                                </div>
                                <div className="text-center">
                                  <div className="font-semibold">발음</div>
                                  <div className={`${message.feedback.pronunciation >= 80 ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {message.feedback.pronunciation}%
                                  </div>
                                </div>
                                <div className="text-center">
                                  <div className="font-semibold">유창성</div>
                                  <div className={`${message.feedback.fluency >= 80 ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {message.feedback.fluency}%
                                  </div>
                                </div>
                              </div>
                              <div className="text-white/80">
                                💡 {message.feedback.suggestions[0]}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {isThinking && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-xl p-4 max-w-[80%]">
                      <div className="flex items-center gap-3">
                        <Bot className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`p-2 rounded-lg transition-colors ${
                      isRecording
                        ? 'bg-red-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-500'
                    }`}
                  >
                    {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                  </button>
                  
                  <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Type your message or use voice input..."
                    className="flex-1 p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-rose-500"
                  />
                  
                  <button
                    onClick={sendMessage}
                    disabled={!currentMessage.trim() || isThinking}
                    className="p-2 bg-rose-500 text-white rounded-lg hover:bg-rose-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
                
                {isRecording && (
                  <div className="mt-2 text-center text-sm text-red-500">
                    🎤 Recording... Speak clearly in English
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}