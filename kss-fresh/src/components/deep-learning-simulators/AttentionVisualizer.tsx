'use client'

import { useState, useEffect } from 'react'
import { Sparkles } from 'lucide-react'

interface AttentionScore {
  from: number
  to: number
  score: number
}

export default function AttentionVisualizer() {
  const [inputText, setInputText] = useState('The cat sat on the mat')
  const [words, setWords] = useState<string[]>([])
  const [selectedWord, setSelectedWord] = useState<number>(0)
  const [attentionScores, setAttentionScores] = useState<AttentionScore[]>([])
  const [numHeads, setNumHeads] = useState(4)
  const [currentHead, setCurrentHead] = useState(0)

  // Process input text
  useEffect(() => {
    const wordList = inputText.trim().split(/\s+/)
    setWords(wordList)
    setSelectedWord(Math.min(selectedWord, wordList.length - 1))
  }, [inputText])

  // Generate attention scores
  useEffect(() => {
    if (words.length === 0) return

    const scores: AttentionScore[] = []

    // Simulate attention patterns for different heads
    words.forEach((_, toIdx) => {
      let score = 0

      switch (currentHead % 4) {
        case 0: // Self-attention pattern
          score = Math.exp(-Math.abs(selectedWord - toIdx) * 0.5)
          break
        case 1: // Look-ahead pattern
          score = toIdx > selectedWord ? Math.exp(-(toIdx - selectedWord) * 0.3) : 0.1
          break
        case 2: // Look-back pattern
          score = toIdx < selectedWord ? Math.exp(-(selectedWord - toIdx) * 0.3) : 0.1
          break
        case 3: // Global pattern
          score = 0.5 + 0.5 * Math.random()
          break
      }

      scores.push({
        from: selectedWord,
        to: toIdx,
        score: score
      })
    })

    // Normalize scores
    const total = scores.reduce((sum, s) => sum + s.score, 0)
    const normalized = scores.map(s => ({
      ...s,
      score: s.score / total
    }))

    setAttentionScores(normalized)
  }, [selectedWord, words, currentHead])

  const getAttentionColor = (score: number) => {
    const intensity = Math.floor(score * 255)
    return `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, ${255 - intensity})`
  }

  const getAttentionOpacity = (score: number) => {
    return 0.2 + score * 0.8
  }

  return (
    <div className="space-y-6">
      {/* Input Controls */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            입력 문장
          </h3>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm resize-none"
            rows={2}
            placeholder="문장을 입력하세요..."
          />
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            {words.length}개 토큰
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Attention Heads: {numHeads}
          </h3>
          <input
            type="range"
            min="1"
            max="8"
            value={numHeads}
            onChange={(e) => setNumHeads(parseInt(e.target.value))}
            className="w-full mb-4"
          />

          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            현재 Head: {currentHead + 1} / {numHeads}
          </h3>
          <input
            type="range"
            min="0"
            max={numHeads - 1}
            value={currentHead}
            onChange={(e) => setCurrentHead(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Attention Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Sparkles className="text-violet-500" size={24} />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
            Self-Attention Weights (Head {currentHead + 1})
          </h3>
        </div>

        {/* Word tokens with selection */}
        <div className="mb-8">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Query 단어 선택:
          </div>
          <div className="flex flex-wrap gap-3">
            {words.map((word, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedWord(idx)}
                className={`px-4 py-2 rounded-lg font-medium text-lg transition-all ${
                  selectedWord === idx
                    ? 'bg-violet-500 text-white shadow-lg scale-110'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {word}
              </button>
            ))}
          </div>
        </div>

        {/* Attention matrix visualization */}
        <div className="mb-8">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Attention Scores:
          </div>

          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${words.length}, minmax(0, 1fr))` }}>
            {attentionScores.map((score, idx) => (
              <div
                key={idx}
                className="aspect-square rounded-lg flex flex-col items-center justify-center text-xs font-semibold border border-gray-300 dark:border-gray-600"
                style={{
                  backgroundColor: getAttentionColor(score.score),
                  opacity: getAttentionOpacity(score.score)
                }}
              >
                <div className="text-white drop-shadow-lg">
                  {words[score.to]}
                </div>
                <div className="text-white drop-shadow-lg text-[10px] mt-1">
                  {(score.score * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Connection visualization */}
        <div className="relative bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Attention Flow:
          </div>

          <svg width="100%" height="200" className="overflow-visible">
            {/* Draw words at top */}
            {words.map((word, idx) => (
              <g key={`top-${idx}`}>
                <circle
                  cx={`${((idx + 0.5) / words.length) * 100}%`}
                  cy="30"
                  r={selectedWord === idx ? "20" : "15"}
                  fill={selectedWord === idx ? "#8b5cf6" : "#9ca3af"}
                  className="transition-all"
                />
                <text
                  x={`${((idx + 0.5) / words.length) * 100}%`}
                  y="35"
                  textAnchor="middle"
                  className="fill-white text-xs font-semibold"
                >
                  {word.length > 5 ? word.slice(0, 4) + '.' : word}
                </text>
              </g>
            ))}

            {/* Draw attention connections */}
            {attentionScores.map((score, idx) => {
              const fromX = ((score.from + 0.5) / words.length) * 100
              const toX = ((score.to + 0.5) / words.length) * 100
              const opacity = score.score

              return (
                <g key={`conn-${idx}`}>
                  <path
                    d={`M ${fromX}% 50 Q ${(fromX + toX) / 2}% 120, ${toX}% 170`}
                    fill="none"
                    stroke={getAttentionColor(score.score)}
                    strokeWidth={Math.max(1, score.score * 5)}
                    opacity={opacity}
                    className="transition-all"
                  />
                </g>
              )
            })}

            {/* Draw words at bottom */}
            {words.map((word, idx) => {
              const score = attentionScores[idx]?.score || 0
              return (
                <g key={`bottom-${idx}`}>
                  <circle
                    cx={`${((idx + 0.5) / words.length) * 100}%`}
                    cy="170"
                    r={10 + score * 10}
                    fill={getAttentionColor(score)}
                    opacity={getAttentionOpacity(score)}
                    className="transition-all"
                  />
                  <text
                    x={`${((idx + 0.5) / words.length) * 100}%`}
                    y="175"
                    textAnchor="middle"
                    className="fill-white text-xs font-semibold"
                  >
                    {word.length > 5 ? word.slice(0, 4) + '.' : word}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* Multi-head comparison */}
        <div className="mt-8">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            All Heads Preview:
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: numHeads }).map((_, headIdx) => (
              <button
                key={headIdx}
                onClick={() => setCurrentHead(headIdx)}
                className={`p-4 rounded-xl border-2 transition-all ${
                  currentHead === headIdx
                    ? 'border-violet-500 bg-violet-50 dark:bg-violet-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-violet-300 dark:hover:border-violet-600'
                }`}
              >
                <div className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Head {headIdx + 1}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {headIdx % 4 === 0 && 'Local'}
                  {headIdx % 4 === 1 && 'Forward'}
                  {headIdx % 4 === 2 && 'Backward'}
                  {headIdx % 4 === 3 && 'Global'}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3">
          💡 Attention 메커니즘 이해하기
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>Query (쿼리):</strong> 선택한 단어가 "무엇을 찾고 있는지"를 나타냄
          </div>
          <div>
            <strong>Key (키):</strong> 각 단어가 "어떤 정보를 가지고 있는지"를 나타냄
          </div>
          <div>
            <strong>Value (값):</strong> 실제로 전달될 정보의 내용
          </div>
          <div>
            <strong>Multi-Head:</strong> 여러 관점에서 동시에 attention을 계산
          </div>
        </div>
      </div>
    </div>
  )
}
