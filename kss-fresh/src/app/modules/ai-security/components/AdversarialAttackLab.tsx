'use client'

import { useState, useRef, useEffect } from 'react'
import { Upload, Play, RotateCcw, AlertTriangle, Shield, Target } from 'lucide-react'

interface AttackResult {
  originalPrediction: string
  adversarialPrediction: string
  confidence: number
  perturbationStrength: number
  success: boolean
}

export default function AdversarialAttackLab() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [attackType, setAttackType] = useState<'fgsm' | 'pgd' | 'cw'>('fgsm')
  const [epsilon, setEpsilon] = useState(0.1)
  const [isAttacking, setIsAttacking] = useState(false)
  const [attackResult, setAttackResult] = useState<AttackResult | null>(null)
  const [showPerturbation, setShowPerturbation] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 샘플 이미지들
  const sampleImages = [
    { id: 'cat', name: '고양이', url: '/api/placeholder/224/224', label: 'cat' },
    { id: 'dog', name: '개', url: '/api/placeholder/224/224', label: 'dog' },
    { id: 'bird', name: '새', url: '/api/placeholder/224/224', label: 'bird' },
    { id: 'car', name: '자동차', url: '/api/placeholder/224/224', label: 'car' }
  ]

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
        setAttackResult(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const simulateAttack = async () => {
    if (!selectedImage) return

    setIsAttacking(true)
    
    // 시뮬레이션: 실제로는 ML 모델이 필요하지만 여기서는 가상의 결과 생성
    await new Promise(resolve => setTimeout(resolve, 2000))

    const originalLabels = ['고양이', '개', '새', '자동차', '비행기']
    const originalPrediction = originalLabels[Math.floor(Math.random() * originalLabels.length)]
    let adversarialPrediction: string
    
    // 공격 성공률 시뮬레이션
    const successRate = attackType === 'fgsm' ? 0.7 : attackType === 'pgd' ? 0.9 : 0.95
    const isSuccessful = Math.random() < successRate

    if (isSuccessful) {
      // 잘못된 예측
      const wrongLabels = originalLabels.filter(label => label !== originalPrediction)
      adversarialPrediction = wrongLabels[Math.floor(Math.random() * wrongLabels.length)]
    } else {
      // 공격 실패, 원래 예측 유지
      adversarialPrediction = originalPrediction
    }

    const result: AttackResult = {
      originalPrediction,
      adversarialPrediction,
      confidence: isSuccessful ? 0.3 + Math.random() * 0.4 : 0.8 + Math.random() * 0.2,
      perturbationStrength: epsilon,
      success: isSuccessful
    }

    setAttackResult(result)
    setIsAttacking(false)

    // 캔버스에 perturbation 시각화
    if (showPerturbation && canvasRef.current) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (ctx) {
        // 가상의 perturbation 패턴 그리기
        const imageData = ctx.createImageData(224, 224)
        for (let i = 0; i < imageData.data.length; i += 4) {
          const noise = (Math.random() - 0.5) * epsilon * 255
          imageData.data[i] = Math.abs(noise)     // R
          imageData.data[i + 1] = Math.abs(noise) // G  
          imageData.data[i + 2] = Math.abs(noise) // B
          imageData.data[i + 3] = 100             // A
        }
        ctx.putImageData(imageData, 0, 0)
      }
    }
  }

  const resetDemo = () => {
    setSelectedImage(null)
    setAttackResult(null)
    setShowPerturbation(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-6">
      {/* 제어 패널 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공격 설정</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              공격 방법
            </label>
            <select
              value={attackType}
              onChange={(e) => setAttackType(e.target.value as any)}
              className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md text-gray-900 dark:text-white"
            >
              <option value="fgsm">FGSM (Fast Gradient Sign Method)</option>
              <option value="pgd">PGD (Projected Gradient Descent)</option>
              <option value="cw">C&W (Carlini & Wagner)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              섭동 강도 (ε): {epsilon.toFixed(3)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.3"
              step="0.001"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-end">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showPerturbation}
                onChange={(e) => setShowPerturbation(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                섭동 시각화
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* 이미지 업로드 및 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">이미지 선택</h3>
        
        <div className="space-y-4">
          {/* 파일 업로드 */}
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              <Upload className="w-4 h-4" />
              이미지 업로드
            </button>
          </div>

          {/* 샘플 이미지 */}
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">또는 샘플 이미지 선택:</p>
            <div className="grid grid-cols-4 gap-2">
              {sampleImages.map((img) => (
                <button
                  key={img.id}
                  onClick={() => {
                    setSelectedImage(img.url)
                    setAttackResult(null)
                  }}
                  className="p-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className="w-full h-16 bg-gray-200 dark:bg-gray-600 rounded mb-1 flex items-center justify-center">
                    <span className="text-xs text-gray-500">{img.name}</span>
                  </div>
                  <p className="text-xs text-center">{img.name}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 이미지 표시 및 공격 실행 */}
      {selectedImage && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">공격 실행</h3>
            <div className="flex gap-2">
              <button
                onClick={simulateAttack}
                disabled={isAttacking}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isAttacking ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    공격 중...
                  </>
                ) : (
                  <>
                    <Target className="w-4 h-4" />
                    공격 실행
                  </>
                )}
              </button>
              <button
                onClick={resetDemo}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 원본 이미지 */}
            <div className="text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">원본 이미지</h4>
              <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
                <img
                  src={selectedImage}
                  alt="Original"
                  className="w-full h-48 object-cover"
                />
              </div>
            </div>

            {/* 섭동 시각화 */}
            {showPerturbation && (
              <div className="text-center">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">섭동 (Perturbation)</h4>
                <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    width={224}
                    height={224}
                    className="w-full h-48"
                  />
                </div>
              </div>
            )}

            {/* 공격된 이미지 */}
            <div className="text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">공격된 이미지</h4>
              <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
                <img
                  src={selectedImage}
                  alt="Adversarial"
                  className="w-full h-48 object-cover"
                  style={{
                    filter: attackResult ? `brightness(${1 + epsilon * 0.5}) contrast(${1 + epsilon * 0.3})` : 'none'
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 공격 결과 */}
      {attackResult && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            {attackResult.success ? (
              <AlertTriangle className="w-6 h-6 text-red-600" />
            ) : (
              <Shield className="w-6 h-6 text-green-600" />
            )}
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              공격 결과
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <h4 className="font-medium text-green-800 dark:text-green-300 mb-2">원본 예측</h4>
                <p className="text-green-700 dark:text-green-200 text-lg">
                  {attackResult.originalPrediction}
                </p>
                <p className="text-sm text-green-600 dark:text-green-400">
                  신뢰도: 95%
                </p>
              </div>

              <div className={`p-4 rounded-lg ${
                attackResult.success 
                  ? 'bg-red-50 dark:bg-red-900/20' 
                  : 'bg-green-50 dark:bg-green-900/20'
              }`}>
                <h4 className={`font-medium mb-2 ${
                  attackResult.success 
                    ? 'text-red-800 dark:text-red-300' 
                    : 'text-green-800 dark:text-green-300'
                }`}>
                  공격 후 예측
                </h4>
                <p className={`text-lg ${
                  attackResult.success 
                    ? 'text-red-700 dark:text-red-200' 
                    : 'text-green-700 dark:text-green-200'
                }`}>
                  {attackResult.adversarialPrediction}
                </p>
                <p className={`text-sm ${
                  attackResult.success 
                    ? 'text-red-600 dark:text-red-400' 
                    : 'text-green-600 dark:text-green-400'
                }`}>
                  신뢰도: {(attackResult.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h4 className="font-medium text-gray-800 dark:text-gray-300 mb-3">공격 정보</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">공격 방법:</span>
                    <span className="text-gray-900 dark:text-white font-mono">
                      {attackType.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">섭동 강도:</span>
                    <span className="text-gray-900 dark:text-white font-mono">
                      ε = {epsilon.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">공격 성공:</span>
                    <span className={`font-medium ${
                      attackResult.success ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {attackResult.success ? '성공' : '실패'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">보안 권장사항</h4>
                <ul className="text-sm text-blue-700 dark:text-blue-200 space-y-1">
                  <li>• 적대적 훈련 적용</li>
                  <li>• 입력 전처리 강화</li>
                  <li>• 모델 앙상블 사용</li>
                  <li>• 이상 탐지 시스템 구축</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 설명 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
          적대적 공격이란?
        </h3>
        <p className="text-blue-800 dark:text-blue-200 leading-relaxed">
          적대적 공격은 머신러닝 모델을 속이기 위해 입력 데이터에 미세한 변화를 가하는 기법입니다. 
          인간의 눈으로는 구별하기 어려운 수준의 변화지만, AI 모델의 예측을 완전히 바꿀 수 있습니다. 
          이러한 공격을 이해하고 방어하는 것은 안전한 AI 시스템 구축에 필수적입니다.
        </p>
      </div>
    </div>
  )
}