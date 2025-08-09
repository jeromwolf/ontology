'use client'

import { useState } from 'react'
import { UserCheck, UserX, Database, AlertTriangle, Shield, BarChart3 } from 'lucide-react'

interface MembershipResult {
  sample: string
  prediction: number
  confidence: number
  isMember: boolean
  actualMember: boolean
}

interface AttributeResult {
  attribute: string
  inferredValue: string
  confidence: number
  actualValue: string
}

export default function PrivacyAttackSimulator() {
  const [attackType, setAttackType] = useState<'membership' | 'attribute'>('membership')
  const [dataset, setDataset] = useState<'medical' | 'financial' | 'social'>('medical')
  const [isAttacking, setIsAttacking] = useState(false)
  const [membershipResults, setMembershipResults] = useState<MembershipResult[]>([])
  const [attributeResults, setAttributeResults] = useState<AttributeResult[]>([])
  const [overallStats, setOverallStats] = useState<{
    accuracy: number
    precision: number
    recall: number
    samplesAnalyzed: number
  } | null>(null)

  const datasetInfo = {
    medical: {
      name: '의료 데이터셋',
      description: '환자 진료 기록 및 진단 정보',
      sensitivity: '극도로 민감',
      samples: ['환자 A: 당뇨병 진단', '환자 B: 고혈압 치료', '환자 C: 정기 검진'],
      attributes: ['나이', '성별', '질병력', '치료 기록']
    },
    financial: {
      name: '금융 데이터셋',
      description: '고객 금융 거래 및 신용 정보',
      sensitivity: '매우 민감',
      samples: ['고객 X: 대출 승인', '고객 Y: 신용카드 발급', '고객 Z: 투자 상담'],
      attributes: ['소득', '신용점수', '거래 내역', '자산 규모']
    },
    social: {
      name: '소셜 데이터셋',
      description: '사용자 소셜 미디어 활동 데이터',
      sensitivity: '민감',
      samples: ['사용자 1: 게시물 분석', '사용자 2: 네트워크 분석', '사용자 3: 선호도 분석'],
      attributes: ['관심사', '정치 성향', '사회적 관계', '활동 패턴']
    }
  }

  const runMembershipInferenceAttack = async () => {
    setIsAttacking(true)
    setMembershipResults([])
    setOverallStats(null)

    const samples = datasetInfo[dataset].samples
    const results: MembershipResult[] = []

    for (let i = 0; i < samples.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800))

      // 멤버십 추론 시뮬레이션
      const actualMember = Math.random() > 0.5
      const prediction = actualMember ? 
        0.6 + Math.random() * 0.4 : // 실제 멤버인 경우 높은 예측값
        Math.random() * 0.5 // 비멤버인 경우 낮은 예측값
      
      const confidence = Math.abs(prediction - 0.5) * 2
      const isMember = prediction > 0.5

      const result: MembershipResult = {
        sample: samples[i],
        prediction,
        confidence,
        isMember,
        actualMember
      }

      results.push(result)
      setMembershipResults([...results])
    }

    // 통계 계산
    const correctPredictions = results.filter(r => r.isMember === r.actualMember).length
    const truePositives = results.filter(r => r.isMember && r.actualMember).length
    const falsePositives = results.filter(r => r.isMember && !r.actualMember).length
    const falseNegatives = results.filter(r => !r.isMember && r.actualMember).length

    const stats = {
      accuracy: (correctPredictions / results.length) * 100,
      precision: truePositives / (truePositives + falsePositives) * 100 || 0,
      recall: truePositives / (truePositives + falseNegatives) * 100 || 0,
      samplesAnalyzed: results.length
    }

    setOverallStats(stats)
    setIsAttacking(false)
  }

  const runAttributeInferenceAttack = async () => {
    setIsAttacking(true)
    setAttributeResults([])
    setOverallStats(null)

    const attributes = datasetInfo[dataset].attributes
    const results: AttributeResult[] = []

    for (let i = 0; i < attributes.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))

      // 속성 추론 시뮬레이션
      const possibleValues = {
        '나이': ['20대', '30대', '40대', '50대+'],
        '성별': ['남성', '여성'],
        '소득': ['저소득', '중소득', '고소득'],
        '관심사': ['스포츠', '음악', '여행', '기술'],
        '질병력': ['없음', '고혈압', '당뇨', '기타'],
        '신용점수': ['우수', '보통', '주의'],
        '정치 성향': ['진보', '중도', '보수']
      }

      const attributeName = attributes[i]
      const values = possibleValues[attributeName as keyof typeof possibleValues] || ['값A', '값B', '값C']
      
      const actualValue = values[Math.floor(Math.random() * values.length)]
      const inferredValue = Math.random() > 0.3 ? actualValue : values[Math.floor(Math.random() * values.length)]
      const confidence = inferredValue === actualValue ? 0.7 + Math.random() * 0.3 : 0.3 + Math.random() * 0.4

      const result: AttributeResult = {
        attribute: attributeName,
        inferredValue,
        confidence,
        actualValue
      }

      results.push(result)
      setAttributeResults([...results])
    }

    // 통계 계산
    const correctInferences = results.filter(r => r.inferredValue === r.actualValue).length
    const stats = {
      accuracy: (correctInferences / results.length) * 100,
      precision: (correctInferences / results.length) * 100,
      recall: (correctInferences / results.length) * 100,
      samplesAnalyzed: results.length
    }

    setOverallStats(stats)
    setIsAttacking(false)
  }

  const startAttack = () => {
    if (attackType === 'membership') {
      runMembershipInferenceAttack()
    } else {
      runAttributeInferenceAttack()
    }
  }

  const resetSimulation = () => {
    setIsAttacking(false)
    setMembershipResults([])
    setAttributeResults([])
    setOverallStats(null)
  }

  return (
    <div className="space-y-6">
      {/* 설정 패널 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공격 설정</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              공격 유형
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="attackType"
                  value="membership"
                  checked={attackType === 'membership'}
                  onChange={(e) => setAttackType(e.target.value as any)}
                  className="mr-2"
                />
                <span className="text-gray-700 dark:text-gray-300">멤버십 추론 공격</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="attackType"
                  value="attribute"
                  checked={attackType === 'attribute'}
                  onChange={(e) => setAttackType(e.target.value as any)}
                  className="mr-2"
                />
                <span className="text-gray-700 dark:text-gray-300">속성 추론 공격</span>
              </label>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              타겟 데이터셋
            </label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value as any)}
              className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md text-gray-900 dark:text-white"
            >
              <option value="medical">의료 데이터</option>
              <option value="financial">금융 데이터</option>
              <option value="social">소셜 데이터</option>
            </select>
          </div>
        </div>
      </div>

      {/* 데이터셋 정보 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              {datasetInfo[dataset].name}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {datasetInfo[dataset].description}
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">샘플 데이터</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  {datasetInfo[dataset].samples.map((sample, i) => (
                    <li key={i}>• {sample}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">민감한 속성</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  {datasetInfo[dataset].attributes.map((attr, i) => (
                    <li key={i}>• {attr}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
          
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            datasetInfo[dataset].sensitivity === '극도로 민감' 
              ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
              : datasetInfo[dataset].sensitivity === '매우 민감'
              ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300'
              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
          }`}>
            {datasetInfo[dataset].sensitivity}
          </div>
        </div>
      </div>

      {/* 공격 실행 버튼 */}
      <div className="flex gap-4">
        <button
          onClick={startAttack}
          disabled={isAttacking}
          className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {attackType === 'membership' ? <UserCheck className="w-5 h-5" /> : <BarChart3 className="w-5 h-5" />}
          {isAttacking ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              공격 진행 중...
            </>
          ) : (
            <>
              {attackType === 'membership' ? '멤버십 추론 시작' : '속성 추론 시작'}
            </>
          )}
        </button>
        
        <button
          onClick={resetSimulation}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          초기화
        </button>
      </div>

      {/* 멤버십 추론 결과 */}
      {membershipResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">멤버십 추론 결과</h3>
          
          <div className="space-y-3">
            {membershipResults.map((result, index) => (
              <div key={index} className={`p-4 rounded-lg border-2 ${
                result.isMember === result.actualMember
                  ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                  : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 dark:text-white">
                      {result.sample}
                    </p>
                    <div className="flex items-center gap-4 mt-2 text-sm">
                      <span className="flex items-center gap-1">
                        {result.isMember ? <UserCheck className="w-4 h-4 text-blue-600" /> : <UserX className="w-4 h-4 text-gray-600" />}
                        추론: {result.isMember ? '멤버' : '비멤버'}
                      </span>
                      <span className="flex items-center gap-1">
                        {result.actualMember ? <UserCheck className="w-4 h-4 text-green-600" /> : <UserX className="w-4 h-4 text-gray-600" />}
                        실제: {result.actualMember ? '멤버' : '비멤버'}
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        신뢰도: {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className={`text-2xl ${
                    result.isMember === result.actualMember ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {result.isMember === result.actualMember ? '✓' : '✗'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 속성 추론 결과 */}
      {attributeResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">속성 추론 결과</h3>
          
          <div className="space-y-3">
            {attributeResults.map((result, index) => (
              <div key={index} className={`p-4 rounded-lg border-2 ${
                result.inferredValue === result.actualValue
                  ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                  : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 dark:text-white mb-2">
                      {result.attribute}
                    </p>
                    <div className="flex items-center gap-6 text-sm">
                      <span className="text-blue-600 dark:text-blue-400">
                        추론값: <strong>{result.inferredValue}</strong>
                      </span>
                      <span className="text-green-600 dark:text-green-400">
                        실제값: <strong>{result.actualValue}</strong>
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        신뢰도: {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className={`text-2xl ${
                    result.inferredValue === result.actualValue ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {result.inferredValue === result.actualValue ? '✓' : '✗'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 통계 */}
      {overallStats && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공격 성능 통계</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                {overallStats.accuracy.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">정확도</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">
                {overallStats.precision.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">정밀도</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-1">
                {overallStats.recall.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">재현율</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">
                {overallStats.samplesAnalyzed}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">분석 샘플</div>
            </div>
          </div>
        </div>
      )}

      {/* 프라이버시 위험도 평가 */}
      {overallStats && (
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border border-red-200 dark:border-red-800">
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-6 h-6 text-red-600 mt-1" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                프라이버시 위험도 평가
              </h3>
              
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-red-800 dark:text-red-200">위험도</span>
                  <span className="text-red-800 dark:text-red-200">
                    {overallStats.accuracy > 80 ? '매우 높음' : 
                     overallStats.accuracy > 60 ? '높음' : 
                     overallStats.accuracy > 40 ? '보통' : '낮음'}
                  </span>
                </div>
                <div className="w-full bg-red-200 dark:bg-red-800 rounded-full h-2">
                  <div
                    className="bg-red-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${overallStats.accuracy}%` }}
                  />
                </div>
              </div>

              <p className="text-red-800 dark:text-red-200 mb-4">
                {overallStats.accuracy > 70 
                  ? `높은 정확도(${overallStats.accuracy.toFixed(1)}%)로 개인정보가 추론되었습니다. 심각한 프라이버시 침해 위험이 있습니다.`
                  : `보통 정확도(${overallStats.accuracy.toFixed(1)}%)로 일부 정보가 노출되었습니다. 추가적인 보안 조치가 필요합니다.`}
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">잠재적 영향</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 개인정보 노출</li>
                    <li>• 차별 및 편견 강화</li>
                    <li>• 신원 도용 위험</li>
                    <li>• 법적 책임 문제</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">대응 방안</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 차등 프라이버시 적용</li>
                    <li>• 데이터 최소화 원칙</li>
                    <li>• 익명화 기법 강화</li>
                    <li>• 접근 권한 제한</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 방어 기법 설명 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
          프라이버시 공격 방어 기법
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">차등 프라이버시</h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              쿼리 결과에 적절한 노이즈를 추가하여 개별 데이터의 존재 여부를 숨깁니다.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">연합 학습</h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              원본 데이터를 중앙 서버로 전송하지 않고 분산된 방식으로 모델을 훈련시킵니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}