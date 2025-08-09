'use client'

import { useState } from 'react'
import { Shield, Target, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

interface DefenseResult {
  mechanism: string
  effectiveness: number
  robustness: number
  performance_impact: number
  cost: number
}

interface AttackScenario {
  id: string
  name: string
  description: string
  difficulty: 'low' | 'medium' | 'high'
  target: string
}

export default function DefenseMechanismsTester() {
  const [selectedAttack, setSelectedAttack] = useState<string>('adversarial')
  const [defenses, setDefenses] = useState<string[]>(['adversarial-training'])
  const [isTesting, setIsTesting] = useState(false)
  const [testResults, setTestResults] = useState<DefenseResult[]>([])
  const [overallScore, setOverallScore] = useState<number | null>(null)

  const attackScenarios: AttackScenario[] = [
    {
      id: 'adversarial',
      name: '적대적 예제 공격',
      description: 'FGSM, PGD 등을 통한 이미지 분류 모델 공격',
      difficulty: 'medium',
      target: '이미지 분류기'
    },
    {
      id: 'model-extraction',
      name: '모델 추출 공격',
      description: '블랙박스 API를 통한 모델 복제 시도',
      difficulty: 'high',
      target: '클라우드 API'
    },
    {
      id: 'poisoning',
      name: '데이터 중독 공격',
      description: '훈련 데이터에 악의적 샘플 주입',
      difficulty: 'high',
      target: '훈련 파이프라인'
    },
    {
      id: 'membership-inference',
      name: '멤버십 추론 공격',
      description: '훈련 데이터 멤버십 정보 추론',
      difficulty: 'low',
      target: '프라이버시'
    },
    {
      id: 'backdoor',
      name: '백도어 공격',
      description: '특정 트리거에 반응하는 숨겨진 기능 삽입',
      difficulty: 'high',
      target: '모델 무결성'
    }
  ]

  const defenseMechanisms = {
    'adversarial-training': {
      name: '적대적 훈련',
      description: '적대적 예제를 포함하여 모델 훈련',
      effectiveness: { adversarial: 85, 'model-extraction': 30, poisoning: 40, 'membership-inference': 50, backdoor: 35 },
      performance_impact: 15,
      cost: 3
    },
    'differential-privacy': {
      name: '차등 프라이버시',
      description: '훈련 과정에 노이즈 추가',
      effectiveness: { adversarial: 40, 'model-extraction': 70, poisoning: 50, 'membership-inference': 90, backdoor: 45 },
      performance_impact: 25,
      cost: 2
    },
    'input-preprocessing': {
      name: '입력 전처리',
      description: '입력 데이터 정규화 및 필터링',
      effectiveness: { adversarial: 60, 'model-extraction': 20, poisoning: 30, 'membership-inference': 25, backdoor: 40 },
      performance_impact: 5,
      cost: 1
    },
    'ensemble-methods': {
      name: '앙상블 방법',
      description: '여러 모델의 예측 결합',
      effectiveness: { adversarial: 70, 'model-extraction': 80, poisoning: 60, 'membership-inference': 60, backdoor: 50 },
      performance_impact: 30,
      cost: 4
    },
    'detection-systems': {
      name: '이상 탐지 시스템',
      description: '비정상적인 입력이나 쿼리 패턴 탐지',
      effectiveness: { adversarial: 75, 'model-extraction': 85, poisoning: 70, 'membership-inference': 65, backdoor: 80 },
      performance_impact: 10,
      cost: 3
    },
    'model-distillation': {
      name: '모델 증류',
      description: '큰 모델의 지식을 작은 모델로 전이',
      effectiveness: { adversarial: 55, 'model-extraction': 60, poisoning: 45, 'membership-inference': 70, backdoor: 35 },
      performance_impact: 20,
      cost: 2
    },
    'randomization': {
      name: '무작위화',
      description: '모델 출력에 무작위성 추가',
      effectiveness: { adversarial: 45, 'model-extraction': 75, poisoning: 35, 'membership-inference': 80, backdoor: 40 },
      performance_impact: 18,
      cost: 1
    },
    'certified-defenses': {
      name: '인증된 방어',
      description: '수학적으로 보장된 방어 메커니즘',
      effectiveness: { adversarial: 95, 'model-extraction': 40, poisoning: 60, 'membership-inference': 55, backdoor: 45 },
      performance_impact: 35,
      cost: 5
    }
  }

  const runDefenseTest = async () => {
    setIsTesting(true)
    setTestResults([])
    setOverallScore(null)

    const results: DefenseResult[] = []

    for (const defenseId of defenses) {
      await new Promise(resolve => setTimeout(resolve, 1000))

      const defense = defenseMechanisms[defenseId as keyof typeof defenseMechanisms]
      const effectiveness = defense.effectiveness[selectedAttack as keyof typeof defense.effectiveness]
      
      // 노이즈 추가로 더 현실적인 결과 생성
      const noiseRange = 10
      const noisyEffectiveness = Math.max(0, Math.min(100, 
        effectiveness + (Math.random() - 0.5) * noiseRange
      ))

      const result: DefenseResult = {
        mechanism: defense.name,
        effectiveness: noisyEffectiveness,
        robustness: Math.max(0, noisyEffectiveness - Math.random() * 15),
        performance_impact: defense.performance_impact + (Math.random() - 0.5) * 10,
        cost: defense.cost
      }

      results.push(result)
      setTestResults([...results])
    }

    // 전체 점수 계산
    const avgEffectiveness = results.reduce((sum, r) => sum + r.effectiveness, 0) / results.length
    const avgPerformanceImpact = results.reduce((sum, r) => sum + r.performance_impact, 0) / results.length
    const totalCost = results.reduce((sum, r) => sum + r.cost, 0)
    
    // 종합 점수 (효과성 - 성능 영향 - 비용 패널티)
    const score = Math.max(0, avgEffectiveness - avgPerformanceImpact * 0.5 - totalCost * 2)
    setOverallScore(score)
    setIsTesting(false)
  }

  const resetTest = () => {
    setIsTesting(false)
    setTestResults([])
    setOverallScore(null)
  }

  const toggleDefense = (defenseId: string) => {
    setDefenses(prev => 
      prev.includes(defenseId)
        ? prev.filter(d => d !== defenseId)
        : [...prev, defenseId]
    )
  }

  const selectedAttackInfo = attackScenarios.find(a => a.id === selectedAttack)

  return (
    <div className="space-y-6">
      {/* 공격 시나리오 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공격 시나리오 선택</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {attackScenarios.map((attack) => (
            <button
              key={attack.id}
              onClick={() => setSelectedAttack(attack.id)}
              className={`p-4 text-left rounded-lg border-2 transition-colors ${
                selectedAttack === attack.id
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-red-300 dark:hover:border-red-600'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-red-600" />
                <h4 className="font-medium text-gray-900 dark:text-white">{attack.name}</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {attack.description}
              </p>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  attack.difficulty === 'low' 
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : attack.difficulty === 'medium'
                    ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                }`}>
                  {attack.difficulty === 'low' ? '낮음' : 
                   attack.difficulty === 'medium' ? '보통' : '높음'}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {attack.target}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 방어 메커니즘 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">방어 메커니즘 선택</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(defenseMechanisms).map(([id, defense]) => (
            <div key={id} className="flex items-start gap-3">
              <input
                type="checkbox"
                id={id}
                checked={defenses.includes(id)}
                onChange={() => toggleDefense(id)}
                className="mt-1"
              />
              <label htmlFor={id} className="flex-1 cursor-pointer">
                <div className="flex items-center gap-2 mb-1">
                  <Shield className="w-4 h-4 text-blue-600" />
                  <h4 className="font-medium text-gray-900 dark:text-white">{defense.name}</h4>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  {defense.description}
                </p>
                <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                  <span>성능 영향: {defense.performance_impact}%</span>
                  <span>비용: {'$'.repeat(defense.cost)}</span>
                </div>
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* 선택된 공격 정보 */}
      {selectedAttackInfo && (
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border border-red-200 dark:border-red-800">
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-6 h-6 text-red-600 mt-1" />
            <div>
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                테스트 대상: {selectedAttackInfo.name}
              </h3>
              <p className="text-red-800 dark:text-red-200 mb-2">
                {selectedAttackInfo.description}
              </p>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-red-700 dark:text-red-300">
                  타겟: {selectedAttackInfo.target}
                </span>
                <span className="text-red-700 dark:text-red-300">
                  난이도: {selectedAttackInfo.difficulty === 'low' ? '낮음' : 
                           selectedAttackInfo.difficulty === 'medium' ? '보통' : '높음'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 테스트 실행 */}
      <div className="flex gap-4">
        <button
          onClick={runDefenseTest}
          disabled={isTesting || defenses.length === 0}
          className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Shield className="w-5 h-5" />
          {isTesting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              테스트 진행 중...
            </>
          ) : (
            '방어 효과 테스트'
          )}
        </button>
        
        <button
          onClick={resetTest}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          초기화
        </button>
      </div>

      {/* 테스트 결과 */}
      {testResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">방어 효과 분석</h3>
          
          <div className="space-y-4">
            {testResults.map((result, index) => (
              <div key={index} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {result.mechanism}
                  </h4>
                  <div className="flex items-center gap-2">
                    {result.effectiveness > 70 ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : result.effectiveness > 40 ? (
                      <AlertTriangle className="w-5 h-5 text-yellow-600" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                    <span className={`font-medium ${
                      result.effectiveness > 70 ? 'text-green-600' :
                      result.effectiveness > 40 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {result.effectiveness > 70 ? '효과적' :
                       result.effectiveness > 40 ? '보통' : '비효과적'}
                    </span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">효과성</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-green-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${result.effectiveness}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {result.effectiveness.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">견고성</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${result.robustness}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {result.robustness.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">성능 영향</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-yellow-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${result.performance_impact}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {result.performance_impact.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">비용</div>
                    <div className="flex items-center gap-1">
                      {Array.from({ length: 5 }, (_, i) => (
                        <div
                          key={i}
                          className={`w-3 h-3 rounded-full ${
                            i < result.cost ? 'bg-red-600' : 'bg-gray-300 dark:bg-gray-600'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 종합 평가 */}
      {overallScore !== null && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">종합 평가</h3>
          
          <div className="text-center mb-6">
            <div className={`text-6xl font-bold mb-2 ${
              overallScore > 70 ? 'text-green-600' :
              overallScore > 40 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {overallScore.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              종합 방어 점수 (100점 만점)
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
              <h4 className="font-medium text-green-800 dark:text-green-300 mb-1">강점</h4>
              <p className="text-sm text-green-700 dark:text-green-200">
                {overallScore > 70 ? '우수한 방어 체계' :
                 overallScore > 40 ? '적절한 보안 수준' : '기본적인 방어 체계'}
              </p>
            </div>
            
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <AlertTriangle className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
              <h4 className="font-medium text-yellow-800 dark:text-yellow-300 mb-1">개선점</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-200">
                {overallScore > 70 ? '비용 최적화 검토' :
                 overallScore > 40 ? '추가 방어 기법 도입' : '전면적인 보안 강화 필요'}
              </p>
            </div>
            
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Shield className="w-8 h-8 text-blue-600 mx-auto mb-2" />
              <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-1">권장사항</h4>
              <p className="text-sm text-blue-700 dark:text-blue-200">
                {overallScore > 70 ? '현재 체계 유지 및 모니터링' :
                 overallScore > 40 ? '추가 방어 레이어 고려' : '우선순위 방어 기법 도입'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 설명 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
          방어 메커니즘 평가 기준
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">평가 지표</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>• <strong>효과성</strong>: 공격 차단 성공률</li>
              <li>• <strong>견고성</strong>: 다양한 공격에 대한 안정성</li>
              <li>• <strong>성능 영향</strong>: 시스템 성능 저하 정도</li>
              <li>• <strong>비용</strong>: 구현 및 운영 비용</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">권장 조합</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>• <strong>기본</strong>: 입력 전처리 + 이상 탐지</li>
              <li>• <strong>고급</strong>: 적대적 훈련 + 앙상블</li>
              <li>• <strong>프라이버시</strong>: 차등 프라이버시 + 무작위화</li>
              <li>• <strong>최고급</strong>: 인증된 방어 + 다중 레이어</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}