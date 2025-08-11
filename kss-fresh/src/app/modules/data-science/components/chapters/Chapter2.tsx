'use client'

import { useState } from 'react'
import { 
  Brain, TrendingUp, AlertCircle, Info, Target,
  BarChart3, LineChart, Calculator, Zap,
  CheckCircle, XCircle, HelpCircle, Activity,
  ChevronRight, Play, FileText, Lightbulb
} from 'lucide-react'

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter2({ onComplete }: ChapterProps) {
  const [activeTab, setActiveTab] = useState('hypothesis')
  const [showPValueDemo, setShowPValueDemo] = useState(false)
  const [sampleSize, setSampleSize] = useState(100)
  const [confidenceLevel, setConfidenceLevel] = useState(95)

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">통계적 사고와 추론</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          가설 검정, p-value, 신뢰구간, 베이지안 추론의 핵심을 마스터하기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-blue-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">통계적 사고의 기초</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">불확실성과 변동성 이해</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">가설 검정 마스터</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">H0, H1, 유의수준, 검정력</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">p-value의 올바른 해석</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">흔한 오해와 올바른 사용법</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">베이지안 추론 입문</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">사전확률과 사후확률</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 통계적 사고란? */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. 통계적 사고란?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-blue-500" />
            통계적 사고의 핵심 원칙
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">1. 변동성은 어디에나 존재한다</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                모든 프로세스와 측정에는 변동이 있으며, 이를 이해하고 정량화하는 것이 핵심입니다.
              </p>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">2. 데이터는 맥락 속에서 이해해야 한다</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                숫자 자체보다 그것이 수집된 방법과 의미하는 바가 더 중요합니다.
              </p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">3. 상관관계 ≠ 인과관계</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                두 변수가 함께 움직인다고 해서 하나가 다른 하나의 원인은 아닙니다.
              </p>
            </div>
            
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-orange-700 dark:text-orange-400 mb-2">4. 표본은 모집단을 대표해야 한다</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                편향된 표본은 잘못된 결론으로 이어집니다. 대표성이 핵심입니다.
              </p>
            </div>
          </div>
        </div>

        {/* 통계적 추론의 유형 */}
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-900/50 dark:to-gray-800/50 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4">통계적 추론의 두 가지 접근법</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-indigo-600 dark:text-indigo-400">빈도주의적 접근</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-indigo-400 mt-0.5" />
                  <span>확률을 장기적 빈도로 해석</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-indigo-400 mt-0.5" />
                  <span>모수는 고정된 값</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-indigo-400 mt-0.5" />
                  <span>p-value와 신뢰구간 사용</span>
                </li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-purple-600 dark:text-purple-400">베이지안 접근</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span>확률을 믿음의 정도로 해석</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span>모수는 확률분포를 가짐</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span>사전확률과 사후확률 사용</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 2. 가설 검정 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. 가설 검정 (Hypothesis Testing)</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4">가설 검정의 단계</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                1
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">가설 설정</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  귀무가설(H₀): 차이가 없다, 효과가 없다<br/>
                  대립가설(H₁): 차이가 있다, 효과가 있다
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                2
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">유의수준 결정</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  일반적으로 α = 0.05 (5%) 사용<br/>
                  제1종 오류를 범할 확률의 상한선
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                3
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">검정통계량 계산</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  t-test, z-test, χ²-test 등<br/>
                  데이터로부터 계산한 요약 통계량
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                4
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">p-value 계산</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  H₀가 참일 때, 관측된 결과보다 극단적인 결과를 얻을 확률
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                5
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">결론</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  p-value &lt; α → H₀ 기각<br/>
                  p-value ≥ α → H₀ 기각 실패
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 오류의 종류 */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">통계적 오류의 종류</h3>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-300 dark:border-gray-600">
                  <th className="py-3 px-4"></th>
                  <th className="py-3 px-4 text-center">H₀가 참</th>
                  <th className="py-3 px-4 text-center">H₀가 거짓</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">H₀ 기각</td>
                  <td className="py-3 px-4 text-center bg-red-100 dark:bg-red-900/30">
                    <span className="font-semibold text-red-700 dark:text-red-400">제1종 오류 (α)</span><br/>
                    <span className="text-sm">거짓 양성</span>
                  </td>
                  <td className="py-3 px-4 text-center bg-green-100 dark:bg-green-900/30">
                    <span className="font-semibold text-green-700 dark:text-green-400">올바른 결정</span><br/>
                    <span className="text-sm">검정력 (1-β)</span>
                  </td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">H₀ 기각 실패</td>
                  <td className="py-3 px-4 text-center bg-green-100 dark:bg-green-900/30">
                    <span className="font-semibold text-green-700 dark:text-green-400">올바른 결정</span><br/>
                    <span className="text-sm">신뢰수준 (1-α)</span>
                  </td>
                  <td className="py-3 px-4 text-center bg-orange-100 dark:bg-orange-900/30">
                    <span className="font-semibold text-orange-700 dark:text-orange-400">제2종 오류 (β)</span><br/>
                    <span className="text-sm">거짓 음성</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 3. p-value 이해하기 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. p-value의 올바른 이해</h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg mb-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="text-yellow-600 mt-1" />
            <div>
              <p className="font-semibold text-yellow-800 dark:text-yellow-400">p-value는 무엇이 아닌가?</p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>❌ 귀무가설이 참일 확률이 아닙니다</li>
                <li>❌ 대립가설이 참일 확률이 아닙니다</li>
                <li>❌ 효과의 크기를 나타내지 않습니다</li>
                <li>❌ 결과의 중요성을 나타내지 않습니다</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg mb-6">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-600 mt-1" />
            <div>
              <p className="font-semibold text-green-800 dark:text-green-400">p-value는 무엇인가?</p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                ✓ 귀무가설이 참이라고 가정했을 때, 우리가 관측한 데이터(또는 더 극단적인 데이터)를 
                얻을 확률입니다.
              </p>
            </div>
          </div>
        </div>

        {/* p-value 시뮬레이션 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4">p-value 시뮬레이션 데모</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                동전 던지기 실험: 동전이 공정한지 검정
              </p>
              <button
                onClick={() => setShowPValueDemo(!showPValueDemo)}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                시뮬레이션 실행
              </button>
              
              {showPValueDemo && (
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                  <p className="text-sm">
                    <strong>실험:</strong> 100번 던져서 65번 앞면<br/>
                    <strong>H₀:</strong> p = 0.5 (공정한 동전)<br/>
                    <strong>계산된 p-value:</strong> 0.0035<br/>
                    <strong>결론:</strong> α=0.05에서 H₀ 기각
                  </p>
                </div>
              )}
            </div>
            
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-sm text-gray-300 overflow-x-auto">
{`# Python 코드
from scipy import stats

# 관측값
n_trials = 100
n_heads = 65
p_null = 0.5

# 이항 검정
p_value = stats.binom_test(
    n_heads, n_trials, 
    p_null, alternative='two-sided'
)

print(f"p-value: {p_value:.4f}")
# Output: p-value: 0.0035`}</pre>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 신뢰구간 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. 신뢰구간 (Confidence Intervals)</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Calculator className="text-purple-500" />
            신뢰구간의 이해
          </h3>
          
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            95% 신뢰구간의 의미: "동일한 방법으로 100번 표본을 추출하여 신뢰구간을 구하면, 
            그 중 약 95개의 구간이 모수를 포함할 것이다."
          </p>

          {/* 신뢰구간 계산기 */}
          <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">신뢰구간 계산기</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">표본 크기</label>
                <input
                  type="number"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  className="w-full px-3 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-600"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">신뢰수준 (%)</label>
                <select
                  value={confidenceLevel}
                  onChange={(e) => setConfidenceLevel(Number(e.target.value))}
                  className="w-full px-3 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-600"
                >
                  <option value={90}>90%</option>
                  <option value={95}>95%</option>
                  <option value={99}>99%</option>
                </select>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <p className="text-sm">
                <strong>표준오차:</strong> σ/√{sampleSize}<br/>
                <strong>신뢰구간 폭:</strong> ± {confidenceLevel === 90 ? '1.645' : confidenceLevel === 95 ? '1.96' : '2.576'} × 표준오차
              </p>
            </div>
          </div>
        </div>

        {/* 신뢰구간 vs 예측구간 */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">신뢰구간</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              모수(평균, 비율 등)의 추정에 대한 불확실성을 나타냅니다.
              표본 크기가 커질수록 좁아집니다.
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">예측구간</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              개별 관측값의 예측에 대한 불확실성을 나타냅니다.
              항상 신뢰구간보다 넓습니다.
            </p>
          </div>
        </div>
      </section>

      {/* 5. 베이지안 추론 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 베이지안 추론 입문</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4">베이즈 정리</h3>
          
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg mb-4">
            <p className="text-center text-lg font-mono">
              P(H|D) = P(D|H) × P(H) / P(D)
            </p>
            <div className="grid grid-cols-4 gap-2 mt-4 text-sm text-center">
              <div>
                <p className="font-semibold">P(H|D)</p>
                <p className="text-gray-600 dark:text-gray-400">사후확률</p>
              </div>
              <div>
                <p className="font-semibold">P(D|H)</p>
                <p className="text-gray-600 dark:text-gray-400">우도</p>
              </div>
              <div>
                <p className="font-semibold">P(H)</p>
                <p className="text-gray-600 dark:text-gray-400">사전확률</p>
              </div>
              <div>
                <p className="font-semibold">P(D)</p>
                <p className="text-gray-600 dark:text-gray-400">정규화 상수</p>
              </div>
            </div>
          </div>

          {/* 베이지안 예제 */}
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">예제: 질병 진단</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              희귀병의 유병률이 0.1%이고, 검사의 정확도가 99%일 때,
              양성 판정을 받은 사람이 실제로 병에 걸렸을 확률은?
            </p>
            
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-sm font-mono">
                P(병|양성) = (0.99 × 0.001) / [(0.99 × 0.001) + (0.01 × 0.999)]<br/>
                ≈ 0.09 = 9%
              </p>
            </div>
            
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
              💡 직관과 다르게 9%밖에 되지 않습니다! 이것이 기저율 오류입니다.
            </p>
          </div>
        </div>

        {/* 베이지안 vs 빈도주의 */}
        <div className="bg-gray-900 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">베이지안 분석 코드 예제</h3>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 동전 던지기 베이지안 분석
# 사전분포: Beta(1, 1) - 균일분포
alpha_prior, beta_prior = 1, 1

# 데이터: 10번 중 7번 앞면
n_trials = 10
n_heads = 7

# 사후분포: Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_trials - n_heads)

# 시각화
x = np.linspace(0, 1, 1000)
prior = stats.beta.pdf(x, alpha_prior, beta_prior)
posterior = stats.beta.pdf(x, alpha_post, beta_post)

plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'b--', label='사전분포 Beta(1,1)', lw=2)
plt.plot(x, posterior, 'r-', label=f'사후분포 Beta({alpha_post},{beta_post})', lw=2)
plt.axvline(0.5, color='gray', linestyle=':', label='p=0.5')
plt.fill_between(x, posterior, alpha=0.3, color='red')

# 95% 신용구간
credible_interval = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
plt.axvline(credible_interval[0], color='red', linestyle='--', alpha=0.5)
plt.axvline(credible_interval[1], color='red', linestyle='--', alpha=0.5)

plt.xlabel('θ (앞면이 나올 확률)')
plt.ylabel('확률밀도')
plt.title('베이지안 추론: 동전 던지기')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"사후평균: {alpha_post/(alpha_post+beta_post):.3f}")
print(f"95% 신용구간: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")`}</code>
          </pre>
        </div>
      </section>

      {/* 6. 실전 가이드라인 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">6. 실전 통계 분석 가이드라인</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <CheckCircle className="text-green-600" />
              DO: 권장사항
            </h3>
            <ul className="space-y-2 text-sm">
              <li>✓ 분석 전에 가설을 명확히 정의</li>
              <li>✓ 효과 크기와 신뢰구간 함께 보고</li>
              <li>✓ 다중 검정 시 보정 적용</li>
              <li>✓ 시각화로 결과 보완</li>
              <li>✓ 가정사항 확인 (정규성, 등분산성 등)</li>
              <li>✓ 재현 가능한 분석 코드 작성</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <XCircle className="text-red-600" />
              DON'T: 피해야 할 것
            </h3>
            <ul className="space-y-2 text-sm">
              <li>❌ p-hacking (데이터 조작)</li>
              <li>❌ p-value만으로 중요성 판단</li>
              <li>❌ 통계적 유의성 = 실질적 중요성</li>
              <li>❌ 사후 가설 설정</li>
              <li>❌ 선택적 보고</li>
              <li>❌ 표본 크기 무시</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 퀴즈 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🧠 통계적 사고 체크리스트</h2>
          <p className="mb-6">
            이제 통계적 추론의 기초를 배웠습니다. 다음 개념들을 확실히 이해했는지 확인해보세요:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <label className="flex items-center gap-3 text-white/90">
              <input type="checkbox" className="w-4 h-4" />
              <span>가설 검정의 5단계 프로세스</span>
            </label>
            <label className="flex items-center gap-3 text-white/90">
              <input type="checkbox" className="w-4 h-4" />
              <span>p-value의 정확한 의미</span>
            </label>
            <label className="flex items-center gap-3 text-white/90">
              <input type="checkbox" className="w-4 h-4" />
              <span>제1종, 제2종 오류의 차이</span>
            </label>
            <label className="flex items-center gap-3 text-white/90">
              <input type="checkbox" className="w-4 h-4" />
              <span>베이즈 정리의 구성 요소</span>
            </label>
          </div>
          <div className="mt-6">
            <button 
              onClick={onComplete}
              className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              챕터 완료하기
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}