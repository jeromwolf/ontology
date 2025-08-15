'use client';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">현대 포트폴리오 이론 (MPT)</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          해리 마코위츠가 개발한 MPT는 동일한 위험 수준에서 최대 수익을, 
          또는 동일한 수익 수준에서 최소 위험을 추구하는 최적 포트폴리오 구성 이론입니다.
        </p>
        
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6 mb-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">MPT의 핵심 개념</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400 font-bold">📊</span>
              <div>
                <strong>분산투자</strong>: 서로 다른 자산에 투자하여 위험 분산
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400 font-bold">📈</span>
              <div>
                <strong>효율적 프론티어</strong>: 각 위험 수준에서 최대 수익을 제공하는 포트폴리오들의 집합
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400 font-bold">⚖️</span>
              <div>
                <strong>위험-수익 트레이드오프</strong>: 높은 수익을 위해서는 높은 위험을 감수해야 함
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">포트폴리오 수익률</h3>
            <div className="text-lg font-mono font-bold mb-2 text-center">
              R(p) = Σ w(i) × R(i)
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              각 자산의 가중평균으로 계산. w(i)는 자산 i의 비중, R(i)는 자산 i의 수익률
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">포트폴리오 위험</h3>
            <div className="text-lg font-mono font-bold mb-2 text-center">
              σ(p) = √(Σ w(i)² × σ(i)² + 2Σ w(i)w(j)σ(ij))
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              개별 자산의 위험과 상관관계를 고려한 포트폴리오 전체의 위험
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">상관계수와 분산효과</h2>
        <div className="space-y-4">
          <p className="text-gray-700 dark:text-gray-300">
            상관계수는 두 자산의 가격 움직임이 얼마나 유사한지를 나타내는 지표입니다.
            -1과 +1 사이의 값을 가지며, 분산투자 효과는 상관계수가 낮을수록 커집니다.
          </p>
          
          <div className="grid gap-4">
            <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">
                완전 음의 상관관계 (ρ = -1)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                한 자산이 오를 때 다른 자산은 정확히 반대로 움직임. 
                이론적으로 위험을 완전히 제거할 수 있으나 현실에서는 거의 존재하지 않음.
              </p>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
                무상관 (ρ = 0)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                두 자산의 움직임이 독립적. 
                분산투자 효과가 가장 명확하게 나타나는 이상적인 경우.
              </p>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">
                완전 양의 상관관계 (ρ = +1)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                두 자산이 완전히 동일하게 움직임. 
                분산투자 효과가 전혀 없어 위험 감소 불가능.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">자산 배분 전략</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">전략적 자산 배분 (SAA)</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              장기적 관점에서 투자 목표와 위험 성향에 따라 자산군별 비중을 결정
            </p>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 투자자의 나이, 투자기간 고려</li>
              <li>• 주식 : 채권 = (100-나이) : 나이</li>
              <li>• 정기적 리밸런싱으로 비중 유지</li>
              <li>• 장기적 안정성 추구</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">전술적 자산 배분 (TAA)</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              시장 상황과 경제 전망에 따라 단기적으로 자산 비중을 조정
            </p>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 시장 사이클과 밸류에이션 고려</li>
              <li>• 경기 국면별 자산 비중 조정</li>
              <li>• 능동적 관리로 초과 수익 추구</li>
              <li>• 더 높은 거래 비용과 위험</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">리밸런싱 전략</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            시간이 지나면서 자산별 성과 차이로 인해 목표 비중에서 벗어나게 됩니다.
            정기적인 리밸런싱을 통해 목표 비중을 유지하는 것이 중요합니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">📅 시간 기준</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                분기별, 반기별, 연 1회 등 정해진 주기마다 리밸런싱
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">📊 비중 기준</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                목표 비중에서 ±5% 이상 벗어날 때 리밸런싱
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🔄 혼합 기준</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                시간과 비중 기준을 함께 고려하는 방식
              </p>
            </div>
          </div>
          
          <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">💡 리밸런싱의 효과</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 고평가된 자산은 매도, 저평가된 자산은 매수 (Buy Low, Sell High)</li>
              <li>• 장기적으로 변동성 감소와 수익률 향상 효과</li>
              <li>• 감정적 판단을 배제한 기계적 거래</li>
              <li>• 거래 비용과 세금 비용 고려 필요</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}