'use client'

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">퀀트 투자의 진화</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          퀀트 투자는 수학적, 통계적 모델을 사용하여 투자 결정을 내리는 방법론입니다.
          최근 AI와 머신러닝 기술의 발전으로 더욱 정교한 전략이 가능해졌습니다.
        </p>
        
        <div className="grid gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">1세대: 통계적 모델</h3>
            <p className="text-gray-700 dark:text-gray-300">
              회귀분석, 팩터 모델 등 전통적 통계 기법을 활용한 체계적 투자
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">2세대: 머신러닝</h3>
            <p className="text-gray-700 dark:text-gray-300">
              랜덤포레스트, SVM, XGBoost 등을 활용한 비선형 패턴 발굴
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">3세대: 딥러닝 & AI</h3>
            <p className="text-gray-700 dark:text-gray-300">
              CNN, LSTM, Transformer 등을 활용한 복잡한 패턴 인식과 예측
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">머신러닝을 이용한 주가 예측</h2>
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">🧠 LSTM (Long Short-Term Memory)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              시계열 데이터의 장기 의존성을 학습할 수 있는 순환 신경망의 한 종류
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">장점</h4>
                <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 장기간의 패턴 학습 가능</li>
                  <li>• 시계열 데이터에 특화</li>
                  <li>• 기울기 소실 문제 해결</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">활용 사례</h4>
                <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 주가 방향성 예측</li>
                  <li>• 변동성 예측</li>
                  <li>• 거래량 패턴 분석</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">🔄 Transformer for Finance</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              어텐션 메커니즘을 활용하여 다양한 시점의 정보를 종합적으로 분석
            </p>
            <div className="space-y-2 text-sm">
              <div><strong>Time Series Transformer:</strong> 과거 가격 패턴의 어텐션 가중치 학습</div>
              <div><strong>Multi-Modal Transformer:</strong> 가격, 뉴스, 거래량 등 다중 정보 통합</div>
              <div><strong>Cross-Asset Attention:</strong> 다른 자산 간의 상관관계 학습</div>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">📊 앙상블 모델링</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              여러 모델의 예측을 결합하여 더 안정적이고 정확한 예측 달성
            </p>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <strong>Voting:</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">다수결 원리</span>
              </div>
              <div>
                <strong>Weighted Average:</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">성과 기반 가중평균</span>
              </div>
              <div>
                <strong>Stacking:</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">메타모델 학습</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">뉴스와 소셜미디어 감정 분석</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6 mb-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">NLP 기반 감정 분석</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            뉴스 기사, 소셜미디어, 애널리스트 리포트 등 텍스트 데이터에서 시장 감정을 추출하여 
            투자 신호로 활용하는 기법입니다.
          </p>
          
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded text-sm font-medium">
                Step 1
              </span>
              <div>
                <strong>데이터 수집</strong>: 뉴스 API, 트위터 API, Reddit 등에서 실시간 텍스트 수집
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded text-sm font-medium">
                Step 2
              </span>
              <div>
                <strong>전처리</strong>: 불용어 제거, 정규화, 토큰화 등 텍스트 정제
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded text-sm font-medium">
                Step 3
              </span>
              <div>
                <strong>감정 분석</strong>: BERT, FinBERT 등을 활용한 감정 점수 계산
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded text-sm font-medium">
                Step 4
              </span>
              <div>
                <strong>투자 신호</strong>: 감정 점수를 기술적/기본적 지표와 결합하여 매매 신호 생성
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">📰 뉴스 감정 분석</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 실적 발표, 공시 내용 분석</li>
              <li>• 애널리스트 리포트 감정 추출</li>
              <li>• 경제 뉴스의 시장 영향도 측정</li>
              <li>• CEO 발언, 컨퍼런스콜 분석</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">📱 소셜미디어 분석</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 트위터, 레딧의 종목 관련 언급</li>
              <li>• 밈주식 현상 조기 감지</li>
              <li>• 인플루언서 의견의 영향력 측정</li>
              <li>• 소매투자자 심리 파악</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">대체 데이터 활용</h2>
        <div className="space-y-4">
          <p className="text-gray-700 dark:text-gray-300">
            전통적인 재무 데이터 외에 다양한 대체 데이터를 활용하여 
            경쟁 우위를 확보하는 것이 현대 퀀트 투자의 핵심입니다.
          </p>
          
          <div className="grid gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">🛰️ 위성 이미지 데이터</h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-2">
                위성에서 촬영한 이미지를 분석하여 경제 활동 지표를 추출
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 쇼핑몰, 공장 주차장의 차량 수 변화</li>
                <li>• 유전, 항구의 활동량 모니터링</li>
                <li>• 농작물 수확량 예측</li>
                <li>• 도시 개발, 건설 현황 파악</li>
              </ul>
            </div>
            
            <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">💳 신용카드 데이터</h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-2">
                익명화된 소비 패턴 데이터로 기업 실적을 선행 예측
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 소매업체별 매출 추이 추적</li>
                <li>• 업종별 소비 트렌드 파악</li>
                <li>• 지역별 경제 활동 측정</li>
                <li>• 계절성, 이벤트 영향 분석</li>
              </ul>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">📊 웹 스크래핑 데이터</h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-2">
                온라인에서 수집 가능한 모든 데이터를 투자 신호로 변환
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 구인구직 사이트의 채용 공고 수</li>
                <li>• 부동산 사이트의 매물 정보</li>
                <li>• 앱스토어 다운로드 순위</li>
                <li>• 검색 키워드 트렌드</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">백테스팅과 성과 평가</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">견고한 백테스팅 프레임워크</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            AI 투자 전략의 실효성을 검증하기 위해서는 과거 데이터를 활용한 
            체계적이고 엄격한 백테스팅 과정이 필수입니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3">⚠️ 백테스팅 함정들</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>Look-ahead Bias:</strong> 미래 정보를 과거에 사용</li>
                <li><strong>Survivorship Bias:</strong> 상장폐지된 종목 제외</li>
                <li><strong>Data Snooping:</strong> 과도한 최적화로 인한 과적합</li>
                <li><strong>Transaction Cost:</strong> 수수료, 슬리피지 미반영</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3">✅ 올바른 백테스팅</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>Walk-Forward Analysis:</strong> 점진적 학습과 검증</li>
                <li><strong>Out-of-Sample Test:</strong> 별도 검증 데이터셋 활용</li>
                <li><strong>Cross-Validation:</strong> 시계열 교차검증</li>
                <li><strong>Monte Carlo:</strong> 다양한 시나리오 시뮬레이션</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">성과 지표</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">공식</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">해석</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">Sharpe Ratio</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">(수익률 - 무위험수익률) / 변동성</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">위험 대비 수익률. 1 이상이면 양호</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">Maximum Drawdown</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">최고점 대비 최대 하락률</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">심리적 견딜 수 있는 손실 수준</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">Calmar Ratio</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">연간 수익률 / MDD</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">하락 위험 대비 수익률</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">Information Ratio</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">초과수익률 / 추적오차</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-xs">벤치마크 대비 일관된 초과 성과</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}