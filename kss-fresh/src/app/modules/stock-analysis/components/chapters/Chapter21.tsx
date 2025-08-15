'use client';

import { useState } from 'react';

export default function Chapter21() {
  const [goals, setGoals] = useState({
    period: '5',
    targetReturn: '10',
    riskTolerance: 'moderate'
  });

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">나만의 투자 계획 수립</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          성공적인 투자의 첫걸음은 명확한 계획 수립입니다. 
          자신의 상황과 목표에 맞는 투자 계획을 세워보고, 꾸준히 실천하는 방법을 배워봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 투자 목표 설정하기</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">SMART 목표 설정법</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">S</span>
                <h4 className="font-semibold">Specific (구체적)</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 pl-11">
                ❌ "돈을 많이 벌고 싶다"<br/>
                ✅ "은퇴자금 3억원을 모으고 싶다"
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">M</span>
                <h4 className="font-semibold">Measurable (측정가능)</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 pl-11">
                ❌ "수익률을 높이겠다"<br/>
                ✅ "연평균 10% 수익률을 달성하겠다"
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">A</span>
                <h4 className="font-semibold">Achievable (달성가능)</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 pl-11">
                ❌ "1년에 100% 수익"<br/>
                ✅ "연 10-15% 수익률 목표"
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">R</span>
                <h4 className="font-semibold">Relevant (관련성)</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 pl-11">
                ❌ "남들처럼 투자하기"<br/>
                ✅ "내 은퇴 계획에 맞는 장기투자"
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">T</span>
                <h4 className="font-semibold">Time-bound (기한설정)</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 pl-11">
                ❌ "언젠가는 부자가 되겠다"<br/>
                ✅ "20년 후 은퇴자금 3억원 달성"
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 자금 계획 세우기</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">투자 가능 자금 계산</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span>월 수입</span>
                <span className="font-mono">+ 400만원</span>
              </div>
              <div className="flex justify-between items-center">
                <span>고정 지출</span>
                <span className="font-mono text-red-600">- 250만원</span>
              </div>
              <div className="flex justify-between items-center">
                <span>비상금 적립</span>
                <span className="font-mono text-red-600">- 50만원</span>
              </div>
              <div className="border-t pt-2 mt-2">
                <div className="flex justify-between items-center font-semibold">
                  <span>투자 가능액</span>
                  <span className="font-mono text-green-600">= 100만원</span>
                </div>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
              💡 비상금은 생활비 3-6개월분을 별도 보관
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">투자금 배분 전략</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded">
                <span>적립식 투자</span>
                <span className="font-semibold">60%</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded">
                <span>거치식 투자</span>
                <span className="font-semibold">30%</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded">
                <span>기회 포착용</span>
                <span className="font-semibold">10%</span>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
              💡 적립식은 매월 정액, 거치식은 목돈 투자
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 투자 성향 진단</h2>
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">나의 투자 성향은?</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">🛡️ 안정형 (Conservative)</h4>
              <ul className="text-sm space-y-1">
                <li>• 원금 보전이 최우선</li>
                <li>• 목표 수익률: 연 3-5%</li>
                <li>• 추천: 우량 배당주, 채권형 펀드</li>
                <li>• 주식 비중: 20-30%</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">⚖️ 중립형 (Moderate)</h4>
              <ul className="text-sm space-y-1">
                <li>• 적절한 위험과 수익 추구</li>
                <li>• 목표 수익률: 연 6-10%</li>
                <li>• 추천: 대형주 + 중형주 조합</li>
                <li>• 주식 비중: 50-60%</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">🚀 공격형 (Aggressive)</h4>
              <ul className="text-sm space-y-1">
                <li>• 높은 수익률 추구</li>
                <li>• 목표 수익률: 연 15% 이상</li>
                <li>• 추천: 성장주, 테마주</li>
                <li>• 주식 비중: 80% 이상</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📝 투자 계획서 작성</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">나의 투자 계획서 템플릿</h3>
          <div className="bg-white dark:bg-gray-700 rounded-lg p-6 space-y-4">
            <div>
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">1. 투자 목표</h4>
              <div className="pl-4 space-y-1 text-sm">
                <p>• 단기 (1년): 투자 원금 1,200만원 달성</p>
                <p>• 중기 (5년): 투자 자산 1억원 달성</p>
                <p>• 장기 (20년): 은퇴자금 5억원 달성</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">2. 투자 전략</h4>
              <div className="pl-4 space-y-1 text-sm">
                <p>• 매월 100만원 적립식 투자</p>
                <p>• 분산투자: 국내주식 60%, 해외주식 30%, 현금 10%</p>
                <p>• 리밸런싱: 분기별 1회</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">3. 리스크 관리</h4>
              <div className="pl-4 space-y-1 text-sm">
                <p>• 개별 종목 최대 투자 한도: 전체의 15%</p>
                <p>• 손절 기준: -10% 도달 시</p>
                <p>• 비상금: 생활비 6개월분 별도 보유</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">4. 학습 계획</h4>
              <div className="pl-4 space-y-1 text-sm">
                <p>• 매주 재무제표 1개 기업 분석</p>
                <p>• 월 1회 투자 서적 읽기</p>
                <p>• 분기별 포트폴리오 성과 분석</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 투자 원칙 정하기</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">
            흔들리지 않는 나만의 투자 원칙
          </h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1️⃣</span>
              <div>
                <p className="font-semibold">이해하지 못하는 것에는 투자하지 않는다</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  워런 버핏의 제1원칙. 모르는 산업, 이해 안 되는 기업은 피한다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">2️⃣</span>
              <div>
                <p className="font-semibold">감정을 배제하고 계획대로 실행한다</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  공포와 탐욕을 이기는 것이 성공 투자의 핵심이다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">3️⃣</span>
              <div>
                <p className="font-semibold">장기 투자를 기본으로 한다</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  단기 시세차익보다 기업의 성장과 함께하는 투자를 추구한다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">4️⃣</span>
              <div>
                <p className="font-semibold">꾸준히 공부하고 기록한다</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시장은 계속 변한다. 학습과 기록으로 투자 실력을 키운다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎓 Foundation 과정을 마치며</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">축하합니다! 이제 당신은:</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                차트를 읽고 기술적 분석을 할 수 있습니다
              </p>
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                재무제표를 보고 기업을 평가할 수 있습니다
              </p>
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                산업 분석과 기업 비교가 가능합니다
              </p>
            </div>
            <div className="space-y-2">
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                포트폴리오를 구성하고 관리할 수 있습니다
              </p>
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                리스크를 관리하며 투자할 수 있습니다
              </p>
              <p className="flex items-center gap-2">
                <span className="text-green-500">✅</span>
                나만의 투자 계획과 원칙이 있습니다
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg">
            <p className="text-center text-lg font-semibold mb-2">
              🎯 다음 단계는 Advanced Program입니다!
            </p>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400">
              퀀트 투자, 파생상품, 글로벌 투자 등 더 깊이 있는 내용을 학습하게 됩니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}