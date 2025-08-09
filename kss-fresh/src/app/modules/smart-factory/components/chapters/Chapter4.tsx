'use client'

import { 
  DollarSign
} from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="space-y-8">
      {/* TCO 계산 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">💰 TCO (Total Cost of Ownership) 계산</h3>
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border border-red-200 dark:border-red-800">
            <h4 className="font-semibold text-red-800 dark:text-red-300 mb-4">초기 투자비용</h4>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between border-b border-red-200 dark:border-red-700 pb-1">
                <span className="text-red-700 dark:text-red-400">하드웨어</span>
                <span className="font-semibold text-red-800 dark:text-red-300">40%</span>
              </div>
              <div className="flex justify-between border-b border-red-200 dark:border-red-700 pb-1">
                <span className="text-red-700 dark:text-red-400">소프트웨어</span>
                <span className="font-semibold text-red-800 dark:text-red-300">30%</span>
              </div>
              <div className="flex justify-between border-b border-red-200 dark:border-red-700 pb-1">
                <span className="text-red-700 dark:text-red-400">컨설팅/구축</span>
                <span className="font-semibold text-red-800 dark:text-red-300">20%</span>
              </div>
              <div className="flex justify-between border-b border-red-200 dark:border-red-700 pb-1">
                <span className="text-red-700 dark:text-red-400">교육/훈련</span>
                <span className="font-semibold text-red-800 dark:text-red-300">10%</span>
              </div>
            </div>
            <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/40 rounded">
              <h5 className="text-xs font-semibold text-red-800 dark:text-red-300">세부 비용 (100억 기준)</h5>
              <ul className="text-xs text-red-700 dark:text-red-400 mt-2 space-y-1">
                <li>• 센서/PLC: 15억</li>
                <li>• 네트워크 장비: 10억</li>
                <li>• 로봇/자동화: 15억</li>
                <li>• MES/ERP: 20억</li>
                <li>• AI 플랫폼: 10억</li>
                <li>• 구축 서비스: 20억</li>
                <li>• 교육/훈련: 10억</li>
              </ul>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-4">운영비용 (연간)</h4>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between border-b border-orange-200 dark:border-orange-700 pb-1">
                <span className="text-orange-700 dark:text-orange-400">소프트웨어 라이선스</span>
                <span className="font-semibold text-orange-800 dark:text-orange-300">12%</span>
              </div>
              <div className="flex justify-between border-b border-orange-200 dark:border-orange-700 pb-1">
                <span className="text-orange-700 dark:text-orange-400">클라우드 비용</span>
                <span className="font-semibold text-orange-800 dark:text-orange-300">8%</span>
              </div>
              <div className="flex justify-between border-b border-orange-200 dark:border-orange-700 pb-1">
                <span className="text-orange-700 dark:text-orange-400">IT 인력</span>
                <span className="font-semibold text-orange-800 dark:text-orange-300">60%</span>
              </div>
              <div className="flex justify-between border-b border-orange-200 dark:border-orange-700 pb-1">
                <span className="text-orange-700 dark:text-orange-400">전력/통신</span>
                <span className="font-semibold text-orange-800 dark:text-orange-300">20%</span>
              </div>
            </div>
            <div className="mt-4 p-3 bg-orange-100 dark:bg-orange-900/40 rounded">
              <h5 className="text-xs font-semibold text-orange-800 dark:text-orange-300">연간 운영비 (15억)</h5>
              <ul className="text-xs text-orange-700 dark:text-orange-400 mt-2 space-y-1">
                <li>• SW 라이선스: 1.8억</li>
                <li>• 클라우드: 1.2억</li>
                <li>• IT 엔지니어 5명: 9억</li>
                <li>• 전력/통신: 3억</li>
              </ul>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-4">유지보수비용</h4>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between border-b border-yellow-200 dark:border-yellow-700 pb-1">
                <span className="text-yellow-700 dark:text-yellow-400">하드웨어 보수</span>
                <span className="font-semibold text-yellow-800 dark:text-yellow-300">40%</span>
              </div>
              <div className="flex justify-between border-b border-yellow-200 dark:border-yellow-700 pb-1">
                <span className="text-yellow-700 dark:text-yellow-400">소프트웨어 업데이트</span>
                <span className="font-semibold text-yellow-800 dark:text-yellow-300">30%</span>
              </div>
              <div className="flex justify-between border-b border-yellow-200 dark:border-yellow-700 pb-1">
                <span className="text-yellow-700 dark:text-yellow-400">기술 지원</span>
                <span className="font-semibold text-yellow-800 dark:text-yellow-300">20%</span>
              </div>
              <div className="flex justify-between border-b border-yellow-200 dark:border-yellow-700 pb-1">
                <span className="text-yellow-700 dark:text-yellow-400">교육/재훈련</span>
                <span className="font-semibold text-yellow-800 dark:text-yellow-300">10%</span>
              </div>
            </div>
            <div className="mt-4 p-3 bg-yellow-100 dark:bg-yellow-900/40 rounded">
              <h5 className="text-xs font-semibold text-yellow-800 dark:text-yellow-300">연간 유지보수 (8억)</h5>
              <ul className="text-xs text-yellow-700 dark:text-yellow-400 mt-2 space-y-1">
                <li>• 장비 정기보수: 3.2억</li>
                <li>• SW 업그레이드: 2.4억</li>
                <li>• 기술 지원: 1.6억</li>
                <li>• 추가 교육: 0.8억</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* ROI 측정 지표 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">📈 ROI 측정 지표 15가지</h3>
        <div className="grid lg:grid-cols-3 gap-8">
          <div>
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-4">생산성 지표</h4>
            <div className="space-y-3">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-900 dark:text-blue-300">OEE 향상</h5>
                <p className="text-sm text-blue-700 dark:text-blue-400">75% → 95% (+20%p)</p>
                <div className="text-xs text-blue-600 dark:text-blue-500 mt-1">연간 효과: 25억 원</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-900 dark:text-blue-300">생산량 증대</h5>
                <p className="text-sm text-blue-700 dark:text-blue-400">시간당 생산량 +30%</p>
                <div className="text-xs text-blue-600 dark:text-blue-500 mt-1">연간 효과: 30억 원</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-900 dark:text-blue-300">인력 효율성</h5>
                <p className="text-sm text-blue-700 dark:text-blue-400">작업자 1인당 생산성 +25%</p>
                <div className="text-xs text-blue-600 dark:text-blue-500 mt-1">연간 효과: 15억 원</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-900 dark:text-blue-300">리드타임 단축</h5>
                <p className="text-sm text-blue-700 dark:text-blue-400">주문-출하 시간 -40%</p>
                <div className="text-xs text-blue-600 dark:text-blue-500 mt-1">연간 효과: 8억 원</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <h5 className="font-medium text-blue-900 dark:text-blue-300">셋업 시간 감소</h5>
                <p className="text-sm text-blue-700 dark:text-blue-400">라인 전환 시간 -60%</p>
                <div className="text-xs text-blue-600 dark:text-blue-500 mt-1">연간 효과: 5억 원</div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-4">품질 & 비용 지표</h4>
            <div className="space-y-3">
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                <h5 className="font-medium text-green-900 dark:text-green-300">불량률 감소</h5>
                <p className="text-sm text-green-700 dark:text-green-400">2% → 0.5% (-75%)</p>
                <div className="text-xs text-green-600 dark:text-green-500 mt-1">연간 효과: 12억 원</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                <h5 className="font-medium text-green-900 dark:text-green-300">재작업 비용</h5>
                <p className="text-sm text-green-700 dark:text-green-400">품질비용 -65%</p>
                <div className="text-xs text-green-600 dark:text-green-500 mt-1">연간 효과: 8억 원</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                <h5 className="font-medium text-green-900 dark:text-green-300">재고 비용</h5>
                <p className="text-sm text-green-700 dark:text-green-400">재고 수준 -30%</p>
                <div className="text-xs text-green-600 dark:text-green-500 mt-1">연간 효과: 10억 원</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                <h5 className="font-medium text-green-900 dark:text-green-300">에너지 효율</h5>
                <p className="text-sm text-green-700 dark:text-green-400">전력 사용량 -20%</p>
                <div className="text-xs text-green-600 dark:text-green-500 mt-1">연간 효과: 6억 원</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                <h5 className="font-medium text-green-900 dark:text-green-300">유지보수 비용</h5>
                <p className="text-sm text-green-700 dark:text-green-400">예방정비로 -40%</p>
                <div className="text-xs text-green-600 dark:text-green-500 mt-1">연간 효과: 4억 원</div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-4">운영 & 혁신 지표</h4>
            <div className="space-y-3">
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-900 dark:text-purple-300">다운타임 감소</h5>
                <p className="text-sm text-purple-700 dark:text-purple-400">계획외 정지 -70%</p>
                <div className="text-xs text-purple-600 dark:text-purple-500 mt-1">연간 효과: 20억 원</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-900 dark:text-purple-300">안전사고 감소</h5>
                <p className="text-sm text-purple-700 dark:text-purple-400">산업재해 -80%</p>
                <div className="text-xs text-purple-600 dark:text-purple-500 mt-1">연간 효과: 2억 원</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-900 dark:text-purple-300">고객 만족도</h5>
                <p className="text-sm text-purple-700 dark:text-purple-400">납기 준수율 +15%</p>
                <div className="text-xs text-purple-600 dark:text-purple-500 mt-1">연간 효과: 5억 원</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-900 dark:text-purple-300">신제품 출시</h5>
                <p className="text-sm text-purple-700 dark:text-purple-400">개발 기간 -35%</p>
                <div className="text-xs text-purple-600 dark:text-purple-500 mt-1">연간 효과: 3억 원</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
                <h5 className="font-medium text-purple-900 dark:text-purple-300">데이터 활용도</h5>
                <p className="text-sm text-purple-700 dark:text-purple-400">실시간 의사결정</p>
                <div className="text-xs text-purple-600 dark:text-purple-500 mt-1">연간 효과: 무형가치</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ROI 계산 예시 */}
      <div className="bg-slate-50 dark:bg-slate-800 p-8 border border-slate-200 dark:border-slate-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-6">🧮 실제 ROI 계산 예시</h3>
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
          <h4 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">중소제조업체 A사 (연매출 500억원)</h4>
          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <h5 className="font-semibold text-red-700 dark:text-red-400 mb-3">투자 비용 (5년간)</h5>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>초기 투자</span>
                  <span className="font-semibold">50억원</span>
                </div>
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>운영비 (5년)</span>
                  <span className="font-semibold">75억원</span>
                </div>
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>유지보수 (5년)</span>
                  <span className="font-semibold">40억원</span>
                </div>
                <div className="flex justify-between border-b-2 border-gray-400 pb-1 font-bold">
                  <span>총 투자</span>
                  <span className="text-red-600 dark:text-red-400">165억원</span>
                </div>
              </div>
            </div>
            <div>
              <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">절감 효과 (연간)</h5>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>생산성 향상</span>
                  <span className="font-semibold">25억원</span>
                </div>
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>품질비용 절감</span>
                  <span className="font-semibold">12억원</span>
                </div>
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>에너지/재고 절감</span>
                  <span className="font-semibold">8억원</span>
                </div>
                <div className="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-1">
                  <span>기타 효과</span>
                  <span className="font-semibold">5억원</span>
                </div>
                <div className="flex justify-between border-b-2 border-gray-400 pb-1 font-bold">
                  <span>연간 절감액</span>
                  <span className="text-blue-600 dark:text-blue-400">50억원</span>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-6 p-6 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <div className="grid md:grid-cols-4 gap-6 text-center">
              <div>
                <span className="text-sm text-green-700 dark:text-green-400">투자회수기간</span>
                <div className="text-2xl font-bold text-green-800 dark:text-green-300">3.3년</div>
                <div className="text-xs text-green-600 dark:text-green-500">165억 ÷ 50억</div>
              </div>
              <div>
                <span className="text-sm text-blue-700 dark:text-blue-400">5년 NPV</span>
                <div className="text-2xl font-bold text-blue-800 dark:text-blue-300">+85억원</div>
                <div className="text-xs text-blue-600 dark:text-blue-500">할인율 8% 적용</div>
              </div>
              <div>
                <span className="text-sm text-purple-700 dark:text-purple-400">IRR</span>
                <div className="text-2xl font-bold text-purple-800 dark:text-purple-300">22.1%</div>
                <div className="text-xs text-purple-600 dark:text-purple-500">내부수익률</div>
              </div>
              <div>
                <span className="text-sm text-orange-700 dark:text-orange-400">ROI (5년)</span>
                <div className="text-2xl font-bold text-orange-800 dark:text-orange-300">151%</div>
                <div className="text-xs text-orange-600 dark:text-orange-500">(250-165)÷165</div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h6 className="font-semibold text-gray-900 dark:text-white mb-2">💡 CFO 설득 포인트</h6>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 3.3년 회수기간으로 중기 전략에 적합</li>
                <li>• IRR 22.1%로 회사 투자기준(15%) 상회</li>
                <li>• 5년 누적 수익 85억원으로 신규 투자 여력 확보</li>
                <li>• 리스크 대비 높은 수익성 확보 (Sharpe Ratio 1.8)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 리스크 평가 */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 p-6 rounded-lg">
        <h3 className="text-xl font-semibold text-yellow-900 dark:text-yellow-300 mb-4">⚠️ 리스크 평가 및 대응 전략</h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">기술적 리스크</h4>
            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">레거시 시스템 통합 실패</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 30%, 영향: 중상</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 단계적 통합, API 우선 연동</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">데이터 품질 이슈</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 40%, 영향: 중</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 데이터 정제 우선, 품질 검증</div>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">조직적 리스크</h4>
            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">직원 저항 및 적응 실패</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 50%, 영향: 상</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 변화관리 프로그램, 인센티브</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">전문 인력 부족</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 60%, 영향: 중상</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 외부 파트너십, 교육 투자</div>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">시장 리스크</h4>
            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">시장 환경 변화</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 20%, 영향: 상</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 모듈화 설계, 유연성 확보</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                <div className="font-medium text-yellow-900 dark:text-yellow-300">경쟁사 기술 추격</div>
                <div className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">확률: 70%, 영향: 중</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">대응: 지속적 혁신, 선제적 투자</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}