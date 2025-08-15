'use client';

export default function Chapter14() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">추세의 기본 이해 📈📉</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식의 방향성을 파악하는 것이 투자 성공의 첫걸음입니다!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 상승 추세 (Bull Trend)</h2>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">특징</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 고점과 저점이 계속 높아짐</li>
            <li>• 이동평균선이 상향 정렬</li>
            <li>• 거래량이 상승 시 증가</li>
            <li>• 조정 후에도 다시 상승</li>
          </ul>
          <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded">
            <strong>투자 전략:</strong> 추세를 따라 매수, 조정 시 추가 매수 검토
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📉 하락 추세 (Bear Trend)</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">특징</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 고점과 저점이 계속 낮아짐</li>
            <li>• 이동평균선이 하향 정렬</li>
            <li>• 거래량이 하락 시 증가</li>
            <li>• 반등 후에도 다시 하락</li>
          </ul>
          <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 rounded">
            <strong>투자 전략:</strong> 매도 우선, 신규 매수는 신중하게
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">➡️ 횡보 추세 (Sideways)</h2>
        <div className="bg-gray-50 dark:bg-gray-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">특징</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 일정한 범위 내에서 등락 반복</li>
            <li>• 이동평균선이 수평에 가까움</li>
            <li>• 거래량이 평소보다 적음</li>
            <li>• 명확한 방향성 부재</li>
          </ul>
          <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-900/30 rounded">
            <strong>투자 전략:</strong> 박스권 매매 또는 추세 돌파 대기
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 추세 판단 체크리스트</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>최근 고점이 이전 고점보다 높은가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>최근 저점이 이전 저점보다 높은가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>주가가 주요 이동평균선 위에 있는가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>이동평균선들이 상향 정렬되어 있는가?</span>
            </label>
          </div>
          <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
            <strong>3개 이상 체크되면</strong> 상승 추세 가능성 높음
          </div>
        </div>
      </section>
    </div>
  )
}