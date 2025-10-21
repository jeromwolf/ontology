export default function Chapter4() {
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-2xl font-bold mb-4">챕터 개요</h2>
        <p className="text-gray-700 dark:text-gray-300">
          이 챕터의 내용이 곧 업데이트됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">학습 내용</h2>
        <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
          <li>핵심 개념 이해</li>
          <li>실습 예제</li>
          <li>응용 사례</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">요약</h2>
        <p className="text-gray-700 dark:text-gray-300">
          이 챕터에서 다루는 핵심 내용을 정리합니다.
        </p>
      </section>
    </div>
  )
}
