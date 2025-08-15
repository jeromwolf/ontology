'use client';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-blue-800 dark:text-blue-200 mb-4">
          그래프 데이터 모델링
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            효과적인 그래프 모델링은 성능과 유지보수성의 핵심입니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">모델링 원칙</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">노드로 표현</h4>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• 독립적으로 존재하는 엔티티</li>
              <li>• 여러 속성을 가진 객체</li>
              <li>• 다른 엔티티와 관계를 맺는 대상</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">관계로 표현</h4>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• 노드 간의 상호작용</li>
              <li>• 동작이나 이벤트</li>
              <li>• 시간적 연결</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">실전 모델링 예제</h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h4 className="font-semibold mb-3">소셜 네트워크 모델</h4>
          <pre className="bg-white dark:bg-gray-900 p-4 rounded text-sm overflow-x-auto">
            <code>{`// 사용자와 게시물
(:User {id, name, email})-[:POSTED]->(:Post {id, content, timestamp})
(:User)-[:FOLLOWS]->(:User)
(:User)-[:LIKES]->(:Post)
(:Post)-[:TAGGED]->(:Hashtag {name})

// 친구 추천 쿼리
MATCH (user:User {name: 'Alice'})-[:FOLLOWS]->(friend)-[:FOLLOWS]->(suggestion)
WHERE NOT (user)-[:FOLLOWS]->(suggestion)
AND user <> suggestion
RETURN suggestion, COUNT(*) as mutualFriends
ORDER BY mutualFriends DESC`}</code>
          </pre>
        </div>
      </section>
    </div>
  )
}