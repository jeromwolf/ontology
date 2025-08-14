'use client'

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">Cypher 고급 기능 ⚡</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          APOC, 동적 쿼리, 성능 최적화로 Cypher의 진정한 힘을 발휘해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔧 APOC: Awesome Procedures on Cypher</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            APOC는 Neo4j의 가장 강력한 확장 라이브러리입니다. 500개 이상의 프로시저와 함수를 제공하여
            복잡한 그래프 작업을 간단하게 해결할 수 있습니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-blue-600 dark:text-blue-400 mb-3">데이터 변환과 가공</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// JSON 파싱과 변환</div>
                <div>WITH apoc.convert.fromJsonMap('{"name":"John","age":30}') AS data</div>
                <div>CREATE (p:Person {`{data}`})</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 날짜와 시간 처리</div>
                <div>RETURN apoc.date.format(timestamp(), 'yyyy-MM-dd')</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 문자열 처리</div>
                <div>RETURN apoc.text.camelCase('hello_world') // helloWorld</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">외부 데이터 연동</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// HTTP API 호출</div>
                <div>CALL apoc.load.json('https://api.example.com/data')</div>
                <div>YIELD value</div>
                <div>CREATE (n:Node {`{value}`})</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// CSV 파일 로드</div>
                <div>CALL apoc.load.csv('file:///data.csv')</div>
                <div>YIELD map AS row</div>
                <div>CREATE (p:Person {`{row}`})</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">주요 APOC 카테고리</h3>
            <div className="grid md:grid-cols-4 gap-4 text-sm">
              <div className="text-center p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
                <div className="font-semibold">데이터 변환</div>
                <div className="text-xs mt-1">convert.*, text.*</div>
              </div>
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="font-semibold">외부 연동</div>
                <div className="text-xs mt-1">load.*, export.*</div>
              </div>
              <div className="text-center p-3 bg-purple-100 dark:bg-purple-900/30 rounded">
                <div className="font-semibold">그래프 분석</div>
                <div className="text-xs mt-1">path.*, algo.*</div>
              </div>
              <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded">
                <div className="font-semibold">유틸리티</div>
                <div className="text-xs mt-1">util.*, help.*</div>
              </div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">💡 APOC 실무 팁</h4>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              APOC는 Neo4j Enterprise에서 일부 제한이 있을 수 있으므로, 
              프로덕션 환경에서는 사용 가능한 프로시저를 미리 확인하세요.
              `CALL apoc.help('keyword')`로 관련 함수를 찾을 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 UNWIND와 리스트 처리</h2>
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">리스트를 행으로 변환하기</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">기본 UNWIND 사용법</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 단순 리스트 언와인드</div>
              <div>UNWIND [1,2,3,4,5] AS number</div>
              <div>RETURN number</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 복잡한 객체 리스트 처리</div>
              <div>UNWIND [</div>
              <div>  {`{name: 'Alice', age: 30}`},</div>
              <div>  {`{name: 'Bob', age: 25}`}</div>
              <div>] AS person</div>
              <div>CREATE (p:Person {`{person}`})</div>
              <div>RETURN p</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">배치 처리</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// 큰 데이터를 배치로 처리</div>
                <div>CALL apoc.periodic.iterate(</div>
                <div>  "MATCH (n:Person) RETURN n",</div>
                <div>  "SET n.updated = timestamp()",</div>
                <div>  {`{batchSize: 1000}`}</div>
                <div>)</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">조건부 UNWIND</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// NULL 값 필터링</div>
                <div>WITH [1, null, 3, null, 5] AS numbers</div>
                <div>UNWIND numbers AS num</div>
                <div>WHERE num IS NOT NULL</div>
                <div>RETURN num</div>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">🎯 UNWIND 활용 시나리오</h4>
            <ul className="text-sm text-green-800 dark:text-green-300 space-y-1">
              <li>• 대량 데이터 import 시 배치 처리</li>
              <li>• JSON 배열을 개별 노드로 변환</li>
              <li>• 동적으로 생성된 리스트 처리</li>
              <li>• 복잡한 중첩 데이터 구조 평면화</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔗 CALL과 서브쿼리</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">복잡한 로직을 모듈화하기</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">서브쿼리 기본</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>MATCH (p:Person)</div>
                <div>CALL {`{`}</div>
                <div className="ml-2">WITH p</div>
                <div className="ml-2">MATCH (p)-[:FRIEND]-({'>'})f)</div>
                <div className="ml-2">RETURN count(f) AS friends</div>
                <div>{`}`}</div>
                <div>RETURN p.name, friends</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">프로시저 호출</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// PageRank 알고리즘 실행</div>
                <div>CALL gds.pageRank.stream('myGraph')</div>
                <div>YIELD nodeId, score</div>
                <div>RETURN gds.util.asNode(nodeId).name,</div>
                <div>       score</div>
                <div>ORDER BY score DESC</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">고급 서브쿼리 패턴</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 조건부 로직 분리</div>
              <div>MATCH (user:User)</div>
              <div>CALL {`{`}</div>
              <div className="ml-2">WITH user</div>
              <div className="ml-2">WHERE user.premium = true</div>
              <div className="ml-2">MATCH (user)-[:PURCHASED]-({'>'})p)</div>
              <div className="ml-2">RETURN collect(p) AS premium_products</div>
              <div className="ml-2">UNION</div>
              <div className="ml-2">WITH user</div>
              <div className="ml-2">WHERE user.premium = false</div>
              <div className="ml-2">MATCH (user)-[:VIEWED]-({'>'})p)</div>
              <div className="ml-2">RETURN collect(p) AS viewed_products</div>
              <div>{`}`}</div>
              <div>RETURN user.name, premium_products, viewed_products</div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🚀 서브쿼리의 장점</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300 space-y-1">
              <div>• 복잡한 로직을 단계별로 분리</div>
              <div>• 코드 가독성과 유지보수성 향상</div>
              <div>• 조건부 실행으로 성능 최적화</div>
              <div>• 재사용 가능한 로직 블록 생성</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 쿼리 성능 분석과 최적화</h2>
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">PROFILE과 EXPLAIN으로 쿼리 튜닝하기</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">EXPLAIN</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>용도:</strong> 실행 계획 확인</div>
                <div>• <strong>특징:</strong> 실제 실행하지 않음</div>
                <div>• <strong>비용:</strong> 무료, 빠름</div>
                <div>• <strong>정보:</strong> 예상 비용과 카디널리티</div>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono mt-3">
                <div>EXPLAIN</div>
                <div>MATCH (p:Person)-[:FRIEND]-(f)</div>
                <div>WHERE p.age {'>'} 25</div>
                <div>RETURN p, f</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">PROFILE</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>용도:</strong> 실제 성능 측정</div>
                <div>• <strong>특징:</strong> 쿼리를 실제 실행</div>
                <div>• <strong>비용:</strong> 실제 리소스 사용</div>
                <div>• <strong>정보:</strong> 실제 시간과 레코드 수</div>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono mt-3">
                <div>PROFILE</div>
                <div>MATCH (p:Person)-[:FRIEND]-(f)</div>
                <div>WHERE p.age {'>'} 25</div>
                <div>RETURN p, f</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">성능 최적화 체크리스트</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <h5 className="font-semibold text-green-600 mb-2">✅ 좋은 패턴</h5>
                <ul className="space-y-1">
                  <li>• 레이블과 인덱스 활용</li>
                  <li>• 필터링을 먼저 적용</li>
                  <li>• 필요한 속성만 RETURN</li>
                  <li>• LIMIT 사용으로 결과 제한</li>
                  <li>• WITH로 중간 결과 정리</li>
                </ul>
              </div>
              <div>
                <h5 className="font-semibold text-red-600 mb-2">❌ 피해야 할 패턴</h5>
                <ul className="space-y-1">
                  <li>• 전체 그래프 스캔</li>
                  <li>• 불필요한 OPTIONAL MATCH</li>
                  <li>• 복잡한 중첩 패턴</li>
                  <li>• 카티션 곱 (Cartesian Product)</li>
                  <li>• WHERE 없는 관계 패턴</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-orange-100 dark:bg-orange-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">🎯 성능 최적화 전략</h4>
            <p className="text-sm text-orange-800 dark:text-orange-300">
              1. 인덱스 생성 → 2. 쿼리 리팩토링 → 3. 데이터 모델 개선 → 4. 하드웨어 확장
              순서로 접근하여 비용 대비 효과를 극대화하세요.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔄 트랜잭션 제어</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">안전한 데이터 처리를 위한 트랜잭션</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">배치 처리 패턴</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 안전한 대량 데이터 처리</div>
              <div>CALL apoc.periodic.commit(</div>
              <div>  "MATCH (n:OldLabel)</div>
              <div>   WHERE NOT n:Processed</div>
              <div>   WITH n LIMIT $limit</div>
              <div>   SET n:NewLabel, n:Processed</div>
              <div>   RETURN count(*)",</div>
              <div>  {`{limit: 1000}`}</div>
              <div>)</div>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-2">⚠️ 트랜잭션 주의사항</h4>
            <div className="text-sm text-indigo-800 dark:text-indigo-300 space-y-1">
              <div>• 너무 큰 트랜잭션은 메모리 부족 야기</div>
              <div>• 긴 트랜잭션은 락 경합 증가</div>
              <div>• 실패 시 전체 롤백 고려</div>
              <div>• 배치 크기를 적절히 조정</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 오늘 배운 것 정리</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>APOC:</strong> 500+ 프로시저로 Cypher 기능 확장</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>UNWIND:</strong> 리스트를 행으로 변환하여 배치 처리</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>서브쿼리:</strong> 복잡한 로직을 모듈화</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>성능 분석:</strong> PROFILE/EXPLAIN으로 쿼리 최적화</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>트랜잭션:</strong> 안전한 대량 데이터 처리</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}