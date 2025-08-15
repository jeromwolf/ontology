'use client';

import { Code } from 'lucide-react';
import dynamic from 'next/dynamic';

const CypherPlayground = dynamic(() => import('../CypherPlayground'), {
  ssr: false,
  loading: () => <div className="animate-pulse bg-gray-200 h-64 rounded-lg"></div>
})

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-blue-800 dark:text-blue-200 mb-4 flex items-center gap-2">
          <Code className="w-6 h-6" />
          Cypher 쿼리 언어
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            Cypher는 Neo4j의 선언적 그래프 쿼리 언어입니다. SQL과 유사하지만 그래프 패턴을 직관적으로 표현합니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">기본 패턴</h3>
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">노드 패턴</h4>
            <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 노드 표현
()                          // 익명 노드
(n)                         // 변수명 n
(:Person)                   // Person 레이블
(p:Person)                  // 변수 p, Person 레이블
(p:Person {name: 'Alice'})  // 속성 포함`}</code>
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">관계 패턴</h4>
            <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 관계 표현
-->                         // 방향성 관계
-[]-                       // 양방향 관계
-[:KNOWS]->                // KNOWS 타입
-[r:KNOWS]->               // 변수 r
-[:KNOWS {since: 2020}]->  // 속성 포함`}</code>
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">패턴 조합</h4>
            <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 전체 패턴
(alice:Person {name: 'Alice'})-[:KNOWS]->(bob:Person {name: 'Bob'})

// 체인 패턴
(alice)-[:KNOWS]->(bob)-[:WORKS_AT]->(company:Company)`}</code>
            </pre>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">CRUD 작업</h3>
        <div className="space-y-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">CREATE - 생성</h4>
            <pre className="bg-white dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 노드 생성
CREATE (p:Person {name: 'Alice', age: 30})

// 관계 생성
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS {since: 2020}]->(b)

// 전체 패턴 생성
CREATE (alice:Person {name: 'Alice'})-[:KNOWS]->(bob:Person {name: 'Bob'})`}</code>
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">MATCH - 조회</h4>
            <pre className="bg-white dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 모든 Person 노드
MATCH (p:Person)
RETURN p

// 특정 조건
MATCH (p:Person {name: 'Alice'})
RETURN p.name, p.age

// 관계 패턴
MATCH (p1:Person)-[:KNOWS]->(p2:Person)
RETURN p1.name, p2.name`}</code>
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">SET - 수정</h4>
            <pre className="bg-white dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 속성 수정
MATCH (p:Person {name: 'Alice'})
SET p.age = 31

// 여러 속성
MATCH (p:Person {name: 'Alice'})
SET p.age = 31, p.city = 'Seoul'

// 레이블 추가
MATCH (p:Person {name: 'Alice'})
SET p:Developer`}</code>
            </pre>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">DELETE - 삭제</h4>
            <pre className="bg-white dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
              <code>{`// 노드 삭제
MATCH (p:Person {name: 'Alice'})
DELETE p

// 관계 삭제
MATCH (a:Person)-[r:KNOWS]->(b:Person)
DELETE r

// 노드와 관계 모두 삭제
MATCH (p:Person {name: 'Alice'})
DETACH DELETE p`}</code>
            </pre>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Cypher 실습</h3>
        <CypherPlayground />
      </section>
    </div>
  )
}