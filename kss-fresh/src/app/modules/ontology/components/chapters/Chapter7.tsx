'use client';

import React from 'react';
import { SparqlPlayground } from '@/components/sparql-playground/SparqlPlayground';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 7: SPARQL - 온톨로지 질의 언어</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            SPARQL(SPARQL Protocol and RDF Query Language)은 RDF 데이터를 조회하고 조작하는 
            표준 질의 언어입니다. SQL과 유사하지만 그래프 데이터에 특화되어 있습니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL 기본 구조</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold mb-3">기본 쿼리 형식</h3>
          <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
            <div className="text-purple-600">PREFIX</div>
            <div className="text-blue-600">SELECT</div> <span className="text-gray-600">?variable</span>
            <div className="text-green-600">WHERE</div> {`{`}
            <div className="ml-4">트리플 패턴들...</div>
            {`}`}
            <div className="text-orange-600">ORDER BY</div> <span className="text-gray-600">?variable</span>
            <div className="text-red-600">LIMIT</div> <span className="text-gray-600">10</span>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL 쿼리 타입</h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">SELECT</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              테이블 형식으로 결과 반환 (가장 많이 사용)
            </p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-2">CONSTRUCT</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              새로운 RDF 그래프 생성
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">ASK</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              패턴 존재 여부를 true/false로 반환
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">DESCRIBE</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              리소스에 대한 설명 반환
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL 예제</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h3 className="font-semibold mb-3">1. 기본 SELECT 쿼리</h3>
            <div className="font-mono text-sm">
              <div className="text-gray-500"># 모든 사람의 이름 조회</div>
              <div>PREFIX foaf: &lt;http://xmlns.com/foaf/0.1/&gt;</div>
              <div className="mt-2">SELECT ?person ?name</div>
              <div>WHERE {`{`}</div>
              <div className="ml-4">?person rdf:type foaf:Person .</div>
              <div className="ml-4">?person foaf:name ?name .</div>
              <div>{`}`}</div>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h3 className="font-semibold mb-3">2. FILTER 사용</h3>
            <div className="font-mono text-sm">
              <div className="text-gray-500"># 30세 이상인 사람 조회</div>
              <div>SELECT ?person ?name ?age</div>
              <div>WHERE {`{`}</div>
              <div className="ml-4">?person foaf:name ?name .</div>
              <div className="ml-4">?person foaf:age ?age .</div>
              <div className="ml-4">FILTER (?age &gt;= 30)</div>
              <div>{`}`}</div>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h3 className="font-semibold mb-3">3. OPTIONAL 패턴</h3>
            <div className="font-mono text-sm">
              <div className="text-gray-500"># 이메일은 있을 수도 없을 수도</div>
              <div>SELECT ?person ?name ?email</div>
              <div>WHERE {`{`}</div>
              <div className="ml-4">?person foaf:name ?name .</div>
              <div className="ml-4">OPTIONAL {`{`} ?person foaf:email ?email {`}`}</div>
              <div>{`}`}</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL Playground</h2>
        <p className="mb-4">
          실시간으로 SPARQL 쿼리를 작성하고 결과를 확인해보세요!
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <SparqlPlayground />
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">고급 SPARQL 기능</h2>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-2">집계 함수</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT 등
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-sm">
              SELECT ?author (COUNT(?book) AS ?bookCount)<br/>
              WHERE {`{`} ?book dc:creator ?author {`}`}<br/>
              GROUP BY ?author<br/>
              ORDER BY DESC(?bookCount)
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-2">프로퍼티 경로</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              복잡한 관계를 간단하게 표현
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-sm">
              # 친구의 친구<br/>
              ?person foaf:knows/foaf:knows ?friendOfFriend .<br/>
              <br/>
              # 1개 이상의 타입<br/>
              ?resource rdf:type+ ?class .
            </div>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💡</span>
          SPARQL 팁
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• 쿼리 최적화를 위해 구체적인 패턴을 먼저 작성</li>
          <li>• LIMIT로 개발 중 결과 수 제한</li>
          <li>• OPTIONAL은 성능에 영향을 줄 수 있으므로 신중히 사용</li>
          <li>• 네임스페이스를 PREFIX로 정의하여 가독성 향상</li>
        </ul>
      </section>
    </div>
  )
}