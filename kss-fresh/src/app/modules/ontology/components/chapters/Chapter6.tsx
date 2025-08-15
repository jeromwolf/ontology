'use client';

import React from 'react';
import { InferenceEngine } from '@/components/rdf-editor/components/InferenceEngine';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 6: OWL - 표현력 있는 온톨로지</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            OWL(Web Ontology Language)은 RDFS보다 훨씬 풍부한 표현력을 제공하는 온톨로지 언어입니다.
            복잡한 개념과 관계를 정확하게 모델링할 수 있습니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">OWL의 세 가지 하위 언어</h2>
        
        <div className="space-y-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-2">OWL Lite</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              가장 단순한 형태로, 분류 계층과 간단한 제약사항만 표현
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • 카디널리티: 0 또는 1만 가능<br/>
              • 계산 복잡도: 낮음
            </p>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">OWL DL (Description Logic)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              완전성과 결정가능성을 보장하면서 최대한의 표현력 제공
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • 모든 추론이 유한 시간 내 완료<br/>
              • 실무에서 가장 많이 사용
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">OWL Full</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              RDF와 완전히 호환되며 최대의 표현력 제공
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • 결정 불가능한 경우 존재<br/>
              • 메타 클래스 허용
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">OWL의 주요 구성 요소</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">클래스 표현</h3>
            <ul className="space-y-2 text-sm">
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:Class</code> - 클래스 정의</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:equivalentClass</code> - 동등 클래스</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:disjointWith</code> - 배타적 클래스</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:complementOf</code> - 여집합</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">속성 표현</h3>
            <ul className="space-y-2 text-sm">
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:ObjectProperty</code> - 객체 속성</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:DatatypeProperty</code> - 데이터 속성</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:inverseOf</code> - 역관계</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:TransitiveProperty</code> - 이행성</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">제약사항</h3>
            <ul className="space-y-2 text-sm">
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:allValuesFrom</code> - 전칭 제약</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:someValuesFrom</code> - 존재 제약</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:hasValue</code> - 값 제약</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:cardinality</code> - 개수 제약</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">논리 연산</h3>
            <ul className="space-y-2 text-sm">
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:intersectionOf</code> - 교집합</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:unionOf</code> - 합집합</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:oneOf</code> - 열거형</li>
              <li><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">owl:complementOf</code> - 여집합</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">OWL 예제: 가족 온톨로지</h2>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 font-mono text-sm overflow-x-auto">
          <div className="space-y-4">
            <div>
              <span className="text-gray-500"># 클래스 정의</span><br/>
              <span className="text-blue-600">:Person</span> <span className="text-green-600">rdf:type</span> <span className="text-purple-600">owl:Class</span> .<br/>
              <br/>
              <span className="text-blue-600">:Male</span> <span className="text-green-600">rdf:type</span> <span className="text-purple-600">owl:Class</span> ;<br/>
              <span className="ml-8 text-green-600">rdfs:subClassOf</span> <span className="text-purple-600">:Person</span> ;<br/>
              <span className="ml-8 text-green-600">owl:disjointWith</span> <span className="text-purple-600">:Female</span> .<br/>
              <br/>
              <span className="text-blue-600">:Parent</span> <span className="text-green-600">owl:equivalentClass</span> [<br/>
              <span className="ml-4 text-green-600">rdf:type</span> <span className="text-purple-600">owl:Restriction</span> ;<br/>
              <span className="ml-4 text-green-600">owl:onProperty</span> <span className="text-purple-600">:hasChild</span> ;<br/>
              <span className="ml-4 text-green-600">owl:minCardinality</span> <span className="text-purple-600">1</span><br/>
              ] .
            </div>
            
            <div>
              <span className="text-gray-500"># 속성 정의</span><br/>
              <span className="text-blue-600">:hasParent</span> <span className="text-green-600">rdf:type</span> <span className="text-purple-600">owl:ObjectProperty</span> ;<br/>
              <span className="ml-12 text-green-600">owl:inverseOf</span> <span className="text-purple-600">:hasChild</span> ;<br/>
              <span className="ml-12 text-green-600">rdfs:domain</span> <span className="text-purple-600">:Person</span> ;<br/>
              <span className="ml-12 text-green-600">rdfs:range</span> <span className="text-purple-600">:Person</span> .<br/>
              <br/>
              <span className="text-blue-600">:hasAncestor</span> <span className="text-green-600">rdf:type</span> <span className="text-purple-600">owl:TransitiveProperty</span> .
            </div>
            
            <div>
              <span className="text-gray-500"># 복잡한 클래스 정의</span><br/>
              <span className="text-blue-600">:Father</span> <span className="text-green-600">owl:equivalentClass</span> [<br/>
              <span className="ml-4 text-green-600">owl:intersectionOf</span> (<br/>
              <span className="ml-8 text-purple-600">:Male</span><br/>
              <span className="ml-8 text-purple-600">:Parent</span><br/>
              <span className="ml-4">)</span><br/>
              ] .
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">추론 엔진 시뮬레이터</h2>
        <p className="mb-4">
          OWL 온톨로지의 추론 과정을 시각화해보세요!
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <InferenceEngine triples={[
            { subject: ':John', predicate: ':hasParent', object: ':Mary' },
            { subject: ':Mary', predicate: ':hasParent', object: ':Susan' },
            { subject: ':John', predicate: ':marriedTo', object: ':Jane' },
            { subject: ':Dog', predicate: 'rdfs:subClassOf', object: ':Animal' },
            { subject: ':Buddy', predicate: 'rdf:type', object: ':Dog' }
          ]} />
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">OWL 제약사항 예제</h2>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. 개수 제약 (Cardinality)</h3>
            <div className="font-mono text-sm mb-3">
              <span className="text-gray-500"># 모든 사람은 정확히 2명의 생물학적 부모를 가진다</span><br/>
              :Person rdfs:subClassOf [<br/>
              <span className="ml-4">owl:onProperty :hasBiologicalParent ;</span><br/>
              <span className="ml-4">owl:cardinality 2</span><br/>
              ] .
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. 값 제약 (Value Restriction)</h3>
            <div className="font-mono text-sm mb-3">
              <span className="text-gray-500"># 채식주의자는 육류를 먹지 않는다</span><br/>
              :Vegetarian rdfs:subClassOf [<br/>
              <span className="ml-4">owl:onProperty :eats ;</span><br/>
              <span className="ml-4">owl:allValuesFrom :NonMeatFood</span><br/>
              ] .
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">3. 존재 제약 (Existential)</h3>
            <div className="font-mono text-sm mb-3">
              <span className="text-gray-500"># 부모는 적어도 한 명의 자녀가 있다</span><br/>
              :Parent owl:equivalentClass [<br/>
              <span className="ml-4">owl:onProperty :hasChild ;</span><br/>
              <span className="ml-4">owl:someValuesFrom :Person</span><br/>
              ] .
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">추론 엔진 시뮬레이터</h2>
        <p className="mb-4">
          OWL의 추론 능력을 직접 체험해보세요! 간단한 트리플을 입력하면 자동으로 새로운 사실들이 추론됩니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <InferenceEngine 
            triples={[
              { subject: ':김철수', predicate: ':hasParent', object: ':김부모' },
              { subject: ':김부모', predicate: ':hasParent', object: ':김조부모' },
              { subject: ':이영희', predicate: ':marriedTo', object: ':김철수' },
              { subject: ':김철수', predicate: ':teaches', object: ':컴퓨터과학' }
            ]}
          />
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💡</span>
          OWL 사용 시 주의사항
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• 표현력과 추론 복잡도는 트레이드오프 관계</li>
          <li>• OWL DL을 사용하면 대부분의 요구사항 충족 가능</li>
          <li>• 복잡한 공리는 추론 성능에 큰 영향</li>
          <li>• 온톨로지 설계 시 목적에 맞는 적절한 수준 선택 중요</li>
        </ul>
      </section>
    </div>
  )
}