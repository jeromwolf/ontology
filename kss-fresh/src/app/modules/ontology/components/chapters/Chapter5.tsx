'use client';

import React from 'react';
import { RDFTripleEditor } from '@/components/rdf-editor/RDFTripleEditor';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 5: RDFS - 스키마와 계층구조</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            RDF Schema (RDFS)는 RDF 데이터를 위한 어휘를 정의하는 언어입니다. 
            클래스와 속성의 계층구조를 표현하고, 도메인과 레인지를 명시하여 
            더 풍부한 의미를 표현할 수 있습니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDFS의 필요성</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">RDF의 한계</h3>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• 클래스 개념이 없음</li>
              <li>• 속성의 의미 정의 불가</li>
              <li>• 계층구조 표현 불가</li>
              <li>• 타입 제약 명시 불가</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">RDFS의 해결책</h3>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• 클래스와 서브클래스 정의</li>
              <li>• 속성의 도메인과 레인지</li>
              <li>• 계층적 분류 체계</li>
              <li>• 의미적 제약 표현</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 RDFS 어휘</h2>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. 클래스 관련</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <span className="text-indigo-600"># 클래스 정의</span><br/>
              :Person rdf:type rdfs:Class .<br/>
              :Student rdf:type rdfs:Class .<br/>
              <br/>
              <span className="text-indigo-600"># 서브클래스 관계</span><br/>
              :Student rdfs:subClassOf :Person .<br/>
              :GraduateStudent rdfs:subClassOf :Student .
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. 속성 관련</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <span className="text-green-600"># 속성 정의</span><br/>
              :hasAge rdf:type rdf:Property .<br/>
              :enrolledIn rdf:type rdf:Property .<br/>
              <br/>
              <span className="text-green-600"># 도메인과 레인지</span><br/>
              :hasAge rdfs:domain :Person .<br/>
              :hasAge rdfs:range xsd:integer .<br/>
              <br/>
              :enrolledIn rdfs:domain :Student .<br/>
              :enrolledIn rdfs:range :Course .
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">3. 메타데이터</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <span className="text-purple-600"># 라벨과 설명</span><br/>
              :Person rdfs:label "사람"@ko .<br/>
              :Person rdfs:label "Person"@en .<br/>
              :Person rdfs:comment "인간을 나타내는 클래스"@ko .<br/>
              <br/>
              <span className="text-purple-600"># 참조 정보</span><br/>
              :Person rdfs:seeAlso &lt;http://dbpedia.org/resource/Person&gt; .<br/>
              :Person rdfs:isDefinedBy &lt;http://example.org/ontology#&gt; .
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">계층구조 설계</h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8">
          <h3 className="font-semibold mb-4">대학 온톨로지 예제</h3>
          <div className="bg-white dark:bg-gray-800 rounded p-6 font-mono text-sm">
            <span className="text-blue-600"># 클래스 계층구조</span><br/>
            :Person rdf:type rdfs:Class .<br/>
            :Student rdfs:subClassOf :Person .<br/>
            :Professor rdfs:subClassOf :Person .<br/>
            :UndergraduateStudent rdfs:subClassOf :Student .<br/>
            :GraduateStudent rdfs:subClassOf :Student .<br/>
            <br/>
            <span className="text-green-600"># 속성 계층구조</span><br/>
            :teaches rdf:type rdf:Property .<br/>
            :lecturesIn rdfs:subPropertyOf :teaches .<br/>
            :supervisesIn rdfs:subPropertyOf :teaches .<br/>
            <br/>
            <span className="text-purple-600"># 도메인/레인지 제약</span><br/>
            :teaches rdfs:domain :Professor .<br/>
            :teaches rdfs:range :Course .<br/>
            :enrolledIn rdfs:domain :Student .<br/>
            :enrolledIn rdfs:range :Course .
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDFS 추론 규칙</h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">rdfs2: 도메인 추론</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              속성을 사용하면 주어가 도메인 클래스의 인스턴스
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
              :john :teaches :CS101 .<br/>
              :teaches rdfs:domain :Professor .<br/>
              <span className="text-green-600">⇒ :john rdf:type :Professor .</span>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">rdfs3: 레인지 추론</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              속성을 사용하면 목적어가 레인지 클래스의 인스턴스
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
              :mary :enrolledIn :CS101 .<br/>
              :enrolledIn rdfs:range :Course .<br/>
              <span className="text-green-600">⇒ :CS101 rdf:type :Course .</span>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">rdfs9: 서브클래스 추론</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              서브클래스의 인스턴스는 상위 클래스의 인스턴스
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
              :mary rdf:type :Student .<br/>
              :Student rdfs:subClassOf :Person .<br/>
              <span className="text-green-600">⇒ :mary rdf:type :Person .</span>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">rdfs11: 이행성</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              서브클래스 관계의 이행성
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
              :A rdfs:subClassOf :B .<br/>
              :B rdfs:subClassOf :C .<br/>
              <span className="text-green-600">⇒ :A rdfs:subClassOf :C .</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실습: RDFS 온톨로지 구축</h2>
        <p className="mb-4">
          RDF Triple Editor를 사용하여 RDFS 스키마를 정의하고 인스턴스를 생성해보세요.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <RDFTripleEditor />
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDFS 모범 사례</h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">설계 가이드라인</h3>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <div>
                <strong>명확한 클래스 계층구조</strong><br/>
                <span className="text-sm">is-a 관계를 기반으로 한 논리적 분류</span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <div>
                <strong>일관된 명명 규칙</strong><br/>
                <span className="text-sm">클래스는 대문자, 속성은 소문자로 시작</span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <div>
                <strong>도메인/레인지 명시</strong><br/>
                <span className="text-sm">모든 속성에 적절한 제약 정의</span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <div>
                <strong>다국어 라벨 제공</strong><br/>
                <span className="text-sm">rdfs:label을 활용한 국제화</span>
              </div>
            </li>
          </ul>
        </div>
      </section>

      <section className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          요약
        </h2>
        <p className="text-gray-700 dark:text-gray-300">
          RDFS는 RDF의 표현력을 확장하여 클래스, 속성, 계층구조를 정의할 수 있게 합니다.
          도메인과 레인지를 통해 의미적 제약을 표현하고, 추론 규칙을 통해 
          명시되지 않은 지식을 자동으로 도출할 수 있습니다.
          다음 챕터에서는 더 강력한 표현력을 제공하는 OWL을 학습합니다.
        </p>
      </section>
    </div>
  )
}