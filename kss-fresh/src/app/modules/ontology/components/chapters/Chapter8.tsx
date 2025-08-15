'use client';

import React from 'react';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 8: Protégé 마스터하기</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            Protégé는 스탠포드 대학에서 개발한 오픈소스 온톨로지 편집기로, 
            온톨로지 개발의 사실상 표준 도구입니다. 이번 챕터에서는 Protégé의 
            핵심 기능과 실전 활용법을 마스터합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Protégé 소개</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">주요 특징</h3>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• OWL 2.0 완벽 지원</li>
              <li>• 직관적인 GUI 인터페이스</li>
              <li>• 다양한 Reasoner 통합</li>
              <li>• 플러그인 아키텍처</li>
              <li>• 협업 기능 지원</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">지원 형식</h3>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• RDF/XML</li>
              <li>• Turtle</li>
              <li>• OWL/XML</li>
              <li>• Manchester Syntax</li>
              <li>• JSON-LD</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Protégé 인터페이스 구성</h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8">
          <h3 className="font-semibold mb-4">주요 탭과 기능</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">1. Active Ontology 탭</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                온톨로지의 메타데이터 (IRI, 버전, 주석 등) 관리
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">2. Classes 탭</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                클래스 계층구조 생성 및 편집, 논리적 표현식 정의
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">3. Object Properties 탭</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                객체 속성 정의, 도메인/레인지 설정, 특성 부여
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">4. Data Properties 탭</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                데이터 속성 정의, 데이터 타입 지정
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">5. Individuals 탭</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                인스턴스 생성 및 속성값 할당
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">클래스 정의하기</h2>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Manchester Syntax 사용법</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <span className="text-indigo-600"># 단순 클래스 정의</span><br/>
              Class: Person<br/>
              <br/>
              <span className="text-indigo-600"># 서브클래스 관계</span><br/>
              Class: Student<br/>
              SubClassOf: Person<br/>
              <br/>
              <span className="text-indigo-600"># 복잡한 클래스 표현</span><br/>
              Class: Parent<br/>
              EquivalentTo: Person and hasChild some Person<br/>
              <br/>
              <span className="text-indigo-600"># 제약사항 추가</span><br/>
              Class: Bachelor<br/>
              SubClassOf: Person and (not married)<br/>
              DisjointWith: MarriedPerson
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">속성 정의하기</h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Object Properties</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              ObjectProperty: hasParent<br/>
              Domain: Person<br/>
              Range: Person<br/>
              Characteristics: Functional<br/>
              <br/>
              ObjectProperty: hasAncestor<br/>
              Characteristics: Transitive<br/>
              <br/>
              ObjectProperty: marriedTo<br/>
              Characteristics: Symmetric
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Data Properties</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              DataProperty: hasAge<br/>
              Domain: Person<br/>
              Range: xsd:integer<br/>
              <br/>
              DataProperty: hasName<br/>
              Domain: Person<br/>
              Range: xsd:string<br/>
              Characteristics: Functional<br/>
              <br/>
              DataProperty: hasEmail<br/>
              Range: xsd:string[pattern ".+@.+"]
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Reasoner 활용</h2>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">Reasoner 시작하기</h3>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li>Reasoner 메뉴에서 원하는 reasoner 선택 (HermiT, Pellet 등)</li>
              <li>"Start reasoner" 클릭</li>
              <li>추론된 계층구조 확인 (노란색 하이라이트)</li>
              <li>일관성 검사 결과 확인</li>
            </ol>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">추론 결과 해석</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <span className="text-yellow-600">노란색 표시</span>: 추론된 관계</li>
              <li>• <span className="text-red-600">빨간색 표시</span>: 비일관성 검출</li>
              <li>• <span className="text-blue-600">파란색 표시</span>: 동치 클래스</li>
              <li>• <span className="text-gray-600">회색 표시</span>: 만족 불가능한 클래스</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">유용한 플러그인</h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">OntoGraf</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              온톨로지를 그래프 형태로 시각화. 클래스와 속성 관계를 
              직관적으로 파악할 수 있습니다.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">SPARQL Query</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Protégé 내에서 직접 SPARQL 쿼리를 실행하고 
              결과를 확인할 수 있습니다.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Ontology Debugger</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              온톨로지의 논리적 오류를 찾아내고 수정 방법을 
              제안해줍니다.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">WebProtégé</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              웹 브라우저에서 온톨로지를 편집하고 팀원들과 
              실시간 협업이 가능합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실전 팁</h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">효율적인 작업을 위한 단축키</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">탐색</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• <kbd>Ctrl/Cmd + F</kbd>: 엔티티 검색</li>
                <li>• <kbd>Alt + ↑/↓</kbd>: 계층구조 탐색</li>
                <li>• <kbd>Ctrl/Cmd + G</kbd>: 사용처 찾기</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">편집</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• <kbd>Ctrl/Cmd + N</kbd>: 새 엔티티</li>
                <li>• <kbd>Ctrl/Cmd + D</kbd>: 복제</li>
                <li>• <kbd>Delete</kbd>: 삭제</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🛠️</span>
          실습 과제
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Protégé를 사용하여 다음 작업을 수행해보세요:
        </p>
        <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
          <li>Pizza 온톨로지 튜토리얼 완성하기</li>
          <li>자신만의 도메인 온톨로지 생성 (최소 10개 클래스)</li>
          <li>HermiT reasoner로 일관성 검사 수행</li>
          <li>OntoGraf로 시각화하여 구조 검토</li>
        </ol>
      </section>
    </div>
  )
}