'use client';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 3: 시맨틱 웹과 온톨로지</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            "웹을 거대한 데이터베이스로" - 팀 버너스-리의 시맨틱 웹 비전
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">웹의 진화</h2>
        
        <div className="space-y-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="font-bold text-blue-600 dark:text-blue-400">1.0</span>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Web 1.0: 읽기 전용 웹</h3>
              <p className="text-gray-600 dark:text-gray-400">
                정적 HTML 페이지, 일방향 정보 전달, 하이퍼링크로 연결된 문서들
              </p>
            </div>
          </div>
          
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="font-bold text-green-600 dark:text-green-400">2.0</span>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Web 2.0: 참여와 공유의 웹</h3>
              <p className="text-gray-600 dark:text-gray-400">
                동적 콘텐츠, 소셜 미디어, 사용자 생성 콘텐츠, API와 매시업
              </p>
            </div>
          </div>
          
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="font-bold text-purple-600 dark:text-purple-400">3.0</span>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Web 3.0: 시맨틱 웹</h3>
              <p className="text-gray-600 dark:text-gray-400">
                기계가 이해하는 웹, 데이터의 의미와 관계 표현, 지능적인 정보 처리
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">시맨틱 웹의 핵심 개념</h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <h3 className="font-semibold mb-3">현재 웹의 한계</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">문제점</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• HTML은 표현에만 집중</li>
                <li>• 데이터의 의미 파악 불가</li>
                <li>• 자동화된 처리 어려움</li>
                <li>• 정보 통합의 한계</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">시맨틱 웹의 해결책</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 메타데이터 추가</li>
                <li>• 표준화된 의미 표현</li>
                <li>• 기계 가독성 확보</li>
                <li>• 자동 추론 가능</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">시맨틱 웹 기술 스택</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="space-y-2">
            <div className="bg-purple-100 dark:bg-purple-900 rounded p-3 text-center font-semibold">
              Trust / Proof / Crypto
            </div>
            <div className="bg-indigo-100 dark:bg-indigo-900 rounded p-3 text-center font-semibold">
              Logic / Rules (SWRL, RIF)
            </div>
            <div className="bg-blue-100 dark:bg-blue-900 rounded p-3 text-center font-semibold">
              Ontology (OWL)
            </div>
            <div className="bg-green-100 dark:bg-green-900 rounded p-3 text-center font-semibold">
              Schema (RDFS)
            </div>
            <div className="bg-yellow-100 dark:bg-yellow-900 rounded p-3 text-center font-semibold">
              Data Model (RDF)
            </div>
            <div className="bg-orange-100 dark:bg-orange-900 rounded p-3 text-center font-semibold">
              Syntax (XML, Turtle, JSON-LD)
            </div>
            <div className="bg-red-100 dark:bg-red-900 rounded p-3 text-center font-semibold">
              Identifiers (URI/IRI)
            </div>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-center font-semibold">
              Character Set (Unicode)
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">링크드 데이터 (Linked Data)</h2>
        <p className="mb-4">
          팀 버너스-리가 제안한 시맨틱 웹 구현을 위한 실천 방법론입니다.
        </p>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">링크드 데이터의 4가지 원칙</h3>
          <ol className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm font-bold">1</span>
              <div>
                <strong>URI 사용</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  모든 것에 고유한 URI를 부여하여 식별 가능하게 만든다
                </p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm font-bold">2</span>
              <div>
                <strong>HTTP URI 사용</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  웹 브라우저나 프로그램이 접근할 수 있도록 HTTP URI를 사용한다
                </p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm font-bold">3</span>
              <div>
                <strong>표준 형식 제공</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  URI에 접근했을 때 RDF, SPARQL 등 표준 형식으로 유용한 정보를 제공한다
                </p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <span className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm font-bold">4</span>
              <div>
                <strong>다른 URI와 연결</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  관련된 다른 데이터의 URI를 포함하여 웹을 탐색할 수 있게 한다
                </p>
              </div>
            </li>
          </ol>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">LOD Cloud (Linked Open Data)</h2>
        <p className="mb-4">
          전 세계적으로 공개된 링크드 데이터의 거대한 네트워크입니다.
        </p>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">DBpedia</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Wikipedia의 정보를 구조화한 지식베이스
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">Wikidata</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              협업으로 구축되는 자유 지식베이스
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Schema.org</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              웹 콘텐츠 마크업을 위한 공통 어휘
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지의 역할</h2>
        <p className="mb-4">
          시맨틱 웹에서 온톨로지는 <strong>공통 어휘와 의미 체계</strong>를 제공하는 핵심 역할을 합니다.
        </p>
        
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">온톨로지가 가능하게 하는 것들</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2 text-purple-600 dark:text-purple-400">데이터 통합</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                서로 다른 소스의 데이터를 의미적으로 연결하고 통합
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-2 text-purple-600 dark:text-purple-400">지능형 검색</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                단순 키워드가 아닌 의미 기반 검색 가능
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-2 text-purple-600 dark:text-purple-400">자동 추론</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                명시되지 않은 정보도 논리적으로 도출
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-2 text-purple-600 dark:text-purple-400">지식 공유</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                표준화된 형식으로 지식을 공유하고 재사용
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🚀</span>
          다음 단계
        </h2>
        <p className="text-gray-700 dark:text-gray-300">
          이제 시맨틱 웹의 개념을 이해했으니, 다음 챕터부터는 실제로 이를 구현하는 
          기술들(RDF, RDFS, OWL, SPARQL)을 하나씩 학습하고 실습해보겠습니다.
        </p>
      </section>
    </div>
  )
}