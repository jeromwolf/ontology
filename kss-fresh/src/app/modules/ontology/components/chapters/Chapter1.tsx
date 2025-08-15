'use client';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 1: 온톨로지란 무엇인가?</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg italic">
            "존재하는 것들에 대한 체계적인 설명" - 온톨로지의 가장 간단한 정의
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지의 기원</h2>
        <p className="mb-4">
          온톨로지(Ontology)라는 용어는 그리스어 'ontos'(존재)와 'logos'(학문)에서 유래했습니다.
          원래는 <strong>철학</strong>의 한 분야로, "존재란 무엇인가?"라는 근본적인 질문을 다루었습니다.
        </p>
        
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-3">철학에서 컴퓨터 과학으로</h3>
          <p className="text-gray-700 dark:text-gray-300">
            1990년대부터 컴퓨터 과학자들은 이 개념을 차용하여, 
            <strong>특정 도메인의 지식을 형식적으로 표현하는 방법</strong>으로 발전시켰습니다.
            이제 온톨로지는 AI, 시맨틱 웹, 지식 관리의 핵심 기술이 되었습니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">컴퓨터 과학에서의 온톨로지</h2>
        <p className="mb-6">
          컴퓨터 과학에서 온톨로지는 다음과 같이 정의됩니다:
        </p>
        
        <div className="bg-indigo-100 dark:bg-indigo-900/30 rounded-xl p-6 mb-6 border-l-4 border-indigo-600">
          <p className="font-medium text-lg">
            "특정 도메인 내의 개념들과 그들 간의 관계를 명시적이고 형식적으로 정의한 명세(specification)"
          </p>
          <p className="text-sm mt-2 text-gray-600 dark:text-gray-400">
            - Tom Gruber, 1993
          </p>
        </div>

        <h3 className="text-xl font-semibold mb-3">온톨로지의 구성 요소</h3>
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">개념(Concepts)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              도메인의 기본 요소들<br/>
              예: 사람, 회사, 제품
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">관계(Relations)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              개념들 간의 연결<br/>
              예: 근무한다, 생산한다
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">공리(Axioms)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              도메인의 규칙과 제약<br/>
              예: 모든 사람은 하나의 생일을 가진다
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">왜 온톨로지가 필요한가?</h2>
        
        <div className="space-y-6">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-green-600 dark:text-green-400 font-bold">1</span>
            </div>
            <div>
              <h4 className="font-semibold mb-2">지식의 공유와 재사용</h4>
              <p className="text-gray-600 dark:text-gray-400">
                동일한 도메인에서 작업하는 사람들이 공통된 이해를 가질 수 있습니다.
                한 번 만든 온톨로지는 여러 시스템에서 재사용할 수 있습니다.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-green-600 dark:text-green-400 font-bold">2</span>
            </div>
            <div>
              <h4 className="font-semibold mb-2">시스템 간 상호운용성</h4>
              <p className="text-gray-600 dark:text-gray-400">
                서로 다른 시스템이 동일한 온톨로지를 사용하면 데이터를 쉽게 교환할 수 있습니다.
                의미적 충돌 없이 정보를 통합할 수 있습니다.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-green-600 dark:text-green-400 font-bold">3</span>
            </div>
            <div>
              <h4 className="font-semibold mb-2">추론과 지식 발견</h4>
              <p className="text-gray-600 dark:text-gray-400">
                명시적으로 저장되지 않은 지식도 논리적 추론을 통해 도출할 수 있습니다.
                숨겨진 관계나 패턴을 발견할 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 예시: 가족 온톨로지</h2>
        <p className="mb-4">
          간단한 가족 관계 온톨로지를 통해 개념을 이해해봅시다:
        </p>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 font-mono text-sm">
          <div className="mb-4">
            <span className="text-indigo-600 dark:text-indigo-400">// 클래스 정의</span><br/>
            <span className="text-green-600 dark:text-green-400">Class:</span> Person<br/>
            <span className="text-green-600 dark:text-green-400">Class:</span> Male <span className="text-gray-500">subClassOf</span> Person<br/>
            <span className="text-green-600 dark:text-green-400">Class:</span> Female <span className="text-gray-500">subClassOf</span> Person<br/>
          </div>
          
          <div className="mb-4">
            <span className="text-indigo-600 dark:text-indigo-400">// 속성 정의</span><br/>
            <span className="text-blue-600 dark:text-blue-400">Property:</span> hasParent <span className="text-gray-500">(domain: Person, range: Person)</span><br/>
            <span className="text-blue-600 dark:text-blue-400">Property:</span> hasChild <span className="text-gray-500">(inverse of hasParent)</span><br/>
            <span className="text-blue-600 dark:text-blue-400">Property:</span> hasSibling <span className="text-gray-500">(symmetric)</span><br/>
          </div>
          
          <div>
            <span className="text-indigo-600 dark:text-indigo-400">// 추론 규칙</span><br/>
            <span className="text-purple-600 dark:text-purple-400">Rule:</span> If X hasParent Y and Z hasParent Y, then X hasSibling Z
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💡</span>
          핵심 포인트
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• 온톨로지는 지식을 컴퓨터가 이해할 수 있는 형태로 표현합니다</li>
          <li>• 개념, 관계, 규칙의 세 가지 요소로 구성됩니다</li>
          <li>• 지식 공유, 상호운용성, 추론 능력을 제공합니다</li>
          <li>• 실제 세계의 복잡한 관계를 모델링할 수 있습니다</li>
        </ul>
      </section>
    </div>
  )
}