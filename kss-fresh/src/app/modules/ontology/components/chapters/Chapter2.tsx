'use client'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 2: 온톨로지의 핵심 개념</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            온톨로지를 구성하는 핵심 요소들을 깊이 있게 살펴보고,
            이들이 어떻게 조합되어 지식을 표현하는지 알아봅시다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">1. 클래스(Classes)</h2>
        <p className="mb-4">
          클래스는 온톨로지의 가장 기본적인 구성 요소로, <strong>동일한 특성을 가진 개체들의 집합</strong>을 나타냅니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">클래스의 특징</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 계층 구조를 형성 (상위/하위 클래스)</li>
              <li>• 다중 상속 가능</li>
              <li>• 개체들의 템플릿 역할</li>
              <li>• 속성과 제약사항 정의</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">예시: 동물 분류</h3>
            <div className="font-mono text-sm">
              <div className="ml-0">Animal</div>
              <div className="ml-4">├─ Mammal</div>
              <div className="ml-8">│  ├─ Dog</div>
              <div className="ml-8">│  └─ Cat</div>
              <div className="ml-4">└─ Bird</div>
              <div className="ml-8">   ├─ Eagle</div>
              <div className="ml-8">   └─ Sparrow</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">2. 속성(Properties)</h2>
        <p className="mb-4">
          속성은 클래스나 개체들 간의 <strong>관계를 표현</strong>하거나 <strong>특성을 나타냅니다</strong>.
        </p>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">객체 속성 (Object Properties)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              개체와 개체 사이의 관계를 나타냅니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <div>• Person <span className="text-blue-600">hasChild</span> Person</div>
              <div>• Student <span className="text-blue-600">enrolledIn</span> Course</div>
              <div>• Company <span className="text-blue-600">employs</span> Employee</div>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">데이터 속성 (Data Properties)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              개체의 리터럴 값(문자열, 숫자 등)을 나타냅니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <div>• Person <span className="text-green-600">hasAge</span> Integer</div>
              <div>• Product <span className="text-green-600">hasPrice</span> Decimal</div>
              <div>• Book <span className="text-green-600">hasTitle</span> String</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">3. 개체(Individuals)</h2>
        <p className="mb-4">
          개체는 클래스의 <strong>실제 인스턴스</strong>입니다. 실제 세계의 구체적인 사물이나 개념을 나타냅니다.
        </p>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
          <h3 className="font-semibold mb-3">예시: 회사 온톨로지</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2 text-indigo-600 dark:text-indigo-400">클래스 정의</h4>
              <div className="font-mono text-sm space-y-1">
                <div>Class: Company</div>
                <div>Class: Employee</div>
                <div>Class: Product</div>
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-2 text-indigo-600 dark:text-indigo-400">개체 예시</h4>
              <div className="font-mono text-sm space-y-1">
                <div>Apple : Company</div>
                <div>TimCook : Employee</div>
                <div>iPhone15 : Product</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">4. 공리(Axioms)</h2>
        <p className="mb-4">
          공리는 온톨로지 내에서 <strong>항상 참이라고 가정되는 명제</strong>입니다. 
          도메인의 규칙과 제약사항을 정의합니다.
        </p>
        
        <div className="space-y-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">주요 공리 유형</h3>
            <ul className="space-y-3">
              <li>
                <strong>클래스 공리</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  예: "모든 학생은 사람이다" (Student subClassOf Person)
                </p>
              </li>
              <li>
                <strong>속성 공리</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  예: "hasParent는 hasChild의 역관계다" (hasParent inverseOf hasChild)
                </p>
              </li>
              <li>
                <strong>개체 공리</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  예: "John과 Jane는 서로 다른 개체다" (John differentFrom Jane)
                </p>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">속성의 특성(Property Characteristics)</h2>
        <p className="mb-4">
          속성은 다양한 특성을 가질 수 있으며, 이는 추론에 중요한 역할을 합니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">대칭성 (Symmetric)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              A가 B와 관계가 있으면, B도 A와 같은 관계
            </p>
            <p className="text-sm font-mono mt-2">예: hasFriend, isMarriedTo</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">이행성 (Transitive)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              A→B이고 B→C이면, A→C
            </p>
            <p className="text-sm font-mono mt-2">예: hasAncestor, locatedIn</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">함수성 (Functional)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              각 주체는 최대 하나의 값만 가질 수 있음
            </p>
            <p className="text-sm font-mono mt-2">예: hasBirthDate, hasSSN</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">역함수성 (InverseFunctional)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              각 값은 최대 하나의 주체에만 연결됨
            </p>
            <p className="text-sm font-mono mt-2">예: hasEmail, hasPassport</p>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          실습 예고
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          다음 챕터에서는 이러한 개념들이 시맨틱 웹에서 어떻게 활용되는지 알아보고,
          Chapter 4에서는 RDF Triple Editor를 사용하여 직접 온톨로지를 만들어볼 예정입니다.
        </p>
      </section>
    </div>
  )
}