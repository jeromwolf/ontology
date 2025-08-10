'use client'

export default function Chapter10() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 10: 패턴과 모범 사례</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            검증된 온톨로지 디자인 패턴(ODPs)을 활용하면 고품질 온톨로지를 효율적으로 개발할 수 있습니다. 
            이번 챕터에서는 주요 패턴과 모범 사례, 그리고 피해야 할 안티패턴을 학습합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 디자인 패턴 (ODPs)이란?</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            소프트웨어 공학의 디자인 패턴처럼, ODPs는 반복적으로 나타나는 모델링 문제에 대한 
            검증된 해결책입니다. 이를 통해 일관성 있고 재사용 가능한 온톨로지를 구축할 수 있습니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4 mt-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400">재사용성</h4>
              <p className="text-sm mt-1">검증된 솔루션을 다양한 도메인에 적용</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400">일관성</h4>
              <p className="text-sm mt-1">표준화된 모델링 접근법</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-4">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">품질 향상</h4>
              <p className="text-sm mt-1">모범 사례 기반 설계</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Content ODPs (내용 패턴)</h2>
        
        <div className="space-y-6">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. Part-Whole Pattern (부분-전체 패턴)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              객체와 그 구성 요소 간의 관계를 모델링
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <div className="text-gray-500"># 자동차와 부품의 관계</div>
              :Car rdfs:subClassOf [<br/>
              <span className="ml-4">owl:onProperty :hasPart ;</span><br/>
              <span className="ml-4">owl:someValuesFrom :Engine</span><br/>
              ] .<br/>
              <br/>
              :hasPart rdf:type owl:TransitiveProperty .<br/>
              :partOf owl:inverseOf :hasPart .
            </div>
            <div className="mt-3 text-sm">
              <strong>사용 예:</strong> 제품 구조, 조직 계층, 지리적 포함 관계
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. Participation Pattern (참여 패턴)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              이벤트나 활동에 대한 참여를 모델링
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <div className="text-gray-500"># 회의 참석 모델링</div>
              :Meeting rdfs:subClassOf :Event .<br/>
              :Person rdfs:subClassOf :Agent .<br/>
              <br/>
              :participatesIn rdfs:domain :Agent ;<br/>
              <span className="ml-15">rdfs:range :Event .</span><br/>
              <br/>
              :hasParticipant owl:inverseOf :participatesIn .
            </div>
            <div className="mt-3 text-sm">
              <strong>사용 예:</strong> 프로젝트 참여, 이벤트 출석, 프로세스 관여
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">3. Time-indexed Situation Pattern (시간 색인 상황 패턴)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              시간에 따라 변하는 속성이나 관계를 모델링
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              <div className="text-gray-500"># 직책이 시간에 따라 변하는 경우</div>
              :EmploymentSituation rdfs:subClassOf :Situation ;<br/>
              <span className="ml-20">:hasEmployee :Person ;</span><br/>
              <span className="ml-20">:hasRole :JobRole ;</span><br/>
              <span className="ml-20">:hasTimeInterval :TimeInterval .</span><br/>
              <br/>
              :john :hasEmployment [<br/>
              <span className="ml-4">a :EmploymentSituation ;</span><br/>
              <span className="ml-4">:hasRole :Manager ;</span><br/>
              <span className="ml-4">:startDate "2020-01-01" ;</span><br/>
              <span className="ml-4">:endDate "2023-12-31"</span><br/>
              ] .
            </div>
            <div className="mt-3 text-sm">
              <strong>사용 예:</strong> 고용 이력, 가격 변동, 상태 변화
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Structural ODPs (구조 패턴)</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. Value Partition Pattern</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              값의 범위를 의미 있는 카테고리로 분할
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              :Size a owl:Class .<br/>
              :Small, :Medium, :Large rdfs:subClassOf :Size ;<br/>
              <span className="ml-24">owl:disjointWith each other .</span><br/>
              <br/>
              :hasSize rdfs:range :Size .
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. List Pattern</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              순서가 있는 항목들을 표현
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
              :RecipeStep rdfs:subClassOf :Step ;<br/>
              <span className="ml-11">:hasNext :RecipeStep ;</span><br/>
              <span className="ml-11">:hasStepNumber xsd:integer .</span><br/>
              <br/>
              :firstStep rdfs:subPropertyOf :hasStep .
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Correspondence ODPs (대응 패턴)</h2>
        
        <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-3">Alignment Pattern (정렬 패턴)</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            서로 다른 온톨로지 간의 개념 매핑
          </p>
          <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
            <div className="text-gray-500"># 두 온톨로지 간 개념 매핑</div>
            onto1:Product owl:equivalentClass onto2:Item .<br/>
            onto1:hasPrice owl:equivalentProperty onto2:cost .<br/>
            <br/>
            <div className="text-gray-500"># 부분적 매핑</div>
            onto1:Vehicle rdfs:subClassOf onto2:Transport .<br/>
            onto1:Car skos:narrowMatch onto2:Automobile .
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">안티패턴 (Anti-patterns)</h2>
        
        <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3 text-red-600 dark:text-red-400">피해야 할 일반적인 실수들</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4 border-l-4 border-red-500">
              <h4 className="font-semibold mb-2">1. 과도한 계층 구조</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                불필요하게 깊은 클래스 계층은 복잡성만 증가시킵니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                <span className="text-red-600"># 나쁜 예</span><br/>
                :Thing > :Object > :PhysicalObject > :LivingThing > :Animal > :Mammal > :Dog > :Beagle<br/>
                <br/>
                <span className="text-green-600"># 좋은 예</span><br/>
                :Animal > :Dog > :Beagle
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded p-4 border-l-4 border-red-500">
              <h4 className="font-semibold mb-2">2. 인스턴스를 클래스로 모델링</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                개별 객체를 클래스로 만드는 실수
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                <span className="text-red-600"># 나쁜 예</span><br/>
                :JohnSmith rdfs:subClassOf :Person .<br/>
                <br/>
                <span className="text-green-600"># 좋은 예</span><br/>
                :johnSmith rdf:type :Person .
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded p-4 border-l-4 border-red-500">
              <h4 className="font-semibold mb-2">3. 순환 정의</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                개념이 자기 자신을 참조하여 정의되는 경우
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                <span className="text-red-600"># 나쁜 예</span><br/>
                :Parent owl:equivalentClass [ owl:onProperty :hasChild; owl:someValuesFrom :Child ] .<br/>
                :Child owl:equivalentClass [ owl:onProperty :hasParent; owl:someValuesFrom :Parent ] .
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">모범 사례 체크리스트</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-green-600 dark:text-green-400">설계 원칙</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>명확하고 일관된 명명 규칙 사용</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>적절한 수준의 추상화 유지</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>모듈화와 재사용성 고려</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>문서화와 주석 충실히 작성</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-blue-600 dark:text-blue-400">구현 지침</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">✓</span>
                  <span>표준 온톨로지 재사용 우선</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">✓</span>
                  <span>추론 성능을 고려한 설계</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">✓</span>
                  <span>정기적인 일관성 검증</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">✓</span>
                  <span>버전 관리 체계 구축</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          실습: 패턴 적용하기
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          다음 시나리오에 적절한 ODP를 선택하고 적용해보세요:
        </p>
        <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
          <li>직원의 부서 소속이 시간에 따라 변경되는 경우</li>
          <li>제품과 그 구성 부품들의 관계</li>
          <li>프로젝트에 여러 팀원이 참여하는 상황</li>
          <li>의류 사이즈를 S, M, L로 분류</li>
        </ol>
      </section>
    </div>
  )
}