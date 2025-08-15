'use client';

import dynamic from 'next/dynamic';

// Lazy load RDF Triple Editor
const RDFTripleEditor = dynamic(() => 
  import('@/components/rdf-editor/RDFTripleEditor').then(mod => ({ default: mod.RDFTripleEditor })), 
  { 
    ssr: false,
    loading: () => <div className="h-96 flex items-center justify-center">RDF Editor 로딩 중...</div>
  }
)

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 4: RDF - 지식 표현의 기초</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            RDF(Resource Description Framework)는 시맨틱 웹의 기초가 되는 데이터 모델입니다.
            모든 지식을 <strong>주어-술어-목적어</strong>의 트리플로 표현합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDF 트리플의 구조</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-6">
          <div className="flex items-center justify-between gap-4">
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mb-2">
                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">S</span>
              </div>
              <h3 className="font-semibold">Subject</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">주어</p>
              <p className="text-xs mt-1">리소스 (URI)</p>
            </div>
            
            <div className="text-gray-400 text-2xl">→</div>
            
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center mb-2">
                <span className="text-2xl font-bold text-green-600 dark:text-green-400">P</span>
              </div>
              <h3 className="font-semibold">Predicate</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">술어</p>
              <p className="text-xs mt-1">속성 (URI)</p>
            </div>
            
            <div className="text-gray-400 text-2xl">→</div>
            
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center mb-2">
                <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">O</span>
              </div>
              <h3 className="font-semibold">Object</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">목적어</p>
              <p className="text-xs mt-1">리소스 또는 리터럴</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
          <h3 className="font-semibold mb-3">예시 트리플</h3>
          <div className="font-mono text-sm space-y-2">
            <div>
              <span className="text-blue-600">:TimBernersLee</span>
              <span className="text-green-600 mx-2">:invented</span>
              <span className="text-purple-600">:WorldWideWeb</span> .
            </div>
            <div>
              <span className="text-blue-600">:Seoul</span>
              <span className="text-green-600 mx-2">:isCapitalOf</span>
              <span className="text-purple-600">:SouthKorea</span> .
            </div>
            <div>
              <span className="text-blue-600">:Einstein</span>
              <span className="text-green-600 mx-2">:bornIn</span>
              <span className="text-purple-600">"1879"^^xsd:integer</span> .
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDF Triple Editor 실습</h2>
        <p className="mb-4">
          아래 에디터를 사용하여 직접 RDF 트리플을 만들어보세요!
        </p>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 mb-4">
          <h3 className="font-semibold mb-2">💡 목적어의 두 가지 유형</h3>
          <div className="space-y-2 text-sm">
            <div>
              <strong>리소스 (Resource)</strong>: URI로 식별되는 개체
              <div className="text-gray-600 dark:text-gray-400">예: :Seoul, :Korea, http://example.org/person/john</div>
            </div>
            <div>
              <strong>리터럴 (Literal)</strong>: 실제 데이터 값
              <div className="text-gray-600 dark:text-gray-400">예: "서울", "25"^^xsd:integer, "2024-01-01"^^xsd:date</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <RDFTripleEditor />
          
          <div className="mt-4 text-center">
            <a
              href="/rdf-editor"
              target="_blank"
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              전체 화면에서 RDF Editor 열기
            </a>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDF 직렬화 형식</h2>
        <p className="mb-4">
          RDF는 다양한 형식으로 표현될 수 있습니다. 각 형식은 같은 정보를 다른 방식으로 표현합니다.
        </p>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">1. Turtle (Terse RDF Triple Language)</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              가장 인간 친화적이고 간결한 형식
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <div>@prefix ex: &lt;http://example.org/&gt; .</div>
              <div>@prefix foaf: &lt;http://xmlns.com/foaf/0.1/&gt; .</div>
              <div className="mt-2">ex:john foaf:name "John Doe" ;</div>
              <div className="ml-8">foaf:age 30 ;</div>
              <div className="ml-8">foaf:knows ex:jane .</div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">2. RDF/XML</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              XML 기반의 표준 형식 (장황하지만 호환성 좋음)
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm overflow-x-auto">
              <div>&lt;rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"</div>
              <div className="ml-9">xmlns:foaf="http://xmlns.com/foaf/0.1/"&gt;</div>
              <div className="ml-2">&lt;rdf:Description rdf:about="http://example.org/john"&gt;</div>
              <div className="ml-4">&lt;foaf:name&gt;John Doe&lt;/foaf:name&gt;</div>
              <div className="ml-4">&lt;foaf:age rdf:datatype="http://www.w3.org/2001/XMLSchema#integer"&gt;30&lt;/foaf:age&gt;</div>
              <div className="ml-2">&lt;/rdf:Description&gt;</div>
              <div>&lt;/rdf:RDF&gt;</div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">3. JSON-LD</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              웹 개발자 친화적인 JSON 형식
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <div>{`{`}</div>
              <div className="ml-2">"@context": {`{`}</div>
              <div className="ml-4">"foaf": "http://xmlns.com/foaf/0.1/"</div>
              <div className="ml-2">{`},`}</div>
              <div className="ml-2">"@id": "http://example.org/john",</div>
              <div className="ml-2">"foaf:name": "John Doe",</div>
              <div className="ml-2">"foaf:age": 30</div>
              <div>{`}`}</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">술어(Predicate)는 어떻게 정하나요?</h2>
        <p className="mb-4">
          RDF에서 술어는 주로 두 가지 방법으로 사용합니다: 표준 온톨로지를 가져다 쓰거나, 직접 정의합니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">📚 표준 온톨로지 사용 (90%)</h3>
            <div className="space-y-2 text-sm">
              <div className="font-mono bg-white dark:bg-gray-800 p-2 rounded">
                <div className="text-gray-600 dark:text-gray-400"># 사람/조직 정보</div>
                <div>foaf:name "홍길동"</div>
                <div>foaf:knows :김철수</div>
              </div>
              <div className="font-mono bg-white dark:bg-gray-800 p-2 rounded">
                <div className="text-gray-600 dark:text-gray-400"># 문서 정보</div>
                <div>dc:title "RDF 가이드"</div>
                <div>dc:creator "저자명"</div>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">🔧 커스텀 술어 정의 (10%)</h3>
            <div className="space-y-2 text-sm">
              <div className="font-mono bg-white dark:bg-gray-800 p-2 rounded">
                <div className="text-gray-600 dark:text-gray-400"># 회사 전용</div>
                <div>my:employeeId "E12345"</div>
                <div>my:department "개발팀"</div>
              </div>
              <div className="font-mono bg-white dark:bg-gray-800 p-2 rounded">
                <div className="text-gray-600 dark:text-gray-400"># 도메인 특화</div>
                <div>med:diagnosis "감기"</div>
                <div>edu:courseCode "CS101"</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-3">🌟 자주 사용하는 표준 온톨로지</h3>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div>
              <strong>FOAF</strong> (<code>foaf:</code>)
              <div className="text-gray-600 dark:text-gray-400">name, knows, mbox, homepage</div>
            </div>
            <div>
              <strong>Dublin Core</strong> (<code>dc:</code>)
              <div className="text-gray-600 dark:text-gray-400">title, creator, date, subject</div>
            </div>
            <div>
              <strong>Schema.org</strong> (<code>schema:</code>)
              <div className="text-gray-600 dark:text-gray-400">Person, Organization, Article</div>
            </div>
            <div>
              <strong>RDF Schema</strong> (<code>rdfs:</code>)
              <div className="text-gray-600 dark:text-gray-400">label, comment, subClassOf</div>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
          <h3 className="font-semibold mb-3">💡 실제 사용 예시: 표준 + 커스텀 혼합</h3>
          <div className="font-mono text-sm">
            <div className="text-gray-600 dark:text-gray-400"># 1. 네임스페이스 선언</div>
            <div>@prefix foaf: &lt;http://xmlns.com/foaf/0.1/&gt; .</div>
            <div>@prefix my: &lt;http://mycompany.com/ont#&gt; .</div>
            <div className="mt-2 text-gray-600 dark:text-gray-400"># 2. 실제 사용</div>
            <div>:john</div>
            <div className="ml-4">foaf:name "John Kim" ;      <span className="text-gray-600 dark:text-gray-400"># 표준</span></div>
            <div className="ml-4">foaf:mbox "john@company.com" ; <span className="text-gray-600 dark:text-gray-400"># 표준</span></div>
            <div className="ml-4">my:employeeId "E12345" ;   <span className="text-gray-600 dark:text-gray-400"># 커스텀</span></div>
            <div className="ml-4">my:team "개발1팀" .        <span className="text-gray-600 dark:text-gray-400"># 커스텀</span></div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">자주 쓰는 온톨로지 치트시트</h2>
        <p className="mb-4">
          실무에서 가장 많이 사용하는 표준 온톨로지와 주요 속성들을 정리했습니다.
        </p>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">
              FOAF (Friend of a Friend) - 사람/조직 정보
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">주요 속성</h4>
                <div className="font-mono text-sm space-y-1">
                  <div><code className="text-blue-600">foaf:name</code> - 이름</div>
                  <div><code className="text-blue-600">foaf:mbox</code> - 이메일</div>
                  <div><code className="text-blue-600">foaf:homepage</code> - 홈페이지</div>
                  <div><code className="text-blue-600">foaf:knows</code> - 아는 사람</div>
                  <div><code className="text-blue-600">foaf:age</code> - 나이</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">사용 예시</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div>:person a foaf:Person ;</div>
                  <div className="ml-4">foaf:name "홍길동" ;</div>
                  <div className="ml-4">foaf:mbox &lt;mailto:hong@kr&gt; .</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">
              Dublin Core (DC) - 문서/출판물 메타데이터
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">핵심 15개 요소</h4>
                <div className="font-mono text-sm space-y-1">
                  <div><code className="text-green-600">dc:title</code> - 제목</div>
                  <div><code className="text-green-600">dc:creator</code> - 작성자</div>
                  <div><code className="text-green-600">dc:date</code> - 날짜</div>
                  <div><code className="text-green-600">dc:subject</code> - 주제</div>
                  <div><code className="text-green-600">dc:language</code> - 언어</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">사용 예시</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div>:book a :Document ;</div>
                  <div className="ml-4">dc:title "RDF 입문" ;</div>
                  <div className="ml-4">dc:creator "김작가" ;</div>
                  <div className="ml-4">dc:date "2024-01-01" .</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">
              Schema.org - 웹 콘텐츠 (Google 권장)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">주요 타입과 속성</h4>
                <div className="font-mono text-sm space-y-1">
                  <div><code className="text-purple-600">schema:Person</code> - 사람</div>
                  <div><code className="text-purple-600">schema:name</code> - 이름</div>
                  <div><code className="text-purple-600">schema:author</code> - 저자</div>
                  <div><code className="text-purple-600">schema:datePublished</code> - 발행일</div>
                  <div><code className="text-purple-600">schema:price</code> - 가격</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">사용 예시</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div>:article a schema:Article ;</div>
                  <div className="ml-4">schema:headline "뉴스 제목" ;</div>
                  <div className="ml-4">schema:author :john .</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">
              SKOS - 분류/카테고리 체계
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">계층 구조 표현</h4>
                <div className="font-mono text-sm space-y-1">
                  <div><code className="text-orange-600">skos:prefLabel</code> - 대표 레이블</div>
                  <div><code className="text-orange-600">skos:altLabel</code> - 대체 레이블</div>
                  <div><code className="text-orange-600">skos:broader</code> - 상위 개념</div>
                  <div><code className="text-orange-600">skos:narrower</code> - 하위 개념</div>
                  <div><code className="text-orange-600">skos:related</code> - 관련 개념</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">사용 예시</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div>:animal skos:prefLabel "동물" ;</div>
                  <div className="ml-4">skos:narrower :dog, :cat .</div>
                  <div>:dog skos:broader :animal .</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
          <h3 className="font-semibold mb-2">💡 실무 팁</h3>
          <ul className="space-y-1 text-sm">
            <li>• 모르겠으면 <strong>Schema.org</strong>부터 확인 (Google이 관리해서 가장 포괄적)</li>
            <li>• 각 온톨로지는 <strong>공식 문서</strong>가 있음 (예: xmlns.com/foaf/spec/)</li>
            <li>• <strong>Protégé</strong> 같은 온톨로지 에디터로 자동완성 지원받기</li>
            <li>• 여러 온톨로지를 <strong>혼합</strong>해서 사용하는 것이 일반적</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDF 네임스페이스</h2>
        <p className="mb-4">
          네임스페이스는 어휘의 충돌을 방지하고 재사용성을 높입니다.
        </p>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">주요 표준 네임스페이스</h3>
          <div className="space-y-2 font-mono text-sm">
            <div>
              <span className="text-indigo-600 dark:text-indigo-400">rdf:</span>
              <span className="text-gray-600 dark:text-gray-400"> http://www.w3.org/1999/02/22-rdf-syntax-ns#</span>
            </div>
            <div>
              <span className="text-indigo-600 dark:text-indigo-400">rdfs:</span>
              <span className="text-gray-600 dark:text-gray-400"> http://www.w3.org/2000/01/rdf-schema#</span>
            </div>
            <div>
              <span className="text-indigo-600 dark:text-indigo-400">owl:</span>
              <span className="text-gray-600 dark:text-gray-400"> http://www.w3.org/2002/07/owl#</span>
            </div>
            <div>
              <span className="text-indigo-600 dark:text-indigo-400">foaf:</span>
              <span className="text-gray-600 dark:text-gray-400"> http://xmlns.com/foaf/0.1/</span>
            </div>
            <div>
              <span className="text-indigo-600 dark:text-indigo-400">dc:</span>
              <span className="text-gray-600 dark:text-gray-400"> http://purl.org/dc/elements/1.1/</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RDF 그래프</h2>
        <p className="mb-4">
          여러 트리플이 모여 그래프를 형성합니다. 노드는 리소스, 엣지는 관계를 나타냅니다.
        </p>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-8">
          <div className="text-center text-gray-600 dark:text-gray-400">
            <p className="mb-4">다음 챕터에서는 RDF 그래프를 3D로 시각화하는 도구를 사용해볼 예정입니다!</p>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
              <span className="text-2xl">🎯</span>
              <span>Knowledge Graph 시각화 예고</span>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💡</span>
          핵심 정리
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• RDF는 모든 지식을 Subject-Predicate-Object 트리플로 표현</li>
          <li>• URI를 사용하여 전역적으로 고유한 식별자 제공</li>
          <li>• 다양한 직렬화 형식 지원 (Turtle, RDF/XML, JSON-LD 등)</li>
          <li>• 네임스페이스로 어휘 충돌 방지 및 재사용성 확보</li>
          <li>• 여러 트리플이 연결되어 지식 그래프 형성</li>
        </ul>
      </section>
    </div>
  )
}