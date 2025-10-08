'use client';

import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: 'Foundational Papers',
            icon: 'paper',
            color: 'border-indigo-500',
            items: [
              {
                title: 'The Semantic Web',
                authors: 'Tim Berners-Lee, James Hendler, Ora Lassila',
                year: '2001',
                description: 'Semantic Web의 비전을 제시한 역사적 논문 (Scientific American)',
                link: 'https://www-sop.inria.fr/acacia/cours/essi2006/Scientific%20American_%20Feature%20Article_%20The%20Semantic%20Web_%20May%202001.pdf'
              },
              {
                title: 'Linked Data - Design Issues',
                authors: 'Tim Berners-Lee',
                year: '2006',
                description: 'Linked Data의 4가지 원칙을 최초로 정의한 문서',
                link: 'https://www.w3.org/DesignIssues/LinkedData.html'
              },
              {
                title: 'A Survey of the First 20 Years of Research on Semantic Web and Linked Data',
                authors: 'Amit Sheth, Prateek Jain',
                year: '2018',
                description: 'Semantic Web 20년의 발전사를 정리한 종합 서베이',
                link: 'https://cacm.acm.org/magazines/2018/11/232219-a-survey-of-the-first-20-years-of-research-on-semantic-web-and-linked-data/'
              },
              {
                title: 'Ontology-Based Information Integration',
                authors: 'Maurizio Lenzerini',
                year: '2002',
                description: '온톨로지 기반 정보 통합의 이론적 기초 (ICDT)',
                link: 'https://dl.acm.org/doi/10.5555/646136.680093'
              }
            ]
          },
          {
            title: 'W3C Standards & Specifications',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'RDF 1.1 Primer',
                authors: 'W3C',
                year: '2014',
                description: 'RDF (Resource Description Framework) 공식 표준',
                link: 'https://www.w3.org/TR/rdf11-primer/'
              },
              {
                title: 'OWL 2 Web Ontology Language Primer',
                authors: 'W3C',
                year: '2012',
                description: 'OWL 2 Web Ontology Language 공식 표준',
                link: 'https://www.w3.org/TR/owl2-primer/'
              },
              {
                title: 'SPARQL 1.1 Query Language',
                authors: 'W3C',
                year: '2013',
                description: 'SPARQL 쿼리 언어 공식 표준',
                link: 'https://www.w3.org/TR/sparql11-query/'
              },
              {
                title: 'JSON-LD 1.1: A JSON-based Serialization for Linked Data',
                authors: 'W3C',
                year: '2020',
                description: 'JSON 기반 Linked Data 직렬화 포맷',
                link: 'https://www.w3.org/TR/json-ld11/'
              }
            ]
          },
          {
            title: 'Linked Open Data Resources',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'DBpedia: A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia',
                authors: 'Jens Lehmann, Robert Isele, Max Jakob, et al.',
                year: '2015',
                description: 'Wikipedia의 구조화된 지식베이스 DBpedia',
                link: 'https://www.semantic-web-journal.net/content/dbpedia-large-scale-multilingual-knowledge-base-extracted-wikipedia-1'
              },
              {
                title: 'Wikidata: A Free Collaborative Knowledge Base',
                authors: 'Denny Vrandečić, Markus Krötzsch',
                year: '2014',
                description: 'Wikimedia의 협업 지식베이스',
                link: 'https://cacm.acm.org/magazines/2014/10/178785-wikidata/fulltext'
              },
              {
                title: 'Schema.org: Evolution of Structured Data on the Web',
                authors: 'R.V. Guha, Dan Brickley, Steve Macbeth',
                year: '2016',
                description: 'Google, Microsoft, Yahoo가 만든 구조화 데이터 어휘',
                link: 'https://dl.acm.org/doi/10.1145/2844544'
              },
              {
                title: 'The Linked Open Data Cloud',
                description: 'LOD 클라우드 데이터셋 카탈로그와 시각화',
                link: 'https://lod-cloud.net/'
              }
            ]
          },
          {
            title: 'Textbooks & Learning Resources',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Semantic Web for the Working Ontologist',
                authors: 'Dean Allemang, James Hendler, Fabien Gandon',
                year: '2020',
                description: 'Semantic Web 실무자를 위한 종합 가이드 (3rd Edition)',
                link: 'https://www.workingontologist.org/'
              },
              {
                title: 'A Semantic Web Primer',
                authors: 'Grigoris Antoniou, Paul Groth, Frank van Harmelen, Rinke Hoekstra',
                year: '2012',
                description: 'Semantic Web 기술 입문서 (3rd Edition, MIT Press)',
                link: 'https://mitpress.mit.edu/9780262018289/'
              },
              {
                title: 'Linked Data: Evolving the Web into a Global Data Space',
                authors: 'Tom Heath, Christian Bizer',
                year: '2011',
                description: 'Linked Data의 개념과 실무 완벽 가이드',
                link: 'https://linkeddatabook.com/'
              },
              {
                title: 'Knowledge Graphs',
                authors: 'Aidan Hogan, Eva Blomqvist, Michael Cochez, et al.',
                year: '2021',
                description: '지식 그래프의 이론과 실무 (ACM Computing Surveys)',
                link: 'https://arxiv.org/abs/2003.02320'
              }
            ]
          }
        ]}
      />
    </div>
  )
}