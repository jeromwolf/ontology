'use client'

export default function Chapter12() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 12: 뉴스 온톨로지: 지식 그래프</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            뉴스 데이터를 활용한 지식 그래프를 구축해봅시다. 뉴스 기사에서 인물, 조직, 이벤트를 추출하고, 
            NLP 기술과 온톨로지를 연계하여 뉴스 도메인의 지식 그래프를 생성합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">프로젝트 개요</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">뉴스 지식 그래프 목표</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 뉴스 기사의 구조화된 표현</li>
            <li>• 개체명 인식(NER)과 관계 추출</li>
            <li>• 시간적 이벤트 모델링</li>
            <li>• 뉴스 간 연관성 분석</li>
            <li>• 팩트 체킹과 신뢰도 평가</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">뉴스 온톨로지 스키마</h2>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 font-mono text-sm">
          <div className="space-y-4">
            <div>
              <div className="text-purple-600"># 네임스페이스 정의</div>
              <div>@prefix news: &lt;http://example.org/news-ontology#&gt; .</div>
              <div>@prefix schema: &lt;http://schema.org/&gt; .</div>
              <div>@prefix foaf: &lt;http://xmlns.com/foaf/0.1/&gt; .</div>
              <div>@prefix time: &lt;http://www.w3.org/2006/time#&gt; .</div>
            </div>
            
            <div>
              <div className="text-purple-600"># 핵심 클래스</div>
              <div className="text-blue-600">news:NewsArticle</div>
              <div className="ml-4">rdfs:subClassOf schema:NewsArticle ;</div>
              <div className="ml-4">rdfs:label "뉴스 기사"@ko .</div>
              
              <div className="text-blue-600 mt-2">news:Person</div>
              <div className="ml-4">rdfs:subClassOf foaf:Person ;</div>
              <div className="ml-4">rdfs:label "인물"@ko .</div>
              
              <div className="text-blue-600 mt-2">news:Organization</div>
              <div className="ml-4">rdfs:subClassOf foaf:Organization ;</div>
              <div className="ml-4">rdfs:label "조직"@ko .</div>
              
              <div className="text-blue-600 mt-2">news:Event</div>
              <div className="ml-4">rdfs:subClassOf schema:Event ;</div>
              <div className="ml-4">rdfs:label "이벤트"@ko .</div>
              
              <div className="text-blue-600 mt-2">news:Topic</div>
              <div className="ml-4">rdfs:subClassOf skos:Concept ;</div>
              <div className="ml-4">rdfs:label "주제"@ko .</div>
            </div>
            
            <div>
              <div className="text-purple-600"># 관계 속성</div>
              <div className="text-green-600">news:mentions</div>
              <div className="ml-4">rdfs:domain news:NewsArticle ;</div>
              <div className="ml-4">rdfs:range [ owl:unionOf (news:Person news:Organization news:Event) ] .</div>
              
              <div className="text-green-600 mt-2">news:hasAuthor</div>
              <div className="ml-4">rdfs:domain news:NewsArticle ;</div>
              <div className="ml-4">rdfs:range news:Person .</div>
              
              <div className="text-green-600 mt-2">news:relatedTo</div>
              <div className="ml-4">rdfs:domain news:NewsArticle ;</div>
              <div className="ml-4">rdfs:range news:NewsArticle ;</div>
              <div className="ml-4">owl:SymmetricProperty .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">NLP 파이프라인 통합</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">개체명 인식 (NER)</h3>
            <div className="space-y-3">
              <p className="text-sm">뉴스 텍스트에서 자동으로 추출:</p>
              <ul className="text-sm space-y-1">
                <li>• <strong>PER</strong>: 인물 (정치인, 연예인, CEO 등)</li>
                <li>• <strong>ORG</strong>: 조직 (기업, 정부기관, 단체)</li>
                <li>• <strong>LOC</strong>: 장소 (도시, 국가, 건물)</li>
                <li>• <strong>DATE</strong>: 날짜 및 시간</li>
                <li>• <strong>MONEY</strong>: 금액 정보</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">관계 추출</h3>
            <div className="space-y-3">
              <p className="text-sm">문장에서 관계 패턴 식별:</p>
              <ul className="text-sm space-y-1">
                <li>• <strong>인수/합병</strong>: A가 B를 인수하다</li>
                <li>• <strong>인사이동</strong>: A가 B의 CEO로 임명되다</li>
                <li>• <strong>파트너십</strong>: A와 B가 협력하다</li>
                <li>• <strong>법적분쟁</strong>: A가 B를 고소하다</li>
                <li>• <strong>투자관계</strong>: A가 B에 투자하다</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 뉴스 데이터 모델링</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold mb-4">뉴스 기사 인스턴스 예시</h3>
          <div className="font-mono text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded">
            <div className="text-purple-600"># 뉴스 기사</div>
            <div>news:article_2024_001</div>
            <div className="ml-2">a news:NewsArticle ;</div>
            <div className="ml-2">news:headline "삼성전자, AI 칩 개발에 10조원 투자 발표" ;</div>
            <div className="ml-2">news:publishedDate "2024-01-15T09:00:00" ;</div>
            <div className="ml-2">news:publisher news:YonhapNews ;</div>
            <div className="ml-2">news:category news:Technology ;</div>
            <div className="ml-2">news:mentions news:Samsung_Electronics ,</div>
            <div className="ml-2 pl-14">news:AI_Chip ,</div>
            <div className="ml-2 pl-14">news:Investment_10T ;</div>
            <div className="ml-2">news:sentiment "positive" ;</div>
            <div className="ml-2">news:credibilityScore 0.95 .</div>
            
            <div className="text-purple-600 mt-4"># 관련 엔티티</div>
            <div>news:Samsung_Electronics</div>
            <div className="ml-2">a news:Organization ;</div>
            <div className="ml-2">foaf:name "삼성전자" ;</div>
            <div className="ml-2">news:industry "Technology" ;</div>
            <div className="ml-2">news:stockSymbol "005930" .</div>
            
            <div className="mt-2">news:Investment_10T</div>
            <div className="ml-2">a news:Event ;</div>
            <div className="ml-2">news:eventType "Investment" ;</div>
            <div className="ml-2">news:amount "10000000000000" ;</div>
            <div className="ml-2">news:currency "KRW" .</div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">시간적 모델링</h2>
        
        <div className="space-y-4">
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">이벤트 타임라인</h3>
            <div className="font-mono text-sm">
              <div className="text-purple-600"># 시간 기반 이벤트 연결</div>
              <div>news:Timeline_Samsung_2024</div>
              <div className="ml-2">a time:TemporalEntity ;</div>
              <div className="ml-2">time:hasBeginning "2024-01-01" ;</div>
              <div className="ml-2">news:contains news:CES_Announcement ,</div>
              <div className="ml-2 pl-14">news:AI_Investment ,</div>
              <div className="ml-2 pl-14">news:Quarterly_Earnings .</div>
              
              <div className="mt-4 text-purple-600"># 이벤트 순서 관계</div>
              <div>news:CES_Announcement</div>
              <div className="ml-2">time:before news:AI_Investment ;</div>
              <div className="ml-2">news:leadTo news:Stock_Rise .</div>
            </div>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">뉴스 클러스터링</h3>
            <div className="font-mono text-sm">
              <div className="text-purple-600"># 관련 뉴스 그룹화</div>
              <div>news:Cluster_AI_Investment</div>
              <div className="ml-2">a news:NewsCluster ;</div>
              <div className="ml-2">rdfs:label "AI 투자 관련 뉴스" ;</div>
              <div className="ml-2">news:hasArticle news:article_2024_001 ,</div>
              <div className="ml-2 pl-16">news:article_2024_002 ,</div>
              <div className="ml-2 pl-16">news:article_2024_003 ;</div>
              <div className="ml-2">news:dominantTopic news:Artificial_Intelligence ;</div>
              <div className="ml-2">news:timespan "P7D" .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">지식 그래프 쿼리</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. 특정 인물/조직 관련 뉴스 검색</h3>
            <div className="font-mono text-sm">
              PREFIX news: &lt;http://example.org/news-ontology#&gt;<br/>
              SELECT ?article ?headline ?date WHERE {`{`}<br/>
              <span className="ml-2">?article a news:NewsArticle ;</span><br/>
              <span className="ml-2 pl-9">news:mentions news:Samsung_Electronics ;</span><br/>
              <span className="ml-2 pl-9">news:headline ?headline ;</span><br/>
              <span className="ml-2 pl-9">news:publishedDate ?date .</span><br/>
              <span className="ml-2">FILTER(?date {'>='} "2024-01-01")</span><br/>
              {`}`} ORDER BY DESC(?date) LIMIT 20
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. 관계 네트워크 분석</h3>
            <div className="font-mono text-sm">
              SELECT ?person ?org (COUNT(?article) as ?coMentions) WHERE {`{`}<br/>
              <span className="ml-2">?article a news:NewsArticle ;</span><br/>
              <span className="ml-2 pl-9">news:mentions ?person ;</span><br/>
              <span className="ml-2 pl-9">news:mentions ?org .</span><br/>
              <span className="ml-2">?person a news:Person .</span><br/>
              <span className="ml-2">?org a news:Organization .</span><br/>
              <span className="ml-2">FILTER(?person != ?org)</span><br/>
              {`}`} GROUP BY ?person ?org<br/>
              HAVING(?coMentions {'>'} 5)<br/>
              ORDER BY DESC(?coMentions)
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">3. 트렌드 분석</h3>
            <div className="font-mono text-sm">
              SELECT ?topic ?month (COUNT(?article) as ?count) WHERE {`{`}<br/>
              <span className="ml-2">?article a news:NewsArticle ;</span><br/>
              <span className="ml-2 pl-9">news:hasTopic ?topic ;</span><br/>
              <span className="ml-2 pl-9">news:publishedDate ?date .</span><br/>
              <span className="ml-2">BIND(MONTH(?date) as ?month)</span><br/>
              <span className="ml-2">FILTER(YEAR(?date) = 2024)</span><br/>
              {`}`} GROUP BY ?topic ?month<br/>
              ORDER BY ?topic ?month
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">팩트 체킹과 신뢰도</h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">신뢰도 평가 모델</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">소스 신뢰도 요소</h4>
              <ul className="text-sm space-y-1">
                <li>• 언론사 신뢰도 점수</li>
                <li>• 기자 전문성 평가</li>
                <li>• 인용 출처 검증</li>
                <li>• 과거 정확도 이력</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">콘텐츠 검증 요소</h4>
              <ul className="text-sm space-y-1">
                <li>• 교차 검증 (다수 언론사)</li>
                <li>• 공식 발표 대조</li>
                <li>• 통계 데이터 확인</li>
                <li>• 전문가 의견 참조</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4 font-mono text-sm bg-white dark:bg-gray-800 p-3 rounded">
            <div className="text-purple-600"># 신뢰도 추론 규칙</div>
            <div>news:VerifiedArticle owl:equivalentClass [</div>
            <div className="ml-4">owl:intersectionOf (</div>
            <div className="ml-8">news:NewsArticle</div>
            <div className="ml-8">[owl:hasValue news:credibilityScore ;</div>
            <div className="ml-10">owl:minInclusive 0.8]</div>
            <div className="ml-8">[owl:minCardinality 2 ;</div>
            <div className="ml-10">owl:onProperty news:confirmedBy]</div>
            <div className="ml-4">)</div>
            <div>] .</div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">📰</span>
          실전 프로젝트 과제
        </h2>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
            <div>
              <h4 className="font-medium">실시간 뉴스 수집기 구현</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                RSS 피드나 API를 통해 뉴스를 수집하고 자동으로 온톨로지에 추가
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
            <div>
              <h4 className="font-medium">이벤트 임팩트 분석</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                주요 이벤트가 관련 기업/인물에 미치는 영향을 추적하고 시각화
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
            <div>
              <h4 className="font-medium">가짜뉴스 탐지 시스템</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                온톨로지 기반 팩트 체킹과 신뢰도 평가 알고리즘 구현
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}