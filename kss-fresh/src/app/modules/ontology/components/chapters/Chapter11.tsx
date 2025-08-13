'use client'

export default function Chapter11() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 11: 금융 온톨로지: 주식 시장</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            주식 시장 도메인의 온톨로지를 구축해봅시다. FIBO(Financial Industry Business Ontology)를 
            참조하여 주식, 거래, 투자자, 기업 등의 개념을 모델링하고, 금융 데이터의 의미적 연결을 구현합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">프로젝트 개요</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">주식 시장 온톨로지 목표</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 금융 상품과 거래 주체의 체계적 모델링</li>
            <li>• 시장 데이터와 기업 정보의 의미적 연결</li>
            <li>• 투자 전략과 리스크 관리 지식 표현</li>
            <li>• 규제 준수 및 보고 체계 구축</li>
            <li>• 실시간 시장 분석을 위한 추론 규칙</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">FIBO 표준 참조</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">FIBO 핵심 모듈</h3>
            <ul className="space-y-2 text-sm">
              <li>• <strong>Foundations (FND)</strong>: 기본 개념과 관계</li>
              <li>• <strong>Business Entities (BE)</strong>: 기업과 법인</li>
              <li>• <strong>Financial Business (FBC)</strong>: 금융 비즈니스</li>
              <li>• <strong>Securities (SEC)</strong>: 증권과 파생상품</li>
              <li>• <strong>Indicators (IND)</strong>: 시장 지표</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">활용 이점</h3>
            <ul className="space-y-2 text-sm">
              <li>• 국제 표준 준수로 상호운용성 확보</li>
              <li>• 검증된 금융 개념 모델 재사용</li>
              <li>• 규제 보고 자동화 지원</li>
              <li>• 다국적 금융 데이터 통합 용이</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 스키마 설계</h2>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 font-mono text-sm">
          <div className="space-y-4">
            <div>
              <div className="text-purple-600"># 네임스페이스 정의</div>
              <div>@prefix stock: &lt;http://example.org/stock-ontology#&gt; .</div>
              <div>@prefix fibo-fnd: &lt;https://spec.edmcouncil.org/fibo/FND/&gt; .</div>
              <div>@prefix fibo-sec: &lt;https://spec.edmcouncil.org/fibo/SEC/&gt; .</div>
              <div>@prefix rdfs: &lt;http://www.w3.org/2000/01/rdf-schema#&gt; .</div>
              <div>@prefix owl: &lt;http://www.w3.org/2002/07/owl#&gt; .</div>
            </div>
            
            <div>
              <div className="text-purple-600"># 핵심 클래스 정의</div>
              <div className="text-blue-600">stock:Stock</div>
              <div className="ml-4">rdfs:subClassOf fibo-sec:Equity ;</div>
              <div className="ml-4">rdfs:label "주식"@ko ;</div>
              <div className="ml-4">rdfs:comment "기업의 소유권을 나타내는 증권" .</div>
              
              <div className="text-blue-600 mt-2">stock:StockExchange</div>
              <div className="ml-4">rdfs:subClassOf fibo-fbc:Exchange ;</div>
              <div className="ml-4">rdfs:label "증권거래소"@ko .</div>
              
              <div className="text-blue-600 mt-2">stock:Investor</div>
              <div className="ml-4">rdfs:subClassOf fibo-fnd:Party ;</div>
              <div className="ml-4">rdfs:label "투자자"@ko .</div>
              
              <div className="text-blue-600 mt-2">stock:TradingOrder</div>
              <div className="ml-4">rdfs:subClassOf fibo-sec:Order ;</div>
              <div className="ml-4">rdfs:label "거래주문"@ko .</div>
            </div>
            
            <div>
              <div className="text-purple-600"># 속성 정의</div>
              <div className="text-green-600">stock:tickerSymbol</div>
              <div className="ml-4">rdfs:domain stock:Stock ;</div>
              <div className="ml-4">rdfs:range xsd:string ;</div>
              <div className="ml-4">rdfs:label "종목코드"@ko .</div>
              
              <div className="text-green-600 mt-2">stock:marketCap</div>
              <div className="ml-4">rdfs:domain stock:Company ;</div>
              <div className="ml-4">rdfs:range xsd:decimal ;</div>
              <div className="ml-4">rdfs:label "시가총액"@ko .</div>
              
              <div className="text-green-600 mt-2">stock:currentPrice</div>
              <div className="ml-4">rdfs:domain stock:Stock ;</div>
              <div className="ml-4">rdfs:range xsd:decimal ;</div>
              <div className="ml-4">rdfs:label "현재가"@ko .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 데이터 인스턴스</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">삼성전자 주식 데이터</h3>
            <div className="font-mono text-sm">
              stock:Samsung_Electronics<br/>
              <span className="ml-2">a stock:Stock ;</span><br/>
              <span className="ml-2">stock:tickerSymbol "005930" ;</span><br/>
              <span className="ml-2">stock:exchange stock:KOSPI ;</span><br/>
              <span className="ml-2">stock:currentPrice 71000 ;</span><br/>
              <span className="ml-2">stock:priceUnit "KRW" ;</span><br/>
              <span className="ml-2">stock:sector "Technology" ;</span><br/>
              <span className="ml-2">stock:issuedBy stock:Samsung_Corp .</span>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">거래 주문 데이터</h3>
            <div className="font-mono text-sm">
              stock:Order_20240101_001<br/>
              <span className="ml-2">a stock:BuyOrder ;</span><br/>
              <span className="ml-2">stock:orderType "limit" ;</span><br/>
              <span className="ml-2">stock:targetStock stock:Samsung_Electronics ;</span><br/>
              <span className="ml-2">stock:quantity 100 ;</span><br/>
              <span className="ml-2">stock:limitPrice 70000 ;</span><br/>
              <span className="ml-2">stock:orderedBy stock:Investor_123 ;</span><br/>
              <span className="ml-2">stock:orderTime "2024-01-01T09:00:00" .</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">투자자 분류 체계</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold mb-4">OWL을 활용한 투자자 분류</h3>
          <div className="font-mono text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded">
            <div className="text-purple-600"># 개인투자자</div>
            <div>stock:RetailInvestor rdfs:subClassOf stock:Investor .</div>
            
            <div className="text-purple-600 mt-4"># 기관투자자</div>
            <div>stock:InstitutionalInvestor rdfs:subClassOf stock:Investor .</div>
            
            <div className="text-purple-600 mt-4"># 외국인투자자</div>
            <div>stock:ForeignInvestor owl:equivalentClass [</div>
            <div className="ml-4">owl:intersectionOf (</div>
            <div className="ml-8">stock:Investor</div>
            <div className="ml-8">[a owl:Restriction ;</div>
            <div className="ml-12">owl:onProperty stock:nationality ;</div>
            <div className="ml-12">owl:not [ owl:hasValue "KR" ]]</div>
            <div className="ml-4">)</div>
            <div>] .</div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">시장 지표 모델링</h2>
        
        <div className="space-y-4">
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">기술적 지표 온톨로지</h3>
            <div className="font-mono text-sm">
              <div className="text-purple-600"># 이동평균선</div>
              <div>stock:MovingAverage a owl:Class ;</div>
              <div className="ml-4">rdfs:subClassOf stock:TechnicalIndicator .</div>
              
              <div className="mt-3">stock:MA20 a stock:MovingAverage ;</div>
              <div className="ml-4">stock:period 20 ;</div>
              <div className="ml-4">stock:calculatedFrom stock:ClosingPrice .</div>
              
              <div className="text-purple-600 mt-4"># RSI (상대강도지수)</div>
              <div>stock:RSI a owl:Class ;</div>
              <div className="ml-4">rdfs:subClassOf stock:MomentumIndicator ;</div>
              <div className="ml-4">stock:range "[0, 100]" .</div>
            </div>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">추론 규칙: 매매 신호</h3>
            <div className="font-mono text-sm">
              <div className="text-purple-600"># 골든크로스 정의</div>
              <div>stock:GoldenCross owl:equivalentClass [</div>
              <div className="ml-4">owl:intersectionOf (</div>
              <div className="ml-8">[a owl:Restriction ;</div>
              <div className="ml-12">owl:onProperty stock:MA20 ;</div>
              <div className="ml-12">owl:hasValue ?shortMA ]</div>
              <div className="ml-8">[a owl:Restriction ;</div>
              <div className="ml-12">owl:onProperty stock:MA60 ;</div>
              <div className="ml-12">owl:hasValue ?longMA ]</div>
              <div className="ml-8">[?shortMA {'>'} ?longMA]</div>
              <div className="ml-4">)</div>
              <div>] .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL 쿼리 실습</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. 시가총액 상위 10개 기업 조회</h3>
            <div className="font-mono text-sm">
              PREFIX stock: &lt;http://example.org/stock-ontology#&gt;<br/>
              SELECT ?company ?name ?marketCap WHERE {`{`}<br/>
              <span className="ml-2">?company a stock:ListedCompany ;</span><br/>
              <span className="ml-2 pl-9">rdfs:label ?name ;</span><br/>
              <span className="ml-2 pl-9">stock:marketCap ?marketCap .</span><br/>
              {`}`} ORDER BY DESC(?marketCap) LIMIT 10
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. 특정 섹터의 평균 PER 계산</h3>
            <div className="font-mono text-sm">
              SELECT ?sector (AVG(?per) as ?avgPER) WHERE {`{`}<br/>
              <span className="ml-2">?stock a stock:Stock ;</span><br/>
              <span className="ml-2 pl-7">stock:sector ?sector ;</span><br/>
              <span className="ml-2 pl-7">stock:priceEarningsRatio ?per .</span><br/>
              {`}`} GROUP BY ?sector ORDER BY ?avgPER
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">3. 거래량 급증 종목 탐지</h3>
            <div className="font-mono text-sm">
              SELECT ?stock ?name ?volumeRatio WHERE {`{`}<br/>
              <span className="ml-2">?stock a stock:Stock ;</span><br/>
              <span className="ml-2 pl-7">rdfs:label ?name ;</span><br/>
              <span className="ml-2 pl-7">stock:todayVolume ?today ;</span><br/>
              <span className="ml-2 pl-7">stock:avgVolume20 ?avg20 .</span><br/>
              <span className="ml-2">BIND(?today/?avg20 AS ?volumeRatio)</span><br/>
              <span className="ml-2">FILTER(?volumeRatio {'>'} 2.0)</span><br/>
              {`}`} ORDER BY DESC(?volumeRatio)
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💹</span>
          실전 프로젝트 과제
        </h2>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
            <div>
              <h4 className="font-medium">포트폴리오 온톨로지 구축</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                투자 포트폴리오의 구성, 리스크, 수익률을 모델링하세요
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
            <div>
              <h4 className="font-medium">기업 재무제표 통합</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                XBRL 데이터를 온톨로지로 변환하고 분석하세요
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
            <div>
              <h4 className="font-medium">실시간 알림 시스템</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                SPARQL 규칙 기반 투자 알림 시스템을 설계하세요
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}