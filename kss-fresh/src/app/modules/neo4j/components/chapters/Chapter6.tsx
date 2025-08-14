'use client'

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">KSS 도메인 통합 🌐</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          온톨로지, LLM, RAG의 강력한 기능들을 Neo4j 그래프로 통합하여
          차세대 지식 플랫폼을 구축하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔗 온톨로지 데이터 통합</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            RDF 트리플, OWL 온톨로지, SKOS 분류체계를 Neo4j의 프로퍼티 그래프로 변환하여
            더 직관적이고 성능이 뛰어난 지식 그래프를 구축합니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-blue-600 dark:text-blue-400 mb-3">RDF to Neo4j 변환</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// RDF 트리플 임포트</div>
                <div>CALL n10s.rdf.import.fetch(</div>
                <div>  'https://example.com/ontology.ttl',</div>
                <div>  'Turtle'</div>
                <div>)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 클래스 계층구조 매핑</div>
                <div>MATCH (c:Class)</div>
                <div>WHERE c.uri STARTS WITH 'http://kss.com/'</div>
                <div>MERGE (domain:Domain {name: 'KSS'})</div>
                <div>MERGE (c)-[:BELONGS_TO]->(domain)</div>
                <div>RETURN c.label, c.uri</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">온톨로지 관계 모델링</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 개념 간 관계 생성</div>
                <div>MATCH (concept:Concept)</div>
                <div>MATCH (related:Concept)</div>
                <div>WHERE concept.broader = related.uri</div>
                <div>MERGE (concept)-[:BROADER_THAN]->(related)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 동의어 관계</div>
                <div>MERGE (concept)-[:SYNONYM]-(related)</div>
                <div>WHERE concept.altLabel CONTAINS related.prefLabel</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">온톨로지 추론 규칙</h3>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 전이적 관계 추론</div>
              <div>MATCH path = (a:Concept)-[:IS_A*]->(c:Concept)</div>
              <div>WHERE a.name = 'Electric Vehicle' </div>
              <div>  AND c.name = 'Thing'</div>
              <div>WITH a, c, length(path) as distance</div>
              <div>MERGE (a)-[r:INFERRED_IS_A {distance: distance}]->(c)</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 역관계 자동 생성</div>
              <div>MATCH (a)-[r:HAS_PART]->(b)</div>
              <div>MERGE (b)-[:PART_OF]->(a)</div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">💡 온톨로지 통합 이점</h4>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              • SPARQL보다 직관적인 Cypher 쿼리 • 빠른 그래프 순회 성능
              • 유연한 스키마 확장 • 실시간 추론 가능
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🤖 LLM 데이터 연동</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">언어 모델의 지식을 그래프로 구조화</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">엔티티 추출 및 저장</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// LLM 응답에서 엔티티 추출</div>
                <div>WITH apoc.ml.openai.extract({</div>
                <div>  text: $llm_response,</div>
                <div>  prompt: "Extract entities and relations"</div>
                <div>}) AS extraction</div>
                <div></div>
                <div>UNWIND extraction.entities AS entity</div>
                <div>MERGE (e:Entity {</div>
                <div>  name: entity.name,</div>
                <div>  type: entity.type</div>
                <div>})</div>
                <div>SET e.confidence = entity.confidence</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">임베딩 벡터 저장</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 텍스트 임베딩 생성</div>
                <div>MATCH (doc:Document)</div>
                <div>WITH doc, apoc.ml.openai.embedding(</div>
                <div>  doc.content, </div>
                <div>  {model: 'text-embedding-ada-002'}</div>
                <div>) AS embedding</div>
                <div>SET doc.embedding = embedding</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 벡터 인덱스 생성</div>
                <div>CREATE VECTOR INDEX doc_embeddings</div>
                <div>FOR (d:Document) ON d.embedding</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">대화 히스토리 그래프</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 대화 세션 모델링</div>
              <div>CREATE (session:ChatSession {</div>
              <div>  id: randomUUID(),</div>
              <div>  startTime: datetime()</div>
              <div>})</div>
              <div></div>
              <div>CREATE (msg:Message {</div>
              <div>  role: 'user',</div>
              <div>  content: $user_input,</div>
              <div>  timestamp: datetime()</div>
              <div>})</div>
              <div>CREATE (session)-[:HAS_MESSAGE {order: 1}]->(msg)</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 컨텍스트 연결</div>
              <div>MATCH (prev:Message)<-[:HAS_MESSAGE]-(session)</div>
              <div>CREATE (prev)-[:FOLLOWED_BY]->(msg)</div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🚀 LLM + Graph 시너지</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300 space-y-1">
              <div>• 구조화된 지식으로 환각 현상 감소</div>
              <div>• 대화 맥락의 영구 저장 및 추적</div>
              <div>• 팩트 체크를 위한 지식 그래프 활용</div>
              <div>• 멀티턴 대화의 효과적 관리</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 RAG 시스템 구축</h2>
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">검색 증강 생성을 위한 그래프 기반 아키텍처</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">문서 청킹과 인덱싱</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 문서를 청크로 분할</div>
              <div>CALL apoc.load.json($document_path) YIELD value</div>
              <div>WITH value.content AS content</div>
              <div>UNWIND apoc.text.split(content, '.', 1000) AS chunk</div>
              <div>CREATE (c:Chunk {</div>
              <div>  text: chunk,</div>
              <div>  embedding: apoc.ml.openai.embedding(chunk)</div>
              <div>})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 청크 간 관계 생성</div>
              <div>MATCH (c1:Chunk), (c2:Chunk)</div>
              <div>WHERE id(c1) < id(c2)</div>
              <div>WITH c1, c2, gds.similarity.cosine(</div>
              <div>  c1.embedding, c2.embedding</div>
              <div>) AS similarity</div>
              <div>WHERE similarity > 0.8</div>
              <div>MERGE (c1)-[:SIMILAR {score: similarity}]->(c2)</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">하이브리드 검색</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 벡터 + 키워드 검색</div>
                <div>WITH apoc.ml.openai.embedding($query) AS qVec</div>
                <div>MATCH (c:Chunk)</div>
                <div>WHERE c.text CONTAINS $keyword</div>
                <div>  OR gds.similarity.cosine(</div>
                <div>    c.embedding, qVec) > 0.7</div>
                <div>RETURN c.text, </div>
                <div>  gds.similarity.cosine(</div>
                <div>    c.embedding, qVec) AS score</div>
                <div>ORDER BY score DESC LIMIT 5</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">컨텍스트 확장</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 관련 청크 수집</div>
                <div>MATCH (c:Chunk)</div>
                <div>WHERE c.id IN $relevant_chunks</div>
                <div>MATCH path = (c)-[:SIMILAR*1..2]-(related)</div>
                <div>WITH c, collect(DISTINCT related) AS context</div>
                <div>RETURN c.text + ' ' + </div>
                <div>  reduce(s='', r IN context | </div>
                <div>    s + ' ' + r.text</div>
                <div>  ) AS expanded_context</div>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">📊 GraphRAG 아키텍처</h4>
            <div className="text-sm text-green-800 dark:text-green-300">
              <div>1️⃣ <strong>인덱싱:</strong> 문서 → 청크 → 임베딩 → 그래프 저장</div>
              <div>2️⃣ <strong>검색:</strong> 쿼리 → 벡터/키워드 검색 → 컨텍스트 확장</div>
              <div>3️⃣ <strong>생성:</strong> 확장된 컨텍스트 → LLM → 답변 생성</div>
              <div>4️⃣ <strong>피드백:</strong> 사용자 평가 → 그래프 가중치 업데이트</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔮 통합 시나리오</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">온톨로지 + LLM + RAG의 완벽한 조화</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">지능형 Q&A 시스템</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 통합 쿼리 파이프라인</div>
              <div>CALL {`{`}</div>
              <div>  // 1. 온톨로지에서 개념 확인</div>
              <div>  MATCH (concept:Concept)</div>
              <div>  WHERE concept.label CONTAINS $query</div>
              <div>  OPTIONAL MATCH (concept)-[:RELATED_TO]-(related)</div>
              <div>  WITH collect(DISTINCT related.label) AS concepts</div>
              <div>  </div>
              <div>  // 2. RAG로 관련 문서 검색</div>
              <div>  CALL db.index.vector.queryNodes(</div>
              <div>    'doc_embeddings', 5, $query_embedding</div>
              <div>  ) YIELD node AS doc, score</div>
              <div>  </div>
              <div>  // 3. LLM 대화 히스토리 참조</div>
              <div>  MATCH (session:ChatSession)-[:HAS_MESSAGE]->(m)</div>
              <div>  WHERE session.userId = $userId</div>
              <div>  WITH concepts, collect(doc) AS docs, </div>
              <div>       collect(m) AS history</div>
              <div>  </div>
              <div>  // 4. 통합 컨텍스트 생성</div>
              <div>  RETURN {</div>
              <div>    ontology: concepts,</div>
              <div>    documents: [d IN docs | d.content],</div>
              <div>    history: [h IN history | h.content]</div>
              <div>  } AS context</div>
              <div>{`}`}</div>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-2">✨ 통합의 시너지 효과</h4>
            <div className="text-sm text-indigo-800 dark:text-indigo-300 space-y-1">
              <div>• <strong>정확성:</strong> 온톨로지로 검증된 팩트 기반 응답</div>
              <div>• <strong>맥락성:</strong> RAG로 최신 정보 반영</div>
              <div>• <strong>개인화:</strong> LLM 대화 히스토리 활용</div>
              <div>• <strong>확장성:</strong> 그래프 구조로 무한 확장 가능</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 오늘 배운 것 정리</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>온톨로지 통합:</strong> RDF/OWL을 Neo4j 프로퍼티 그래프로 변환</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>LLM 연동:</strong> 대화 히스토리와 임베딩 벡터 관리</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>RAG 구축:</strong> 그래프 기반 검색 증강 생성 시스템</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>통합 아키텍처:</strong> 세 기술의 시너지로 지능형 시스템 구현</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}