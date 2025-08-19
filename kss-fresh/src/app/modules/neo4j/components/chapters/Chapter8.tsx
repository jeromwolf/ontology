'use client';

import React from 'react';
import { Rocket, Code, Users, Target } from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">실전 프로젝트 🚀</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          추천 시스템, 사기 탐지, 지식 그래프 구축 등 실제 비즈니스 문제를
          Neo4j로 해결하는 종합 프로젝트를 진행합니다!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 프로젝트 1: 실시간 추천 시스템</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            사용자의 행동 패턴과 관계 네트워크를 분석하여 개인화된 추천을 제공하는
            협업 필터링과 컨텐츠 기반 하이브리드 추천 시스템을 구축합니다.
          </p>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
            <h3 className="font-bold text-purple-600 dark:text-purple-400 mb-3">데이터 모델 설계</h3>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 노드 생성</div>
              <div>CREATE (u:User {id: 'user1', name: 'Alice', age: 28})</div>
              <div>CREATE (p:Product {id: 'prod1', name: 'Neo4j Book', category: 'Tech'})</div>
              <div>CREATE (c:Category {name: 'Tech', description: 'Technology'})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 관계 생성</div>
              <div>CREATE (u)-[:PURCHASED {rating: 5, date: datetime()}]->(p)</div>
              <div>CREATE (u)-[:VIEWED {duration: 120, bounce: false}]->(p)</div>
              <div>CREATE (p)-[:BELONGS_TO]->(c)</div>
              <div>CREATE (u1)-[:FOLLOWS {since: date('2024-01-01')}]->(u2)</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">협업 필터링</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 유사 사용자 찾기</div>
                <div>MATCH (u1:User {id: $userId})</div>
                <div>MATCH (u1)-[r1:PURCHASED]->(p:Product)</div>
                <div>    <-[r2:PURCHASED]-(u2:User)</div>
                <div>WHERE u1 <> u2</div>
                <div>WITH u1, u2, </div>
                <div>  COUNT(p) AS shared,</div>
                <div>  SUM(ABS(r1.rating - r2.rating)) AS diff</div>
                <div>WHERE shared > 3</div>
                <div>RETURN u2, shared, </div>
                <div>  1.0 - (diff / (shared * 5.0)) AS similarity</div>
                <div>ORDER BY similarity DESC</div>
                <div>LIMIT 10</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">컨텐츠 기반</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 속성 기반 추천</div>
                <div>MATCH (u:User {id: $userId})</div>
                <div>    -[:PURCHASED]->(p1:Product)</div>
                <div>    -[:BELONGS_TO]->(c:Category)</div>
                <div>MATCH (p2:Product)-[:BELONGS_TO]->(c)</div>
                <div>WHERE NOT EXISTS {</div>
                <div>  (u)-[:PURCHASED|VIEWED]-(p2)</div>
                <div>}</div>
                <div>WITH p2, COUNT(c) AS categories,</div>
                <div>  COLLECT(DISTINCT c.name) AS cats</div>
                <div>RETURN p2.name, categories, cats</div>
                <div>ORDER BY categories DESC</div>
                <div>LIMIT 20</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">하이브리드 추천 엔진</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 통합 추천 스코어 계산</div>
              <div>CALL {`{`}</div>
              <div>  // 협업 필터링 스코어</div>
              <div>  MATCH (u:User {id: $userId})-[:SIMILAR]-(similar)</div>
              <div>        -[r:PURCHASED]->(p:Product)</div>
              <div>  WHERE NOT EXISTS((u)-[:PURCHASED|VIEWED]-(p))</div>
              <div>  RETURN p, AVG(r.rating) * 0.6 AS cfScore</div>
              <div>  UNION</div>
              <div>  // 컨텐츠 기반 스코어</div>
              <div>  MATCH (u:User {id: $userId})-[:PURCHASED]->(liked)</div>
              <div>        -[:SIMILAR_TO]-(p:Product)</div>
              <div>  WHERE NOT EXISTS((u)-[:PURCHASED|VIEWED]-(p))</div>
              <div>  RETURN p, COUNT(*) * 0.4 AS cbScore</div>
              <div>{`}`}</div>
              <div>WITH p, SUM(cfScore + cbScore) AS totalScore</div>
              <div>RETURN p.name, p.category, totalScore</div>
              <div>ORDER BY totalScore DESC LIMIT 10</div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">📊 성능 최적화 포인트</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300 space-y-1">
              <div>• 사용자별 유사도 사전 계산 및 캐싱</div>
              <div>• 인기 상품과 신규 상품의 균형 조정</div>
              <div>• 실시간 업데이트를 위한 증분 처리</div>
              <div>• A/B 테스트로 가중치 최적화</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 프로젝트 2: 금융 사기 탐지 시스템</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">그래프 패턴으로 이상 거래 실시간 탐지</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
            <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">사기 패턴 모델링</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 순환 거래 탐지</div>
              <div>MATCH path = (a:Account)-[:TRANSFER*3..6]->(a)</div>
              <div>WHERE ALL(r IN relationships(path) WHERE </div>
              <div>  r.timestamp > datetime() - duration('PT1H') AND</div>
              <div>  r.amount > 10000</div>
              <div>)</div>
              <div>RETURN path, </div>
              <div>  [n IN nodes(path) | n.accountId] AS accounts,</div>
              <div>  REDUCE(s = 0, r IN relationships(path) | s + r.amount) AS total</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 머니 뮬 탐지</div>
              <div>MATCH (source:Account)-[t:TRANSFER]->(mule:Account)</div>
              <div>WHERE t.timestamp > datetime() - duration('P1D')</div>
              <div>WITH mule, COUNT(DISTINCT source) AS inCount, </div>
              <div>     SUM(t.amount) AS inAmount</div>
              <div>WHERE inCount > 5 AND inAmount > 50000</div>
              <div>MATCH (mule)-[out:TRANSFER]->(dest:Account)</div>
              <div>WHERE out.timestamp > datetime() - duration('PT2H')</div>
              <div>RETURN mule, inCount, inAmount, COUNT(dest) AS outCount</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">위험 점수 계산</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 다차원 위험 평가</div>
                <div>MATCH (a:Account {id: $accountId})</div>
                <div>OPTIONAL MATCH (a)-[t:TRANSFER]-()</div>
                <div>WITH a,</div>
                <div>  COUNT(t) AS txCount,</div>
                <div>  AVG(t.amount) AS avgAmount,</div>
                <div>  STDEV(t.amount) AS stdAmount</div>
                <div>RETURN a.id,</div>
                <div>  CASE</div>
                <div>    WHEN txCount > 100 THEN 0.3</div>
                <div>    WHEN stdAmount > avgAmount THEN 0.5</div>
                <div>    ELSE 0.1</div>
                <div>  END AS riskScore</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">실시간 알림</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 이상 패턴 감지</div>
                <div>CALL apoc.trigger.add(</div>
                <div>  'fraud-detection',</div>
                <div>  'MATCH (a:Account)</div>
                <div>   WHERE $createdNodes(a)</div>
                <div>   WITH a</div>
                <div>   MATCH (a)-[t:TRANSFER]->()</div>
                <div>   WHERE t.amount > a.dailyLimit</div>
                <div>   CREATE (alert:Alert {</div>
                <div>     type: "LIMIT_EXCEEDED",</div>
                <div>     account: a.id,</div>
                <div>     amount: t.amount,</div>
                <div>     timestamp: datetime()</div>
                <div>   })',</div>
                <div>  {phase: 'after'}</div>
                <div>)</div>
              </div>
            </div>
          </div>

          <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">🚨 탐지 규칙 라이브러리</h4>
            <div className="text-sm text-red-800 dark:text-red-300 space-y-1">
              <div>• <strong>순환 거래:</strong> 3-6 홉 내 자금 순환</div>
              <div>• <strong>계좌 체인:</strong> 연속적인 소액 이체</div>
              <div>• <strong>휴면 계좌 악용:</strong> 갑작스러운 활동 증가</div>
              <div>• <strong>속도 위반:</strong> 비정상적 거래 빈도</div>
              <div>• <strong>네트워크 이상:</strong> 새로운 연결 패턴</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🧠 프로젝트 3: 엔터프라이즈 지식 그래프</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">조직의 모든 지식을 연결하는 통합 플랫폼</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3">지식 그래프 스키마</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 엔티티 정의</div>
              <div>CREATE CONSTRAINT concept_unique</div>
              <div>FOR (c:Concept) REQUIRE c.uri IS UNIQUE;</div>
              <div></div>
              <div>CREATE (c1:Concept {name: 'Machine Learning', uri: 'ml:001'})</div>
              <div>CREATE (c2:Concept {name: 'Deep Learning', uri: 'ml:002'})</div>
              <div>CREATE (doc:Document {title: 'ML Guide', type: 'PDF'})</div>
              <div>CREATE (person:Expert {name: 'Dr. Kim', field: 'AI'})</div>
              <div>CREATE (proj:Project {name: 'AI Platform', status: 'Active'})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 관계 정의</div>
              <div>CREATE (c2)-[:IS_SUBCONCEPT_OF]->(c1)</div>
              <div>CREATE (doc)-[:COVERS_TOPIC]->(c1)</div>
              <div>CREATE (person)-[:EXPERT_IN]->(c1)</div>
              <div>CREATE (proj)-[:USES_TECHNOLOGY]->(c2)</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">지식 검색</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 의미 기반 검색</div>
                <div>MATCH (query:Concept {name: $search})</div>
                <div>CALL gds.alpha.allShortestPaths.stream({</div>
                <div>  nodeProjection: 'Concept',</div>
                <div>  relationshipProjection: '*',</div>
                <div>  sourceNode: id(query),</div>
                <div>  delta: 3.0</div>
                <div>})</div>
                <div>YIELD targetNodeId, distance</div>
                <div>WITH gds.util.asNode(targetNodeId) AS related,</div>
                <div>     distance</div>
                <div>WHERE distance <= 2</div>
                <div>MATCH (related)<-[:COVERS_TOPIC]-(doc)</div>
                <div>RETURN DISTINCT doc, distance</div>
                <div>ORDER BY distance</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3">전문가 찾기</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 주제별 전문가 네트워크</div>
                <div>MATCH (topic:Concept {name: $topic})</div>
                <div>MATCH path = (topic)<-[:EXPERT_IN|</div>
                <div>  AUTHORED|CONTRIBUTED*1..2]-(person)</div>
                <div>WITH person, </div>
                <div>  COUNT(DISTINCT path) AS connections,</div>
                <div>  COLLECT(DISTINCT topic) AS expertise</div>
                <div>MATCH (person)-[:WORKS_ON]->(project)</div>
                <div>RETURN person.name,</div>
                <div>  connections AS relevance,</div>
                <div>  expertise,</div>
                <div>  COLLECT(project.name) AS projects</div>
                <div>ORDER BY relevance DESC</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">지식 갭 분석</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 문서화되지 않은 중요 주제 찾기</div>
              <div>MATCH (c:Concept)</div>
              <div>WHERE NOT EXISTS((c)<-[:COVERS_TOPIC]-(:Document))</div>
              <div>MATCH (c)<-[:REQUIRES|USES*1..2]-(p:Project)</div>
              <div>WHERE p.status = 'Active'</div>
              <div>WITH c, COUNT(DISTINCT p) AS projectCount</div>
              <div>RETURN c.name AS missingTopic,</div>
              <div>       projectCount AS impactedProjects</div>
              <div>ORDER BY projectCount DESC</div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">💡 지식 그래프 활용</h4>
            <div className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
              <div>• <strong>스마트 검색:</strong> 의미 기반 문서 추천</div>
              <div>• <strong>전문가 매칭:</strong> 프로젝트별 최적 인력 배치</div>
              <div>• <strong>지식 갭 분석:</strong> 교육 필요 영역 도출</div>
              <div>• <strong>영향도 분석:</strong> 기술 변경 시 파급 효과 예측</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 프로젝트 체크리스트</h2>
        <div className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/20 dark:to-slate-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">성공적인 Neo4j 프로젝트를 위한 단계별 가이드</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-3">📐 설계 단계</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>비즈니스 요구사항 명확히 정의</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>그래프 모델 스케치 및 검증</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>쿼리 패턴 사전 식별</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>성능 목표 수립</span>
                </li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-3">🚀 구현 단계</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>인덱스 전략 수립 및 생성</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>데이터 임포트 파이프라인 구축</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>핵심 쿼리 최적화</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>모니터링 대시보드 설정</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 코스 완주를 축하합니다!</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <p className="text-lg mb-4">
            Neo4j의 핵심 개념부터 실전 프로젝트까지 모든 과정을 마스터하셨습니다! 🎉
          </p>
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">🏆</span>
              <span><strong>그래프 모델링:</strong> 복잡한 관계를 직관적으로 표현</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">🏆</span>
              <span><strong>Cypher 마스터:</strong> 기본부터 고급 기능까지 완벽 구사</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">🏆</span>
              <span><strong>알고리즘 활용:</strong> PageRank, 최단경로, 커뮤니티 탐지</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">🏆</span>
              <span><strong>성능 최적화:</strong> 대규모 그래프도 빠르게 처리</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">🏆</span>
              <span><strong>실전 경험:</strong> 추천, 사기탐지, 지식그래프 구축</span>
            </li>
          </ul>
          
          <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <p className="text-center text-blue-800 dark:text-blue-200 font-semibold">
              이제 여러분은 Neo4j 전문가입니다! 
              실제 프로젝트에서 그래프의 힘을 발휘해보세요! 💪
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}