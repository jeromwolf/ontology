'use client';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">그래프 알고리즘과 분석 📊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          그래프의 구조를 이해하고 숨겨진 패턴을 발견하는 고급 분석 기법을 마스터하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏆 PageRank: 중요도 측정의 대가</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Google의 검색 알고리즘으로 유명한 PageRank는 노드의 중요도를 측정하는 가장 강력한 알고리즘입니다.
            링크의 품질과 양을 종합적으로 고려하여 권위를 계산합니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-yellow-600 dark:text-yellow-400 mb-3">GDS PageRank 실행</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 그래프 투영 생성</div>
                <div>CALL gds.graph.project(</div>
                <div>  'webGraph',</div>
                <div>  'Page',</div>
                <div>  'LINKS_TO'</div>
                <div>)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// PageRank 스트리밍</div>
                <div>CALL gds.pageRank.stream('webGraph')</div>
                <div>YIELD nodeId, score</div>
                <div>RETURN gds.util.asNode(nodeId).url AS page,</div>
                <div>       score</div>
                <div>ORDER BY score DESC LIMIT 10</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-orange-600 dark:text-orange-400 mb-3">파라미터 조정</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// 고급 설정</div>
                <div>CALL gds.pageRank.stream('webGraph', {`{`}</div>
                <div>  maxIterations: 20,</div>
                <div>  dampingFactor: 0.85,</div>
                <div>  tolerance: 0.0000001</div>
                <div>{`}`})</div>
                <div>YIELD nodeId, score</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 개인화된 PageRank</div>
                <div>MATCH (start:Page {`{name: 'Homepage'}`})</div>
                <div>CALL gds.pageRank.stream('webGraph', {`{`}</div>
                <div>  sourceNodes: [id(start)]</div>
                <div>{`}`})</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">PageRank 이해하기</h3>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="text-center p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded">
                <div className="font-semibold">Damping Factor</div>
                <div className="text-xs mt-1">보통 0.85 (85% 확률로 링크 따라감)</div>
              </div>
              <div className="text-center p-3 bg-orange-100 dark:bg-orange-900/30 rounded">
                <div className="font-semibold">Max Iterations</div>
                <div className="text-xs mt-1">수렴까지 보통 10-20회 반복</div>
              </div>
              <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="font-semibold">Tolerance</div>
                <div className="text-xs mt-1">수렴 기준 (0.0000001)</div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">🎯 PageRank 활용 사례</h4>
            <p className="text-sm text-yellow-800 dark:text-yellow-300">
              웹페이지 랭킹, 소셜 네트워크 인플루언서 발견, 추천 시스템, 사기 탐지,
              생물학적 네트워크에서 중요한 단백질 식별 등 다양한 분야에서 활용됩니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">👥 커뮤니티 탐지: 숨겨진 그룹 발견</h2>
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">연결된 노드들의 클러스터 찾기</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">Louvain 알고리즘</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 커뮤니티 탐지 실행</div>
                <div>CALL gds.louvain.stream('socialGraph')</div>
                <div>YIELD nodeId, communityId</div>
                <div>RETURN communityId,</div>
                <div>       collect(gds.util.asNode(nodeId).name)</div>
                <div>         AS members,</div>
                <div>       count(*) AS size</div>
                <div>ORDER BY size DESC</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 모듈성 점수 확인</div>
                <div>CALL gds.louvain.stats('socialGraph')</div>
                <div>YIELD modularity</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">Label Propagation</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// 라벨 전파 방식</div>
                <div>CALL gds.labelPropagation.stream(</div>
                <div>  'socialGraph',</div>
                <div>  {`{ maxIterations: 10 }`}</div>
                <div>)</div>
                <div>YIELD nodeId, communityId</div>
                <div>WITH communityId, collect(nodeId) AS nodes</div>
                <div>WHERE size(nodes) {'>'} 2</div>
                <div>RETURN communityId, size(nodes) AS size</div>
                <div>ORDER BY size DESC</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">커뮤니티 품질 측정</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 커뮤니티별 밀도 계산</div>
              <div>MATCH (n)-[r]-(m)</div>
              <div>WHERE n.communityId = m.communityId</div>
              <div>WITH n.communityId AS community,</div>
              <div>     count(DISTINCT n) AS nodes,</div>
              <div>     count(r) AS internal_edges</div>
              <div>WITH community, nodes,</div>
              <div>     internal_edges * 2.0 / (nodes * (nodes - 1)) AS density</div>
              <div>RETURN community, nodes, density</div>
              <div>ORDER BY density DESC</div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🔍 커뮤니티 탐지 비교</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300 space-y-1">
              <div>• <strong>Louvain:</strong> 빠르고 정확, 대규모 그래프에 적합</div>
              <div>• <strong>Label Propagation:</strong> 매우 빠름, 근사치 허용</div>
              <div>• <strong>Weakly Connected:</strong> 방향 무시하고 연결성만 고려</div>
              <div>• <strong>Strongly Connected:</strong> 방향 고려한 강한 연결 컴포넌트</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🛣️ 최단 경로와 중심성</h2>
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">노드와 경로의 중요도 측정</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">최단 경로 찾기</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// Dijkstra 알고리즘</div>
                <div>MATCH (start:Station {`{name: 'Seoul'}`}),</div>
                <div>      (end:Station {`{name: 'Busan'}`})</div>
                <div>CALL gds.shortestPath.dijkstra.stream(</div>
                <div>  'routeGraph',</div>
                <div>  {`{`}</div>
                <div>    sourceNode: id(start),</div>
                <div>    targetNode: id(end),</div>
                <div>    relationshipWeightProperty: 'distance'</div>
                <div>  {`}`}</div>
                <div>)</div>
                <div>YIELD path, totalCost</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">A* 알고리즘</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// 휴리스틱 기반 최적화</div>
                <div>CALL gds.shortestPath.astar.stream(</div>
                <div>  'mapGraph',</div>
                <div>  {`{`}</div>
                <div>    sourceNode: id(start),</div>
                <div>    targetNode: id(end),</div>
                <div>    latitudeProperty: 'lat',</div>
                <div>    longitudeProperty: 'lon',</div>
                <div>    relationshipWeightProperty: 'distance'</div>
                <div>  {`}`}</div>
                <div>)</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">중심성 지표들</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <h5 className="font-semibold text-blue-600 mb-2">Betweenness Centrality</h5>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-2 font-mono text-xs mb-3">
                  <div>CALL gds.betweenness.stream('network')</div>
                  <div>YIELD nodeId, score</div>
                  <div>RETURN gds.util.asNode(nodeId).name,</div>
                  <div>       score AS betweenness</div>
                  <div>ORDER BY betweenness DESC LIMIT 5</div>
                </div>
                <p className="text-xs">• 다른 노드들 사이의 중개 역할 측정</p>
                <p className="text-xs">• 정보 흐름의 핵심 지점 식별</p>
              </div>
              <div>
                <h5 className="font-semibold text-purple-600 mb-2">Closeness Centrality</h5>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-2 font-mono text-xs mb-3">
                  <div>CALL gds.closeness.stream('network')</div>
                  <div>YIELD nodeId, centrality</div>
                  <div>RETURN gds.util.asNode(nodeId).name,</div>
                  <div>       centrality</div>
                  <div>ORDER BY centrality DESC LIMIT 5</div>
                </div>
                <p className="text-xs">• 모든 노드와의 평균 거리 측정</p>
                <p className="text-xs">• 빠른 정보 전파 가능성 평가</p>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">📊 중심성 지표 선택 가이드</h4>
            <div className="text-sm text-green-800 dark:text-green-300 space-y-1">
              <div>• <strong>Degree:</strong> 직접 연결 수 (인기도)</div>
              <div>• <strong>Betweenness:</strong> 중개 역할 (영향력)</div>
              <div>• <strong>Closeness:</strong> 접근성 (효율성)</div>
              <div>• <strong>PageRank:</strong> 링크 품질 고려 (권위)</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔗 유사성과 추천</h2>
        <div className="bg-gradient-to-r from-pink-50 to-rose-50 dark:from-pink-900/20 dark:to-rose-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">노드 간 유사성으로 패턴 발견</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">Jaccard 유사성</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 공통 이웃 기반 유사성</div>
                <div>CALL gds.nodeSimilarity.stream(</div>
                <div>  'userGraph',</div>
                <div>  {`{`}</div>
                <div>    similarityCutoff: 0.1,</div>
                <div>    topK: 5</div>
                <div>  {`}`}</div>
                <div>)</div>
                <div>YIELD node1Id, node2Id, similarity</div>
                <div>RETURN gds.util.asNode(node1Id).name AS user1,</div>
                <div>       gds.util.asNode(node2Id).name AS user2,</div>
                <div>       similarity</div>
                <div>ORDER BY similarity DESC</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-rose-600 dark:text-rose-400 mb-3">Cosine 유사성</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div>// 벡터 각도 기반 유사성</div>
                <div>MATCH (u1:User)-[r1:RATED]-({'>'})m)</div>
                <div>MATCH (u2:User)-[r2:RATED]-({'>'})m)</div>
                <div>WHERE u1 {'<'} u2</div>
                <div>WITH u1, u2,</div>
                <div>     collect([r1.rating, r2.rating]) AS pairs</div>
                <div>WITH u1, u2,</div>
                <div>     gds.similarity.cosine(</div>
                <div>       [p IN pairs | p[0]],</div>
                <div>       [p IN pairs | p[1]]</div>
                <div>     ) AS similarity</div>
                <div>WHERE similarity {'>'} 0.5</div>
                <div>RETURN u1.name, u2.name, similarity</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">그래프 기반 추천 시스템</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 협업 필터링 + 그래프 워크</div>
              <div>MATCH (user:User {`{id: $userId}`})</div>
              <div>CALL {`{`}</div>
              <div className="ml-2">// 1단계: 유사한 사용자 찾기</div>
              <div className="ml-2">MATCH (user)-[:SIMILAR]-(similar:User)</div>
              <div className="ml-2">// 2단계: 이들이 좋아한 아이템 수집</div>
              <div className="ml-2">MATCH (similar)-[:LIKED]-({'>'})item)</div>
              <div className="ml-2">WHERE NOT (user)-[:LIKED|DISLIKED]-(item)</div>
              <div className="ml-2">// 3단계: 점수 계산</div>
              <div className="ml-2">WITH item, count(*) AS score,</div>
              <div className="ml-2">     collect(similar.name) AS recommenders</div>
              <div className="ml-2">RETURN item, score, recommenders</div>
              <div>{`}`}</div>
              <div>RETURN item.title, score</div>
              <div>ORDER BY score DESC LIMIT 10</div>
            </div>
          </div>

          <div className="bg-pink-100 dark:bg-pink-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-pink-800 dark:text-pink-200 mb-2">💡 추천 시스템 최적화 팁</h4>
            <div className="text-sm text-pink-800 dark:text-pink-300 space-y-1">
              <div>• 콜드 스타트 문제: 컨텐츠 기반 필터링과 하이브리드 접근</div>
              <div>• 다양성 확보: 인기도와 참신함의 균형</div>
              <div>• 실시간 업데이트: 증분 계산으로 성능 최적화</div>
              <div>• A/B 테스트: 추천 품질 지속적 개선</div>
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
              <span><strong>PageRank:</strong> 링크 기반 중요도 측정 (구글 검색의 핵심)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>커뮤니티 탐지:</strong> Louvain, Label Propagation으로 그룹 발견</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>최단 경로:</strong> Dijkstra, A* 알고리즘으로 최적 경로 탐색</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>중심성 지표:</strong> Degree, Betweenness, Closeness로 영향력 측정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>유사성과 추천:</strong> Jaccard, Cosine으로 패턴 기반 추천</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}