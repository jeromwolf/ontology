'use client'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">성능 최적화와 운영 ⚡</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          대규모 그래프 데이터베이스를 효율적으로 운영하고
          최고의 성능을 끌어내는 전문 기술을 마스터하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 인덱스 전략과 최적화</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            적절한 인덱스 전략은 Neo4j 성능의 핵심입니다. 쿼리 패턴에 맞는 
            인덱스를 설계하여 검색 속도를 수백 배 향상시킬 수 있습니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-blue-600 dark:text-blue-400 mb-3">단일 속성 인덱스</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// BTREE 인덱스 생성</div>
                <div>CREATE INDEX user_email_idx</div>
                <div>FOR (u:User) ON (u.email)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// TEXT 인덱스 (전문 검색)</div>
                <div>CREATE TEXT INDEX product_desc_idx</div>
                <div>FOR (p:Product) ON (p.description)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// POINT 인덱스 (공간 데이터)</div>
                <div>CREATE POINT INDEX location_idx</div>
                <div>FOR (l:Location) ON (l.coordinates)</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">복합 인덱스</h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 다중 속성 인덱스</div>
                <div>CREATE INDEX person_name_age_idx</div>
                <div>FOR (p:Person) ON (p.name, p.age)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 제약 조건 인덱스</div>
                <div>CREATE CONSTRAINT unique_user_id</div>
                <div>FOR (u:User) REQUIRE u.id IS UNIQUE</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 존재 제약</div>
                <div>CREATE CONSTRAINT user_email_exists</div>
                <div>FOR (u:User) REQUIRE u.email IS NOT NULL</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">인덱스 분석 및 모니터링</h3>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 인덱스 사용 통계 확인</div>
              <div>CALL db.index.usage("user_email_idx")</div>
              <div>YIELD count, trackedSince</div>
              <div>RETURN count AS uses, trackedSince</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 사용되지 않는 인덱스 찾기</div>
              <div>SHOW INDEXES YIELD name, state, populationPercent</div>
              <div>WHERE populationPercent < 100 OR state <> 'ONLINE'</div>
              <div>RETURN name, state, populationPercent</div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">💡 인덱스 베스트 프랙티스</h4>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              • 빈번히 검색되는 속성에 우선 인덱스 생성
              • 카디널리티가 높은 속성은 인덱스 효과 제한적
              • 복합 인덱스는 쿼리 패턴에 맞춰 순서 고려
              • 주기적으로 인덱스 사용율 모니터링
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 쿼리 튜닝 기법</h2>
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">성능을 극대화하는 고급 쿼리 최적화 기법</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">필터링 최적화</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-red-600 dark:text-red-400">// ❌ 비효율적</div>
                <div>MATCH (p:Person)</div>
                <div>WHERE toInteger(p.age) > 30</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// ✅ 효율적</div>
                <div>MATCH (p:Person)</div>
                <div>WHERE p.age > 30</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// ✅ 인덱스 활용</div>
                <div>MATCH (p:Person)</div>
                <div>USING INDEX p:Person(age)</div>
                <div>WHERE p.age > 30</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">패턴 최적화</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-red-600 dark:text-red-400">// ❌ 카티션 곱</div>
                <div>MATCH (a), (b)</div>
                <div>WHERE a.type = 'A' AND b.type = 'B'</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// ✅ 필터 먼저</div>
                <div>MATCH (a {type: 'A'})</div>
                <div>MATCH (b {type: 'B'})</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// ✅ 패턴 사용</div>
                <div>MATCH (a:TypeA)-[:CONNECTS]-(b:TypeB)</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">메모리 최적화</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// WITH로 중간 결과 정리</div>
                <div>MATCH (p:Person)-[:KNOWS]-(friend)</div>
                <div>WITH p, count(friend) AS friendCount</div>
                <div>WHERE friendCount > 100</div>
                <div>MATCH (p)-[:WORKS_AT]->(c:Company)</div>
                <div>RETURN p.name, c.name, friendCount</div>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// LIMIT로 조기 종료</div>
                <div>MATCH (p:Person)</div>
                <div>WHERE p.age > 25</div>
                <div>WITH p LIMIT 1000</div>
                <div>MATCH (p)-[:BOUGHT]->(prod:Product)</div>
                <div>RETURN p, collect(prod) AS products</div>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">🎯 쿼리 튜닝 체크리스트</h4>
            <div className="text-sm text-green-800 dark:text-green-300 space-y-1">
              <div>✅ 레이블 사용으로 타입 필터링</div>
              <div>✅ WHERE 절 최소화와 인덱스 활용</div>
              <div>✅ WITH로 중간 결과 크기 제한</div>
              <div>✅ 불필요한 OPTIONAL MATCH 회피</div>
              <div>✅ 배치 처리로 메모리 부하 분산</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏛️ 클러스터 구성과 확장</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">고가용성과 확장성을 위한 클러스터 아키텍처</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">Causal Cluster 구성</h4>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="text-center p-3 bg-purple-100 dark:bg-purple-900/30 rounded">
                <div className="font-semibold">Core Servers</div>
                <div className="text-xs mt-1">• Raft 프로토콜</div>
                <div className="text-xs">• 데이터 일관성 보장</div>
                <div className="text-xs">• 최소 3대 권장</div>
              </div>
              <div className="text-center p-3 bg-pink-100 dark:bg-pink-900/30 rounded">
                <div className="font-semibold">Read Replicas</div>
                <div className="text-xs mt-1">• 읽기 부하 분산</div>
                <div className="text-xs">• 비동기 복제</div>
                <div className="text-xs">• 지역별 배치 가능</div>
              </div>
              <div className="text-center p-3 bg-indigo-100 dark:bg-indigo-900/30 rounded">
                <div className="font-semibold">Load Balancer</div>
                <div className="text-xs mt-1">• 트래픽 분배</div>
                <div className="text-xs">• 헬스 체크</div>
                <div className="text-xs">• 자동 페일오버</div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">클러스터 설정</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400"># neo4j.conf</div>
                <div>dbms.mode=CORE</div>
                <div>dbms.default_listen_address=0.0.0.0</div>
                <div></div>
                <div># 클러스터 멤버</div>
                <div>dbms.cluster.discovery.endpoints=</div>
                <div>  server1:5000,server2:5000,server3:5000</div>
                <div></div>
                <div># 메모리 설정</div>
                <div>dbms.memory.heap.max_size=32G</div>
                <div>dbms.memory.pagecache.size=64G</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">모니터링</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 클러스터 상태 확인</div>
                <div>CALL dbms.cluster.overview()</div>
                <div>YIELD id, addresses, role, groups</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 리더 확인</div>
                <div>CALL dbms.cluster.role()</div>
                <div>YIELD role</div>
                <div>RETURN role</div>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🔄 클러스터 운영 팁</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300 space-y-1">
              <div>• 코어 서버는 홀수개로 구성 (3, 5, 7)</div>
              <div>• 네트워크 대역폭과 레이턴시 최적화</div>
              <div>• 정기적인 백업과 복구 테스트</div>
              <div>• 모니터링 대시보드 구축</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔒 보안과 권한 관리</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">엔터프라이즈급 보안 설정</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">사용자 및 역할 관리</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 사용자 생성</div>
                <div>CREATE USER alice</div>
                <div>SET PASSWORD 'secure_password'</div>
                <div>CHANGE NOT REQUIRED</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 역할 할당</div>
                <div>GRANT ROLE reader TO alice</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 세부 권한</div>
                <div>GRANT MATCH {*} ON GRAPH * TO reader</div>
                <div>DENY DELETE ON GRAPH * TO reader</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">암호화 설정</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400"># SSL/TLS 설정</div>
                <div>dbms.ssl.policy.default.enabled=true</div>
                <div>dbms.ssl.policy.default.base_directory=</div>
                <div>  certificates/default</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400"># 데이터 암호화</div>
                <div>dbms.directories.data=data</div>
                <div>dbms.security.auth_enabled=true</div>
                <div>dbms.security.encryption_level=REQUIRED</div>
              </div>
            </div>
          </div>

          <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">🔐 보안 체크리스트</h4>
            <div className="text-sm text-red-800 dark:text-red-300 space-y-1">
              <div>✅ 기본 비밀번호 즉시 변경</div>
              <div>✅ SSL/TLS 통신 활성화</div>
              <div>✅ 역할 기반 접근 제어 (RBAC)</div>
              <div>✅ 감사 로그 활성화</div>
              <div>✅ 정기적인 보안 패치</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📦 백업과 복구</h2>
        <div className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/20 dark:to-slate-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">데이터 보호와 비즈니스 연속성 보장</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-3">백업 전략</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400"># 온라인 백업</div>
                <div>neo4j-admin backup \</div>
                <div>  --backup-dir=/backup/2024-01-15 \</div>
                <div>  --name=daily-backup \</div>
                <div>  --from=localhost:6362</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400"># 증분 백업</div>
                <div>neo4j-admin backup \</div>
                <div>  --backup-dir=/backup/incremental \</div>
                <div>  --name=inc-backup \</div>
                <div>  --incremental</div>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400"># 복구</div>
                <div>neo4j-admin restore \</div>
                <div>  --from=/backup/2024-01-15 \</div>
                <div>  --database=neo4j \</div>
                <div>  --force</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400"># 검증</div>
                <div>neo4j-admin check-consistency \</div>
                <div>  --database=neo4j \</div>
                <div>  --verbose</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-100 dark:bg-slate-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-2">📅 백업 베스트 프랙티스</h4>
            <div className="text-sm text-slate-800 dark:text-slate-300 space-y-1">
              <div>• 3-2-1 규칙: 3개 복사본, 2개 다른 미디어, 1개 오프사이트</div>
              <div>• 정기적인 복구 테스트 수행</div>
              <div>• 백업 암호화 및 안전한 저장소</div>
              <div>• 자동화된 백업 스케줄링</div>
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
              <span><strong>인덱스 전략:</strong> 단일/복합 인덱스로 쿼리 속도 향상</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>쿼리 튜닝:</strong> 필터링, 패턴, 메모리 최적화 기법</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>클러스터 구성:</strong> Causal Cluster로 고가용성 확보</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>보안 설정:</strong> RBAC, SSL/TLS, 암호화로 데이터 보호</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>백업/복구:</strong> 정기 백업과 검증으로 비즈니스 연속성</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}