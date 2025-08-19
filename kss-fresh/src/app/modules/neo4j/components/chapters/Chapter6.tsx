'use client';

import React from 'react';
import { Database, Network, Zap, Globe } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">KSS 도메인 통합 🌐</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Knowledge Space Simulator의 다양한 도메인 데이터를 
          Neo4j 그래프로 통합하여 지식의 연결성을 탐험하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 KSS 지식 그래프 아키텍처</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">모든 학습 도메인을 하나의 그래프로</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3">통합 노드 타입</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 1. 도메인 노드</div>
              <div>(:Domain {name: 'Data Science', modules: 12})</div>
              <div>(:Domain {name: 'AI/ML', modules: 8})</div>
              <div>(:Domain {name: 'Blockchain', modules: 5})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 2. 모듈 노드</div>
              <div>(:Module {name: 'Deep Learning', difficulty: 'Advanced'})</div>
              <div>(:Module {name: 'Smart Contracts', difficulty: 'Intermediate'})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 3. 개념 노드</div>
              <div>(:Concept {name: 'Neural Network', category: 'Algorithm'})</div>
              <div>(:Concept {name: 'Consensus', category: 'Protocol'})</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 4. 학습자 노드</div>
              <div>(:Learner {id: 'user123', level: 'Expert'})</div>
              <div>(:Progress {module: 'Deep Learning', completion: 85})</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">도메인 간 관계</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 선수 지식 관계</div>
                <div>(ml:Module)-[:REQUIRES]->(stat:Module)</div>
                <div>(dl:Module)-[:REQUIRES]->(ml:Module)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 연관 개념 관계</div>
                <div>(nn:Concept)-[:RELATED_TO]->(dl:Concept)</div>
                <div>(blockchain:Concept)-[:USES]->(crypto:Concept)</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 학습 경로 관계</div>
                <div>(learner)-[:COMPLETED]->(module)</div>
                <div>(learner)-[:NEXT_RECOMMENDED]->(module)</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">메타데이터 활용</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 학습 분석 속성</div>
                <div>MATCH (l:Learner)-[r:STUDIED]->(m:Module)</div>
                <div>SET r.duration = 3600,</div>
                <div>    r.score = 92,</div>
                <div>    r.timestamp = datetime()</div>
                <div></div>
                <div className="text-green-600 dark:text-green-400">// 난이도 가중치</div>
                <div>MATCH (m1:Module)-[r:REQUIRES]->(m2:Module)</div>
                <div>SET r.weight = CASE</div>
                <div>  WHEN m2.difficulty = 'Advanced' THEN 0.8</div>
                <div>  WHEN m2.difficulty = 'Intermediate' THEN 0.5</div>
                <div>  ELSE 0.3 END</div>
              </div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">💡 통합의 가치</h4>
            <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
              <li>• 도메인 간 지식 연결성 시각화</li>
              <li>• 개인화된 학습 경로 자동 생성</li>
              <li>• 선수 지식 갭 자동 탐지</li>
              <li>• 학습 커뮤니티 네트워크 분석</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 지능형 학습 추천 시스템</h2>
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">그래프 기반 개인화 추천 알고리즘</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">협업 필터링 + 콘텐츠 기반</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">// 유사 학습자 찾기</div>
              <div>MATCH (me:Learner {id: $userId})</div>
              <div>      -[:COMPLETED]->(m:Module)</div>
              <div>      <-[:COMPLETED]-(other:Learner)</div>
              <div>WITH me, other, COUNT(m) AS shared</div>
              <div>WHERE shared > 5</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 추천 모듈 도출</div>
              <div>MATCH (other)-[:COMPLETED]->(rec:Module)</div>
              <div>WHERE NOT EXISTS((me)-[:COMPLETED]->(rec))</div>
              <div>WITH rec, COUNT(other) AS popularity</div>
              <div></div>
              <div className="text-green-600 dark:text-green-400">// 선수 지식 체크</div>
              <div>MATCH (rec)-[:REQUIRES*]->(prereq:Module)</div>
              <div>WHERE NOT EXISTS((me)-[:COMPLETED]->(prereq))</div>
              <div>WITH rec, popularity, COLLECT(prereq) AS missing</div>
              <div></div>
              <div>RETURN rec.name AS recommendation,</div>
              <div>       popularity AS score,</div>
              <div>       SIZE(missing) AS prerequisites_needed</div>
              <div>ORDER BY score DESC, prerequisites_needed ASC</div>
              <div>LIMIT 5</div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">학습 경로 최적화</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 목표까지 최단 경로</div>
                <div>MATCH (me:Learner {id: $userId})</div>
                <div>MATCH (goal:Module {name: $target})</div>
                <div>MATCH path = shortestPath(</div>
                <div>  (me)-[:COMPLETED|REQUIRES*]-(goal)</div>
                <div>)</div>
                <div>WITH [n IN nodes(path) WHERE</div>
                <div>  n:Module AND NOT EXISTS(</div>
                <div>    (me)-[:COMPLETED]->(n)</div>
                <div>  )] AS todo</div>
                <div>RETURN todo</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">학습 성과 예측</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// ML 피처 추출</div>
                <div>MATCH (l:Learner)-[r:STUDIED]->(m:Module)</div>
                <div>WITH l, </div>
                <div>  AVG(r.score) AS avg_score,</div>
                <div>  COUNT(m) AS modules_done,</div>
                <div>  SUM(r.duration) AS total_time</div>
                <div>MATCH (l)-[:INTERESTED_IN]->(c:Concept)</div>
                <div>WITH l, avg_score, modules_done,</div>
                <div>     total_time, COUNT(c) AS interests</div>
                <div>RETURN l.id, avg_score, modules_done,</div>
                <div>       total_time/modules_done AS pace,</div>
                <div>       interests AS diversity</div>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">🎓 추천 시스템 특징</h4>
            <div className="text-sm text-green-800 dark:text-green-300 space-y-1">
              <div>• 실시간 학습 패턴 분석</div>
              <div>• 난이도 적응형 추천</div>
              <div>• 선수 지식 자동 보완</div>
              <div>• 동료 학습자 매칭</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 학습 분석 대시보드</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">실시간 학습 인사이트</h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">개인 통계</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 학습 현황</div>
                <div>MATCH (me:Learner {id: $userId})</div>
                <div>MATCH (me)-[:COMPLETED]->(m:Module)</div>
                <div>RETURN COUNT(m) AS completed,</div>
                <div>  SUM(m.credits) AS total_credits,</div>
                <div>  COLLECT(DISTINCT m.domain) AS domains</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-3">커뮤니티 랭킹</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 상위 학습자</div>
                <div>MATCH (l:Learner)-[:COMPLETED]->(m)</div>
                <div>WITH l, COUNT(m) AS modules,</div>
                <div>     AVG(m.difficulty) AS avg_diff</div>
                <div>RETURN l.name, modules,</div>
                <div>  modules * avg_diff AS score</div>
                <div>ORDER BY score DESC LIMIT 10</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">트렌드 분석</h4>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
                <div className="text-green-600 dark:text-green-400">// 인기 모듈</div>
                <div>MATCH (m:Module)<-[r:STUDYING]-()</div>
                <div>WHERE r.timestamp > </div>
                <div>  datetime() - duration('P7D')</div>
                <div>RETURN m.name,</div>
                <div>  COUNT(r) AS weekly_learners</div>
                <div>ORDER BY weekly_learners DESC</div>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">📈 분석 가능 지표</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-purple-800 dark:text-purple-300">
              <ul className="space-y-1">
                <li>• 학습 속도 및 패턴 분석</li>
                <li>• 강점/약점 도메인 파악</li>
                <li>• 최적 학습 시간대 분석</li>
              </ul>
              <ul className="space-y-1">
                <li>• 동료 학습자 비교 분석</li>
                <li>• 목표 달성 예측</li>
                <li>• 학습 효율성 점수</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🛠️ KSS 통합 실습</h2>
        <div className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/20 dark:to-slate-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">직접 구현해보는 KSS 지식 그래프</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-3">전체 도메인 임포트</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono overflow-x-auto">
              <div className="text-green-600 dark:text-green-400">// KSS 전체 구조 생성</div>
              <div>CREATE</div>
              <div>// 도메인 생성</div>
              <div>(ai:Domain {name: 'AI/ML', color: '#3B82F6'}),</div>
              <div>(data:Domain {name: 'Data Science', color: '#10B981'}),</div>
              <div>(blockchain:Domain {name: 'Blockchain', color: '#F59E0B'}),</div>
              <div>(quantum:Domain {name: 'Quantum Computing', color: '#8B5CF6'}),</div>
              <div></div>
              <div>// 모듈 생성</div>
              <div>(dl:Module {name: 'Deep Learning', domain: 'AI/ML', difficulty: 3}),</div>
              <div>(ml:Module {name: 'Machine Learning', domain: 'AI/ML', difficulty: 2}),</div>
              <div>(stats:Module {name: 'Statistics', domain: 'Data Science', difficulty: 1}),</div>
              <div>(smart:Module {name: 'Smart Contracts', domain: 'Blockchain', difficulty: 2}),</div>
              <div></div>
              <div>// 관계 생성</div>
              <div>(ai)-[:CONTAINS]->(dl),</div>
              <div>(ai)-[:CONTAINS]->(ml),</div>
              <div>(data)-[:CONTAINS]->(stats),</div>
              <div>(blockchain)-[:CONTAINS]->(smart),</div>
              <div>(dl)-[:REQUIRES]->(ml),</div>
              <div>(ml)-[:REQUIRES]->(stats)</div>
            </div>
          </div>

          <div className="bg-slate-100 dark:bg-slate-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-2">💡 실습 과제</h4>
            <ol className="text-sm text-slate-800 dark:text-slate-300 space-y-2">
              <li>1. 학습자 프로필과 진도 데이터 추가</li>
              <li>2. 도메인 간 연결 관계 탐색</li>
              <li>3. 개인화 추천 쿼리 작성</li>
              <li>4. 학습 네트워크 시각화</li>
            </ol>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 오늘 배운 것 정리</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>통합 아키텍처:</strong> 모든 KSS 도메인을 하나의 그래프로 연결</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>추천 시스템:</strong> 협업 필터링과 그래프 알고리즘 결합</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>학습 분석:</strong> 실시간 인사이트와 예측 모델</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>실전 활용:</strong> KSS 플랫폼에 즉시 적용 가능한 그래프 모델</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}