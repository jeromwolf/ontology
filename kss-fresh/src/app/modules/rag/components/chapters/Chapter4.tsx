'use client';

import VectorSearchDemo from '../VectorSearchDemo';
import References from '@/components/common/References';

// Chapter 4: Vector Search
export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색의 원리</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          벡터 검색은 고차원 공간에서 가장 가까운 이웃을 찾는 과정입니다.
          쿼리 벡터와 문서 벡터 간의 거리를 계산하여 가장 유사한 문서를 찾습니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색 실습</h2>
        <VectorSearchDemo />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 벡터 데이터베이스 비교</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Pinecone</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 완전 관리형 서비스</li>
              <li>✓ 실시간 업데이트</li>
              <li>✓ 하이브리드 검색 지원</li>
              <li>✗ 클라우드 전용</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Weaviate</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 오픈소스</li>
              <li>✓ GraphQL API</li>
              <li>✓ 모듈식 아키텍처</li>
              <li>✓ 온프레미스 가능</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Chroma</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 경량 임베디드 DB</li>
              <li>✓ Python 네이티브</li>
              <li>✓ 개발자 친화적</li>
              <li>✗ 대규모 확장 제한</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Qdrant</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ Rust 기반 고성능</li>
              <li>✓ 필터링 기능 강력</li>
              <li>✓ 클라우드 & 온프레미스</li>
              <li>✓ gRPC API</li>
            </ul>
          </div>
        </div>
      </section>

      {/* GraphDB vs VectorDB 비교 섹션 */}
      <section>
        <h2 className="text-2xl font-bold mb-4">GraphDB vs VectorDB: 언제 무엇을 사용할까?</h2>
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            RAG 시스템 구축 시 <strong>GraphDB(그래프 데이터베이스)</strong>와 <strong>VectorDB(벡터 데이터베이스)</strong> 중 어떤 것을 선택해야 할까요?
            각각의 강점을 이해하고 상황에 맞게 선택하는 것이 중요합니다.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* GraphDB 카드 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg border-2 border-purple-200 dark:border-purple-700">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center text-2xl">
                🕸️
              </div>
              <div>
                <h3 className="font-bold text-xl text-gray-900 dark:text-white">GraphDB (Neo4j)</h3>
                <p className="text-sm text-purple-600 dark:text-purple-400">관계 중심 데이터베이스</p>
              </div>
            </div>

            <div className="space-y-3 mb-4">
              <div>
                <h4 className="font-semibold text-emerald-700 dark:text-emerald-400 mb-2">✅ 강점</h4>
                <ul className="space-y-1.5 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>복잡한 관계 추론</strong>: 엔티티 간 다단계 관계 쿼리 (A → B → C)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>온톨로지 구조</strong>: 도메인 지식을 명시적으로 모델링</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>Cypher 쿼리</strong>: 직관적인 그래프 쿼리 언어</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>논리 기반 추론</strong>: 규칙 기반 추론 엔진 활용</span>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ 약점</h4>
                <ul className="space-y-1.5 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>데이터 전처리 복잡</strong>: 엔티티 추출, 관계 정의 필요</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>유사도 검색 약함</strong>: 의미적 유사성 계산 어려움</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>Cypher 학습 곡선</strong>: SQL과 다른 쿼리 패러다임</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>대규모 데이터 제한</strong>: 수천만 노드 이상 시 성능 저하</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
              <p className="text-xs font-semibold text-purple-800 dark:text-purple-300 mb-1">💡 최적 사용 사례</p>
              <p className="text-xs text-gray-700 dark:text-gray-400">
                지식 그래프, 의료/금융 도메인, 법률 문서, 추론 기반 QA
              </p>
            </div>
          </div>

          {/* VectorDB 카드 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg border-2 border-blue-200 dark:border-blue-700">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center text-2xl">
                🔢
              </div>
              <div>
                <h3 className="font-bold text-xl text-gray-900 dark:text-white">VectorDB (Pinecone/Chroma)</h3>
                <p className="text-sm text-blue-600 dark:text-blue-400">유사도 기반 데이터베이스</p>
              </div>
            </div>

            <div className="space-y-3 mb-4">
              <div>
                <h4 className="font-semibold text-emerald-700 dark:text-emerald-400 mb-2">✅ 강점</h4>
                <ul className="space-y-1.5 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>의미적 유사도 검색</strong>: 코사인 유사도로 관련 문서 찾기</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>대량 데이터 처리</strong>: 수억~수십억 벡터 지원</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>간단한 전처리</strong>: 임베딩만 생성하면 끝</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 mt-0.5">▸</span>
                    <span><strong>빠른 검색 속도</strong>: HNSW 알고리즘으로 ms 단위 응답</span>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ 약점</h4>
                <ul className="space-y-1.5 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>복잡한 추론 불가</strong>: 다단계 관계 쿼리 어려움</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>관계 정보 부족</strong>: 엔티티 간 연결 표현 제한</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>블랙박스 검색</strong>: 왜 이 문서가 검색됐는지 설명 어려움</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-0.5">▸</span>
                    <span><strong>임베딩 모델 의존</strong>: 모델 품질이 검색 품질 결정</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
              <p className="text-xs font-semibold text-blue-800 dark:text-blue-300 mb-1">💡 최적 사용 사례</p>
              <p className="text-xs text-gray-700 dark:text-gray-400">
                일반 문서 검색, 챗봇, 추천 시스템, 대규모 콘텐츠 라이브러리
              </p>
            </div>
          </div>
        </div>

        {/* 비교 표 */}
        <div className="overflow-x-auto mb-6">
          <table className="w-full bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-sm">
            <thead className="bg-gray-100 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white">비교 항목</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-purple-700 dark:text-purple-400">GraphDB (Neo4j)</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-blue-700 dark:text-blue-400">VectorDB (Pinecone)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">검색 방식</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">Cypher 쿼리 (명시적)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">벡터 유사도 (암묵적)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">확장성</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">수백만 노드 (수직 확장)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">수십억 벡터 (수평 확장)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">데이터 준비</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">복잡 (엔티티/관계 추출)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">간단 (임베딩만 생성)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">쿼리 복잡도</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">높음 (Cypher 학습 필요)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">낮음 (간단한 API)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">비용 (클라우드)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">높음 ($500+/월)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">중간 ($100~$300/월)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">설명 가능성</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">높음 (경로 추적 가능)</td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">낮음 (블랙박스)</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* GraphRAG: 하이브리드 접근 */}
        <div className="bg-gradient-to-r from-purple-100 via-pink-100 to-blue-100 dark:from-purple-900/30 dark:via-pink-900/30 dark:to-blue-900/30 rounded-xl p-6 border-2 border-purple-300 dark:border-purple-600">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-white dark:bg-gray-800 rounded-lg flex items-center justify-center text-2xl flex-shrink-0">
              🚀
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">GraphRAG: 두 세계의 장점을 결합</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                <strong>GraphRAG</strong>는 GraphDB의 추론 능력과 VectorDB의 유사도 검색을 결합한 하이브리드 접근법입니다.
              </p>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 dark:text-purple-400 mt-0.5 font-bold">1단계:</span>
                  <span><strong>VectorDB로 후보 검색</strong> - 의미적으로 관련된 문서 찾기</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-pink-600 dark:text-pink-400 mt-0.5 font-bold">2단계:</span>
                  <span><strong>GraphDB로 관계 확장</strong> - 엔티티 간 연결 탐색</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-0.5 font-bold">3단계:</span>
                  <span><strong>LLM으로 답변 생성</strong> - 풍부한 컨텍스트 활용</span>
                </li>
              </ul>
              <div className="mt-4 pt-4 border-t border-purple-300 dark:border-purple-700">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  💡 <strong>실습</strong>: 아래 <strong>GraphRAG Explorer 시뮬레이터</strong>에서 하이브리드 검색을 직접 체험해보세요!
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 실전 문제 사례 */}
        <div className="mt-6 bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 rounded-r-lg p-4">
          <h4 className="font-semibold text-amber-800 dark:text-amber-300 mb-2 flex items-center gap-2">
            <span>⚠️</span>
            <span>실전 문제 사례: Neo4j + Pandas 통합</span>
          </h4>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
            Neo4j와 Pandas를 함께 사용할 때 <strong>접계점(join point) 불일치</strong> 문제가 발생할 수 있습니다:
          </p>
          <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400 ml-4">
            <li>• Neo4j는 노드 ID 기반, Pandas는 컬럼 기반 조인</li>
            <li>• 데이터 타입 불일치로 누락 발생 (문자열 vs 정수)</li>
            <li>• 대량 데이터 전송 시 메모리 부족</li>
          </ul>
          <p className="text-sm text-emerald-700 dark:text-emerald-400 mt-2">
            <strong>해결책</strong>: Neo4j의 <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">APOC</code> 라이브러리 사용,
            또는 VectorDB로 마이그레이션 검토
          </p>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-blue-800 dark:text-blue-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">벡터 검색의 원리와 거리 측정 방법 (코사인 유사도, 유클리드)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">주요 벡터 데이터베이스 비교 (Pinecone, Weaviate, Chroma, Qdrant)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>GraphDB vs VectorDB</strong> 선택 가이드 - 관계 추론 vs 유사도 검색
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>GraphRAG</strong> 하이브리드 접근법 - 두 세계의 장점 결합
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">하이브리드 검색 (벡터 + 키워드) 전략</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">인덱싱 알고리즘과 검색 성능 최적화</span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 벡터 검색 핵심 논문 (Vector Search Papers)',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Efficient and Robust Approximate Nearest Neighbor Search Using HNSW',
                authors: 'Yu. A. Malkov, D. A. Yashunin',
                year: '2018',
                description: '현대 벡터 검색의 표준 - Hierarchical Navigable Small World 알고리즘',
                link: 'https://arxiv.org/abs/1603.09320'
              },
              {
                title: 'FAISS: A Library for Efficient Similarity Search',
                authors: 'Jeff Johnson, Matthijs Douze, Hervé Jégou (Meta AI)',
                year: '2019',
                description: 'Meta의 오픈소스 벡터 검색 라이브러리 - 10억개 벡터 검색',
                link: 'https://arxiv.org/abs/1702.08734'
              },
              {
                title: 'ScaNN: Efficient Vector Similarity Search',
                authors: 'Google Research',
                year: '2020',
                description: 'Google의 초고속 벡터 검색 알고리즘',
                link: 'https://arxiv.org/abs/1908.10396'
              },
              {
                title: 'Hybrid Search: Combining BM25 and Vector Search',
                authors: 'Various Contributors',
                year: '2023',
                description: '키워드 검색과 벡터 검색의 최적 결합 방법',
                link: 'https://www.pinecone.io/learn/hybrid-search-intro/'
              }
            ]
          },
          {
            title: '🛠️ 벡터 데이터베이스 (Vector Databases)',
            icon: 'tools',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Pinecone',
                description: '완전 관리형 벡터 DB - 실시간 업데이트, 하이브리드 검색 (무료 티어 有)',
                link: 'https://www.pinecone.io/'
              },
              {
                title: 'Weaviate',
                description: '오픈소스 벡터 DB - GraphQL API, 모듈식 아키텍처',
                link: 'https://weaviate.io/'
              },
              {
                title: 'Qdrant',
                description: 'Rust 기반 고성능 벡터 DB - 강력한 필터링, gRPC API',
                link: 'https://qdrant.tech/'
              },
              {
                title: 'Chroma',
                description: '경량 임베디드 벡터 DB - Python 네이티브, 개발자 친화적',
                link: 'https://www.trychroma.com/'
              },
              {
                title: 'Milvus',
                description: '대규모 벡터 검색 - 수십억 벡터 지원, 클라우드 네이티브',
                link: 'https://milvus.io/'
              },
              {
                title: 'pgvector',
                description: 'PostgreSQL용 벡터 확장 - 기존 DB에 벡터 검색 추가',
                link: 'https://github.com/pgvector/pgvector'
              }
            ]
          },
          {
            title: '📖 벡터 DB 선택 가이드 (Selection Guides)',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Vector Database Comparison',
                description: '12개 벡터 DB 성능/비용/기능 종합 비교',
                link: 'https://superlinked.com/vector-db-comparison/'
              },
              {
                title: 'Pinecone vs Weaviate vs Qdrant',
                description: '3대 벡터 DB 벤치마크 (속도, 정확도, 비용)',
                link: 'https://weaviate.io/blog/vector-database-benchmarks'
              },
              {
                title: 'Choosing a Vector Database',
                description: '사용 사례별 최적 벡터 DB 선택 전략',
                link: 'https://www.datastax.com/guides/what-is-a-vector-database'
              },
              {
                title: 'LangChain Vector Store Integration',
                description: '40+ 벡터 DB 통합 코드 예제',
                link: 'https://python.langchain.com/docs/integrations/vectorstores/'
              }
            ]
          },
          {
            title: '⚡ 성능 최적화 (Performance Optimization)',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'HNSW vs IVF vs Product Quantization',
                description: '인덱싱 알고리즘 성능 비교 및 튜닝 가이드',
                link: 'https://www.pinecone.io/learn/series/faiss/vector-indexes/'
              },
              {
                title: 'Hybrid Search Implementation',
                description: 'BM25 + 벡터 검색 결합으로 정확도 40% 향상',
                link: 'https://qdrant.tech/articles/hybrid-search/'
              },
              {
                title: 'Query Optimization Strategies',
                description: 'Top-K 검색, Re-ranking, 필터링 최적화',
                link: 'https://weaviate.io/developers/weaviate/search'
              }
            ]
          }
        ]}
      />
    </div>
  )
}