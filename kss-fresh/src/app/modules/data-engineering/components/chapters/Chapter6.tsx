'use client'

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">데이터 모델링과 웨어하우징 🏗️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Kimball vs Inmon, Star Schema, Data Vault 2.0으로 최적의 데이터 구조를 설계해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 데이터 웨어하우스 개요</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            데이터 웨어하우스는 의사결정을 지원하기 위해 여러 소스의 데이터를 통합, 정리, 저장하는 시스템입니다.
            OLTP(Online Transaction Processing)와는 다른 OLAP(Online Analytical Processing) 목적으로 설계됩니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">OLTP (운영계)</h3>
              <ul className="text-sm space-y-1">
                <li>• 일상적인 업무 처리</li>
                <li>• 짧고 빈번한 트랜잭션</li>
                <li>• 정규화된 구조</li>
                <li>• 높은 일관성 (ACID)</li>
                <li>• 예: 주문 처리, 계정 관리</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">OLAP (분석계)</h3>
              <ul className="text-sm space-y-1">
                <li>• 의사결정 지원 분석</li>
                <li>• 복잡한 집계 쿼리</li>
                <li>• 비정규화된 구조</li>
                <li>• 읽기 최적화</li>
                <li>• 예: 매출 분석, 트렌드 예측</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏛️ Kimball vs Inmon: 두 가지 접근법</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">Kimball 방법론 (Bottom-Up)</h3>
            
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">핵심 특징</h4>
                <ul className="text-xs space-y-1">
                  <li>• 비즈니스 프로세스 중심</li>
                  <li>• 차원 모델링 (Dimensional Modeling)</li>
                  <li>• 빠른 구현과 ROI</li>
                  <li>• 부서별 데이터 마트</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">장점</h4>
                <ul className="text-xs space-y-1 text-green-600">
                  <li>✅ 빠른 개발과 배포</li>
                  <li>✅ 사용자 친화적</li>
                  <li>✅ 비교적 저렴한 비용</li>
                  <li>✅ 점진적 확장 가능</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">단점</h4>
                <ul className="text-xs space-y-1 text-red-600">
                  <li>❌ 데이터 불일치 가능성</li>
                  <li>❌ 전사적 관점 부족</li>
                  <li>❌ ETL 로직 중복</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">Inmon 방법론 (Top-Down)</h3>
            
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">핵심 특징</h4>
                <ul className="text-xs space-y-1">
                  <li>• 전사적 데이터 모델</li>
                  <li>• 정규화된 구조</li>
                  <li>• 단일 진실 공급원 (SSOT)</li>
                  <li>• 중앙집중식 EDW</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">장점</h4>
                <ul className="text-xs space-y-1 text-green-600">
                  <li>✅ 데이터 일관성 보장</li>
                  <li>✅ 전사적 통합 관점</li>
                  <li>✅ 확장성과 유연성</li>
                  <li>✅ 중복 데이터 최소화</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-2">단점</h4>
                <ul className="text-xs space-y-1 text-red-600">
                  <li>❌ 긴 개발 기간</li>
                  <li>❌ 높은 초기 비용</li>
                  <li>❌ 복잡한 쿼리</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⭐ Star Schema: 차원 모델링의 핵심</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">별 모양 스키마 구조</h3>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">팩트 테이블 (Fact Table)</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>역할:</strong> 비즈니스 이벤트/측정값 저장</div>
                <div>• <strong>구조:</strong> 외래키 + 측정값</div>
                <div>• <strong>특징:</strong> 좁고 길며, 자주 삽입됨</div>
                <div>• <strong>예시:</strong> 매출액, 수량, 주문건수</div>
              </div>
              
              <div className="mt-3 p-2 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">
                <div>sales_fact</div>
                <div>├── date_key (FK)</div>
                <div>├── product_key (FK)</div>
                <div>├── customer_key (FK)</div>
                <div>├── sales_amount</div>
                <div>└── quantity</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-yellow-600 dark:text-yellow-400 mb-3">차원 테이블 (Dimension Table)</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>역할:</strong> 맥락과 설명 정보 제공</div>
                <div>• <strong>구조:</strong> 기본키 + 속성들</div>
                <div>• <strong>특징:</strong> 넓고 짧으며, 상대적으로 안정</div>
                <div>• <strong>예시:</strong> 시간, 제품, 고객, 지역</div>
              </div>
              
              <div className="mt-3 p-2 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">
                <div>product_dim</div>
                <div>├── product_key (PK)</div>
                <div>├── product_name</div>
                <div>├── category</div>
                <div>├── brand</div>
                <div>└── price</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Star Schema의 장점</h4>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="text-2xl mb-2">🚀</div>
                <div className="font-semibold">빠른 쿼리 성능</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">적은 조인, 효율적 집계</div>
              </div>
              <div className="text-center">
                <div className="text-2xl mb-2">🧠</div>
                <div className="font-semibold">직관적 구조</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">비즈니스 사용자 친화적</div>
              </div>
              <div className="text-center">
                <div className="text-2xl mb-2">🔧</div>
                <div className="font-semibold">BI 도구 최적화</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">대부분 도구에서 지원</div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">🎯 Snowflake Schema</h4>
            <p className="text-sm text-yellow-800 dark:text-yellow-300">
              차원 테이블을 추가로 정규화한 구조. 저장 공간은 절약되지만 
              조인이 복잡해져서 일반적으로 Star Schema를 더 선호합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏦 Data Vault 2.0: 차세대 모델링</h2>
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Data Vault의 3가지 구성 요소</h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">Hub</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>역할:</strong> 비즈니스 키 저장</div>
                <div>• <strong>구조:</strong> 해시키 + 비즈니스키 + 메타데이터</div>
                <div>• <strong>특징:</strong> 불변, 고유 식별자</div>
                <div>• <strong>예시:</strong> Customer Hub, Product Hub</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-2">Link</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>역할:</strong> Hub 간의 관계</div>
                <div>• <strong>구조:</strong> 해시키 + Hub 키들 + 메타데이터</div>
                <div>• <strong>특징:</strong> 다대다 관계 지원</div>
                <div>• <strong>예시:</strong> Customer-Product Link</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-2">Satellite</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>역할:</strong> 상세 속성과 이력</div>
                <div>• <strong>구조:</strong> 부모키 + 속성들 + 시간정보</div>
                <div>• <strong>특징:</strong> 변경 이력 추적</div>
                <div>• <strong>예시:</strong> Customer Details</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Data Vault의 장점</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <ul className="space-y-1">
                  <li className="text-green-600">✅ 높은 유연성과 확장성</li>
                  <li className="text-green-600">✅ 감사 추적성 (Auditability)</li>
                  <li className="text-green-600">✅ 병렬 로딩 가능</li>
                  <li className="text-green-600">✅ 소스 시스템 변경에 강함</li>
                </ul>
              </div>
              <div>
                <ul className="space-y-1">
                  <li className="text-red-600">❌ 복잡한 쿼리</li>
                  <li className="text-red-600">❌ 높은 저장 비용</li>
                  <li className="text-red-600">❌ BI 도구 최적화 필요</li>
                  <li className="text-red-600">❌ 학습 곡선 존재</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🎯 Data Vault를 언제 사용할까?</h4>
            <p className="text-sm text-purple-800 dark:text-purple-300">
              높은 규제 요구사항, 빈번한 소스 변경, 복잡한 비즈니스 관계가 있는 
              대기업 환경에서 특히 유용합니다. 금융, 통신, 제약 업계에서 많이 사용됩니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 모델링 방법론 선택 가이드</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h3 className="font-semibold text-green-600 mb-2">Kimball + Star Schema</h3>
              <div className="text-sm space-y-1">
                <div><strong>적합한 경우:</strong></div>
                <div>• 빠른 BI 구축</div>
                <div>• 명확한 비즈니스 프로세스</div>
                <div>• 중소규모 프로젝트</div>
                <div>• 셀프서비스 분석 중심</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 mb-2">Inmon + 정규화</h3>
              <div className="text-sm space-y-1">
                <div><strong>적합한 경우:</strong></div>
                <div>• 전사적 데이터 통합</div>
                <div>• 높은 데이터 일관성 요구</div>
                <div>• 대규모 엔터프라이즈</div>
                <div>• 장기적 확장성 중시</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h3 className="font-semibold text-purple-600 mb-2">Data Vault 2.0</h3>
              <div className="text-sm space-y-1">
                <div><strong>적합한 경우:</strong></div>
                <div>• 높은 규제 요구사항</div>
                <div>• 복잡한 데이터 관계</div>
                <div>• 빈번한 소스 변경</div>
                <div>• 완전한 감사 추적 필요</div>
              </div>
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
              <span><strong>Kimball vs Inmon:</strong> Bottom-up vs Top-down 접근법</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Star Schema:</strong> 직관적이고 성능 우수한 차원 모델</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Data Vault 2.0:</strong> 유연하고 감사 가능한 모델링</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>선택 기준:</strong> 요구사항과 조직 특성에 맞는 방법론</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}