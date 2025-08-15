'use client';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">현대적 데이터 아키텍처 패턴 🏗️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          람다/카파 아키텍처, 데이터 메시, 레이크하우스로 확장 가능한 데이터 플랫폼을 설계해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏛️ 데이터 아키텍처 패턴 개요</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            현대의 데이터 아키텍처는 대용량, 다양한 형태, 실시간 처리 요구사항을 모두 만족해야 합니다.
            각 패턴은 고유한 장단점과 적용 시나리오를 가지고 있어요.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">전통적 아키텍처</h3>
              <ul className="text-sm space-y-1">
                <li>• 배치 처리 중심</li>
                <li>• ETL → DW → BI</li>
                <li>• 정형 데이터 위주</li>
                <li>• 높은 일관성</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-green-600 dark:text-green-400 mb-2">현대적 아키텍처</h3>
              <ul className="text-sm space-y-1">
                <li>• 실시간 + 배치 하이브리드</li>
                <li>• ELT → 레이크 → 분석</li>
                <li>• 다양한 데이터 형태</li>
                <li>• 결과적 일관성</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 람다 아키텍처 (Lambda Architecture)</h2>
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">3개 레이어로 구성된 하이브리드 아키텍처</h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-purple-200 dark:border-purple-600">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">배치 레이어 (Batch Layer)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                마스터 데이터셋을 관리하고 배치 뷰를 생성
              </p>
              <div className="text-xs space-y-1">
                <p><strong>도구:</strong> Hadoop, Spark</p>
                <p><strong>특징:</strong> 완전성, 정확성</p>
                <p><strong>주기:</strong> 시간/일 단위</p>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-green-200 dark:border-green-600">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-2">스피드 레이어 (Speed Layer)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                실시간 데이터를 처리하고 실시간 뷰를 생성
              </p>
              <div className="text-xs space-y-1">
                <p><strong>도구:</strong> Storm, Spark Streaming</p>
                <p><strong>특징:</strong> 저지연, 근사치</p>
                <p><strong>주기:</strong> 초/분 단위</p>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-orange-200 dark:border-orange-600">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-2">서빙 레이어 (Serving Layer)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                배치와 실시간 뷰를 병합하여 쿼리 제공
              </p>
              <div className="text-xs space-y-1">
                <p><strong>도구:</strong> ElephantDB, Druid</p>
                <p><strong>특징:</strong> 임의 읽기, 빠른 쿼리</p>
                <p><strong>응답:</strong> 밀리초 단위</p>
              </div>
            </div>
          </div>

          <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">💡 람다 아키텍처의 핵심</h4>
            <p className="text-sm text-yellow-800 dark:text-yellow-300">
              Query = merge(BatchView(all data), RealTimeView(recent data))
              <br />
              배치 뷰는 완전하지만 느리고, 실시간 뷰는 빠르지만 불완전합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🌊 카파 아키텍처 (Kappa Architecture)</h2>
        <div className="bg-gradient-to-r from-teal-50 to-green-50 dark:from-teal-900/20 dark:to-green-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">스트림 처리 중심의 단순한 아키텍처</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">카파 아키텍처 구조</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded"></div>
                  <span>데이터 소스</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <div className="w-2 h-2 bg-gray-400 rounded"></div>
                  <span>↓</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded"></div>
                  <span>이벤트 로그 (Kafka)</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <div className="w-2 h-2 bg-gray-400 rounded"></div>
                  <span>↓</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded"></div>
                  <span>스트림 처리 엔진</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <div className="w-2 h-2 bg-gray-400 rounded"></div>
                  <span>↓</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-orange-500 rounded"></div>
                  <span>뷰 저장소</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-3">람다 vs 카파</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">복잡성</span>
                  <span className="text-green-600">카파 승리</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">정확성</span>
                  <span className="text-purple-600">람다 승리</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">유지보수</span>
                  <span className="text-green-600">카파 승리</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">디버깅</span>
                  <span className="text-green-600">카파 승리</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">재처리</span>
                  <span className="text-green-600">카파 승리</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-teal-100 dark:bg-teal-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-teal-800 dark:text-teal-200 mb-2">🚀 카파의 핵심 아이디어</h4>
            <p className="text-sm text-teal-800 dark:text-teal-300">
              "모든 것은 스트림이다" - 과거 데이터도 스트림으로 재처리할 수 있다면, 
              배치와 실시간을 구분할 필요가 없다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🕸️ 데이터 메시 (Data Mesh)</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">도메인 중심의 분산 데이터 아키텍처</h3>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="space-y-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400">4가지 핵심 원칙</h4>
              <div className="space-y-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm">1. 도메인 소유권 (Domain Ownership)</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">각 도메인팀이 자신의 데이터를 소유하고 관리</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm">2. 데이터 제품으로서의 데이터</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">데이터를 제품처럼 디자인, 개발, 배포</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm">3. 셀프서비스 플랫폼</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">도메인팀이 독립적으로 작업할 수 있는 플랫폼</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm">4. 연합 거버넌스</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">글로벌 표준과 로컬 자율성의 균형</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400">Data Product 구조</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <div className="space-y-2 text-sm">
                  <div className="font-semibold">📊 Data Product</div>
                  <div className="ml-4 space-y-1">
                    <div>• 📥 입력 포트 (Input Ports)</div>
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400">외부에서 데이터 수신</div>
                    
                    <div>• ⚙️ 변환 로직 (Transformation)</div>
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400">비즈니스 로직 적용</div>
                    
                    <div>• 📤 출력 포트 (Output Ports)</div>
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400">API, 이벤트, 파일 등</div>
                    
                    <div>• 📋 메타데이터</div>
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400">스키마, SLA, 계보 정보</div>
                    
                    <div>• 🔍 관찰가능성</div>
                    <div className="ml-4 text-xs text-gray-600 dark:text-gray-400">품질 메트릭, 로그</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-2">🎯 데이터 메시를 언제 사용할까?</h4>
            <p className="text-sm text-indigo-800 dark:text-indigo-300">
              조직이 크고(100+ 엔지니어), 도메인이 복잡하며, 데이터팀이 병목이 되는 상황에서 효과적입니다.
              작은 조직에서는 오히려 복잡성만 증가할 수 있어요.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏠 데이터 레이크하우스 (Data Lakehouse)</h2>
        <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">데이터 레이크와 웨어하우스의 장점을 결합</h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-l-4 border-blue-500">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">데이터 레이크</h4>
              <div className="text-sm space-y-1">
                <div className="text-green-600">✅ 저비용 저장</div>
                <div className="text-green-600">✅ 스키마 유연성</div>
                <div className="text-green-600">✅ 다양한 형태</div>
                <div className="text-red-600">❌ 성능 문제</div>
                <div className="text-red-600">❌ ACID 미지원</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-l-4 border-purple-500">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">데이터 웨어하우스</h4>
              <div className="text-sm space-y-1">
                <div className="text-green-600">✅ 고성능 쿼리</div>
                <div className="text-green-600">✅ ACID 지원</div>
                <div className="text-green-600">✅ 일관된 스키마</div>
                <div className="text-red-600">❌ 높은 비용</div>
                <div className="text-red-600">❌ 구조 변경 어려움</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-l-4 border-emerald-500">
              <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-2">레이크하우스</h4>
              <div className="text-sm space-y-1">
                <div className="text-green-600">✅ 저비용 + 고성능</div>
                <div className="text-green-600">✅ ACID + 유연성</div>
                <div className="text-green-600">✅ 스트리밍 + 배치</div>
                <div className="text-green-600">✅ ML + BI 통합</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-3">레이크하우스를 가능하게 하는 기술들</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="font-semibold text-sm">오픈 테이블 포맷</div>
                <div className="text-xs space-y-1">
                  <div>• <strong>Delta Lake:</strong> Databricks</div>
                  <div>• <strong>Apache Iceberg:</strong> Netflix</div>
                  <div>• <strong>Apache Hudi:</strong> Uber</div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="font-semibold text-sm">핵심 기능</div>
                <div className="text-xs space-y-1">
                  <div>• ACID 트랜잭션</div>
                  <div>• 타임 트래블</div>
                  <div>• 스키마 진화</div>
                  <div>• 메타데이터 레이어</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">🏗️ 레이크하우스 아키텍처 예시</h4>
            <p className="text-sm text-emerald-800 dark:text-emerald-300">
              S3 (저장) + Delta Lake (테이블 포맷) + Spark (컴퓨트) + Unity Catalog (거버넌스)
              = 완전한 레이크하우스 솔루션
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 아키텍처 선택 가이드</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">🚀 스타트업/소규모 팀</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4 text-sm">
                <div className="mb-2"><strong>추천:</strong> 레이크하우스 (Databricks/Snowflake)</div>
                <div className="mb-2"><strong>이유:</strong> 관리 복잡성 최소화, 빠른 구축</div>
                <div><strong>시작:</strong> dbt + Snowflake + Fivetran</div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">🏢 대기업/복잡한 도메인</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4 text-sm">
                <div className="mb-2"><strong>추천:</strong> 데이터 메시 + 레이크하우스</div>
                <div className="mb-2"><strong>이유:</strong> 도메인별 자율성, 확장성</div>
                <div><strong>시작:</strong> 도메인 분리 → 플랫폼 구축 → 점진적 확산</div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">⚡ 실시간 중심</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4 text-sm">
                <div className="mb-2"><strong>추천:</strong> 카파 아키텍처</div>
                <div className="mb-2"><strong>이유:</strong> 단순함, 일관된 처리 모델</div>
                <div><strong>시작:</strong> Kafka + Spark Streaming/Flink</div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">🔄 배치 + 실시간</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4 text-sm">
                <div className="mb-2"><strong>추천:</strong> 람다 아키텍처</div>
                <div className="mb-2"><strong>이유:</strong> 각각의 최적화, 정확성 보장</div>
                <div><strong>시작:</strong> Spark (배치) + Spark Streaming (실시간)</div>
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
              <span><strong>람다:</strong> 배치+실시간 하이브리드, 정확성 중시</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>카파:</strong> 스트림 중심, 단순함 중시</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>데이터 메시:</strong> 도메인 분산, 조직 확장성</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>레이크하우스:</strong> 레이크+웨어하우스 장점 결합</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>선택 기준:</strong> 조직 크기, 요구사항, 기술 성숙도</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}