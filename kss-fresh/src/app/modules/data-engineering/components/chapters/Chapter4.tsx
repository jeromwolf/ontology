'use client';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">배치 데이터 처리와 ETL/ELT ⚙️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Apache Spark, dbt, Airflow로 대규모 데이터를 효율적으로 처리하는 방법을 마스터해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔄 ETL vs ELT: 패러다임 변화</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">전통적 ETL</h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">📥 Extract</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">소스 시스템에서 데이터 추출</p>
              </div>
              <div className="text-center text-gray-400">↓</div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">🔄 Transform</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">별도 서버에서 데이터 변환</p>
              </div>
              <div className="text-center text-gray-400">↓</div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">📤 Load</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">데이터 웨어하우스로 적재</p>
              </div>
            </div>
            <div className="mt-4 text-sm">
              <div className="text-red-600">❌ 변환 서버 필요</div>
              <div className="text-red-600">❌ 스키마 사전 정의</div>
              <div className="text-red-600">❌ 복잡한 파이프라인</div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">현대적 ELT</h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">📥 Extract</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">소스 시스템에서 데이터 추출</p>
              </div>
              <div className="text-center text-gray-400">↓</div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">📤 Load</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">데이터 레이크/웨어하우스 직접 적재</p>
              </div>
              <div className="text-center text-gray-400">↓</div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="font-semibold text-sm mb-1">🔄 Transform</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">목적지에서 직접 변환 (SQL)</p>
              </div>
            </div>
            <div className="mt-4 text-sm">
              <div className="text-green-600">✅ 클라우드 컴퓨팅 활용</div>
              <div className="text-green-600">✅ 스키마 유연성</div>
              <div className="text-green-600">✅ 간단한 파이프라인</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ Apache Spark: 대규모 데이터 처리의 왕</h2>
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Spark의 핵심 개념들</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">RDD (Resilient Distributed Dataset)</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>불변성:</strong> 한 번 생성되면 수정 불가</div>
                <div>• <strong>분산:</strong> 클러스터 전체에 분산 저장</div>
                <div>• <strong>복구 가능:</strong> 장애 시 자동 복구</div>
                <div>• <strong>지연 평가:</strong> action이 호출될 때 실행</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">DataFrame & Dataset</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>DataFrame:</strong> 스키마가 있는 RDD</div>
                <div>• <strong>Dataset:</strong> 타입 안전한 DataFrame</div>
                <div>• <strong>Catalyst 옵티마이저:</strong> 쿼리 최적화</div>
                <div>• <strong>Tungsten:</strong> 메모리 효율적 실행</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Spark 실행 모델</h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-500 rounded-lg mx-auto mb-2 flex items-center justify-center text-white font-bold">Driver</div>
                <div className="text-sm">
                  <div className="font-semibold">드라이버</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">애플리케이션 제어</div>
                </div>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-green-500 rounded-lg mx-auto mb-2 flex items-center justify-center text-white font-bold">CM</div>
                <div className="text-sm">
                  <div className="font-semibold">클러스터 매니저</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">리소스 관리</div>
                </div>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-500 rounded-lg mx-auto mb-2 flex items-center justify-center text-white font-bold">Exec</div>
                <div className="text-sm">
                  <div className="font-semibold">실행자</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">실제 작업 처리</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-orange-100 dark:bg-orange-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">💡 Spark 성능 최적화 팁</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="font-semibold mb-1">파티셔닝</div>
                <div className="text-orange-800 dark:text-orange-300">• 데이터 크기에 따른 파티션 수 조정</div>
                <div className="text-orange-800 dark:text-orange-300">• 셔플 최소화</div>
              </div>
              <div>
                <div className="font-semibold mb-1">캐싱</div>
                <div className="text-orange-800 dark:text-orange-300">• 반복 사용 데이터 메모리 캐싱</div>
                <div className="text-orange-800 dark:text-orange-300">• 적절한 저장 레벨 선택</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🛠️ dbt (Data Build Tool): 변환의 새로운 표준</h2>
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">SQL로 데이터 변환하기</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-3">dbt의 핵심 개념</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>모델 (Models):</strong> SQL 쿼리로 정의된 테이블/뷰</div>
                <div>• <strong>테스트 (Tests):</strong> 데이터 품질 검증</div>
                <div>• <strong>문서화 (Docs):</strong> 자동 문서 생성</div>
                <div>• <strong>스냅샷 (Snapshots):</strong> 변경 이력 추적</div>
                <div>• <strong>시드 (Seeds):</strong> 작은 참조 데이터</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">dbt 프로젝트 구조</h4>
              <div className="text-sm font-mono">
                <div>my_dbt_project/</div>
                <div className="ml-2">├── dbt_project.yml</div>
                <div className="ml-2">├── models/</div>
                <div className="ml-4">│   ├── staging/</div>
                <div className="ml-4">│   ├── intermediate/</div>
                <div className="ml-4">│   └── marts/</div>
                <div className="ml-2">├── tests/</div>
                <div className="ml-2">├── macros/</div>
                <div className="ml-2">└── seeds/</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">dbt 모델 예시</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400">-- models/staging/stg_orders.sql</div>
              <div className="mt-2">
                <div>{`{{ config(materialized='view') }}`}</div>
                <div></div>
                <div>select</div>
                <div className="ml-4">order_id,</div>
                <div className="ml-4">customer_id,</div>
                <div className="ml-4">order_date,</div>
                <div className="ml-4">amount</div>
                <div>from {`{{ source('raw', 'orders') }}`}</div>
                <div>where order_date {'>'}= '2023-01-01'</div>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🎯 dbt의 장점</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-purple-800 dark:text-purple-300">• SQL 기반으로 접근성 높음</div>
                <div className="text-purple-800 dark:text-purple-300">• 코드로서의 인프라 (IaC)</div>
                <div className="text-purple-800 dark:text-purple-300">• 자동 문서화와 계보 추적</div>
              </div>
              <div>
                <div className="text-purple-800 dark:text-purple-300">• 내장된 테스팅 프레임워크</div>
                <div className="text-purple-800 dark:text-purple-300">• Git 기반 협업</div>
                <div className="text-purple-800 dark:text-purple-300">• 점진적 배포</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🗓️ Apache Airflow: 워크플로우 오케스트레이션</h2>
        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">DAG로 데이터 파이프라인 관리하기</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-teal-600 dark:text-teal-400 mb-3">Airflow 핵심 개념</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>DAG:</strong> 방향성 비순환 그래프</div>
                <div>• <strong>Task:</strong> 실행 단위 (Python, Bash, SQL 등)</div>
                <div>• <strong>Operator:</strong> Task를 실행하는 방법</div>
                <div>• <strong>Scheduler:</strong> DAG 스케줄링</div>
                <div>• <strong>Executor:</strong> Task 실행 관리</div>
                <div>• <strong>Web UI:</strong> 모니터링 대시보드</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-cyan-600 dark:text-cyan-400 mb-3">주요 Operator들</h4>
              <div className="text-sm space-y-2">
                <div>• <strong>PythonOperator:</strong> Python 함수 실행</div>
                <div>• <strong>BashOperator:</strong> Shell 명령어 실행</div>
                <div>• <strong>SQLOperator:</strong> SQL 쿼리 실행</div>
                <div>• <strong>SparkSubmitOperator:</strong> Spark 작업 제출</div>
                <div>• <strong>S3Operator:</strong> AWS S3 작업</div>
                <div>• <strong>EmailOperator:</strong> 이메일 발송</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Airflow DAG 예시</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm font-mono">
              <div className="text-green-600 dark:text-green-400"># ETL 파이프라인 DAG</div>
              <div className="mt-2">
                <div>dag = DAG(</div>
                <div className="ml-4">'daily_etl',</div>
                <div className="ml-4">schedule_interval='@daily',</div>
                <div className="ml-4">start_date=datetime(2023, 1, 1)</div>
                <div>)</div>
                <div></div>
                <div>extract = PythonOperator(</div>
                <div className="ml-4">task_id='extract_data',</div>
                <div className="ml-4">python_callable=extract_function</div>
                <div>)</div>
                <div></div>
                <div>transform = SparkSubmitOperator(</div>
                <div className="ml-4">task_id='transform_data',</div>
                <div className="ml-4">application='transform.py'</div>
                <div>)</div>
                <div></div>
                <div>load = SQLOperator(</div>
                <div className="ml-4">task_id='load_data',</div>
                <div className="ml-4">sql='INSERT INTO ...'</div>
                <div>)</div>
                <div></div>
                <div>extract {'>'}{'>'}transform {'>'}{'>'}load</div>
              </div>
            </div>
          </div>

          <div className="bg-teal-100 dark:bg-teal-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-teal-800 dark:text-teal-200 mb-2">⚡ Airflow 모범 사례</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-teal-800 dark:text-teal-300">• 멱등성 보장 (같은 입력 → 같은 출력)</div>
                <div className="text-teal-800 dark:text-teal-300">• 원자성 유지 (all-or-nothing)</div>
                <div className="text-teal-800 dark:text-teal-300">• 적절한 리트라이 설정</div>
              </div>
              <div>
                <div className="text-teal-800 dark:text-teal-300">• SLA 모니터링</div>
                <div className="text-teal-800 dark:text-teal-300">• 알람 설정</div>
                <div className="text-teal-800 dark:text-teal-300">• 리소스 풀 활용</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔧 배치 처리 파이프라인 설계 패턴</h2>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">📊 Full Load vs Incremental</h3>
              <div className="space-y-3 text-sm">
                <div className="bg-white dark:bg-gray-700 rounded-lg p-3">
                  <div className="font-semibold text-blue-600">Full Load</div>
                  <div className="text-xs">전체 데이터를 매번 처리</div>
                  <div className="text-xs text-green-600">✅ 단순함, 일관성</div>
                  <div className="text-xs text-red-600">❌ 느림, 리소스 낭비</div>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-lg p-3">
                  <div className="font-semibold text-green-600">Incremental</div>
                  <div className="text-xs">변경된 데이터만 처리</div>
                  <div className="text-xs text-green-600">✅ 빠름, 효율적</div>
                  <div className="text-xs text-red-600">❌ 복잡함, 일관성 주의</div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">🎯 데이터 품질 체크포인트</h3>
              <div className="space-y-2 text-sm">
                <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded p-2">
                  <div className="font-semibold">1. 소스 검증</div>
                  <div className="text-xs">데이터 형식, 필수 컬럼 존재</div>
                </div>
                <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-2">
                  <div className="font-semibold">2. 변환 검증</div>
                  <div className="text-xs">비즈니스 로직, 계산 결과</div>
                </div>
                <div className="bg-green-100 dark:bg-green-900/30 rounded p-2">
                  <div className="font-semibold">3. 적재 검증</div>
                  <div className="text-xs">레코드 수, 중복 체크</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 bg-white dark:bg-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">🏗️ 권장 아키텍처</h3>
            <div className="grid md:grid-cols-4 gap-2 text-sm">
              <div className="text-center p-2 bg-blue-100 dark:bg-blue-900/30 rounded">
                <div className="font-semibold">Raw Data</div>
                <div className="text-xs">원본 그대로</div>
              </div>
              <div className="flex items-center justify-center">→</div>
              <div className="text-center p-2 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="font-semibold">Staging</div>
                <div className="text-xs">정제, 표준화</div>
              </div>
              <div className="flex items-center justify-center">→</div>
            </div>
            <div className="grid md:grid-cols-4 gap-2 text-sm mt-2">
              <div className="text-center p-2 bg-purple-100 dark:bg-purple-900/30 rounded">
                <div className="font-semibold">Intermediate</div>
                <div className="text-xs">비즈니스 로직</div>
              </div>
              <div className="flex items-center justify-center">→</div>
              <div className="text-center p-2 bg-orange-100 dark:bg-orange-900/30 rounded">
                <div className="font-semibold">Marts</div>
                <div className="text-xs">최종 분석용</div>
              </div>
              <div className="flex items-center justify-center">📊</div>
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
              <span><strong>ELT 패러다임:</strong> 클라우드 시대의 새로운 표준</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Apache Spark:</strong> 대규모 병렬 처리의 핵심</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>dbt:</strong> SQL로 하는 모던 데이터 변환</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Airflow:</strong> 안정적인 워크플로우 오케스트레이션</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>설계 패턴:</strong> 확장 가능하고 유지보수 가능한 파이프라인</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}