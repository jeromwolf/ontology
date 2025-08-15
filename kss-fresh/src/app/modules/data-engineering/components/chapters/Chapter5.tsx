'use client';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">실시간 스트림 처리 마스터 🌊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Kafka, Flink, Spark Streaming으로 실시간 파이프라인을 구축하고 운영해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 스트림 처리 vs 배치 처리</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">배치 처리</h3>
            <div className="space-y-3 text-sm">
              <div>📊 <strong>처리 방식:</strong> 정해진 시간에 대량 데이터 처리</div>
              <div>⏰ <strong>지연시간:</strong> 분/시간 단위</div>
              <div>🎯 <strong>정확성:</strong> 높음 (모든 데이터 확인 가능)</div>
              <div>💰 <strong>비용:</strong> 상대적으로 저렴</div>
              <div>🔧 <strong>복잡성:</strong> 낮음</div>
            </div>
            <div className="mt-4">
              <div className="font-semibold text-blue-800 dark:text-blue-200">적합한 경우:</div>
              <div className="text-sm">일일 리포트, 월말 정산, ETL 파이프라인</div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">스트림 처리</h3>
            <div className="space-y-3 text-sm">
              <div>🌊 <strong>처리 방식:</strong> 데이터가 들어오는 즉시 처리</div>
              <div>⚡ <strong>지연시간:</strong> 밀리초/초 단위</div>
              <div>🎲 <strong>정확성:</strong> 근사치 (부분적 정보)</div>
              <div>💸 <strong>비용:</strong> 상대적으로 비쌈</div>
              <div>🔧 <strong>복잡성:</strong> 높음</div>
            </div>
            <div className="mt-4">
              <div className="font-semibold text-green-800 dark:text-green-200">적합한 경우:</div>
              <div className="text-sm">실시간 알림, 사기 탐지, 개인화 추천</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📡 Apache Kafka: 이벤트 스트리밍 플랫폼</h2>
        <div className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Kafka의 핵심 아키텍처</h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-2">Producer</h4>
              <div className="text-sm space-y-1">
                <div>• 메시지를 토픽에 발행</div>
                <div>• 파티션 선택 로직</div>
                <div>• 배치 처리로 성능 최적화</div>
                <div>• Acks 설정으로 신뢰성 조절</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-yellow-600 dark:text-yellow-400 mb-2">Broker</h4>
              <div className="text-sm space-y-1">
                <div>• 메시지 저장 및 관리</div>
                <div>• 토픽과 파티션 관리</div>
                <div>• 클러스터 구성으로 확장성</div>
                <div>• 복제(Replication)로 가용성</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-2">Consumer</h4>
              <div className="text-sm space-y-1">
                <div>• 토픽에서 메시지 구독</div>
                <div>• 컨슈머 그룹으로 병렬 처리</div>
                <div>• 오프셋으로 위치 추적</div>
                <div>• At-least-once 보장</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Kafka 토픽과 파티션</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="font-semibold text-sm mb-2">토픽 (Topic)</div>
                <div className="text-xs space-y-1">
                  <div>• 메시지의 논리적 그룹</div>
                  <div>• 여러 파티션으로 구성</div>
                  <div>• 보존 정책 설정 가능</div>
                </div>
              </div>
              <div>
                <div className="font-semibold text-sm mb-2">파티션 (Partition)</div>
                <div className="text-xs space-y-1">
                  <div>• 순서가 보장되는 메시지 시퀀스</div>
                  <div>• 병렬 처리의 단위</div>
                  <div>• 키 기반 라우팅</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-orange-100 dark:bg-orange-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">🎯 Kafka 성능 최적화</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="font-semibold mb-1">Producer 최적화</div>
                <div className="text-orange-800 dark:text-orange-300">• batch.size와 linger.ms 조절</div>
                <div className="text-orange-800 dark:text-orange-300">• compression.type 설정</div>
              </div>
              <div>
                <div className="font-semibold mb-1">Consumer 최적화</div>
                <div className="text-orange-800 dark:text-orange-300">• fetch.min.bytes 조절</div>
                <div className="text-orange-800 dark:text-orange-300">• 적절한 컨슈머 그룹 크기</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 Apache Flink: 진정한 스트림 처리</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibent mb-4">Flink의 특징과 장점</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3">Native Streaming</h4>
              <div className="text-sm space-y-2">
                <div>• 진정한 스트림 우선 처리</div>
                <div>• 낮고 일관된 지연시간</div>
                <div>• 이벤트 시간 (Event Time) 지원</div>
                <div>• 워터마크 기반 늦은 데이터 처리</div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-3">Exactly-Once</h4>
              <div className="text-sm space-y-2">
                <div>• 정확히 한 번 처리 보장</div>
                <div>• 체크포인트 메커니즘</div>
                <div>• 장애 시 자동 복구</div>
                <div>• 상태 일관성 보장</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Flink vs Spark Streaming</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-2">특성</th>
                    <th className="text-left py-2 text-blue-600">Flink</th>
                    <th className="text-left py-2 text-orange-600">Spark Streaming</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2">처리 모델</td>
                    <td className="py-2">Native Streaming</td>
                    <td className="py-2">Micro-batch</td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2">지연시간</td>
                    <td className="py-2 text-green-600">밀리초</td>
                    <td className="py-2 text-yellow-600">초</td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2">Exactly-Once</td>
                    <td className="py-2 text-green-600">네이티브 지원</td>
                    <td className="py-2 text-yellow-600">구현 필요</td>
                  </tr>
                  <tr>
                    <td className="py-2">생태계</td>
                    <td className="py-2 text-yellow-600">발전 중</td>
                    <td className="py-2 text-green-600">성숙함</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">🔧 Flink 애플리케이션 구조</h4>
            <p className="text-sm text-blue-800 dark:text-blue-300">
              Source → Transformation → Sink 파이프라인으로 구성되며, 
              상태 관리와 윈도우 연산을 통해 복잡한 이벤트 처리가 가능합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ Spark Streaming: 통합된 처리 엔진</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Structured Streaming</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">핵심 개념</h4>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="font-semibold text-purple-600">무한한 테이블</div>
                <div>스트림을 계속 확장되는 테이블로 간주</div>
              </div>
              <div>
                <div className="font-semibold text-pink-600">트리거</div>
                <div>언제 결과를 업데이트할지 정의</div>
              </div>
              <div>
                <div className="font-semibold text-indigo-600">출력 모드</div>
                <div>결과를 어떻게 출력할지 정의</div>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">💡 Spark Streaming 장점</h4>
            <div className="text-sm text-purple-800 dark:text-purple-300">
              배치와 스트림 처리를 같은 API로 처리할 수 있어 코드 재사용성이 높고,
              풍부한 Spark 생태계를 그대로 활용할 수 있습니다.
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
              <span><strong>Kafka:</strong> 대규모 이벤트 스트리밍 플랫폼</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Flink:</strong> 낮은 지연시간의 진정한 스트림 처리</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>Spark Streaming:</strong> 배치와 스트림 통합 처리</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span><strong>설계 패턴:</strong> 확장 가능한 실시간 아키텍처</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}