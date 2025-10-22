export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          분산 시스템 (Distributed Systems)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            대규모 HPC 시스템은 본질적으로 분산 시스템입니다. 수천 개의 노드가 협력하여
            단일 문제를 해결하기 위해서는 분산 알고리즘, 동기화, 내결함성 등의 핵심 개념을 이해해야 합니다.
          </p>
        </div>
      </section>

      {/* 분산 컴퓨팅 모델 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. 분산 컴퓨팅 모델
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                MapReduce 패러다임
              </h4>
              <p className="mb-3 text-gray-700 dark:text-gray-300">
                대규모 데이터를 병렬로 처리하는 프로그래밍 모델입니다.
              </p>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto mb-3">
                <pre className="text-sm text-gray-100">
                  <code>{`// Map 단계: 각 데이터 청크를 독립적으로 처리
map(String key, String value):
    // key: 파일명, value: 파일 내용
    for each word w in value:
        emit(w, 1)

// Reduce 단계: Map 출력을 집계
reduce(String key, Iterator values):
    // key: 단어, values: [1, 1, 1, ...]
    int sum = 0
    for each v in values:
        sum += v
    emit(key, sum)

// 예시 실행:
// Input: "hello world" "hello HPC"
// Map: (hello, 1), (world, 1), (hello, 1), (HPC, 1)
// Reduce: (hello, 2), (world, 1), (HPC, 1)`}</code>
                </pre>
              </div>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                  <h5 className="font-semibold mb-1 text-sm text-gray-900 dark:text-white">장점</h5>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 간단한 프로그래밍 모델</li>
                    <li>• 자동 병렬화</li>
                    <li>• 내결함성 제공</li>
                  </ul>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                  <h5 className="font-semibold mb-1 text-sm text-gray-900 dark:text-white">사용 사례</h5>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 로그 분석</li>
                    <li>• 인덱싱</li>
                    <li>• 그래프 처리</li>
                  </ul>
                </div>
                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                  <h5 className="font-semibold mb-1 text-sm text-gray-900 dark:text-white">구현체</h5>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• Hadoop</li>
                    <li>• Apache Spark</li>
                    <li>• Google MapReduce</li>
                  </ul>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400">
                Bulk Synchronous Parallel (BSP)
              </h4>
              <p className="mb-3 text-gray-700 dark:text-gray-300">
                프로세서들이 동기화 지점에서 데이터를 교환하는 모델입니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>3단계 사이클</strong>:<br/>
                  1. <strong>로컬 연산</strong>: 각 프로세서가 독립적으로 계산<br/>
                  2. <strong>통신</strong>: 프로세서 간 메시지 교환<br/>
                  3. <strong>배리어 동기화</strong>: 모든 프로세서 대기<br/><br/>
                  예시: Apache Giraph, Pregel (그래프 처리)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 분산 알고리즘 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. 분산 알고리즘
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">분산 정렬 (Parallel Sorting)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Sample Sort: 데이터를 샘플링하여 균등하게 분할 후 정렬합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// Sample Sort 알고리즘
1. 각 프로세서에서 로컬 데이터 샘플링
2. 샘플을 모아서 전체 정렬
3. P-1개의 splitter 선택 (P = 프로세서 수)
4. Splitter 기준으로 데이터 분할 및 재분배
5. 각 프로세서에서 로컬 정렬

// MPI 구현 예시
MPI_Gather(local_samples, ..., all_samples, ...)  // 샘플 수집
sort(all_samples)                                  // 샘플 정렬
select_splitters(all_samples, splitters, P-1)     // Splitter 선택
MPI_Alltoall(local_data, ..., redistributed_data, ...) // 재분배
sort(redistributed_data)                          // 로컬 정렬`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">분산 합의 알고리즘 (Consensus)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                여러 노드가 하나의 값에 합의하는 알고리즘입니다.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">알고리즘</th>
                      <th className="p-2">특징</th>
                      <th className="p-2">사용 사례</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono font-semibold">Paxos</td>
                      <td className="p-2">강력한 일관성, 복잡한 구현</td>
                      <td className="p-2">Google Chubby</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono font-semibold">Raft</td>
                      <td className="p-2">이해하기 쉬운 Paxos 대안</td>
                      <td className="p-2">etcd, Consul</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-mono font-semibold">Byzantine</td>
                      <td className="p-2">악의적 노드 허용</td>
                      <td className="p-2">블록체인</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">분산 그래프 알고리즘</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                PageRank: 웹 페이지 중요도를 계산하는 대표적인 분산 알고리즘
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`// Vertex-centric 프로그래밍 (Pregel 모델)
class PageRankVertex {
    double rank = 1.0 / numVertices

    void compute(Iterable<Message> messages) {
        if (superstep > 0) {
            double sum = 0
            for (msg in messages) {
                sum += msg.value
            }
            rank = 0.15 / numVertices + 0.85 * sum
        }

        if (superstep < maxIterations) {
            // 이웃에게 rank 전송
            for (edge in outEdges) {
                sendMessage(edge.target, rank / outDegree)
            }
        } else {
            voteToHalt()
        }
    }
}`}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 동기화와 일관성 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. 동기화와 일관성 (Synchronization & Consistency)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">분산 락 (Distributed Locks)</h5>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// Redis 기반 분산 락 (Redlock 알고리즘)
def acquire_lock(resource_name, ttl):
    identifier = random_string()
    end_time = time() + ttl

    while time() < end_time:
        # 다수의 Redis 인스턴스에서 락 획득 시도
        acquired = 0
        for redis_instance in redis_instances:
            if redis_instance.set(resource_name, identifier,
                                  nx=True, px=ttl):
                acquired += 1

        # 과반수 이상에서 성공하면 락 획득
        if acquired >= (len(redis_instances) // 2 + 1):
            return identifier

        # 실패 시 모든 락 해제
        release_lock(resource_name, identifier)
        sleep(random_delay)

    return None`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">일관성 모델 (Consistency Models)</h5>
              <div className="overflow-x-auto mb-3">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-orange-500">
                      <th className="p-2">모델</th>
                      <th className="p-2">설명</th>
                      <th className="p-2">성능</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Strong Consistency</td>
                      <td className="p-2">모든 노드가 항상 동일한 값 반환</td>
                      <td className="p-2 text-red-600 dark:text-red-400">낮음 (느림)</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Eventual Consistency</td>
                      <td className="p-2">시간이 지나면 일관성 달성</td>
                      <td className="p-2 text-green-600 dark:text-green-400">높음 (빠름)</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Causal Consistency</td>
                      <td className="p-2">인과 관계가 있는 연산만 순서 보장</td>
                      <td className="p-2 text-yellow-600 dark:text-yellow-400">중간</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-semibold">Read-Your-Writes</td>
                      <td className="p-2">자신이 쓴 값은 항상 읽을 수 있음</td>
                      <td className="p-2 text-yellow-600 dark:text-yellow-400">중간</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>CAP 정리</strong>: 분산 시스템은 다음 세 가지 중 두 가지만 보장 가능<br/>
                  • <strong>C</strong>onsistency (일관성): 모든 노드가 동일한 데이터<br/>
                  • <strong>A</strong>vailability (가용성): 모든 요청이 응답<br/>
                  • <strong>P</strong>artition tolerance (분할 허용): 네트워크 분할에도 동작
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 내결함성 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 내결함성 (Fault Tolerance)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">체크포인팅 (Checkpointing)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                주기적으로 프로그램 상태를 저장하여 장애 시 복구합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// MPI 체크포인팅 예시 (BLCR 라이브러리)
#include <mpi.h>

void save_checkpoint(int iteration) {
    char filename[256];
    sprintf(filename, "checkpoint_%d.dat", iteration);

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // 로컬 데이터 저장
    MPI_File_write(fh, local_data, data_size,
                   MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}

// 메인 루프
for (int iter = 0; iter < max_iterations; iter++) {
    compute_iteration(iter);

    // 100 iteration마다 체크포인트
    if (iter % 100 == 0) {
        save_checkpoint(iter);
    }
}`}</code>
                </pre>
              </div>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <strong>장점</strong>:<br/>
                    • 장애 시 최근 상태에서 재시작<br/>
                    • 계산 시간 절약
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <strong>단점</strong>:<br/>
                    • 저장 오버헤드<br/>
                    • 대규모 시스템에서 I/O 병목
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">복제 (Replication)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                여러 노드에 데이터를 복제하여 가용성을 높입니다.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">기법</th>
                      <th className="p-2">복제 계수</th>
                      <th className="p-2">사용 사례</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Primary-Backup</td>
                      <td className="p-2">2-3개</td>
                      <td className="p-2">데이터베이스</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Multi-Paxos</td>
                      <td className="p-2">2f+1 (f=장애 허용)</td>
                      <td className="p-2">분산 합의</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-semibold">Erasure Coding</td>
                      <td className="p-2">1.5배 (효율적)</td>
                      <td className="p-2">HDFS, Ceph</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">장애 감지 (Failure Detection)</h5>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Heartbeat 메커니즘</strong>:<br/>
                  • 주기적으로 살아있음을 알리는 메시지 전송<br/>
                  • Timeout 내에 응답 없으면 장애로 간주<br/>
                  • False positive 최소화 위해 여러 번 확인<br/><br/>
                  <strong>구현 예</strong>: ZooKeeper, etcd, Consul
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          📚 핵심 요약
        </h3>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
          <ul className="space-y-3 text-gray-800 dark:text-gray-200">
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">1.</span>
              <span>MapReduce와 BSP는 대표적인 분산 컴퓨팅 모델이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>분산 알고리즘은 데이터 재분배와 통신 최소화가 핵심이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>일관성 모델은 성능과 정확성 사이의 트레이드오프를 결정한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>체크포인팅과 복제는 내결함성 확보의 핵심 기법이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">5.</span>
              <span>CAP 정리는 분산 시스템 설계 시 고려해야 할 근본적인 제약이다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
