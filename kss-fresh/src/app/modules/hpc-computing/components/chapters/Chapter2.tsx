export default function Chapter2() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          병렬 컴퓨팅 (Parallel Computing)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            병렬 컴퓨팅은 여러 프로세서가 동시에 계산을 수행하여 문제를 빠르게 해결하는 기술입니다.
            OpenMP(공유 메모리)와 MPI(분산 메모리) 두 가지 주요 프로그래밍 모델을 통해
            수백~수천 개의 코어를 활용할 수 있습니다.
          </p>
        </div>
      </section>

      {/* OpenMP 기초 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. OpenMP (Open Multi-Processing)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
            공유 메모리 병렬화
          </h4>
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            OpenMP는 단일 노드 내의 여러 코어를 활용합니다. pragma 지시어를 사용하여 간단하게 병렬화할 수 있습니다.
          </p>

          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto mb-4">
            <pre className="text-sm text-gray-100">
              <code>{`// 병렬 for 루프 예제
#include <omp.h>
#include <stdio.h>

int main() {
    int n = 1000000;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += i * 0.5;
    }

    printf("합계: %f\\n", sum);
    return 0;
}`}</code>
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">주요 지시어</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">#pragma omp parallel</code> - 병렬 영역 생성</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">#pragma omp for</code> - 루프 병렬화</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">reduction</code> - 병렬 리덕션</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">critical</code> - 임계 영역</li>
              </ul>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">데이터 공유 속성</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">shared</code> - 모든 스레드가 공유</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">private</code> - 각 스레드마다 독립 복사본</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">firstprivate</code> - 초기값 복사</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">lastprivate</code> - 마지막 값 저장</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* MPI 기초 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. MPI (Message Passing Interface)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400">
            분산 메모리 병렬화
          </h4>
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            MPI는 여러 노드 간 메시지 전달을 통해 병렬 처리를 수행합니다. 수천 개의 노드로 확장 가능합니다.
          </p>

          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto mb-4">
            <pre className="text-sm text-gray-100">
              <code>{`// MPI Hello World
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("프로세스 %d / %d\\n", rank, size);

    MPI_Finalize();
    return 0;
}`}</code>
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">실행 방법</h5>
            <div className="bg-gray-900 rounded p-2 text-sm">
              <code className="text-green-400">
                mpicc hello_mpi.c -o hello_mpi<br/>
                mpirun -np 4 ./hello_mpi  # 4개 프로세스로 실행
              </code>
            </div>
          </div>
        </div>
      </section>

      {/* Point-to-Point 통신 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. MPI Point-to-Point 통신
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto mb-4">
            <pre className="text-sm text-gray-100">
              <code>{`int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == 0) {
    int data = 100;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    printf("프로세스 0이 데이터 %d를 프로세스 1에 전송\\n", data);
} else if (rank == 1) {
    int received_data;
    MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("프로세스 1이 데이터 %d를 수신\\n", received_data);
}`}</code>
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Blocking 통신</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MPI_Send</code> - 송신 (완료까지 대기)</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MPI_Recv</code> - 수신 (완료까지 대기)</li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">Non-blocking 통신</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MPI_Isend</code> - 비동기 송신</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MPI_Irecv</code> - 비동기 수신</li>
                <li><code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MPI_Wait</code> - 완료 대기</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Collective 통신 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. MPI Collective 통신
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">MPI_Bcast - 브로드캐스트</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                루트 프로세스에서 모든 프로세스로 데이터 전송
              </p>
              <div className="bg-gray-900 rounded p-2">
                <code className="text-sm text-gray-100">
                  MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
                </code>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">MPI_Reduce - 리덕션</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                모든 프로세스의 데이터를 결합하여 루트로 전송
              </p>
              <div className="bg-gray-900 rounded p-2">
                <code className="text-sm text-gray-100">
                  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                </code>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">MPI_Scatter / MPI_Gather</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                데이터 분산 및 수집
              </p>
              <div className="bg-gray-900 rounded p-2">
                <code className="text-sm text-gray-100">
                  MPI_Scatter(send_data, count, MPI_INT, recv_data, count, MPI_INT, 0, MPI_COMM_WORLD);
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Amdahl's Law */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          5. Amdahl's Law (암달의 법칙)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            병렬화로 얻을 수 있는 이론적 최대 속도 향상을 나타냅니다.
          </p>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg mb-4">
            <div className="text-center mb-2">
              <p className="text-lg font-mono text-gray-900 dark:text-white">
                속도 향상 = 1 / (S + P/N)
              </p>
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>S</strong>: 직렬 부분 비율 (병렬화 불가능)</li>
              <li><strong>P</strong>: 병렬 부분 비율 (P = 1 - S)</li>
              <li><strong>N</strong>: 프로세서 개수</li>
            </ul>
          </div>

          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-4 rounded-lg">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">예시</h5>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              프로그램의 10%가 직렬 부분(S=0.1)이라면, 무한대의 프로세서를 사용해도 최대 10배 속도 향상만 가능합니다.<br/>
              100개 프로세서: 1 / (0.1 + 0.9/100) ≈ 9.2배<br/>
              1000개 프로세서: 1 / (0.1 + 0.9/1000) ≈ 9.9배<br/>
              ∞ 프로세서: 1 / 0.1 = 10배 (최대)
            </p>
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
              <span>OpenMP는 공유 메모리 모델로 단일 노드 내 병렬화에 적합하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>MPI는 분산 메모리 모델로 수천 개 노드로 확장 가능하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>Collective 통신으로 효율적인 데이터 교환이 가능하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>Amdahl's Law는 직렬 부분이 전체 속도 향상을 제한한다는 것을 보여준다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
