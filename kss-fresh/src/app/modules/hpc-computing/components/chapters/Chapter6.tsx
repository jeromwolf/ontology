export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          성능 최적화 (Performance Optimization)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            HPC 시스템에서 최고의 성능을 얻기 위해서는 알고리즘, 메모리, 통신, 로드 밸런싱 등
            다양한 측면의 최적화가 필요합니다. 이 챕터에서는 실전에서 바로 적용할 수 있는
            성능 최적화 기법들을 다룹니다.
          </p>
        </div>
      </section>

      {/* 병렬 알고리즘 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. 병렬 알고리즘 최적화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
            작업 분할 (Work Decomposition)
          </h4>
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            효율적인 병렬화를 위해서는 작업을 균등하게 분할하고 통신 오버헤드를 최소화해야 합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">데이터 병렬화 (Data Parallelism)</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 데이터를 여러 프로세서에 분산</li>
                <li>• 각 프로세서가 같은 연산 수행</li>
                <li>• SIMD 패턴에 적합</li>
                <li>• 예: 행렬 곱셈, 이미지 처리</li>
              </ul>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">태스크 병렬화 (Task Parallelism)</h5>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 서로 다른 작업을 병렬 수행</li>
                <li>• 독립적인 태스크 동시 실행</li>
                <li>• 파이프라인 패턴에 적합</li>
                <li>• 예: 데이터 전처리 + 분석</li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`// 데이터 병렬화 예제: 벡터 덧셈
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}

// 태스크 병렬화 예제: 독립적인 함수들
#pragma omp parallel sections
{
    #pragma omp section
    process_data_A();

    #pragma omp section
    process_data_B();

    #pragma omp section
    process_data_C();
}`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* GPU 메모리 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. GPU 메모리 최적화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400">
            메모리 접근 패턴 최적화
          </h4>

          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Coalesced Memory Access</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Warp 내 스레드들이 연속된 메모리 주소에 접근하면 하나의 메모리 트랜잭션으로 처리됩니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// ❌ 나쁜 예: Strided Access (성능 저하)
__global__ void bad_kernel(float *data) {
    int idx = threadIdx.x * STRIDE;  // 불연속 접근
    data[idx] = 0.0f;
}

// ✅ 좋은 예: Coalesced Access (최적)
__global__ void good_kernel(float *data) {
    int idx = threadIdx.x;  // 연속 접근
    data[idx] = 0.0f;
}`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Shared Memory 활용</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Shared Memory는 Global Memory보다 100배 이상 빠릅니다.
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`__global__ void matrix_multiply_shared(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 타일 단위로 처리
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Global → Shared 메모리로 복사
        As[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        __syncthreads();

        // Shared 메모리에서 연산 (빠름!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Bank Conflict 회피</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Shared Memory는 32개 뱅크로 구성됩니다. 같은 뱅크에 동시 접근하면 직렬화됩니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>해결 방법</strong>: Padding을 추가하여 뱅크를 분산시킵니다.<br/>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];</code>
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 통신 오버헤드 최소화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. 통신 오버헤드 최소화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">통신-연산 오버랩 (Communication-Computation Overlap)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                비동기 통신을 사용하여 통신과 연산을 동시에 수행합니다.
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`// MPI Non-blocking Communication
MPI_Request request[4];

// 비동기 송수신 시작 (즉시 리턴)
MPI_Isend(send_buf, count, MPI_DOUBLE, dest, tag, comm, &request[0]);
MPI_Irecv(recv_buf, count, MPI_DOUBLE, src, tag, comm, &request[1]);

// 통신이 진행되는 동안 독립적인 연산 수행
compute_internal_data();  // 🔥 통신과 동시 실행!

// 통신 완료 대기
MPI_Waitall(2, request, MPI_STATUSES_IGNORE);

// 수신된 데이터 처리
compute_boundary_data(recv_buf);`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">메시지 합치기 (Message Aggregation)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                여러 작은 메시지를 하나로 합쳐서 전송 횟수를 줄입니다.
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded">
                  <p className="text-sm font-semibold text-red-600 dark:text-red-400 mb-1">❌ 비효율적</p>
                  <code className="text-xs text-gray-700 dark:text-gray-300">
                    MPI_Send(data1, ...)<br/>
                    MPI_Send(data2, ...)<br/>
                    MPI_Send(data3, ...)<br/>
                    // 3번 통신 (오버헤드 3배)
                  </code>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                  <p className="text-sm font-semibold text-green-600 dark:text-green-400 mb-1">✅ 효율적</p>
                  <code className="text-xs text-gray-700 dark:text-gray-300">
                    memcpy(buffer, data1, ...)<br/>
                    memcpy(buffer+offset1, data2, ...)<br/>
                    memcpy(buffer+offset2, data3, ...)<br/>
                    MPI_Send(buffer, ...)<br/>
                    // 1번 통신
                  </code>
                </div>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Collective 통신 활용</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                MPI Collective 연산은 내부적으로 최적화되어 있어 Point-to-Point보다 빠릅니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>MPI_Allreduce</strong>: O(log N) 시간에 전체 합산</li>
                  <li>• <strong>MPI_Alltoall</strong>: 모든 프로세스 간 데이터 교환 최적화</li>
                  <li>• <strong>MPI_Barrier</strong>: 동기화 (단, 과도한 사용 주의)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 로드 밸런싱 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 로드 밸런싱 (Load Balancing)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            모든 프로세서가 균등하게 작업을 수행하도록 하는 것이 중요합니다.
          </p>

          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">정적 로드 밸런싱 (Static)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                실행 전에 작업량을 미리 분배합니다. 작업량이 균등할 때 효과적입니다.
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`// OpenMP Static Scheduling
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    // 균등한 작업량
    process(data[i]);
}

// 각 스레드가 N/thread_count 만큼 처리`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">동적 로드 밸런싱 (Dynamic)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                실행 중 작업을 동적으로 할당합니다. 불균등한 작업량에 적합합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// OpenMP Dynamic Scheduling (작업 완료 시 새 작업 할당)
#pragma omp parallel for schedule(dynamic, chunk_size)
for (int i = 0; i < N; i++) {
    // 불균등한 작업량
    complex_computation(data[i]);
}

// Guided Scheduling (청크 크기를 점점 줄임)
#pragma omp parallel for schedule(guided)
for (int i = 0; i < N; i++) {
    irregular_workload(i);
}`}</code>
                </pre>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>장점</strong>: 작업 불균형 해소, 유휴 시간 최소화<br/>
                  <strong>단점</strong>: 스케줄링 오버헤드 증가
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Work Stealing</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                작업이 끝난 프로세서가 바쁜 프로세서의 작업을 가져옵니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Intel TBB (Threading Building Blocks), Cilk Plus 등에서 사용<br/>
                  태스크 큐 기반으로 자동 로드 밸런싱
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 성능 프로파일링 도구 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          5. 성능 프로파일링 도구
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            성능 병목 지점을 찾고 최적화 효과를 측정하는 도구들입니다.
          </p>

          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b-2 border-yellow-500">
                    <th className="p-3">도구</th>
                    <th className="p-3">용도</th>
                    <th className="p-3">주요 기능</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-mono font-semibold text-yellow-600 dark:text-yellow-400">NVIDIA Nsight</td>
                    <td className="p-3">CUDA 프로파일링</td>
                    <td className="p-3">커널 분석, 메모리 대역폭, Warp 효율</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-mono font-semibold text-yellow-600 dark:text-yellow-400">nvprof / ncu</td>
                    <td className="p-3">GPU 성능 측정</td>
                    <td className="p-3">실행 시간, 점유율, 메트릭 수집</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-mono font-semibold text-orange-600 dark:text-orange-400">Intel VTune</td>
                    <td className="p-3">CPU 프로파일링</td>
                    <td className="p-3">핫스팟, 캐시 미스, 벡터화 분석</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-mono font-semibold text-orange-600 dark:text-orange-400">Scalasca</td>
                    <td className="p-3">MPI 성능 분석</td>
                    <td className="p-3">통신 패턴, 대기 시간, 로드 밸런스</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-mono font-semibold text-orange-600 dark:text-orange-400">TAU</td>
                    <td className="p-3">병렬 프로그램 분석</td>
                    <td className="p-3">타임라인, 콜 그래프, 이벤트 추적</td>
                  </tr>
                  <tr>
                    <td className="p-3 font-mono font-semibold text-orange-600 dark:text-orange-400">perf / gprof</td>
                    <td className="p-3">일반 프로파일링</td>
                    <td className="p-3">함수별 실행 시간, CPU 사이클</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">NVIDIA Nsight Compute 사용 예시</h5>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`# 커널 프로파일링
ncu --set full ./my_cuda_program

# 메모리 대역폭 분석
ncu --metrics dram__throughput.avg.pct_of_peak ./my_program

# 점유율 확인
ncu --metrics sm__warps_active.avg.pct_of_peak ./my_program`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">성능 최적화 체크리스트</h5>
              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-4 rounded-lg">
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>알고리즘 선택</strong>: 병렬화에 적합한 알고리즘 사용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>메모리 접근</strong>: Coalescing, Shared Memory 활용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>통신 최소화</strong>: 비동기 통신, 메시지 합치기</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>로드 밸런싱</strong>: 모든 프로세서 균등 활용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>프로파일링</strong>: 병목 지점 측정 및 개선</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400">✓</span>
                    <span><strong>반복 최적화</strong>: 측정 → 분석 → 개선 사이클</span>
                  </li>
                </ul>
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
              <span>병렬 알고리즘은 데이터/태스크 병렬화 패턴을 적절히 선택해야 한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>GPU 메모리 최적화는 Coalesced Access, Shared Memory, Bank Conflict 회피가 핵심이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>통신 오버헤드는 비동기 통신과 메시지 합치기로 최소화할 수 있다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>동적 로드 밸런싱은 불균등한 작업량 처리에 효과적이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">5.</span>
              <span>프로파일링 도구로 병목 지점을 정확히 파악하고 개선해야 한다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
