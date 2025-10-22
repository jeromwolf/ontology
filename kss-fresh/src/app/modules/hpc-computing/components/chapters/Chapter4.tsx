export default function Chapter4() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          GPU 아키텍처 심화
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            GPU는 수천 개의 간단한 코어를 사용하여 대규모 병렬 처리를 수행합니다.
            NVIDIA GPU 아키텍처(Ampere, Hopper)의 내부 구조를 이해하면
            CUDA 프로그램의 성능을 극대화할 수 있습니다.
          </p>
        </div>
      </section>

      {/* GPU vs CPU 아키텍처 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. GPU vs CPU 아키텍처 비교
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
              <h4 className="text-xl font-semibold mb-4 text-blue-600 dark:text-blue-400">
                CPU 아키텍처
              </h4>
              <ul className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <div>
                    <strong>코어 수:</strong> 8-64개 (고성능 코어)
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <div>
                    <strong>설계 철학:</strong> 낮은 지연시간, 복잡한 제어 로직
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <div>
                    <strong>캐시:</strong> 큰 L1/L2/L3 캐시 (수 MB)
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <div>
                    <strong>분기 예측:</strong> 고급 분기 예측 하드웨어
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <div>
                    <strong>적합한 작업:</strong> 직렬 처리, 복잡한 로직
                  </div>
                </li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg">
              <h4 className="text-xl font-semibold mb-4 text-orange-600 dark:text-orange-400">
                GPU 아키텍처
              </h4>
              <ul className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>코어 수:</strong> 수천~수만 개 (간단한 코어)
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>설계 철학:</strong> 높은 처리량, 대규모 병렬성
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>캐시:</strong> 작은 캐시, 큰 메모리 대역폭
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>분기 예측:</strong> 최소한의 분기 예측
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>적합한 작업:</strong> 병렬 처리, 데이터 집약적 연산
                  </div>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-yellow-700 dark:text-yellow-400">핵심 차이:</strong> CPU는 "소수의 강력한 일꾼",
              GPU는 "수천 명의 단순 일꾼"으로 비유할 수 있습니다. CPU는 복잡한 문제를 빠르게 해결하고,
              GPU는 단순한 작업을 엄청나게 많이 동시에 처리합니다.
            </p>
          </div>
        </div>
      </section>

      {/* NVIDIA GPU 아키텍처 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. NVIDIA GPU 아키텍처 계층
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-6 rounded-lg">
              <h4 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">
                GPU 칩 (예: A100, H100)
              </h4>
              <div className="pl-4 space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <p>→ <strong>여러 개의 GPC (Graphics Processing Cluster)</strong></p>
                <p className="pl-4">→ <strong>여러 개의 SM (Streaming Multiprocessor)</strong></p>
                <p className="pl-8">→ <strong>여러 개의 CUDA Core</strong></p>
              </div>
            </div>

            <div>
              <h4 className="text-xl font-semibold mb-4 text-yellow-600 dark:text-yellow-400">
                SM (Streaming Multiprocessor) 구조
              </h4>
              <div className="bg-gray-900 rounded-lg p-4 text-sm text-gray-100 space-y-2">
                <p><strong className="text-yellow-400">CUDA Cores:</strong> 64-128개 (FP32 연산)</p>
                <p><strong className="text-yellow-400">Tensor Cores:</strong> AI/딥러닝 가속 (FP16/INT8)</p>
                <p><strong className="text-yellow-400">Warp Scheduler:</strong> 스레드 그룹 스케줄링</p>
                <p><strong className="text-yellow-400">Register File:</strong> 64KB (스레드별 고속 메모리)</p>
                <p><strong className="text-yellow-400">Shared Memory:</strong> 48-164KB (블록 공유)</p>
                <p><strong className="text-yellow-400">L1 Cache:</strong> 128KB</p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b-2 border-yellow-500">
                    <th className="p-3 font-semibold text-gray-900 dark:text-white">GPU 모델</th>
                    <th className="p-3 font-semibold text-gray-900 dark:text-white">SM 개수</th>
                    <th className="p-3 font-semibold text-gray-900 dark:text-white">CUDA Cores</th>
                    <th className="p-3 font-semibold text-gray-900 dark:text-white">메모리</th>
                    <th className="p-3 font-semibold text-gray-900 dark:text-white">성능 (FP32)</th>
                  </tr>
                </thead>
                <tbody className="text-gray-700 dark:text-gray-300">
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-3 font-semibold">RTX 4090</td>
                    <td className="p-3">128 SM</td>
                    <td className="p-3">16,384</td>
                    <td className="p-3">24GB GDDR6X</td>
                    <td className="p-3">82.6 TFLOPS</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
                    <td className="p-3 font-semibold">A100 (40GB)</td>
                    <td className="p-3">108 SM</td>
                    <td className="p-3">6,912</td>
                    <td className="p-3">40GB HBM2e</td>
                    <td className="p-3">19.5 TFLOPS</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700 bg-orange-50 dark:bg-orange-900/10">
                    <td className="p-3 font-semibold">H100 (80GB)</td>
                    <td className="p-3">132 SM</td>
                    <td className="p-3">16,896</td>
                    <td className="p-3">80GB HBM3</td>
                    <td className="p-3">67 TFLOPS</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Warp 실행 모델 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. Warp 실행 모델
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg mb-6">
            <h4 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
              Warp란?
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              GPU의 기본 실행 단위. <strong className="text-yellow-600">32개의 스레드</strong>가
              하나의 Warp를 구성하며, 같은 명령어를 동시에 실행합니다 (SIMT: Single Instruction, Multiple Threads).
            </p>

            <div className="bg-gray-900 rounded p-4 text-sm">
              <code className="text-gray-100">
                블록 크기 = 256 스레드<br/>
                Warp 개수 = 256 / 32 = 8개 Warp<br/>
                → 각 Warp는 32개 스레드를 동시 실행
              </code>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">
                Warp Scheduler
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                각 SM은 여러 Warp를 관리하며, 메모리 접근 대기 시간 동안 다른 Warp를 실행하여
                지연 시간을 숨깁니다 (Latency Hiding).
              </p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                Warp Divergence 문제
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                같은 Warp 내 스레드가 다른 실행 경로를 가지면 직렬화되어 성능이 저하됩니다.
              </p>
              <div className="bg-gray-900 rounded p-3 text-xs">
                <code className="text-red-400">
                  {`// 나쁜 예: Warp의 절반만 실행
if (threadIdx.x % 2 == 0) {
    // Path A (16 스레드)
} else {
    // Path B (16 스레드)
}
// → 32 사이클 소요 (직렬 실행)`}
                </code>
                <br/><br/>
                <code className="text-green-400">
                  {`// 좋은 예: 전체 Warp 동일 실행
int value = data[threadIdx.x];
value = value * 2;  // 모든 스레드 동시 실행
// → 1 사이클 소요`}
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 메모리 대역폭 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 메모리 대역폭 & 최적화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto mb-6">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b-2 border-yellow-500">
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">메모리 타입</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">지연시간</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">대역폭</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">사용 시나리오</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-green-50 dark:bg-green-900/10">
                  <td className="p-3 font-mono">Register</td>
                  <td className="p-3 text-green-600 font-semibold">1 사이클</td>
                  <td className="p-3">~20 TB/s</td>
                  <td className="p-3">지역 변수, 임시 값</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-green-50 dark:bg-green-900/10">
                  <td className="p-3 font-mono">Shared Memory</td>
                  <td className="p-3 text-green-600 font-semibold">~5 사이클</td>
                  <td className="p-3">~15 TB/s</td>
                  <td className="p-3">블록 내 데이터 공유</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
                  <td className="p-3 font-mono">L1 Cache</td>
                  <td className="p-3 text-yellow-600">~30 사이클</td>
                  <td className="p-3">~10 TB/s</td>
                  <td className="p-3">자동 캐싱</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
                  <td className="p-3 font-mono">L2 Cache</td>
                  <td className="p-3 text-yellow-600">~200 사이클</td>
                  <td className="p-3">~5 TB/s</td>
                  <td className="p-3">전역 캐싱</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-red-50 dark:bg-red-900/10">
                  <td className="p-3 font-mono">Global Memory</td>
                  <td className="p-3 text-red-600 font-semibold">~500 사이클</td>
                  <td className="p-3">1-3 TB/s</td>
                  <td className="p-3">대용량 데이터</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-6 rounded-lg">
            <h5 className="font-semibold mb-3 text-gray-900 dark:text-white">
              최적화 전략
            </h5>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>1. Coalesced Access:</strong> 연속된 메모리 주소 접근으로 트랜잭션 최소화</li>
              <li><strong>2. Shared Memory 활용:</strong> Global Memory 접근 횟수 줄이기</li>
              <li><strong>3. Bank Conflict 회피:</strong> Shared Memory 접근 시 같은 뱅크 회피</li>
              <li><strong>4. Prefetching:</strong> 데이터 미리 로드하여 지연 시간 숨기기</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Tensor Core */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          5. Tensor Core (AI 가속)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            Tensor Core는 딥러닝 행렬 연산(GEMM)을 가속화하는 전용 하드웨어입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                CUDA Core (일반 연산)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                4×4 행렬 곱셈 시간: ~64 사이클
              </p>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                (각 원소마다 곱셈과 덧셈 필요)
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">
                Tensor Core (AI 가속)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                4×4 행렬 곱셈 시간: ~1 사이클
              </p>
              <div className="text-xs text-green-600 dark:text-green-400 font-semibold">
                (64배 빠름! FP16/INT8 연산)
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-orange-600 dark:text-orange-400">H100 Tensor Core 성능:</strong>
              FP16 기준 2,000 TFLOPS (일반 CUDA Core 대비 30배 빠름)
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
              <span>GPU는 수천 개의 간단한 코어로 대규모 병렬 처리에 특화되어 있다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>SM은 Warp(32 스레드) 단위로 실행하며, Warp Divergence를 최소화해야 한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>메모리 계층을 이해하고 Shared Memory를 활용하여 성능을 최적화한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>Tensor Core는 딥러닝 연산을 수십 배 가속화한다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
