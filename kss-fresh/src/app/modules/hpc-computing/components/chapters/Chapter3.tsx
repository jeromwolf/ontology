export default function Chapter3() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          CUDA 프로그래밍
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            CUDA(Compute Unified Device Architecture)는 NVIDIA GPU에서 병렬 처리를 수행하기 위한
            프로그래밍 플랫폼입니다. 수천 개의 코어를 활용하여 CPU 대비 10~100배 빠른 성능을 낼 수 있습니다.
          </p>
        </div>
      </section>

      {/* CUDA 기초 개념 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. CUDA 프로그래밍 모델
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-3 text-yellow-600 dark:text-yellow-400">Host (CPU)</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 직렬 코드 실행</li>
                <li>• 메모리 할당/해제</li>
                <li>• 커널 함수 호출</li>
                <li>• 결과 데이터 수집</li>
              </ul>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-3 text-orange-600 dark:text-orange-400">Device (GPU)</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 병렬 커널 실행</li>
                <li>• 수천 개 스레드 동시 처리</li>
                <li>• 고속 메모리 접근</li>
                <li>• 대량 데이터 연산</li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`// 첫 번째 CUDA 프로그램: 벡터 덧셈
#include <stdio.h>

// GPU에서 실행되는 커널 함수 (__global__)
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    // Host 메모리 할당
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Device 메모리 할당
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Host → Device 데이터 복사
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 커널 실행 (1024개 블록, 블록당 256 스레드)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Device → Host 결과 복사
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* 스레드 계층 구조 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. CUDA 스레드 계층 구조
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Thread (스레드)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                가장 작은 실행 단위. <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">threadIdx.x/y/z</code>로 식별
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Block (블록)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                스레드의 그룹 (최대 1024개). <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">blockIdx.x/y/z</code>로 식별.
                같은 블록 내 스레드는 Shared Memory 공유 가능
              </p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Grid (그리드)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                블록의 그룹. 하나의 커널 호출이 하나의 Grid를 생성
              </p>
            </div>
          </div>

          <div className="mt-6 bg-gray-900 rounded-lg p-4">
            <pre className="text-sm text-gray-100">
              <code>{`// 스레드 인덱스 계산
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 그리드 예시
dim3 threadsPerBlock(16, 16);  // 블록당 256 스레드
dim3 blocksPerGrid(64, 64);     // 4096개 블록
myKernel<<<blocksPerGrid, threadsPerBlock>>>(...);

// 총 스레드 수 = 64 × 64 × 16 × 16 = 1,048,576개`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* 메모리 계층 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. CUDA 메모리 계층
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b-2 border-yellow-500">
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">메모리 타입</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">범위</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">속도</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">크기</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-green-50 dark:bg-green-900/10">
                  <td className="p-3 font-mono">Register</td>
                  <td className="p-3">스레드</td>
                  <td className="p-3 text-green-600 font-semibold">매우 빠름</td>
                  <td className="p-3">~64KB/SM</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-green-50 dark:bg-green-900/10">
                  <td className="p-3 font-mono">Shared Memory</td>
                  <td className="p-3">블록</td>
                  <td className="p-3 text-green-600 font-semibold">매우 빠름</td>
                  <td className="p-3">~48KB/SM</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Local Memory</td>
                  <td className="p-3">스레드</td>
                  <td className="p-3 text-yellow-600">중간</td>
                  <td className="p-3">512KB/스레드</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Global Memory</td>
                  <td className="p-3">전체</td>
                  <td className="p-3 text-red-600">느림</td>
                  <td className="p-3">8-80GB</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Constant Memory</td>
                  <td className="p-3">전체(읽기전용)</td>
                  <td className="p-3 text-yellow-600">중간</td>
                  <td className="p-3">64KB</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-6 bg-gray-900 rounded-lg p-4">
            <pre className="text-sm text-gray-100">
              <code>{`// Shared Memory 사용 예제
__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    __shared__ float As[16][16];  // Shared Memory 선언
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    // Global → Shared 메모리 로드
    As[threadIdx.y][threadIdx.x] = A[row * N + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[threadIdx.y * N + col];
    __syncthreads();  // 블록 내 동기화

    // Shared Memory에서 빠르게 계산
    float sum = 0.0f;
    for (int k = 0; k < 16; k++) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    C[row * N + col] = sum;
}`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* 성능 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. CUDA 성능 최적화 기법
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">1. Coalesced Memory Access (병합된 메모리 접근)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                연속된 스레드가 연속된 메모리 주소에 접근하면 하나의 트랜잭션으로 병합됩니다.
              </p>
              <div className="bg-gray-900 rounded p-2 text-xs">
                <code className="text-green-400">// 좋은 예: arr[threadIdx.x]</code><br/>
                <code className="text-red-400">// 나쁜 예: arr[threadIdx.x * stride]</code>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">2. Occupancy 최적화</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                SM당 활성 Warp 수를 최대화하여 지연 시간을 숨깁니다.
                블록당 스레드 수와 레지스터/Shared Memory 사용량의 균형이 중요합니다.
              </p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">3. Warp Divergence 최소화</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                같은 Warp(32 스레드) 내에서 다른 실행 경로를 가지면 성능이 저하됩니다.
              </p>
              <div className="bg-gray-900 rounded p-2 text-xs">
                <code className="text-red-400">
                  {`if (threadIdx.x % 2 == 0) { /* path A */ }  // 나쁨
else { /* path B */ }`}
                </code>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">4. Stream을 통한 병렬 실행</h4>
              <div className="bg-gray-900 rounded p-2 mt-2 text-xs">
                <code className="text-gray-100">
                  {`cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);  // 비동기 실행
kernel2<<<grid, block, 0, stream2>>>(...);  // 동시 실행 가능`}
                </code>
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
              <span>CUDA는 Grid → Block → Thread 계층 구조로 수만 개 스레드를 관리한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>Shared Memory를 활용하여 Global Memory 접근을 최소화해야 한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>Coalesced Memory Access로 메모리 대역폭을 최대화한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>Warp Divergence를 피하고 Occupancy를 최적화해야 한다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
