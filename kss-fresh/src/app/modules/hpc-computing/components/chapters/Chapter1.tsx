export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          High-Performance Computing (HPC) 기초
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            High-Performance Computing(HPC)은 복잡한 계산 문제를 빠르게 해결하기 위해
            수천 개의 프로세서를 병렬로 사용하는 기술입니다. 과학 연구, 날씨 예측,
            신약 개발, AI 모델 학습 등 대규모 계산이 필요한 모든 분야에서 필수적입니다.
          </p>
        </div>
      </section>

      {/* HPC의 정의와 중요성 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. HPC란 무엇인가?
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                핵심 개념
              </h4>
              <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-3">
                  <span className="text-yellow-500 mt-1">⚡</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">병렬 처리:</strong> 하나의 큰 문제를 여러 작은 문제로 나누어 동시에 처리
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-yellow-500 mt-1">🖥️</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">슈퍼컴퓨터:</strong> 수천~수백만 개의 CPU 코어를 가진 거대한 컴퓨팅 시스템
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-yellow-500 mt-1">📊</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">FLOPS:</strong> Floating Point Operations Per Second - HPC 성능 측정 단위
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-yellow-500 mt-1">🌐</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">클러스터:</strong> 네트워크로 연결된 여러 컴퓨터를 하나의 시스템처럼 사용
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 성능 측정 단위 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. HPC 성능 측정
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b-2 border-yellow-500">
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">단위</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">연산 횟수/초</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">활용 예시</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-yellow-600">KiloFLOPS</td>
                  <td className="p-3">10³ (천)</td>
                  <td className="p-3">초기 개인용 컴퓨터</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-yellow-600">MegaFLOPS</td>
                  <td className="p-3">10⁶ (백만)</td>
                  <td className="p-3">1980년대 워크스테이션</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-yellow-600">GigaFLOPS</td>
                  <td className="p-3">10⁹ (십억)</td>
                  <td className="p-3">현대 스마트폰</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-yellow-600">TeraFLOPS</td>
                  <td className="p-3">10¹² (조)</td>
                  <td className="p-3">고성능 GPU (RTX 4090: 82.6 TFLOPS)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-yellow-600">PetaFLOPS</td>
                  <td className="p-3">10¹⁵ (천조)</td>
                  <td className="p-3">슈퍼컴퓨터 (Frontier: 1.1 ExaFLOPS)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono font-semibold text-orange-600">ExaFLOPS</td>
                  <td className="p-3">10¹⁸ (백경)</td>
                  <td className="p-3">세계 최고 성능 슈퍼컴퓨터 (2022~)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-yellow-700 dark:text-yellow-400">참고:</strong> 2022년 미국의 Frontier 슈퍼컴퓨터가
              세계 최초로 ExaFLOPS 벽을 돌파했습니다. 이는 1초에 1,000,000,000,000,000,000번의 부동소수점 연산을 수행할 수 있는 성능입니다.
            </p>
          </div>
        </div>
      </section>

      {/* HPC 활용 분야 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. HPC 활용 분야
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400 flex items-center gap-2">
              <span>🧬</span> 과학 연구
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>기후 모델링:</strong> 지구 온난화 시뮬레이션 (수백만 개 변수)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>분자 동역학:</strong> 단백질 폴딩, 신약 개발</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>천체물리학:</strong> 우주 시뮬레이션, 블랙홀 연구</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>유전체학:</strong> DNA 시퀀싱, 게놈 분석</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400 flex items-center gap-2">
              <span>🤖</span> 인공지능
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>대규모 모델 학습:</strong> GPT-4, Claude 같은 LLM 학습</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>딥러닝:</strong> 수천 개 GPU로 병렬 학습</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>강화학습:</strong> AlphaGo, AlphaFold 같은 시스템</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>컴퓨터 비전:</strong> 자율주행, 의료 영상 분석</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400 flex items-center gap-2">
              <span>🏭</span> 산업 응용
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>공학 시뮬레이션:</strong> 자동차, 항공기 설계 (CFD)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>에너지:</strong> 석유 탐사, 원자로 시뮬레이션</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>금융:</strong> 리스크 분석, 고빈도 거래 (HFT)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <span><strong>제조:</strong> 디지털 트윈, 공정 최적화</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400 flex items-center gap-2">
              <span>💊</span> 의료/제약
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>신약 개발:</strong> 약물-단백질 상호작용 시뮬레이션</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>맞춤 의료:</strong> 유전체 분석 기반 치료</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>의료 영상:</strong> MRI, CT 이미지 재구성</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>전염병 모델링:</strong> COVID-19 확산 예측</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* HPC 시스템 구성 요소 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. HPC 시스템 구성 요소
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                🖥️ 계산 노드 (Compute Nodes)
              </h4>
              <div className="pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                <p><strong>CPU 노드:</strong> Intel Xeon, AMD EPYC 프로세서 (16-128 코어)</p>
                <p><strong>GPU 노드:</strong> NVIDIA A100, H100 (수만 개 CUDA 코어)</p>
                <p><strong>메모리:</strong> 노드당 256GB ~ 2TB RAM</p>
                <p><strong>스토리지:</strong> 고속 NVMe SSD, 병렬 파일 시스템</p>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                🌐 네트워크 인터커넥트
              </h4>
              <div className="pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                <p><strong>InfiniBand:</strong> 초저지연 (1-2μs), 초고속 (200-400 Gbps)</p>
                <p><strong>Ethernet:</strong> 100 Gbps+ 네트워크 (비용 효율적)</p>
                <p><strong>토폴로지:</strong> Fat-Tree, Dragonfly, Torus 구조</p>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                💾 스토리지 시스템
              </h4>
              <div className="pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                <p><strong>병렬 파일 시스템:</strong> Lustre, GPFS (수 PB 용량)</p>
                <p><strong>Burst Buffer:</strong> NVMe 기반 고속 임시 저장소</p>
                <p><strong>아카이브:</strong> 테이프 라이브러리 (장기 보관)</p>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                ⚙️ 작업 스케줄러
              </h4>
              <div className="pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                <p><strong>Slurm:</strong> 가장 널리 사용되는 오픈소스 스케줄러</p>
                <p><strong>PBS Pro:</strong> 상용 스케줄러 (고급 기능)</p>
                <p><strong>LSF:</strong> IBM Spectrum LSF (엔터프라이즈용)</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* TOP500 슈퍼컴퓨터 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          5. 세계 TOP 슈퍼컴퓨터 (2024)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b-2 border-yellow-500">
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">순위</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">시스템명</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">국가</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">성능</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">프로세서</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
                  <td className="p-3 font-bold text-yellow-600">1</td>
                  <td className="p-3 font-semibold">Frontier</td>
                  <td className="p-3">🇺🇸 미국 (ORNL)</td>
                  <td className="p-3 font-mono">1.194 ExaFLOPS</td>
                  <td className="p-3">AMD EPYC + AMD MI250X GPU</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-bold text-gray-600">2</td>
                  <td className="p-3 font-semibold">Aurora</td>
                  <td className="p-3">🇺🇸 미국 (ANL)</td>
                  <td className="p-3 font-mono">1.012 ExaFLOPS</td>
                  <td className="p-3">Intel Xeon + Intel Data Center GPU</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-bold text-gray-600">3</td>
                  <td className="p-3 font-semibold">Eagle</td>
                  <td className="p-3">🇺🇸 미국 (Microsoft)</td>
                  <td className="p-3 font-mono">561 PetaFLOPS</td>
                  <td className="p-3">NVIDIA H100 GPU</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-bold text-gray-600">4</td>
                  <td className="p-3 font-semibold">Fugaku</td>
                  <td className="p-3">🇯🇵 일본 (RIKEN)</td>
                  <td className="p-3 font-mono">442 PetaFLOPS</td>
                  <td className="p-3">Fujitsu A64FX ARM</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-bold text-gray-600">11</td>
                  <td className="p-3 font-semibold">누리온</td>
                  <td className="p-3">🇰🇷 한국 (KISTI)</td>
                  <td className="p-3 font-mono">25.7 PetaFLOPS</td>
                  <td className="p-3">Intel Xeon Phi KNL</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-yellow-700 dark:text-yellow-400">흥미로운 사실:</strong> Frontier 슈퍼컴퓨터는
              세계 모든 노트북을 합친 것보다 약 1,000배 빠른 성능을 가지고 있으며,
              하루 전기료만 약 1억 원이 넘습니다. 하지만 단 1초 만에 인류가 75년간 계산할 수 있는 양을 처리할 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      {/* 실습 예제 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          6. 첫 번째 HPC 프로그램 (병렬 Hello World)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-4 text-gray-700 dark:text-gray-300">
            OpenMP를 사용한 간단한 병렬 프로그램 예제입니다:
          </p>

          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        printf("안녕하세요! 저는 %d번 스레드입니다. (전체: %d개)\\n",
               thread_id, total_threads);
    }

    return 0;
}`}</code>
            </pre>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>컴파일 & 실행:</strong>
            </p>
            <div className="mt-2 bg-gray-900 rounded p-2">
              <code className="text-sm text-green-400">
                gcc -fopenmp hello_parallel.c -o hello_parallel<br/>
                export OMP_NUM_THREADS=8<br/>
                ./hello_parallel
              </code>
            </div>
          </div>

          <div className="mt-4 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>출력 예시:</strong><br/>
              안녕하세요! 저는 0번 스레드입니다. (전체: 8개)<br/>
              안녕하세요! 저는 3번 스레드입니다. (전체: 8개)<br/>
              안녕하세요! 저는 1번 스레드입니다. (전체: 8개)<br/>
              ... (순서는 랜덤)
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
              <span>HPC는 대규모 계산 문제를 병렬 처리로 해결하는 기술이며, ExaFLOPS 시대에 진입했다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>계산 노드, 고속 네트워크, 병렬 스토리지, 작업 스케줄러가 핵심 구성 요소다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>과학, AI, 산업, 의료 등 거의 모든 분야에서 HPC가 필수적으로 사용된다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>OpenMP, MPI, CUDA 같은 병렬 프로그래밍 프레임워크를 통해 HPC를 활용한다</span>
            </li>
          </ul>
        </div>
      </section>

      {/* 다음 단계 */}
      <section>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
            🚀 다음 단계
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            다음 챕터에서는 OpenMP와 MPI를 사용한 병렬 프로그래밍을 본격적으로 다룹니다.
          </p>
        </div>
      </section>
    </div>
  )
}
