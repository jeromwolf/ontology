export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          클러스터 컴퓨팅
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            HPC 클러스터는 수백~수천 개의 컴퓨터(노드)를 고속 네트워크로 연결하여
            하나의 강력한 시스템처럼 작동시킵니다. Slurm 같은 작업 스케줄러를 통해
            수천 명의 사용자가 효율적으로 컴퓨팅 자원을 공유할 수 있습니다.
          </p>
        </div>
      </section>

      {/* 클러스터 아키텍처 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. HPC 클러스터 아키텍처
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg">
              <h4 className="text-xl font-semibold mb-4 text-yellow-600 dark:text-yellow-400">
                클러스터 구성 요소
              </h4>
              <div className="space-y-4 text-gray-700 dark:text-gray-300">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🖥️</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">Login 노드:</strong>
                    <p className="text-sm">사용자 접속, 작업 제출, 코드 컴파일</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">⚡</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">Compute 노드:</strong>
                    <p className="text-sm">실제 계산 수행 (CPU/GPU)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">💾</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">Storage 노드:</strong>
                    <p className="text-sm">병렬 파일 시스템 (Lustre, GPFS)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🌐</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">네트워크:</strong>
                    <p className="text-sm">InfiniBand (200-400 Gbps, 1-2μs 지연)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🎛️</span>
                  <div>
                    <strong className="text-gray-900 dark:text-white">관리 노드:</strong>
                    <p className="text-sm">스케줄러, 모니터링, 보안</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 rounded-lg p-6">
              <h5 className="text-lg font-semibold mb-3 text-yellow-400">
                전형적인 클러스터 구성 예시
              </h5>
              <div className="text-sm text-gray-100 space-y-2">
                <p>• <strong>Login 노드:</strong> 2-4대 (로드 밸런싱)</p>
                <p>• <strong>Compute 노드:</strong> 100-10,000대</p>
                <p className="pl-4">- CPU 노드: 각 2 × 64코어 AMD EPYC</p>
                <p className="pl-4">- GPU 노드: 각 8 × NVIDIA A100</p>
                <p>• <strong>Storage:</strong> 10-100 PB Lustre 파일 시스템</p>
                <p>• <strong>네트워크:</strong> InfiniBand HDR (200 Gbps)</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 작업 스케줄러 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. Slurm 작업 스케줄러
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="mb-6 text-gray-700 dark:text-gray-300">
            Slurm(Simple Linux Utility for Resource Management)은 가장 널리 사용되는
            오픈소스 HPC 스케줄러입니다. 작업 우선순위, 자원 할당, 공정성을 관리합니다.
          </p>

          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                기본 Slurm 명령어
              </h4>
              <div className="space-y-4">
                <div className="bg-gray-900 rounded-lg p-4">
                  <p className="text-yellow-400 mb-2">작업 제출 (sbatch)</p>
                  <code className="text-sm text-gray-100">
                    sbatch my_job.sh
                  </code>
                </div>

                <div className="bg-gray-900 rounded-lg p-4">
                  <p className="text-yellow-400 mb-2">작업 상태 확인 (squeue)</p>
                  <code className="text-sm text-gray-100">
                    squeue -u $USER  # 내 작업만 보기<br/>
                    squeue -p gpu    # GPU 파티션 작업 보기
                  </code>
                </div>

                <div className="bg-gray-900 rounded-lg p-4">
                  <p className="text-yellow-400 mb-2">작업 취소 (scancel)</p>
                  <code className="text-sm text-gray-100">
                    scancel 12345    # Job ID로 취소<br/>
                    scancel -u $USER # 내 모든 작업 취소
                  </code>
                </div>

                <div className="bg-gray-900 rounded-lg p-4">
                  <p className="text-yellow-400 mb-2">노드 정보 (sinfo)</p>
                  <code className="text-sm text-gray-100">
                    sinfo            # 전체 노드 상태<br/>
                    sinfo -N -l      # 상세 노드 정보
                  </code>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-3 text-orange-600 dark:text-orange-400">
                Slurm 작업 스크립트 예제
              </h4>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm text-gray-100">
                  <code>{`#!/bin/bash
#SBATCH --job-name=my_simulation
#SBATCH --nodes=4                  # 4개 노드 요청
#SBATCH --ntasks-per-node=32       # 노드당 32 프로세스
#SBATCH --cpus-per-task=2          # 태스크당 2 CPU
#SBATCH --gres=gpu:4               # 노드당 GPU 4개
#SBATCH --time=24:00:00            # 최대 실행 시간 24시간
#SBATCH --partition=gpu            # GPU 파티션
#SBATCH --output=job_%j.out        # 출력 파일 (%j = Job ID)
#SBATCH --error=job_%j.err         # 에러 파일

# 환경 설정
module load cuda/12.0
module load openmpi/4.1.5

# 작업 실행
mpirun -np 128 ./my_program input.dat

# 총 프로세스 수 = 4 nodes × 32 tasks = 128 MPI processes`}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 네트워크 토폴로지 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. 네트워크 토폴로지
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg">
              <h4 className="font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                Fat-Tree 토폴로지
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                가장 일반적인 구조. 상위 스위치로 갈수록 대역폭이 증가합니다.
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>✓ 높은 대역폭</li>
                <li>✓ 확장성 우수</li>
                <li>✗ 비용이 높음</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg">
              <h4 className="font-semibold mb-3 text-orange-600 dark:text-orange-400">
                Dragonfly 토폴로지
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                그룹 기반 계층 구조. 최신 슈퍼컴퓨터에서 사용합니다.
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>✓ 낮은 지름(Diameter)</li>
                <li>✓ 비용 효율적</li>
                <li>✓ 확장성 매우 우수</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-yellow-700 dark:text-yellow-400">참고:</strong> Frontier 슈퍼컴퓨터는
              Dragonfly 토폴로지를 사용하여 9,000개 이상의 노드를 연결합니다.
              노드 간 평균 홉 수는 3.5개로 매우 낮습니다.
            </p>
          </div>
        </div>
      </section>

      {/* 병렬 파일 시스템 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 병렬 파일 시스템
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="overflow-x-auto mb-6">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b-2 border-yellow-500">
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">파일 시스템</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">개발사</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">최대 대역폭</th>
                  <th className="p-3 font-semibold text-gray-900 dark:text-white">특징</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
                  <td className="p-3 font-semibold">Lustre</td>
                  <td className="p-3">오픈소스</td>
                  <td className="p-3">1+ TB/s</td>
                  <td className="p-3">가장 널리 사용, 대규모 확장</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">GPFS (Spectrum Scale)</td>
                  <td className="p-3">IBM</td>
                  <td className="p-3">500+ GB/s</td>
                  <td className="p-3">안정성 우수, 상용</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">BeeGFS</td>
                  <td className="p-3">ThinkParQ</td>
                  <td className="p-3">100+ GB/s</td>
                  <td className="p-3">설치 간편, 중소형 클러스터</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg">
            <h5 className="font-semibold mb-3 text-gray-900 dark:text-white">
              Lustre 아키텍처
            </h5>
            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <p><strong>MDS (Metadata Server):</strong> 파일 메타데이터 관리</p>
              <p><strong>OSS (Object Storage Server):</strong> 실제 데이터 저장</p>
              <p><strong>클라이언트:</strong> 각 컴퓨트 노드에서 마운트</p>
              <p className="mt-4 text-xs text-gray-600 dark:text-gray-400">
                → 수천 개 클라이언트가 동시에 1개 파일을 읽고 쓸 수 있음
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 클러스터 모니터링 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          5. 클러스터 모니터링 & 관리
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                성능 모니터링
              </h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Ganglia</li>
                <li>• Prometheus</li>
                <li>• Grafana</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-orange-600 dark:text-orange-400">
                작업 회계
              </h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Slurm Accounting</li>
                <li>• PBS Pro</li>
                <li>• 사용량 리포트</li>
              </ul>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                환경 관리
              </h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Environment Modules</li>
                <li>• Spack (패키지 관리)</li>
                <li>• Singularity (컨테이너)</li>
              </ul>
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
              <span>HPC 클러스터는 Login/Compute/Storage 노드로 구성되며 InfiniBand로 연결된다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>Slurm 스케줄러로 작업을 제출하고 자원을 효율적으로 관리한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>병렬 파일 시스템(Lustre)을 통해 수천 개 노드가 동시에 데이터에 접근한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>Fat-Tree, Dragonfly 같은 네트워크 토폴로지로 확장성을 확보한다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
