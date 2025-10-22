export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          과학 컴퓨팅 응용 (Scientific Computing Applications)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            HPC는 기후 모델링, 분자 동역학, 우주 시뮬레이션 등 복잡한 과학 문제를 해결하는 데 필수적입니다.
            이 챕터에서는 실제 과학 분야에서 HPC를 어떻게 활용하는지 살펴봅니다.
          </p>
        </div>
      </section>

      {/* 물리 시뮬레이션 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. 물리 시뮬레이션 (Physics Simulation)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h4 className="text-xl font-semibold mb-3 text-yellow-600 dark:text-yellow-400">
                분자 동역학 (Molecular Dynamics)
              </h4>
              <p className="mb-3 text-gray-700 dark:text-gray-300">
                수백만 개 원자의 운동을 시뮬레이션하여 단백질 구조, 신약 개발 등에 활용합니다.
              </p>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto mb-3">
                <pre className="text-sm text-gray-100">
                  <code>{`// LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
# 단백질 시뮬레이션 예시

# 초기화
units           real
atom_style      full
boundary        p p p

# 힘장 정의
pair_style      lj/cut 10.0
bond_style      harmonic
angle_style     harmonic

# 병렬 도메인 분할
processors      * * *  # 자동 분할

# 타임스텝 적분
fix             1 all nvt temp 300.0 300.0 100.0
timestep        2.0  # 2 femtoseconds

# 시뮬레이션 실행 (1 nanosecond)
run             500000

# 성능: 10^6 atoms, 1000 cores → ~100 ns/day`}</code>
                </pre>
              </div>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                  <h5 className="font-semibold mb-2 text-sm text-gray-900 dark:text-white">주요 응용</h5>
                  <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 단백질 폴딩 (Protein Folding)</li>
                    <li>• 신약 설계 (Drug Design)</li>
                    <li>• 재료 과학 (나노 물질)</li>
                    <li>• 생물막 시뮬레이션</li>
                  </ul>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                  <h5 className="font-semibold mb-2 text-sm text-gray-900 dark:text-white">HPC 요구사항</h5>
                  <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 고속 네트워크 (InfiniBand)</li>
                    <li>• 대용량 메모리 (TB급)</li>
                    <li>• 장시간 작업 (주/월 단위)</li>
                    <li>• 체크포인팅 필수</li>
                  </ul>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-xl font-semibold mb-3 text-orange-600 dark:text-orange-400">
                유체 역학 (Computational Fluid Dynamics)
              </h4>
              <p className="mb-3 text-gray-700 dark:text-gray-300">
                항공기 설계, 기상 예보, 난류 해석 등에 사용되는 CFD 시뮬레이션입니다.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">응용 분야</th>
                      <th className="p-2">격자 수</th>
                      <th className="p-2">소프트웨어</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2">항공기 설계</td>
                      <td className="p-2">10억+ 셀</td>
                      <td className="p-2 font-mono">OpenFOAM, ANSYS Fluent</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2">기후 모델링</td>
                      <td className="p-2">1000억+ 셀</td>
                      <td className="p-2 font-mono">WRF, CESM</td>
                    </tr>
                    <tr>
                      <td className="p-2">엔진 연소</td>
                      <td className="p-2">1억+ 셀</td>
                      <td className="p-2 font-mono">CONVERGE, StarCCM+</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 수치 해석 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. 수치 해석 (Numerical Analysis)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">선형 시스템 솔버</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                대규모 선형 방정식 Ax = b를 푸는 것은 많은 과학 문제의 핵심입니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// PETSc (Portable, Extensible Toolkit for Scientific Computation)
#include <petscksp.h>

int main(int argc, char **argv) {
    Mat A;
    Vec x, b;
    KSP ksp;

    PetscInitialize(&argc, &argv, NULL, NULL);

    // 100억 × 100억 희소 행렬 생성 (분산 저장)
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10000000000, 10000000000);
    MatSetType(A, MATAIJ);

    // Krylov Subspace 방법 (Conjugate Gradient)
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);

    // 멀티그리드 전처리자 (수렴 가속)
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);

    // 병렬 해결 (MPI)
    KSPSolve(ksp, b, x);

    // 1000+ 코어에서 수십억 미지수 해결 가능
}`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">FFT (Fast Fourier Transform)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                신호 처리, 이미지 분석, 우주론 시뮬레이션에 필수적입니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`// FFTW (Fastest Fourier Transform in the West)
#include <fftw3-mpi.h>

int main(int argc, char **argv) {
    fftw_mpi_init();
    MPI_Init(&argc, &argv);

    // 4096³ 3D FFT (분산 처리)
    ptrdiff_t N0 = 4096, N1 = 4096, N2 = 4096;
    ptrdiff_t local_n0, local_0_start;

    // 로컬 배열 크기 계산
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(
        N0, N1, N2, MPI_COMM_WORLD, &local_n0, &local_0_start
    );

    fftw_complex *data = fftw_alloc_complex(alloc_local);

    // 3D FFT 플랜 생성
    fftw_plan plan = fftw_mpi_plan_dft_3d(
        N0, N1, N2, data, data,
        MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(plan);  // 병렬 실행

    // 성능: ~10 TFLOPS on 1000 cores
}`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">몬테카를로 시뮬레이션</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                무작위 샘플링을 통해 복잡한 적분, 확률 문제를 해결합니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>장점</strong>: 병렬화가 매우 쉬움 (Embarrassingly Parallel)<br/>
                  <strong>응용</strong>: 금융 공학 (옵션 가격), 입자 물리, 방사선 치료 계획<br/>
                  <strong>확장성</strong>: 거의 선형 (10,000+ 코어)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 빅데이터 과학 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. 빅데이터 과학 (Big Data Science)
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">천문학 데이터 분석</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                LSST (Large Synoptic Survey Telescope)는 하루 20TB의 데이터를 생성합니다.
              </p>
              <div className="overflow-x-auto mb-3">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">프로젝트</th>
                      <th className="p-2">데이터량</th>
                      <th className="p-2">HPC 활용</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">SKA (Square Kilometre Array)</td>
                      <td className="p-2">1 Exabyte/day</td>
                      <td className="p-2">실시간 신호 처리</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">LHC (입자가속기)</td>
                      <td className="p-2">50 Petabytes/년</td>
                      <td className="p-2">이벤트 재구성</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-semibold">유전체학 (Genomics)</td>
                      <td className="p-2">100 Petabytes</td>
                      <td className="p-2">시퀀스 정렬</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">기후 데이터 분석</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                CMIP6 (기후 모델 비교 프로젝트)는 20 Petabytes의 시뮬레이션 데이터를 생성합니다.
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`# Dask - 파이썬 병렬 컴퓨팅
import dask.array as da
from dask.distributed import Client

# 1000 노드 클러스터 연결
client = Client('scheduler-address:8786')

# 100 TB 기후 데이터 로드 (lazy evaluation)
temperature = da.from_zarr('/data/cmip6/temperature.zarr')
# Shape: (365*100, 721, 1440) - 100년 일일 데이터

# 병렬 연산 (평균 계산)
annual_mean = temperature.reshape(-1, 365, 721, 1440).mean(axis=1)

# 실제 계산 트리거 (1000 노드에서 병렬 실행)
result = annual_mean.compute()

# I/O 최적화: Zarr, HDF5, NetCDF 포맷 활용`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">생명과학 데이터</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>단일 세포 RNA 시퀀싱</strong>:<br/>
                  • 샘플당 수백만 개 세포<br/>
                  • 세포당 20,000+ 유전자<br/>
                  • 희소 행렬 연산 (98% zeros)<br/>
                  • Scanpy, Seurat (R) 활용<br/><br/>
                  <strong>HPC 최적화</strong>: GPU 가속 (RAPIDS cuML), 분산 처리 (Dask)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 사례 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 실전 사례 연구
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                사례 1: COVID-19 약물 재창출 (2020)
              </h5>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded mb-2">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>목표</strong>: SARS-CoV-2 단백질에 결합하는 기존 약물 찾기<br/>
                  <strong>HPC 사용</strong>: Summit 슈퍼컴퓨터 (200 petaFLOPS)<br/>
                  <strong>계산량</strong>: 10억 개 화합물 도킹 시뮬레이션<br/>
                  <strong>시간</strong>: 12시간 (기존 방법: 수년)<br/>
                  <strong>결과</strong>: 77개 후보 약물 발견 → 임상 시험
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                사례 2: 블랙홀 이미지 생성 (Event Horizon Telescope, 2019)
              </h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded mb-2">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>데이터</strong>: 5 Petabytes (8개 전파망원경)<br/>
                  <strong>처리</strong>: 간섭계 데이터 보정 및 이미지 재구성<br/>
                  <strong>알고리즘</strong>: CLEAN, SMILI (희소 모델링)<br/>
                  <strong>HPC 클러스터</strong>: MIT, MPIfR 슈퍼컴퓨터<br/>
                  <strong>계산 시간</strong>: 수개월 (반복 최적화)<br/>
                  <strong>성과</strong>: 인류 최초 블랙홀 이미지
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">
                사례 3: 핵융합 시뮬레이션 (ITER 프로젝트)
              </h5>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>목표</strong>: 플라즈마 난류 및 안정성 예측<br/>
                  <strong>코드</strong>: GENE, GYRO (자이로운동론적 시뮬레이션)<br/>
                  <strong>격자</strong>: 6차원 위상 공간 (10¹² 그리드 포인트)<br/>
                  <strong>HPC</strong>: 100,000+ CPU 코어, GPU 가속<br/>
                  <strong>결과</strong>: 에너지 손실 메커니즘 이해 → ITER 설계 최적화
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
              <span>분자 동역학과 CFD는 가장 대표적인 HPC 물리 시뮬레이션 응용이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>대규모 선형 시스템과 FFT는 수많은 과학 문제의 핵심 연산이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>빅데이터 과학은 Exabyte 규모의 데이터 처리 능력을 요구한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>HPC는 신약 개발, 기후 연구, 우주 탐사 등 인류의 난제 해결에 필수적이다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">5.</span>
              <span>도메인 특화 소프트웨어와 수치 라이브러리의 조합이 성공의 열쇠다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
