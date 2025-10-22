export default function Chapter9() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          클라우드 HPC (Cloud HPC)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            클라우드는 HPC 환경을 민주화하고 있습니다. AWS, Azure, Google Cloud는 온디맨드로
            수천 개의 CPU/GPU를 제공하여 대규모 슈퍼컴퓨터 없이도 HPC 작업을 수행할 수 있게 합니다.
          </p>
        </div>
      </section>

      {/* AWS HPC */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. AWS HPC 아키텍처
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">EC2 HPC 인스턴스</h5>
              <div className="overflow-x-auto mb-3">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">인스턴스</th>
                      <th className="p-2">사양</th>
                      <th className="p-2">용도</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono font-semibold">hpc7a</td>
                      <td className="p-2">192 vCPU, 768 GB RAM, 300 Gbps EFA</td>
                      <td className="p-2">MPI 워크로드</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono font-semibold">p4d.24xlarge</td>
                      <td className="p-2">8× A100 GPU (40GB), 400 Gbps EFA</td>
                      <td className="p-2">AI/ML 훈련</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-mono font-semibold">c6i.32xlarge</td>
                      <td className="p-2">128 vCPU, 256 GB RAM</td>
                      <td className="p-2">범용 컴퓨팅</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">AWS ParallelCluster</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                HPC 클러스터를 자동으로 생성하고 관리하는 오픈소스 도구입니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`# ParallelCluster 설정 예시 (YAML)
Region: us-east-1
Image:
  Os: alinux2

HeadNode:
  InstanceType: c6i.xlarge
  Networking:
    SubnetId: subnet-xxxx
  Ssh:
    KeyName: my-key

Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: compute
      ComputeResources:
        - Name: hpc-nodes
          InstanceType: hpc7a.96xlarge
          MinCount: 0
          MaxCount: 100
      Networking:
        SubnetIds:
          - subnet-xxxx
        PlacementGroup:
          Enabled: true  # 저지연 통신

# 클러스터 생성
pcluster create-cluster --cluster-name my-hpc-cluster \\
                        --cluster-configuration config.yaml

# Slurm 작업 제출
srun -N 50 -n 4800 ./my_mpi_app`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Elastic Fabric Adapter (EFA)</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>특징</strong>: OS-bypass 통신으로 온프레미스 InfiniBand 수준의 성능<br/>
                  <strong>대역폭</strong>: 최대 400 Gbps<br/>
                  <strong>지연시간</strong>: ~10 μs (마이크로초)<br/>
                  <strong>지원</strong>: libfabric, MPI (OpenMPI, Intel MPI)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Azure HPC */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. Azure HPC 솔루션
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">CycleCloud</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                HPC 클러스터를 웹 인터페이스로 관리하는 Azure의 클러스터 오케스트레이션 도구입니다.
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded mb-2">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>주요 기능</strong>:<br/>
                  • 자동 스케일링 (0 → 수천 노드)<br/>
                  • 비용 최적화 (Spot VM 활용)<br/>
                  • Slurm, PBS, Grid Engine 지원<br/>
                  • BeeGFS, Lustre 병렬 파일시스템<br/>
                  • Active Directory 통합
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Azure Batch</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                대규모 병렬 작업을 자동으로 스케줄링하고 실행하는 관리형 서비스입니다.
              </p>
              <div className="bg-gray-900 rounded p-4">
                <pre className="text-sm text-gray-100">
                  <code>{`# Azure CLI로 Batch 작업 생성
az batch pool create \\
  --id my-hpc-pool \\
  --vm-size Standard_HB120rs_v3 \\  # 120 코어 HPC VM
  --node-count 50 \\
  --image "CentOS 7.9" \\
  --node-agent-sku-id "batch.node.centos 7"

# MPI 작업 제출
az batch job create --id my-job --pool-id my-hpc-pool

az batch task create \\
  --job-id my-job \\
  --task-id mpi-task \\
  --command-line "mpirun -np 6000 ./app" \\
  --resource-files input.dat \\
  --output-files results.dat`}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Google Cloud HPC */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. Google Cloud HPC Toolkit
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">HPC Toolkit (오픈소스)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Terraform 기반으로 HPC 인프라를 코드로 정의하고 배포합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`# HPC Toolkit Blueprint (YAML)
blueprint_name: genomics-cluster

deployment_groups:
  - group: primary
    modules:
      - id: network
        source: modules/network/vpc

      - id: homefs
        source: modules/file-system/filestore
        settings:
          filestore_tier: HIGH_SCALE_SSD
          size_gb: 10240

      - id: compute_cluster
        source: modules/scheduler/schedmd-slurm-gcp-v5-controller
        settings:
          machine_type: c2d-highcpu-112
          instance_count: 100

# 배포
./ghpc deploy genomics-cluster.yaml`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">TPU Pods (AI 가속)</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>TPU v5e Pod</strong>: 256개 TPU 칩 (109 PFLOPS)<br/>
                  <strong>용도</strong>: 대규모 Transformer 모델 훈련<br/>
                  <strong>성능</strong>: GPT-3 (175B) 훈련 시간 대폭 단축<br/>
                  <strong>가격</strong>: GPU 대비 60% 저렴 (특정 워크로드)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 비용 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 클라우드 HPC 비용 최적화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Spot/Preemptible 인스턴스 활용</h5>
              <div className="overflow-x-auto mb-3">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">클라우드</th>
                      <th className="p-2">할인율</th>
                      <th className="p-2">특징</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">AWS Spot</td>
                      <td className="p-2 text-green-600 dark:text-green-400">70-90%</td>
                      <td className="p-2">2분 경고 후 중단</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Azure Spot</td>
                      <td className="p-2 text-green-600 dark:text-green-400">60-80%</td>
                      <td className="p-2">30초 경고</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-semibold">GCP Preemptible</td>
                      <td className="p-2 text-green-600 dark:text-green-400">80%</td>
                      <td className="p-2">24시간 최대 실행</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>전략</strong>: 체크포인팅 + Spot 인스턴스<br/>
                  • 100 iteration마다 체크포인트 저장<br/>
                  • Spot 중단 시 자동 재시작<br/>
                  • 비용 절감: 1000 코어 × 24시간 = $2,000 → $400
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">데이터 전송 비용 최적화</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>문제</strong>: 100 TB 데이터 다운로드 = $9,000 (AWS 기준)<br/>
                  <strong>해결책</strong>:<br/>
                  • 동일 리전 내 계산 수행<br/>
                  • S3/Blob Storage에 결과만 저장<br/>
                  • AWS Snowball (물리적 전송) 활용<br/>
                  • CloudFront CDN 캐싱
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
              <span>클라우드 HPC는 온디맨드 확장성과 유연성을 제공한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>EFA, InfiniBand over Cloud는 온프레미스 수준의 네트워크 성능을 제공한다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>ParallelCluster, CycleCloud, HPC Toolkit로 클러스터 관리를 자동화할 수 있다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>Spot 인스턴스와 체크포인팅 조합으로 비용을 80% 절감 가능하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">5.</span>
              <span>클라우드 HPC는 초기 투자 없이 대규모 계산 능력에 접근할 수 있게 한다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
