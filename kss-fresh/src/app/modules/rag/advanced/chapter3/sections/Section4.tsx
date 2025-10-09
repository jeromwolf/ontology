import { Shield } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
          <Shield className="text-red-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.4 장애 복구와 99.99% 가용성 달성</h2>
          <p className="text-gray-600 dark:text-gray-400">Netflix 수준의 안정성을 위한 전략</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
          <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">다층 방어 아키텍처</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# 장애 복구 시스템 구현
import asyncio
from enum import Enum
from typing import List, Dict, Optional, Callable
import logging
from datetime import datetime, timedelta

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"

class ReplicaManager:
    def __init__(self, primary_nodes: List[str], replica_factor: int = 3):
        """
        복제본 관리자
        - 자동 복제본 승격
        - 데이터 일관성 보장
        - 복구 오케스트레이션
        """
        self.primary_nodes = primary_nodes
        self.replica_factor = replica_factor
        self.replicas: Dict[str, List[str]] = self._init_replicas()
        self.health_status: Dict[str, HealthStatus] = {}
        self.last_sync: Dict[str, datetime] = {}

    def _init_replicas(self) -> Dict[str, List[str]]:
        """각 프라이머리 노드의 복제본 초기화"""
        replicas = {}
        for i, primary in enumerate(self.primary_nodes):
            # 다른 노드들을 복제본으로 할당
            replica_nodes = []
            for j in range(1, self.replica_factor + 1):
                replica_idx = (i + j) % len(self.primary_nodes)
                replica_nodes.append(self.primary_nodes[replica_idx])
            replicas[primary] = replica_nodes
        return replicas

    async def monitor_health(self):
        """노드 상태 모니터링"""
        while True:
            tasks = []
            for node in self.primary_nodes:
                task = self._check_node_health(node)
                tasks.append(task)

            await asyncio.gather(*tasks)
            await asyncio.sleep(5)  # 5초마다 체크

    async def _check_node_health(self, node: str) -> HealthStatus:
        """개별 노드 상태 확인"""
        try:
            # 1. 기본 연결 체크
            response = await self._ping_node(node)
            if not response:
                self.health_status[node] = HealthStatus.DEAD
                await self._handle_node_failure(node)
                return HealthStatus.DEAD

            # 2. 응답 시간 체크
            if response['latency'] > 1000:  # 1초 이상
                self.health_status[node] = HealthStatus.DEGRADED
            elif response['latency'] > 500:  # 500ms 이상
                self.health_status[node] = HealthStatus.UNHEALTHY
            else:
                self.health_status[node] = HealthStatus.HEALTHY

            # 3. 데이터 동기화 체크
            last_sync = self.last_sync.get(node)
            if last_sync and datetime.now() - last_sync > timedelta(minutes=5):
                await self._trigger_sync(node)

            return self.health_status[node]

        except Exception as e:
            logging.error(f"Health check failed for {node}: {e}")
            self.health_status[node] = HealthStatus.UNHEALTHY
            return HealthStatus.UNHEALTHY

    async def _handle_node_failure(self, failed_node: str):
        """노드 장애 처리"""
        logging.critical(f"Node {failed_node} failed! Initiating failover...")

        # 1. 복제본 중 하나를 새 프라이머리로 승격
        replicas = self.replicas.get(failed_node, [])
        new_primary = None

        for replica in replicas:
            if self.health_status.get(replica) == HealthStatus.HEALTHY:
                new_primary = replica
                break

        if not new_primary:
            logging.error(f"No healthy replica found for {failed_node}")
            return

        # 2. 승격 프로세스
        await self._promote_replica(failed_node, new_primary)

        # 3. 클라이언트 라우팅 업데이트
        await self._update_routing_table(failed_node, new_primary)

        # 4. 새로운 복제본 생성
        await self._create_new_replicas(new_primary)

        logging.info(f"Failover complete: {failed_node} -> {new_primary}")

# 분산 트랜잭션 관리
class DistributedTransaction:
    def __init__(self, coordinator_url: str):
        """
        2단계 커밋을 사용한 분산 트랜잭션
        """
        self.coordinator = coordinator_url
        self.participants: List[str] = []
        self.transaction_id: str = None
        self.state = "INITIAL"

    async def begin(self):
        """트랜잭션 시작"""
        self.transaction_id = f"txn_{datetime.now().timestamp()}"
        self.state = "PREPARING"

        # 코디네이터에 트랜잭션 등록
        await self._register_transaction()

    async def add_operation(self, node: str, operation: Dict):
        """트랜잭션에 작업 추가"""
        self.participants.append(node)

        # 각 참여자에게 준비 요청
        prepare_result = await self._prepare_on_node(node, operation)
        if not prepare_result:
            await self.rollback()
            raise Exception(f"Prepare failed on {node}")

    async def commit(self):
        """트랜잭션 커밋"""
        if self.state != "PREPARING":
            raise Exception("Invalid transaction state")

        self.state = "COMMITTING"

        # 모든 참여자에게 커밋 요청
        commit_tasks = []
        for participant in self.participants:
            task = self._commit_on_node(participant)
            commit_tasks.append(task)

        results = await asyncio.gather(*commit_tasks, return_exceptions=True)

        # 하나라도 실패하면 롤백
        if any(isinstance(r, Exception) for r in results):
            await self.rollback()
            raise Exception("Commit failed")

        self.state = "COMMITTED"

    async def rollback(self):
        """트랜잭션 롤백"""
        self.state = "ROLLING_BACK"

        rollback_tasks = []
        for participant in self.participants:
            task = self._rollback_on_node(participant)
            rollback_tasks.append(task)

        await asyncio.gather(*rollback_tasks, return_exceptions=True)
        self.state = "ROLLED_BACK"

# 실전 예제: Chaos Engineering
class ChaosMonkey:
    """
    프로덕션 환경에서 장애 복구 테스트
    Netflix의 Chaos Monkey 구현
    """
    def __init__(self, cluster_manager, enabled: bool = False):
        self.cluster = cluster_manager
        self.enabled = enabled
        self.failure_probability = 0.001  # 0.1% 확률

    async def run(self):
        """주기적으로 무작위 장애 발생"""
        while self.enabled:
            await asyncio.sleep(60)  # 1분마다 실행

            if np.random.random() < self.failure_probability:
                # 무작위 노드 선택
                victim = np.random.choice(self.cluster.nodes)

                # 장애 유형 선택
                failure_type = np.random.choice([
                    'network_partition',
                    'high_latency',
                    'resource_exhaustion',
                    'process_crash'
                ])

                logging.warning(f"Chaos Monkey: Inducing {failure_type} on {victim}")
                await self._induce_failure(victim, failure_type)

    async def _induce_failure(self, node: str, failure_type: str):
        """실제 장애 유발"""
        if failure_type == 'network_partition':
            # 네트워크 격리
            await self._isolate_node(node, duration=30)
        elif failure_type == 'high_latency':
            # 높은 지연 유발
            await self._add_latency(node, latency_ms=5000, duration=60)
        elif failure_type == 'resource_exhaustion':
            # CPU/메모리 소진
            await self._exhaust_resources(node, duration=45)
        elif failure_type == 'process_crash':
            # 프로세스 강제 종료
            await self._kill_process(node)`}
            </pre>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">실제 장애 시나리오와 복구 전략</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">시나리오 1: 데이터센터 전체 장애</h4>
              <div className="text-sm text-gray-700 dark:text-gray-300">
                <p className="mb-2"><strong>상황:</strong> AWS us-east-1 리전 전체 다운</p>
                <p className="mb-2"><strong>복구 전략:</strong></p>
                <ul className="list-disc list-inside space-y-1">
                  <li>자동으로 us-west-2로 트래픽 전환</li>
                  <li>Cross-region 복제본 활성화</li>
                  <li>DNS 업데이트 (Route53 헬스체크)</li>
                  <li>캐시 워밍 시작</li>
                </ul>
                <p className="mt-2"><strong>RTO:</strong> 5분 이내</p>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">시나리오 2: 캐스케이딩 장애</h4>
              <div className="text-sm text-gray-700 dark:text-gray-300">
                <p className="mb-2"><strong>상황:</strong> 하나의 노드 장애가 연쇄적으로 확산</p>
                <p className="mb-2"><strong>복구 전략:</strong></p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Circuit Breaker로 장애 노드 격리</li>
                  <li>백프레셔(Backpressure) 적용</li>
                  <li>Rate Limiting으로 과부하 방지</li>
                  <li>점진적 복구 (10% → 50% → 100%)</li>
                </ul>
                <p className="mt-2"><strong>RTO:</strong> 30초 이내</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
