'use client'

import { Server } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <Server className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.1 고가용성 아키텍처 설계</h2>
          <p className="text-gray-600 dark:text-gray-400">단일 장애 지점(SPOF) 제거</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">엔터프라이즈 RAG 아키텍처</h3>
          <div className="space-y-3 text-blue-700 dark:text-blue-300">
            <p>🔹 <strong>로드 밸런서</strong>: 다중 리전 트래픽 분산</p>
            <p>🔹 <strong>API 게이트웨이</strong>: 인증, 속도 제한, 라우팅</p>
            <p>🔹 <strong>마이크로서비스</strong>: 독립적 스케일링 가능</p>
            <p>🔹 <strong>벡터 DB 클러스터</strong>: 마스터-슬레이브 복제</p>
            <p>🔹 <strong>캐시 레이어</strong>: Redis Sentinel 고가용성</p>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
          <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">멀티 리전 고가용성 시스템</h3>
          <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
            <code>{`import asyncio
from typing import Dict, List, Optional
import httpx
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class ServiceEndpoint:
    region: str
    url: str
    priority: int
    health_score: float = 1.0
    last_check: datetime = None
    is_healthy: bool = True

class HighAvailabilityRAG:
    def __init__(self):
        # 멀티 리전 엔드포인트
        self.endpoints = {
            'llm_service': [
                ServiceEndpoint('us-west', 'https://us-west.api.example.com', 1),
                ServiceEndpoint('us-east', 'https://us-east.api.example.com', 2),
                ServiceEndpoint('eu-west', 'https://eu-west.api.example.com', 3),
                ServiceEndpoint('ap-north', 'https://ap-north.api.example.com', 4)
            ],
            'vector_db': [
                ServiceEndpoint('primary', 'vector-db-primary.example.com:6333', 1),
                ServiceEndpoint('secondary', 'vector-db-secondary.example.com:6333', 2),
                ServiceEndpoint('tertiary', 'vector-db-tertiary.example.com:6333', 3)
            ],
            'cache': [
                ServiceEndpoint('cache-1', 'redis-sentinel-1.example.com:26379', 1),
                ServiceEndpoint('cache-2', 'redis-sentinel-2.example.com:26379', 1),
                ServiceEndpoint('cache-3', 'redis-sentinel-3.example.com:26379', 1)
            ]
        }

        # 헬스 체크 설정
        self.health_check_interval = 30  # seconds
        self.failure_threshold = 3
        self.recovery_threshold = 2

        # 서킷 브레이커 설정
        self.circuit_breakers = {}

        # 메트릭스
        self.metrics = {
            'requests_total': 0,
            'requests_failed': 0,
            'failovers': 0,
            'avg_latency': 0
        }

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """시스템 초기화 및 헬스 체크 시작"""
        # 초기 헬스 체크
        await self._perform_health_checks()

        # 백그라운드 헬스 체크 시작
        asyncio.create_task(self._background_health_monitor())

        self.logger.info("High Availability RAG System initialized")

    async def process_query(self, query: str, user_context: Dict = None) -> Dict:
        """고가용성 쿼리 처리"""
        self.metrics['requests_total'] += 1
        start_time = datetime.now()

        try:
            # 1. 캐시 확인 (다중 캐시 서버)
            cached_result = await self._check_distributed_cache(query)
            if cached_result:
                return cached_result

            # 2. 벡터 검색 (자동 페일오버)
            documents = await self._retrieve_documents_with_failover(query)

            # 3. LLM 호출 (지역 최적화 및 페일오버)
            response = await self._call_llm_with_failover(query, documents)

            # 4. 결과 캐싱 (다중 캐시 서버에 복제)
            await self._cache_result_distributed(query, response)

            # 메트릭 업데이트
            latency = (datetime.now() - start_time).total_seconds()
            self._update_metrics(latency, success=True)

            return response

        except Exception as e:
            self.metrics['requests_failed'] += 1
            self.logger.error(f"Query processing failed: {e}")

            # 그레이스풀 데그레데이션
            return await self._graceful_degradation(query)

    async def _check_distributed_cache(self, query: str) -> Optional[Dict]:
        """분산 캐시 확인 (Quorum Read)"""
        cache_endpoints = self._get_healthy_endpoints('cache')

        if len(cache_endpoints) < 2:
            self.logger.warning("Insufficient healthy cache nodes for quorum read")
            return None

        # 병렬로 여러 캐시 서버 조회
        tasks = []
        for endpoint in cache_endpoints[:3]:  # 최대 3개
            task = self._query_cache(endpoint, query)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Quorum 확인 (과반수 일치)
        valid_results = [r for r in results if not isinstance(r, Exception) and r is not None]

        if len(valid_results) >= 2:
            # 결과가 일치하는지 확인
            if all(r == valid_results[0] for r in valid_results):
                return valid_results[0]

        return None

    async def _retrieve_documents_with_failover(self, query: str) -> List[Dict]:
        """벡터 DB 검색 with 자동 페일오버"""
        vector_endpoints = self._get_healthy_endpoints('vector_db')

        for endpoint in vector_endpoints:
            try:
                if self._is_circuit_open(endpoint.url):
                    continue

                documents = await self._query_vector_db(endpoint, query)

                # 성공하면 헬스 스코어 개선
                self._improve_health_score(endpoint)

                return documents

            except Exception as e:
                self.logger.warning(f"Vector DB query failed on {endpoint.region}: {e}")
                self._degrade_health_score(endpoint)
                self._record_failure(endpoint.url)

                # 다음 엔드포인트로 페일오버
                self.metrics['failovers'] += 1
                continue

        raise Exception("All vector DB endpoints failed")

    async def _call_llm_with_failover(self, query: str, documents: List[Dict]) -> Dict:
        """LLM 호출 with 지능형 라우팅"""
        llm_endpoints = self._get_healthy_endpoints('llm_service')

        # 사용자 위치 기반 최적 엔드포인트 선택
        optimal_endpoint = self._select_optimal_endpoint(llm_endpoints)

        # 재시도 로직
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                endpoint = optimal_endpoint if retry_count == 0 else llm_endpoints[retry_count]

                if self._is_circuit_open(endpoint.url):
                    retry_count += 1
                    continue

                response = await self._query_llm(endpoint, query, documents)

                # 성공
                self._improve_health_score(endpoint)
                return response

            except Exception as e:
                self.logger.warning(f"LLM call failed on {endpoint.region}: {e}")
                self._degrade_health_score(endpoint)
                self._record_failure(endpoint.url)

                retry_count += 1

                # 지수 백오프
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)

        raise Exception("All LLM endpoints failed")

    def _get_healthy_endpoints(self, service: str) -> List[ServiceEndpoint]:
        """건강한 엔드포인트 목록 반환"""
        endpoints = self.endpoints.get(service, [])

        # 건강한 엔드포인트만 필터링
        healthy = [ep for ep in endpoints if ep.is_healthy]

        # 우선순위와 헬스 스코어로 정렬
        healthy.sort(key=lambda x: (x.priority, -x.health_score))

        return healthy

    def _is_circuit_open(self, endpoint_url: str) -> bool:
        """서킷 브레이커 상태 확인"""
        breaker = self.circuit_breakers.get(endpoint_url, {'failures': 0, 'last_failure': None, 'state': 'closed'})

        if breaker['state'] == 'open':
            # 일정 시간 후 half-open 상태로 전환
            if breaker['last_failure'] and (datetime.now() - breaker['last_failure']) > timedelta(minutes=5):
                breaker['state'] = 'half-open'
                return False
            return True

        return False

    def _record_failure(self, endpoint_url: str):
        """실패 기록 및 서킷 브레이커 업데이트"""
        if endpoint_url not in self.circuit_breakers:
            self.circuit_breakers[endpoint_url] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }

        breaker = self.circuit_breakers[endpoint_url]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.now()

        # 임계값 초과시 서킷 오픈
        if breaker['failures'] >= self.failure_threshold:
            breaker['state'] = 'open'
            self.logger.warning(f"Circuit breaker opened for {endpoint_url}")

    async def _graceful_degradation(self, query: str) -> Dict:
        """그레이스풀 데그레데이션"""
        self.logger.info("Entering graceful degradation mode")

        # 1. 로컬 캐시 확인
        local_cache = await self._check_local_cache(query)
        if local_cache:
            return {
                **local_cache,
                'degraded': True,
                'message': 'Using cached response due to system issues'
            }

        # 2. 정적 응답 반환
        return {
            'answer': '죄송합니다. 현재 시스템에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요.',
            'degraded': True,
            'retry_after': 30
        }

    async def _perform_health_checks(self):
        """모든 엔드포인트 헬스 체크"""
        for service, endpoints in self.endpoints.items():
            for endpoint in endpoints:
                is_healthy = await self._check_endpoint_health(endpoint)
                endpoint.is_healthy = is_healthy
                endpoint.last_check = datetime.now()

                if not is_healthy:
                    self.logger.warning(f"Endpoint unhealthy: {endpoint.region} ({service})")

    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> bool:
        """개별 엔드포인트 헬스 체크"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{endpoint.url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
        except:
            return False

    async def _background_health_monitor(self):
        """백그라운드 헬스 모니터링"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._perform_health_checks()

# 재해 복구 시스템
class DisasterRecoverySystem:
    def __init__(self):
        self.backup_regions = ['us-west', 'eu-west', 'ap-southeast']
        self.primary_region = 'us-east'
        self.rpo = 15  # Recovery Point Objective: 15분
        self.rto = 30  # Recovery Time Objective: 30분

    async def initiate_failover(self, failed_region: str) -> Dict:
        """리전 페일오버 실행"""
        self.logger.info(f"Initiating failover from {failed_region}")

        # 1. 트래픽 재라우팅
        await self._update_dns_records(failed_region)

        # 2. 데이터 동기화 확인
        sync_status = await self._verify_data_sync()

        # 3. 서비스 활성화
        activated_region = await self._activate_standby_services()

        # 4. 헬스 체크
        health_status = await self._verify_new_primary(activated_region)

        return {
            'failover_completed': True,
            'new_primary': activated_region,
            'data_loss': sync_status.get('lag_minutes', 0),
            'total_time': datetime.now()
        }

    async def perform_backup(self):
        """정기 백업 수행"""
        backup_tasks = []

        # 1. 벡터 DB 백업
        backup_tasks.append(self._backup_vector_db())

        # 2. 설정 및 메타데이터 백업
        backup_tasks.append(self._backup_configurations())

        # 3. 캐시 스냅샷
        backup_tasks.append(self._backup_cache_snapshot())

        results = await asyncio.gather(*backup_tasks)

        return {
            'backup_completed': all(results),
            'timestamp': datetime.now(),
            'backup_size': sum(r.get('size', 0) for r in results)
        }

# 사용 예제
async def main():
    # 고가용성 RAG 시스템 초기화
    ha_rag = HighAvailabilityRAG()
    await ha_rag.initialize()

    # 쿼리 처리
    try:
        result = await ha_rag.process_query(
            "머신러닝과 딥러닝의 차이점은?",
            user_context={'region': 'us-west', 'priority': 'high'}
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # 시스템 상태 확인
    health_status = {
        'endpoints': ha_rag.endpoints,
        'metrics': ha_rag.metrics,
        'circuit_breakers': ha_rag.circuit_breakers
    }

    print(f"\nSystem Status: {health_status}")

# 실행
# asyncio.run(main())`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}
