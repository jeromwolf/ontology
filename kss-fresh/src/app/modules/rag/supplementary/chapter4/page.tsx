'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Server, Shield, RefreshCw, AlertTriangle, CheckCircle2, Activity } from 'lucide-react'

export default function Chapter4Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/supplementary"
          className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Server size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 4: High Availability & Recovery</h1>
              <p className="text-purple-100 text-lg">99.9% 가동률을 위한 엔터프라이즈급 아키텍처</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Architecture Overview */}
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

        {/* Section 2: Circuit Breaker Pattern */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <RefreshCw className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.2 서킷 브레이커 패턴</h2>
              <p className="text-gray-600 dark:text-gray-400">연쇄 장애 방지 및 자동 복구</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-3">서킷 브레이커 상태</h3>
              <div className="space-y-2 text-orange-700 dark:text-orange-300">
                <p>🟢 <strong>Closed</strong>: 정상 작동, 모든 요청 통과</p>
                <p>🔴 <strong>Open</strong>: 장애 감지, 요청 차단</p>
                <p>🟡 <strong>Half-Open</strong>: 복구 확인, 제한적 요청 허용</p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">고급 서킷 브레이커 구현</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`import time
from enum import Enum
from typing import Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import numpy as np

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # 실패 임계값
    success_threshold: int = 3          # 복구 임계값
    timeout: float = 60.0              # 오픈 상태 유지 시간
    half_open_max_calls: int = 3       # Half-open 상태 최대 호출
    error_rate_threshold: float = 0.5   # 에러율 임계값
    window_size: int = 10              # 슬라이딩 윈도우 크기

class AdvancedCircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # 통계
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_at = datetime.now()
        
        # 슬라이딩 윈도우
        self.call_results = []  # True: 성공, False: 실패
        
        # Half-open 상태 관리
        self.half_open_calls = 0
        
        # 메트릭
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_transitions': []
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """서킷 브레이커를 통한 함수 호출"""
        self.metrics['total_calls'] += 1
        
        # 상태별 처리
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.metrics['rejected_calls'] += 1
                raise Exception(f"Circuit breaker is OPEN for {self.name}")
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.metrics['rejected_calls'] += 1
                raise Exception(f"Circuit breaker is HALF_OPEN with max calls reached")
            self.half_open_calls += 1
        
        # 함수 실행
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """성공 처리"""
        self.metrics['successful_calls'] += 1
        self.success_count += 1
        
        # 슬라이딩 윈도우 업데이트
        self._update_window(True)
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.OPEN:
            # 이론적으로 불가능하지만 안전장치
            self._transition_to_half_open()
    
    def _on_failure(self):
        """실패 처리"""
        self.metrics['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # 슬라이딩 윈도우 업데이트
        self._update_window(False)
        
        if self.state == CircuitState.CLOSED:
            if self._should_open_circuit():
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _update_window(self, success: bool):
        """슬라이딩 윈도우 업데이트"""
        self.call_results.append(success)
        
        # 윈도우 크기 유지
        if len(self.call_results) > self.config.window_size:
            self.call_results.pop(0)
    
    def _should_open_circuit(self) -> bool:
        """서킷 오픈 조건 확인"""
        # 1. 연속 실패 확인
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # 2. 에러율 확인
        if len(self.call_results) >= self.config.window_size:
            error_rate = sum(1 for r in self.call_results if not r) / len(self.call_results)
            if error_rate >= self.config.error_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """리셋 시도 조건 확인"""
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            return elapsed >= self.config.timeout
        return False
    
    def _transition_to_open(self):
        """OPEN 상태로 전환"""
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        self.metrics['state_transitions'].append({
            'from': self.state,
            'to': CircuitState.OPEN,
            'at': datetime.now()
        })
        print(f"🔴 Circuit {self.name} is now OPEN")
    
    def _transition_to_closed(self):
        """CLOSED 상태로 전환"""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.metrics['state_transitions'].append({
            'from': self.state,
            'to': CircuitState.CLOSED,
            'at': datetime.now()
        })
        print(f"🟢 Circuit {self.name} is now CLOSED")
    
    def _transition_to_half_open(self):
        """HALF_OPEN 상태로 전환"""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self.half_open_calls = 0
        self.success_count = 0
        self.failure_count = 0
        self.metrics['state_transitions'].append({
            'from': self.state,
            'to': CircuitState.HALF_OPEN,
            'at': datetime.now()
        })
        print(f"🟡 Circuit {self.name} is now HALF_OPEN")
    
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        error_rate = 0
        if self.call_results:
            error_rate = sum(1 for r in self.call_results if not r) / len(self.call_results)
        
        return {
            'name': self.name,
            'state': self.state.value,
            'metrics': self.metrics,
            'error_rate': f"{error_rate:.1%}",
            'uptime': (datetime.now() - self.state_changed_at).total_seconds()
        }

# RAG 시스템용 서킷 브레이커 매니저
class CircuitBreakerManager:
    def __init__(self):
        self.breakers = {}
        
        # 서비스별 설정
        self.configs = {
            'llm_api': CircuitBreakerConfig(
                failure_threshold=3,
                timeout=30.0,
                error_rate_threshold=0.3
            ),
            'vector_db': CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60.0,
                error_rate_threshold=0.5
            ),
            'embedding_api': CircuitBreakerConfig(
                failure_threshold=5,
                timeout=45.0,
                error_rate_threshold=0.4
            )
        }
    
    def get_breaker(self, service: str) -> AdvancedCircuitBreaker:
        """서비스별 서킷 브레이커 획득"""
        if service not in self.breakers:
            config = self.configs.get(service, CircuitBreakerConfig())
            self.breakers[service] = AdvancedCircuitBreaker(service, config)
        
        return self.breakers[service]
    
    async def call_with_breaker(self, service: str, func: Callable, *args, **kwargs):
        """서킷 브레이커를 통한 안전한 호출"""
        breaker = self.get_breaker(service)
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_status(self) -> Dict:
        """모든 서킷 브레이커 상태"""
        return {
            name: breaker.get_status() 
            for name, breaker in self.breakers.items()
        }

# 실제 사용 예제
async def unreliable_llm_call(prompt: str) -> str:
    """불안정한 LLM API 호출 시뮬레이션"""
    # 30% 확률로 실패
    if np.random.random() < 0.3:
        raise Exception("LLM API timeout")
    
    await asyncio.sleep(0.1)  # API 호출 시뮬레이션
    return f"Response for: {prompt}"

async def test_circuit_breaker():
    """서킷 브레이커 테스트"""
    manager = CircuitBreakerManager()
    
    print("🔧 서킷 브레이커 테스트 시작\n")
    
    # 20번 호출 시도
    for i in range(20):
        try:
            result = await manager.call_with_breaker(
                'llm_api',
                unreliable_llm_call,
                f"Query {i+1}"
            )
            print(f"✅ Call {i+1}: Success - {result}")
        except Exception as e:
            print(f"❌ Call {i+1}: Failed - {e}")
        
        # 상태 확인
        if i % 5 == 4:
            status = manager.get_breaker('llm_api').get_status()
            print(f"\n📊 Status after {i+1} calls:")
            print(f"   State: {status['state']}")
            print(f"   Error Rate: {status['error_rate']}")
            print(f"   Total Calls: {status['metrics']['total_calls']}")
            print()
        
        await asyncio.sleep(0.5)
    
    # 최종 상태
    print("\n📊 Final Circuit Breaker Status:")
    for service, status in manager.get_all_status().items():
        print(f"{service}: {status}")

# asyncio.run(test_circuit_breaker())`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 3: Graceful Degradation */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-yellow-100 dark:bg-yellow-900/20 flex items-center justify-center">
              <Activity className="text-yellow-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.3 그레이스풀 데그레데이션</h2>
              <p className="text-gray-600 dark:text-gray-400">서비스 품질 단계적 하향 조정</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-3">데그레데이션 레벨</h3>
              <div className="space-y-2 text-yellow-700 dark:text-yellow-300">
                <p>📊 <strong>Level 0</strong>: 모든 기능 정상 (100%)</p>
                <p>📉 <strong>Level 1</strong>: 고급 기능 제한 (80%)</p>
                <p>📉 <strong>Level 2</strong>: 캐시 의존 모드 (60%)</p>
                <p>📉 <strong>Level 3</strong>: 정적 응답 (40%)</p>
                <p>🚨 <strong>Level 4</strong>: 서비스 중단 안내 (0%)</p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">적응형 데그레데이션 시스템</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`class GracefulDegradationSystem:
    def __init__(self):
        self.degradation_level = 0
        self.service_health = {
            'llm_api': 1.0,
            'vector_db': 1.0,
            'embedding_service': 1.0,
            'cache_service': 1.0
        }
        
        # 데그레데이션 정책
        self.policies = {
            0: self._level_0_full_service,
            1: self._level_1_reduced_features,
            2: self._level_2_cache_only,
            3: self._level_3_static_responses,
            4: self._level_4_maintenance_mode
        }
        
        # 캐시된 응답
        self.static_responses = {
            'common_questions': self._load_common_qa(),
            'fallback_responses': self._load_fallback_responses()
        }
        
        # 메트릭
        self.degradation_metrics = {
            'requests_at_level': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            'user_satisfaction': 1.0,
            'revenue_impact': 0.0
        }
    
    async def process_request(self, query: str, context: Dict) -> Dict:
        """데그레데이션 레벨에 따른 요청 처리"""
        # 현재 시스템 상태 평가
        self._evaluate_system_health()
        
        # 데그레데이션 레벨 결정
        self.degradation_level = self._determine_degradation_level()
        
        # 메트릭 기록
        self.degradation_metrics['requests_at_level'][self.degradation_level] += 1
        
        # 레벨별 처리
        handler = self.policies[self.degradation_level]
        response = await handler(query, context)
        
        # 응답에 서비스 레벨 정보 추가
        response['service_level'] = {
            'level': self.degradation_level,
            'quality': f"{(4-self.degradation_level)/4*100:.0f}%",
            'features_available': self._get_available_features()
        }
        
        return response
    
    def _evaluate_system_health(self):
        """시스템 건강도 평가"""
        # 각 서비스의 건강도 체크 (실제로는 모니터링 시스템 연동)
        # 여기서는 시뮬레이션
        for service in self.service_health:
            # 가용성, 응답 시간, 에러율 등을 종합
            self.service_health[service] = self._check_service_health(service)
    
    def _determine_degradation_level(self) -> int:
        """데그레데이션 레벨 결정"""
        avg_health = np.mean(list(self.service_health.values()))
        
        if avg_health >= 0.9:
            return 0  # 정상
        elif avg_health >= 0.7:
            return 1  # 경미한 제한
        elif avg_health >= 0.5:
            return 2  # 캐시 중심 모드
        elif avg_health >= 0.3:
            return 3  # 정적 응답
        else:
            return 4  # 서비스 중단
    
    async def _level_0_full_service(self, query: str, context: Dict) -> Dict:
        """Level 0: 모든 기능 활성화"""
        # 정상적인 RAG 파이프라인
        embeddings = await self._generate_embeddings(query)
        documents = await self._search_documents(embeddings)
        enhanced_docs = await self._rerank_documents(documents, query)
        response = await self._generate_response(query, enhanced_docs)
        
        return {
            'answer': response,
            'sources': enhanced_docs[:3],
            'confidence': 0.95,
            'features_used': ['embeddings', 'vector_search', 'reranking', 'llm_generation']
        }
    
    async def _level_1_reduced_features(self, query: str, context: Dict) -> Dict:
        """Level 1: 고급 기능 비활성화"""
        # Reranking 스킵, 간단한 모델 사용
        embeddings = await self._generate_embeddings(query)
        documents = await self._search_documents(embeddings, limit=5)  # 문서 수 제한
        
        # 저렴한 모델로 전환
        response = await self._generate_response(
            query, 
            documents,
            model='gpt-3.5-turbo'  # GPT-4 대신
        )
        
        return {
            'answer': response,
            'sources': documents[:2],
            'confidence': 0.85,
            'features_used': ['embeddings', 'vector_search', 'llm_generation'],
            'disabled_features': ['reranking', 'advanced_models']
        }
    
    async def _level_2_cache_only(self, query: str, context: Dict) -> Dict:
        """Level 2: 캐시 중심 모드"""
        # 캐시된 결과만 사용
        cached_result = await self._search_cache(query)
        
        if cached_result:
            return {
                'answer': cached_result['answer'],
                'sources': cached_result.get('sources', []),
                'confidence': 0.7,
                'cached': True,
                'features_used': ['cache'],
                'disabled_features': ['live_search', 'llm_generation']
            }
        
        # 캐시 미스시 유사 질문 검색
        similar = await self._find_similar_cached_query(query)
        if similar:
            return {
                'answer': similar['answer'],
                'sources': [],
                'confidence': 0.5,
                'cached': True,
                'similar_question': similar['question'],
                'features_used': ['semantic_cache']
            }
        
        # 대안 없을 때
        return await self._level_3_static_responses(query, context)
    
    async def _level_3_static_responses(self, query: str, context: Dict) -> Dict:
        """Level 3: 사전 정의된 정적 응답"""
        # 일반적인 질문 카테고리 분류
        category = self._classify_query(query)
        
        if category in self.static_responses['common_questions']:
            response = self.static_responses['common_questions'][category]
            return {
                'answer': response['answer'],
                'sources': [],
                'confidence': 0.3,
                'static': True,
                'category': category,
                'features_used': ['static_responses']
            }
        
        # 기본 폴백 응답
        return {
            'answer': self.static_responses['fallback_responses']['general'],
            'sources': [],
            'confidence': 0.1,
            'static': True,
            'fallback': True,
            'message': '현재 일부 기능이 제한되어 있습니다. 잠시 후 다시 시도해주세요.'
        }
    
    async def _level_4_maintenance_mode(self, query: str, context: Dict) -> Dict:
        """Level 4: 유지보수 모드"""
        return {
            'answer': None,
            'error': True,
            'maintenance': True,
            'message': '시스템 점검 중입니다. 약 30분 후에 다시 시도해주세요.',
            'expected_recovery': datetime.now() + timedelta(minutes=30),
            'support_contact': 'support@example.com'
        }
    
    def _get_available_features(self) -> List[str]:
        """현재 레벨에서 사용 가능한 기능"""
        features_by_level = {
            0: ['full_rag', 'advanced_models', 'reranking', 'real_time'],
            1: ['basic_rag', 'standard_models', 'real_time'],
            2: ['cache_search', 'semantic_similarity'],
            3: ['static_responses', 'common_qa'],
            4: []
        }
        return features_by_level[self.degradation_level]
    
    def _load_common_qa(self) -> Dict:
        """일반적인 Q&A 로드"""
        return {
            'greeting': {
                'patterns': ['안녕', 'hello', 'hi'],
                'answer': '안녕하세요! 무엇을 도와드릴까요?'
            },
            'help': {
                'patterns': ['도움', 'help', '사용법'],
                'answer': '제가 도와드릴 수 있는 것들입니다...'
            },
            'technical': {
                'patterns': ['오류', 'error', '문제'],
                'answer': '기술적인 문제가 발생했다면 support@example.com으로 문의해주세요.'
            }
        }
    
    def monitor_impact(self) -> Dict:
        """데그레데이션의 비즈니스 영향 모니터링"""
        total_requests = sum(self.degradation_metrics['requests_at_level'].values())
        
        if total_requests == 0:
            return {}
        
        # 레벨별 가중치 (비즈니스 영향)
        weights = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.4, 4: 0.0}
        
        # 사용자 만족도 계산
        satisfaction = sum(
            count * weights[level] 
            for level, count in self.degradation_metrics['requests_at_level'].items()
        ) / total_requests
        
        # 수익 영향 추정
        revenue_impact = (1 - satisfaction) * 100  # 퍼센트
        
        return {
            'total_requests': total_requests,
            'level_distribution': self.degradation_metrics['requests_at_level'],
            'estimated_satisfaction': f"{satisfaction:.1%}",
            'estimated_revenue_impact': f"-{revenue_impact:.1f}%",
            'recommendations': self._generate_recommendations(satisfaction)
        }
    
    def _generate_recommendations(self, satisfaction: float) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if satisfaction < 0.8:
            recommendations.append("⚠️ 긴급: 주요 서비스 복구 필요")
        
        if self.degradation_metrics['requests_at_level'][3] > 100:
            recommendations.append("📊 캐시 적중률 개선 필요")
        
        if self.service_health['llm_api'] < 0.7:
            recommendations.append("🤖 LLM 서비스 이중화 검토")
        
        return recommendations

# 실제 사용 예제
degradation_system = GracefulDegradationSystem()

# 시뮬레이션: 서비스 상태 변화
print("🔄 서비스 데그레데이션 시뮬레이션\n")

# 정상 상태
degradation_system.service_health = {
    'llm_api': 1.0,
    'vector_db': 1.0,
    'embedding_service': 1.0,
    'cache_service': 1.0
}

# 점진적 서비스 저하
health_scenarios = [
    {'llm_api': 1.0, 'vector_db': 1.0, 'embedding_service': 1.0, 'cache_service': 1.0},
    {'llm_api': 0.8, 'vector_db': 0.9, 'embedding_service': 1.0, 'cache_service': 1.0},
    {'llm_api': 0.5, 'vector_db': 0.6, 'embedding_service': 0.8, 'cache_service': 1.0},
    {'llm_api': 0.2, 'vector_db': 0.3, 'embedding_service': 0.4, 'cache_service': 0.9},
    {'llm_api': 0.1, 'vector_db': 0.1, 'embedding_service': 0.1, 'cache_service': 0.5}
]

for i, health in enumerate(health_scenarios):
    print(f"시나리오 {i+1}: 서비스 건강도")
    for service, score in health.items():
        print(f"  {service}: {score:.1f}")
    
    degradation_system.service_health = health
    level = degradation_system._determine_degradation_level()
    
    print(f"  → 데그레데이션 레벨: {level}")
    print(f"  → 서비스 품질: {(4-level)/4*100:.0f}%")
    print(f"  → 사용 가능 기능: {degradation_system._get_available_features()}")
    print()

# 영향 분석
impact = degradation_system.monitor_impact()
print("\n📊 비즈니스 영향 분석:")
print(f"총 요청: {impact.get('total_requests', 0)}")
print(f"예상 만족도: {impact.get('estimated_satisfaction', 'N/A')}")
print(f"수익 영향: {impact.get('estimated_revenue_impact', 'N/A')}")`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 4: Disaster Recovery */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <Shield className="text-red-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.4 재해 복구 계획</h2>
              <p className="text-gray-600 dark:text-gray-400">RPO 15분, RTO 30분 달성</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-3">복구 목표</h3>
              <div className="space-y-2 text-red-700 dark:text-red-300">
                <p>🎯 <strong>RPO (Recovery Point Objective)</strong>: 15분 - 최대 데이터 손실</p>
                <p>⏱️ <strong>RTO (Recovery Time Objective)</strong>: 30분 - 최대 다운타임</p>
                <p>📊 <strong>SLA (Service Level Agreement)</strong>: 99.9% 가용성</p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">자동화된 재해 복구 시스템</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`# Production 체크리스트 템플릿
disaster_recovery_checklist = {
    "사전 준비": [
        "✅ 멀티 리전 아키텍처 구성",
        "✅ 실시간 데이터 복제 설정",
        "✅ 자동 페일오버 스크립트",
        "✅ 백업 검증 자동화",
        "✅ 복구 절차 문서화"
    ],
    
    "모니터링": [
        "✅ 실시간 헬스 체크 (5초 간격)",
        "✅ 리전간 레이턴시 모니터링",
        "✅ 데이터 동기화 지연 추적",
        "✅ 자동 알림 시스템",
        "✅ 대시보드 구축"
    ],
    
    "복구 절차": [
        "1️⃣ 장애 감지 (자동, 1분 이내)",
        "2️⃣ 영향 범위 평가 (2분)",
        "3️⃣ 페일오버 결정 (3분)",
        "4️⃣ DNS 업데이트 (5분)",
        "5️⃣ 서비스 검증 (10분)",
        "6️⃣ 사용자 알림 (15분)"
    ],
    
    "테스트 계획": [
        "🔄 월간 페일오버 드릴",
        "🔄 분기별 전체 복구 테스트",
        "🔄 연간 재해 시뮬레이션",
        "🔄 자동화 스크립트 검증"
    ]
}

# 복구 자동화 스크립트
class AutomatedDisasterRecovery:
    def __init__(self):
        self.regions = {
            'primary': 'us-east-1',
            'secondary': 'us-west-2',
            'tertiary': 'eu-west-1'
        }
        
        self.recovery_steps = []
        self.start_time = None
        
    async def execute_failover(self, failed_region: str):
        """자동 페일오버 실행"""
        self.start_time = datetime.now()
        self.recovery_steps = []
        
        try:
            # 1. 장애 확인
            await self._verify_failure(failed_region)
            
            # 2. 새 Primary 선택
            new_primary = await self._select_new_primary(failed_region)
            
            # 3. 데이터 동기화 확인
            await self._verify_data_consistency(new_primary)
            
            # 4. 트래픽 전환
            await self._switch_traffic(new_primary)
            
            # 5. 서비스 검증
            await self._verify_services(new_primary)
            
            # 6. 알림 발송
            await self._notify_stakeholders(failed_region, new_primary)
            
            recovery_time = (datetime.now() - self.start_time).total_seconds() / 60
            
            return {
                'success': True,
                'recovery_time_minutes': recovery_time,
                'new_primary': new_primary,
                'steps': self.recovery_steps
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': self.recovery_steps
            }
    
    async def _verify_failure(self, region: str):
        """장애 확인"""
        # 실제로는 여러 소스에서 확인
        self.recovery_steps.append({
            'step': 'verify_failure',
            'timestamp': datetime.now(),
            'result': f'{region} confirmed down'
        })
    
    async def _select_new_primary(self, failed_region: str) -> str:
        """새로운 Primary 리전 선택"""
        candidates = [r for r in self.regions.values() if r != failed_region]
        
        # 데이터 신선도와 가용성 기반 선택
        # 실제로는 복잡한 로직 필요
        new_primary = candidates[0]
        
        self.recovery_steps.append({
            'step': 'select_primary',
            'timestamp': datetime.now(),
            'result': f'Selected {new_primary} as new primary'
        })
        
        return new_primary

# 99.9% 가용성 달성 전략
uptime_strategy = {
    "아키텍처": {
        "멀티 리전": "최소 3개 리전에 분산",
        "로드 밸런싱": "지능형 트래픽 분산",
        "데이터 복제": "실시간 크로스 리전 복제",
        "캐싱": "엣지 로케이션 활용"
    },
    
    "모니터링": {
        "헬스 체크": "5초 간격",
        "메트릭 수집": "1분 간격",
        "알림": "다중 채널 (SMS, Email, Slack)",
        "대시보드": "실시간 상태 표시"
    },
    
    "자동화": {
        "페일오버": "자동 실행",
        "스케일링": "예측적 오토스케일링",
        "백업": "증분 백업 15분마다",
        "복구": "원클릭 복구"
    },
    
    "테스트": {
        "카오스 엔지니어링": "주간 실행",
        "부하 테스트": "월간 실행",
        "복구 드릴": "분기별 실행",
        "전체 재해 시뮬레이션": "연간 실행"
    }
}`}</code>
              </pre>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">실전 복구 시나리오</h3>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">시나리오 1: 단일 서비스 장애</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 영향: LLM API 불가</li>
                    <li>• 조치: 백업 프로바이더로 자동 전환</li>
                    <li>• 복구 시간: 30초</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">시나리오 2: 전체 리전 장애</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 영향: Primary 리전 전체 다운</li>
                    <li>• 조치: Secondary 리전으로 완전 페일오버</li>
                    <li>• 복구 시간: 15분</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">시나리오 3: 데이터 센터 재해</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 영향: 물리적 재해로 데이터센터 손실</li>
                    <li>• 조치: 지리적으로 분산된 백업에서 복구</li>
                    <li>• 복구 시간: 30분</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">Production 준비 체크리스트</h3>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>멀티 리전 인프라 구성 완료</span>
                </label>
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>실시간 데이터 복제 설정</span>
                </label>
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>자동 페일오버 스크립트 테스트</span>
                </label>
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>모니터링 및 알림 시스템 구축</span>
                </label>
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>복구 절차 문서화 및 교육</span>
                </label>
                <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <input type="checkbox" className="rounded" />
                  <span>정기 복구 훈련 일정 수립</span>
                </label>
              </div>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          <Link
            href="/modules/rag/supplementary/chapter3"
            className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
          >
            <ArrowLeft size={20} />
            이전: Cost Optimization
          </Link>
          
          <Link
            href="/modules/rag/supplementary"
            className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
          >
            보충 과정 완료
            <CheckCircle2 size={20} />
          </Link>
        </div>
      </div>
    </div>
  )
}