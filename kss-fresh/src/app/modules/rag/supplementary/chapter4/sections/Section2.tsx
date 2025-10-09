'use client'

import { RefreshCw } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
