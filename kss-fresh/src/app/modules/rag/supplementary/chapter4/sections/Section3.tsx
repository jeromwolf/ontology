'use client'

import { Activity } from 'lucide-react'

export default function Section3() {
  return (
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
  )
}
