'use client'

import { TrendingUp } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <TrendingUp className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.2 RAG를 위한 A/B 테스팅</h2>
          <p className="text-gray-600 dark:text-gray-400">데이터 기반 의사결정을 위한 실험 프레임워크</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">프로덕션 A/B 테스팅 시스템</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import hashlib
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

class ExperimentVariant(Enum):
    """실험 변형"""
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"

@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    traffic_allocation: Dict[ExperimentVariant, float]
    success_metrics: List[str]
    guardrail_metrics: List[str]
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95

@dataclass
class ExperimentResult:
    """실험 결과"""
    variant: ExperimentVariant
    user_id: str
    metrics: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGExperimentationFramework:
    def __init__(self):
        """
        RAG A/B 테스팅 프레임워크
        - 트래픽 분배
        - 메트릭 수집
        - 통계적 유의성 검정
        - 실시간 모니터링
        """
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.variant_configs = self._init_variant_configs()

    def _init_variant_configs(self) -> Dict[ExperimentVariant, Dict[str, Any]]:
        """각 변형의 RAG 설정"""
        return {
            ExperimentVariant.CONTROL: {
                'retrieval': {
                    'model': 'bge-large-en',
                    'top_k': 5,
                    'reranking': False
                },
                'generation': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            },
            ExperimentVariant.TREATMENT_A: {
                'retrieval': {
                    'model': 'e5-large-v2',
                    'top_k': 10,
                    'reranking': True,
                    'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                },
                'generation': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            },
            ExperimentVariant.TREATMENT_B: {
                'retrieval': {
                    'model': 'bge-large-en',
                    'top_k': 5,
                    'reranking': True,
                    'reranker': 'colbert-v2'
                },
                'generation': {
                    'model': 'gpt-4',
                    'temperature': 0.3,
                    'max_tokens': 800
                }
            }
        }

    def create_experiment(self, config: ExperimentConfig):
        """새 실험 생성"""
        # 트래픽 할당 합이 1인지 확인
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")

        self.experiments[config.experiment_id] = config
        print(f"Created experiment: {config.name}")

    def get_variant(self, experiment_id: str, user_id: str) -> Optional[ExperimentVariant]:
        """사용자에게 할당할 실험 변형 결정"""
        if experiment_id not in self.experiments:
            return None

        config = self.experiments[experiment_id]

        # 실험 기간 체크
        now = datetime.now()
        if now < config.start_date or now > config.end_date:
            return ExperimentVariant.CONTROL

        # Deterministic assignment based on user_id
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0

        # 트래픽 할당에 따라 변형 선택
        cumulative = 0.0
        for variant, allocation in config.traffic_allocation.items():
            cumulative += allocation
            if bucket < cumulative:
                return variant

        return ExperimentVariant.CONTROL

    async def run_experiment_request(self, experiment_id: str, user_id: str,
                                   query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """실험 요청 실행"""
        # 변형 할당
        variant = self.get_variant(experiment_id, user_id)
        if variant is None:
            variant = ExperimentVariant.CONTROL

        # 변형별 설정 가져오기
        variant_config = self.variant_configs[variant]

        # RAG 실행 (시뮬레이션)
        start_time = datetime.now()
        response = await self._execute_rag_variant(query, context, variant_config)
        latency = (datetime.now() - start_time).total_seconds() * 1000

        # 메트릭 수집
        metrics = {
            'latency': latency,
            'relevance_score': response.get('relevance_score', 0),
            'click_through': 0,  # 나중에 업데이트
            'dwell_time': 0,  # 나중에 업데이트
            'thumbs_up': 0,  # 나중에 업데이트
        }

        # 결과 기록
        result = ExperimentResult(
            variant=variant,
            user_id=user_id,
            metrics=metrics,
            timestamp=datetime.now(),
            metadata={
                'query': query,
                'response': response
            }
        )
        self.results[experiment_id].append(result)

        # 응답에 실험 정보 추가
        response['experiment'] = {
            'id': experiment_id,
            'variant': variant.value
        }

        return response

    async def _execute_rag_variant(self, query: str, context: Dict[str, Any],
                                  variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """변형별 RAG 실행 (시뮬레이션)"""
        # 실제로는 각 설정에 따라 다른 RAG 파이프라인 실행

        # 시뮬레이션을 위한 가짜 응답
        base_score = 0.8
        if variant_config['retrieval']['reranking']:
            base_score += 0.05
        if variant_config['generation']['model'] == 'gpt-4':
            base_score += 0.1

        return {
            'answer': f"Sample answer using {variant_config['generation']['model']}",
            'documents': [{'id': f'doc{i}', 'score': 0.9-i*0.1}
                         for i in range(variant_config['retrieval']['top_k'])],
            'relevance_score': min(base_score + np.random.normal(0, 0.05), 1.0),
            'config': variant_config
        }

    def update_metrics(self, experiment_id: str, user_id: str,
                      metric_updates: Dict[str, float]):
        """사용자 행동 메트릭 업데이트"""
        # 해당 사용자의 최신 결과 찾기
        experiment_results = self.results.get(experiment_id, [])

        for result in reversed(experiment_results):
            if result.user_id == user_id:
                result.metrics.update(metric_updates)
                break

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """실험 결과 분석"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}

        config = self.experiments[experiment_id]
        results = self.results.get(experiment_id, [])

        if len(results) < config.minimum_sample_size:
            return {
                'status': 'insufficient_data',
                'current_sample_size': len(results),
                'required_sample_size': config.minimum_sample_size
            }

        # 변형별 결과 분리
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant].append(result)

        # 각 메트릭에 대한 분석
        analysis = {
            'experiment_id': experiment_id,
            'name': config.name,
            'sample_sizes': {v.value: len(results)
                           for v, results in variant_results.items()},
            'metric_analysis': {}
        }

        # Success metrics 분석
        for metric in config.success_metrics:
            metric_analysis = self._analyze_metric(
                variant_results, metric, config.confidence_level
            )
            analysis['metric_analysis'][metric] = metric_analysis

        # Guardrail metrics 체크
        guardrail_violations = []
        for metric in config.guardrail_metrics:
            violation = self._check_guardrail(variant_results, metric)
            if violation:
                guardrail_violations.append(violation)

        analysis['guardrail_violations'] = guardrail_violations

        # 승자 결정
        analysis['winner'] = self._determine_winner(
            analysis['metric_analysis'], config.success_metrics
        )

        return analysis

    def _analyze_metric(self, variant_results: Dict[ExperimentVariant, List[ExperimentResult]],
                       metric_name: str, confidence_level: float) -> Dict[str, Any]:
        """개별 메트릭 분석"""
        metric_values = {}
        for variant, results in variant_results.items():
            values = [r.metrics.get(metric_name, 0) for r in results]
            metric_values[variant] = values

        control_values = metric_values.get(ExperimentVariant.CONTROL, [])
        if not control_values:
            return {'error': 'No control data'}

        analysis = {
            'means': {},
            'confidence_intervals': {},
            'lifts': {},
            'p_values': {}
        }

        # 각 변형에 대한 분석
        for variant, values in metric_values.items():
            if not values:
                continue

            # 기본 통계
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)

            analysis['means'][variant.value] = mean

            # 신뢰구간
            ci = stats.t.interval(confidence_level, n-1,
                                 loc=mean, scale=std/np.sqrt(n))
            analysis['confidence_intervals'][variant.value] = ci

            # Control 대비 상승률
            if variant != ExperimentVariant.CONTROL:
                control_mean = np.mean(control_values)
                lift = (mean - control_mean) / control_mean * 100
                analysis['lifts'][variant.value] = lift

                # T-test
                t_stat, p_value = stats.ttest_ind(values, control_values)
                analysis['p_values'][variant.value] = p_value

        return analysis

    def _check_guardrail(self, variant_results: Dict[ExperimentVariant, List[ExperimentResult]],
                        metric_name: str) -> Optional[Dict[str, Any]]:
        """Guardrail 메트릭 체크"""
        control_values = [r.metrics.get(metric_name, 0)
                         for r in variant_results.get(ExperimentVariant.CONTROL, [])]

        if not control_values:
            return None

        control_mean = np.mean(control_values)

        violations = []
        for variant, results in variant_results.items():
            if variant == ExperimentVariant.CONTROL:
                continue

            values = [r.metrics.get(metric_name, 0) for r in results]
            if not values:
                continue

            variant_mean = np.mean(values)

            # 5% 이상 성능 저하 체크
            degradation = (control_mean - variant_mean) / control_mean * 100
            if degradation > 5:
                violations.append({
                    'variant': variant.value,
                    'metric': metric_name,
                    'degradation': degradation
                })

        return violations[0] if violations else None

    def _determine_winner(self, metric_analysis: Dict[str, Dict[str, Any]],
                         success_metrics: List[str]) -> Optional[str]:
        """승자 결정"""
        scores = defaultdict(float)

        for metric in success_metrics:
            analysis = metric_analysis.get(metric, {})
            p_values = analysis.get('p_values', {})
            lifts = analysis.get('lifts', {})

            for variant in [ExperimentVariant.TREATMENT_A, ExperimentVariant.TREATMENT_B]:
                variant_name = variant.value

                # 통계적으로 유의하고 개선이 있는 경우
                if (variant_name in p_values and
                    p_values[variant_name] < 0.05 and
                    variant_name in lifts and
                    lifts[variant_name] > 0):
                    scores[variant_name] += lifts[variant_name]

        # 가장 높은 점수의 변형 선택
        if scores:
            winner = max(scores.items(), key=lambda x: x[1])
            return winner[0]

        return None

    def generate_report(self, experiment_id: str) -> str:
        """실험 리포트 생성"""
        analysis = self.analyze_experiment(experiment_id)

        if 'error' in analysis:
            return f"Error: {analysis['error']}"

        report = []
        report.append(f"# Experiment Report: {analysis.get('name', experiment_id)}")
        report.append(f"\n## Sample Sizes")
        for variant, size in analysis['sample_sizes'].items():
            report.append(f"- {variant}: {size}")

        report.append(f"\n## Metric Analysis")
        for metric, metric_analysis in analysis['metric_analysis'].items():
            report.append(f"\n### {metric}")

            # 평균값
            report.append("**Means:**")
            for variant, mean in metric_analysis['means'].items():
                report.append(f"- {variant}: {mean:.4f}")

            # 상승률
            if metric_analysis.get('lifts'):
                report.append("\n**Lifts vs Control:**")
                for variant, lift in metric_analysis['lifts'].items():
                    p_value = metric_analysis['p_values'].get(variant, 1)
                    sig = "✅" if p_value < 0.05 else "❌"
                    report.append(f"- {variant}: {lift:+.2f}% (p={p_value:.4f}) {sig}")

        # Guardrail violations
        if analysis.get('guardrail_violations'):
            report.append("\n## ⚠️ Guardrail Violations")
            for violation in analysis['guardrail_violations']:
                report.append(f"- {violation['variant']}: {violation['metric']} "
                            f"degraded by {violation['degradation']:.2f}%")

        # Winner
        if analysis.get('winner'):
            report.append(f"\n## 🏆 Winner: {analysis['winner']}")
        else:
            report.append("\n## No clear winner yet")

        return "\n".join(report)

# 실시간 실험 대시보드
class ExperimentDashboard:
    def __init__(self, framework: RAGExperimentationFramework):
        """실험 대시보드"""
        self.framework = framework

    def get_live_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """실시간 메트릭 조회"""
        results = self.framework.results.get(experiment_id, [])
        if not results:
            return {}

        # 최근 1시간 데이터
        cutoff = datetime.now() - timedelta(hours=1)
        recent_results = [r for r in results if r.timestamp > cutoff]

        # 변형별 실시간 메트릭
        metrics = defaultdict(lambda: defaultdict(list))
        for result in recent_results:
            for metric, value in result.metrics.items():
                metrics[result.variant.value][metric].append(value)

        # 집계
        live_metrics = {}
        for variant, variant_metrics in metrics.items():
            live_metrics[variant] = {}
            for metric, values in variant_metrics.items():
                live_metrics[variant][metric] = {
                    'current': values[-1] if values else 0,
                    'mean': np.mean(values) if values else 0,
                    'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down'
                }

        return live_metrics

# 사용 예제
print("=== RAG A/B 테스팅 데모 ===\n")

# 프레임워크 초기화
framework = RAGExperimentationFramework()

# 실험 생성
experiment_config = ExperimentConfig(
    experiment_id="rag_reranking_test",
    name="Reranking Algorithm Comparison",
    description="Compare different reranking strategies",
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=7),
    traffic_allocation={
        ExperimentVariant.CONTROL: 0.33,
        ExperimentVariant.TREATMENT_A: 0.33,
        ExperimentVariant.TREATMENT_B: 0.34
    },
    success_metrics=['relevance_score', 'click_through'],
    guardrail_metrics=['latency'],
    minimum_sample_size=100
)

framework.create_experiment(experiment_config)

# 시뮬레이션: 여러 사용자 요청 실행
async def simulate_requests():
    for i in range(150):
        user_id = f"user_{i}"
        query = f"Test query {i % 10}"

        response = await framework.run_experiment_request(
            "rag_reranking_test", user_id, query, {}
        )

        # 시뮬레이션: 사용자 행동
        if response.get('relevance_score', 0) > 0.85:
            framework.update_metrics(
                "rag_reranking_test", user_id,
                {'click_through': 1, 'dwell_time': np.random.randint(10, 60)}
            )

# 실행
asyncio.run(simulate_requests())

# 분석
print("\n" + framework.generate_report("rag_reranking_test"))`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
