'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, TrendingUp } from 'lucide-react'
import References from '@/components/common/References'

export default function Section6() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
            <TrendingUp className="text-purple-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.6 Production 구현 가이드</h2>
            <p className="text-gray-600 dark:text-gray-400">실제 서비스에서의 RAGAS 활용법</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">완전한 모니터링 대시보드</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
              <code>{`# Production 모니터링 시스템
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

class RAGMonitoringDashboard:
    def __init__(self):
        self.metrics_history = []

    def add_evaluation(self, eval_results):
        """평가 결과 추가"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'scores': eval_results,
            'alerts': self._check_alerts(eval_results)
        })

    def _check_alerts(self, scores):
        """알림 조건 확인"""
        alerts = []
        thresholds = {
            'context_relevancy': 0.7,
            'answer_faithfulness': 0.8,
            'answer_relevancy': 0.75
        }

        for metric, threshold in thresholds.items():
            if scores.get(metric, 1.0) < threshold:
                alerts.append({
                    'metric': metric,
                    'score': scores[metric],
                    'threshold': threshold,
                    'severity': 'high' if scores[metric] < threshold * 0.7 else 'medium'
                })

        return alerts

    def render_dashboard(self):
        """Streamlit 대시보드 렌더링"""
        st.title("🚀 RAG System Monitoring Dashboard")

        # 현재 상태 요약
        col1, col2, col3, col4 = st.columns(4)

        latest_scores = self.metrics_history[-1]['scores'] if self.metrics_history else {}

        with col1:
            st.metric(
                "Context Relevancy",
                f"{latest_scores.get('context_relevancy', 0):.2%}",
                delta=self._calculate_delta('context_relevancy')
            )

        with col2:
            st.metric(
                "Answer Faithfulness",
                f"{latest_scores.get('answer_faithfulness', 0):.2%}",
                delta=self._calculate_delta('answer_faithfulness')
            )

        with col3:
            st.metric(
                "Answer Relevancy",
                f"{latest_scores.get('answer_relevancy', 0):.2%}",
                delta=self._calculate_delta('answer_relevancy')
            )

        with col4:
            active_alerts = sum(len(h['alerts']) for h in self.metrics_history[-10:])
            st.metric("Active Alerts", active_alerts, delta=-2 if active_alerts > 0 else 0)

        # 시계열 그래프
        st.subheader("📊 Metrics Over Time")
        self._render_time_series()

        # 알림 섹션
        st.subheader("⚠️ Recent Alerts")
        self._render_alerts()

        # 권장 조치
        st.subheader("💡 Recommended Actions")
        self._render_recommendations()

    def _calculate_delta(self, metric_name):
        """변화량 계산"""
        if len(self.metrics_history) < 2:
            return 0

        current = self.metrics_history[-1]['scores'].get(metric_name, 0)
        previous = self.metrics_history[-2]['scores'].get(metric_name, 0)

        return f"{(current - previous):.1%}"

    def _render_time_series(self):
        """시계열 차트 렌더링"""
        if not self.metrics_history:
            st.info("No data available yet")
            return

        # 데이터 준비
        df = pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                **h['scores']
            }
            for h in self.metrics_history
        ])

        # Plotly 차트
        fig = go.Figure()

        for metric in ['context_relevancy', 'answer_faithfulness', 'answer_relevancy']:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=2)
            ))

        # 임계값 라인 추가
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                      annotation_text="Critical Threshold")
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                      annotation_text="Warning Threshold")

        fig.update_layout(
            title="RAG Metrics Trend",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_alerts(self):
        """최근 알림 표시"""
        recent_alerts = []
        for h in self.metrics_history[-10:]:
            for alert in h['alerts']:
                recent_alerts.append({
                    'Time': h['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Metric': alert['metric'].replace('_', ' ').title(),
                    'Score': f"{alert['score']:.2%}",
                    'Threshold': f"{alert['threshold']:.2%}",
                    'Severity': alert['severity'].upper()
                })

        if recent_alerts:
            df = pd.DataFrame(recent_alerts)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No alerts in recent evaluations!")

    def _render_recommendations(self):
        """권장 조치 제안"""
        if not self.metrics_history:
            return

        latest = self.metrics_history[-1]['scores']

        recommendations = []

        if latest.get('context_relevancy', 1) < 0.7:
            recommendations.append({
                'issue': 'Low Context Relevancy',
                'action': '• 임베딩 모델 재학습\\n• 청킹 전략 개선\\n• 메타데이터 필터링 강화'
            })

        if latest.get('answer_faithfulness', 1) < 0.8:
            recommendations.append({
                'issue': 'Low Answer Faithfulness',
                'action': '• 프롬프트 엔지니어링 개선\\n• 컨텍스트 윈도우 크기 조정\\n• Few-shot 예제 추가'
            })

        if recommendations:
            for rec in recommendations:
                st.warning(f"**{rec['issue']}**")
                st.markdown(rec['action'])
        else:
            st.success("시스템이 최적 상태로 운영 중입니다!")

# 실제 사용 예제
if __name__ == "__main__":
    dashboard = RAGMonitoringDashboard()

    # 시뮬레이션 데이터 추가 (실제로는 실시간 평가 결과)
    import random
    for i in range(24):  # 24시간 데이터
        scores = {
            'context_relevancy': random.uniform(0.65, 0.85),
            'answer_faithfulness': random.uniform(0.75, 0.95),
            'answer_relevancy': random.uniform(0.70, 0.90),
            'context_recall': random.uniform(0.60, 0.80)
        }
        dashboard.add_evaluation(scores)

    # Streamlit 앱 실행
    dashboard.render_dashboard()`}</code>
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">비용 분석</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-blue-200 dark:border-blue-700">
                    <th className="text-left py-2 text-blue-800 dark:text-blue-200">평가 전략</th>
                    <th className="text-right py-2 text-blue-800 dark:text-blue-200">월간 쿼리</th>
                    <th className="text-right py-2 text-blue-800 dark:text-blue-200">평가 비율</th>
                    <th className="text-right py-2 text-blue-800 dark:text-blue-200">예상 비용</th>
                  </tr>
                </thead>
                <tbody className="text-blue-700 dark:text-blue-300">
                  <tr>
                    <td className="py-2">전체 평가</td>
                    <td className="text-right">100,000</td>
                    <td className="text-right">100%</td>
                    <td className="text-right">$2,000</td>
                  </tr>
                  <tr>
                    <td className="py-2">샘플링 (권장)</td>
                    <td className="text-right">100,000</td>
                    <td className="text-right">15%</td>
                    <td className="text-right">$300</td>
                  </tr>
                  <tr>
                    <td className="py-2">중요 쿼리만</td>
                    <td className="text-right">100,000</td>
                    <td className="text-right">5%</td>
                    <td className="text-right">$100</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 RAGAS & 평가 프레임워크',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'RAGAS Official Documentation',
                authors: 'Explodinggradients',
                year: '2024',
                description: 'RAG 평가 프레임워크 - Context Relevancy, Answer Faithfulness, Answer Relevancy 공식 문서',
                link: 'https://docs.ragas.io/'
              },
              {
                title: 'TruLens: LLM Evaluation & Observability',
                authors: 'TruEra',
                year: '2024',
                description: 'LLM 애플리케이션 평가 - Groundedness, Answer Relevance, Context Relevance 측정',
                link: 'https://www.trulens.org/'
              },
              {
                title: 'LangSmith Evaluation',
                authors: 'LangChain',
                year: '2024',
                description: 'LangChain 공식 평가 도구 - 자동화된 테스트, 비교 분석, Production 모니터링',
                link: 'https://docs.smith.langchain.com/evaluation'
              },
              {
                title: 'DeepEval: Unit Testing for LLMs',
                authors: 'Confident AI',
                year: '2024',
                description: 'LLM 단위 테스트 프레임워크 - 14개 평가 메트릭, Pytest 통합, CI/CD 지원',
                link: 'https://docs.confident-ai.com/'
              },
              {
                title: 'Evidently AI: ML Monitoring',
                authors: 'Evidently AI',
                year: '2024',
                description: 'ML 시스템 모니터링 - 데이터 드리프트, 성능 저하 감지, 대시보드 생성',
                link: 'https://www.evidentlyai.com/'
              }
            ]
          },
          {
            title: '📖 RAG 평가 연구 논문',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'ARES: An Automated Evaluation Framework for RAG',
                authors: 'Saad-Falcon et al., Stanford',
                year: '2024',
                description: '자동화된 RAG 평가 - Synthetic 데이터 생성, LLM-as-judge, 인간 평가 대체',
                link: 'https://arxiv.org/abs/2311.09476'
              },
              {
                title: 'Benchmarking Large Language Models in RAG',
                authors: 'Chen et al., Tsinghua University',
                year: '2024',
                description: 'RGB 벤치마크 - 4개 도메인, 다양한 RAG 시나리오, 종합 평가 프레임워크',
                link: 'https://arxiv.org/abs/2309.01431'
              },
              {
                title: 'RAGAS: Automated Evaluation of RAG',
                authors: 'Es et al., Explodinggradients',
                year: '2023',
                description: 'RAGAS 논문 - Reference-free 평가, LLM 기반 메트릭, 자동화된 품질 측정',
                link: 'https://arxiv.org/abs/2309.15217'
              },
              {
                title: 'Evaluating RAG: A Survey',
                authors: 'Liu et al., Microsoft Research',
                year: '2024',
                description: 'RAG 평가 서베이 - 기존 메트릭 분류, 한계점 분석, 미래 방향 제시',
                link: 'https://arxiv.org/abs/2405.17009'
              }
            ]
          },
          {
            title: '🛠️ Production 모니터링 도구',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Weights & Biases: ML Experiment Tracking',
                authors: 'Weights & Biases',
                year: '2024',
                description: 'ML 실험 추적 - 메트릭 시각화, 하이퍼파라미터 최적화, 팀 협업',
                link: 'https://wandb.ai/'
              },
              {
                title: 'MLflow: Open Source ML Platform',
                authors: 'Databricks',
                year: '2024',
                description: 'ML 라이프사이클 관리 - 실험 추적, 모델 레지스트리, 배포 자동화',
                link: 'https://mlflow.org/'
              },
              {
                title: 'Streamlit: Data App Framework',
                authors: 'Snowflake',
                year: '2024',
                description: 'Python 대시보드 - 실시간 모니터링 UI, 빠른 프로토타이핑, 인터랙티브 차트',
                link: 'https://streamlit.io/'
              },
              {
                title: 'Grafana + Prometheus: Metrics Monitoring',
                authors: 'Grafana Labs',
                year: '2024',
                description: '시계열 메트릭 모니터링 - 알림 설정, 실시간 대시보드, 다중 데이터소스 지원',
                link: 'https://grafana.com/'
              },
              {
                title: 'Arize AI: ML Observability Platform',
                authors: 'Arize AI',
                year: '2024',
                description: 'ML 관측성 플랫폼 - 성능 모니터링, 드리프트 감지, 근본 원인 분석',
                link: 'https://arize.com/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
        <Link
          href="/modules/rag/supplementary"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>

        <Link
          href="/modules/rag/supplementary/chapter2"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          다음: Security & Privacy
          <ArrowRight size={20} />
        </Link>
      </div>
    </>
  )
}
