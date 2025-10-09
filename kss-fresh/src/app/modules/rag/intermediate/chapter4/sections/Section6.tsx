'use client'

import { BookOpen } from 'lucide-react'
import References from '@/components/common/References'

export default function Section6() {
  return (
    <>
      <section className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">RAG 성능 최적화 실습</h3>

          <div className="prose prose-sm prose-invert mb-4">
            <p>
              이번 챕터에서 배운 최적화 기법들을 직접 구현하고 성능을 측정해보세요.
              각 최적화 기법이 실제로 얼마나 효과적인지 정량적으로 검증하는 것이 목표입니다.
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📊 과제 1: 성능 벤치마킹</h4>
              <ol className="space-y-2 text-sm">
                <li>1. 기본 RAG 시스템 구축</li>
                <li>2. 캐싱 전/후 성능 측정</li>
                <li>3. 모델 양자화 효과 검증</li>
                <li>4. 배치 처리 vs 단일 처리 비교</li>
                <li>5. 최적화 보고서 작성</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">⚡ 과제 2: 실시간 모니터링 구현</h4>
              <ul className="space-y-1 text-sm">
                <li>• 성능 메트릭 수집 시스템 구축</li>
                <li>• 임계값 기반 알림 시스템</li>
                <li>• 대시보드 UI 개발</li>
                <li>• 자동 최적화 제안 기능</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📱 과제 3: 엣지 RAG 프로토타입</h4>
              <ul className="space-y-1 text-sm">
                <li>• 모바일 환경용 경량화 RAG</li>
                <li>• 오프라인 동작 지원</li>
                <li>• 배터리 효율성 최적화</li>
                <li>• 제한된 메모리에서의 성능 측정</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 성능 최적화 & 캐싱',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Redis Official Documentation',
                authors: 'Redis Labs',
                year: '2025',
                description: '인메모리 캐싱 - RAG 응답 시간 10x 향상',
                link: 'https://redis.io/docs/'
              },
              {
                title: 'PyTorch Performance Tuning Guide',
                authors: 'PyTorch Team',
                year: '2025',
                description: 'GPU 최적화, 배치 처리, 양자화 - 공식 가이드',
                link: 'https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html'
              },
              {
                title: 'FastAPI Async Best Practices',
                authors: 'FastAPI',
                year: '2025',
                description: '비동기 RAG API 구축 - 처리량 20x 증가',
                link: 'https://fastapi.tiangolo.com/async/'
              },
              {
                title: 'Celery Task Queue',
                authors: 'Celery Project',
                year: '2024',
                description: '대규모 배치 처리 - Redis 기반 분산 큐',
                link: 'https://docs.celeryq.dev/en/stable/'
              },
              {
                title: 'ONNX Runtime Optimization',
                authors: 'Microsoft',
                year: '2025',
                description: 'LLM 추론 속도 2-4x 향상 - 프로덕션 최적화',
                link: 'https://onnxruntime.ai/docs/performance/'
              }
            ]
          },
          {
            title: '📖 모델 압축 & 양자화 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Quantization and Training of Neural Networks',
                authors: 'Krishnamoorthi (Google)',
                year: '2018',
                description: 'INT8 양자화 - 4x 압축, 2-3x 속도 향상',
                link: 'https://arxiv.org/abs/1712.05877'
              },
              {
                title: 'GPTQ: Accurate Quantization for LLMs',
                authors: 'Frantar et al., IST Austria',
                year: '2023',
                description: '3-4bit 양자화로 LLM 압축 - 정확도 유지',
                link: 'https://arxiv.org/abs/2210.17323'
              },
              {
                title: 'SmoothQuant: LLM Quantization',
                authors: 'Xiao et al., MIT',
                year: '2023',
                description: '8bit 양자화 - 1% 미만 정확도 손실',
                link: 'https://arxiv.org/abs/2211.10438'
              },
              {
                title: 'DistilBERT: Smaller, Faster, Cheaper',
                authors: 'Sanh et al., Hugging Face',
                year: '2019',
                description: 'Knowledge Distillation - 60% 작고 97% 성능 유지',
                link: 'https://arxiv.org/abs/1910.01108'
              }
            ]
          },
          {
            title: '🛠️ 성능 모니터링 & 프로파일링',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Prometheus + Grafana',
                authors: 'CNCF',
                year: '2025',
                description: 'RAG 시스템 실시간 모니터링 - 메트릭 수집 & 시각화',
                link: 'https://prometheus.io/docs/introduction/overview/'
              },
              {
                title: 'PyTorch Profiler',
                authors: 'PyTorch',
                year: '2025',
                description: 'GPU/CPU 병목 현상 분석 - TensorBoard 통합',
                link: 'https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html'
              },
              {
                title: 'LangSmith Tracing',
                authors: 'LangChain',
                year: '2025',
                description: 'LLM 체인 디버깅 - 단계별 성능 추적',
                link: 'https://docs.smith.langchain.com/'
              },
              {
                title: 'Ray Serve',
                authors: 'Anyscale',
                year: '2025',
                description: 'LLM 분산 서빙 - 오토스케일링 & 로드밸런싱',
                link: 'https://docs.ray.io/en/latest/serve/index.html'
              },
              {
                title: 'TensorRT-LLM',
                authors: 'NVIDIA',
                year: '2025',
                description: 'NVIDIA GPU 최적화 - 8x 추론 속도 향상',
                link: 'https://github.com/NVIDIA/TensorRT-LLM'
              }
            ]
          }
        ]}
      />
    </>
  )
}
