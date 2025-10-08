'use client';

import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* 1. 실전 딥러닝 프로젝트 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          실전 딥러닝 프로젝트: 이론에서 배포까지
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          딥러닝 모델을 실제 프로덕션 환경에 배포하려면 학습뿐만 아니라 데이터 파이프라인, 모델 최적화, 서빙 인프라 등을 고려해야 합니다.
        </p>

        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-6 border border-purple-200 dark:border-purple-700 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-purple-900 dark:text-purple-300">
            🎯 전체 워크플로우
          </h3>
          <div className="grid md:grid-cols-4 gap-3 text-sm text-gray-700 dark:text-gray-300">
            <div className="text-center">
              <div className="font-semibold text-blue-600 dark:text-blue-400">1. 데이터 수집</div>
              <div className="text-xs mt-1">Dataset, Labeling</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-green-600 dark:text-green-400">2. 모델 개발</div>
              <div className="text-xs mt-1">Training, Tuning</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-orange-600 dark:text-orange-400">3. 모델 최적화</div>
              <div className="text-xs mt-1">ONNX, TensorRT</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-purple-600 dark:text-purple-400">4. 배포</div>
              <div className="text-xs mt-1">API, Monitoring</div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. PyTorch vs TensorFlow */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          PyTorch vs TensorFlow
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-3 text-orange-900 dark:text-orange-300 text-lg">🔥 PyTorch</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Dynamic Computational Graph (Eager Execution)</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
              <li>• 직관적이고 Pythonic한 코드</li>
              <li>• 디버깅 용이 (일반 Python 디버거 사용 가능)</li>
              <li>• 연구 커뮤니티에서 압도적 인기</li>
              <li>• TorchScript로 static graph 변환 가능</li>
            </ul>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              ✓ 추천: 연구, 프로토타이핑, 최신 모델 실험
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-300 text-lg">⚡ TensorFlow</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Static Computational Graph (TF 2.0부터 Eager 지원)</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
              <li>• 프로덕션 배포에 최적화 (TF Serving, TF Lite)</li>
              <li>• 모바일/임베디드 지원 우수</li>
              <li>• Google 생태계 통합 (TPU, GCP)</li>
              <li>• Keras API로 간편한 개발</li>
            </ul>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              ✓ 추천: 프로덕션 배포, 모바일/엣지 디바이스
            </div>
          </div>
        </div>
      </section>

      {/* 3. Dataset & DataLoader */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Dataset & DataLoader 구성
        </h2>

        <div className="bg-teal-50 dark:bg-teal-900/20 rounded-xl p-6 border border-teal-200 dark:border-teal-800 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-teal-900 dark:text-teal-300">
            📁 효율적인 데이터 파이프라인
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <strong className="text-gray-900 dark:text-gray-100">1. 데이터 저장 형식</strong>
              <ul className="text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>• <strong>이미지</strong>: HDF5, LMDB, TFRecord</li>
                <li>• <strong>텍스트</strong>: Arrow, Parquet</li>
                <li>• <strong>대용량</strong>: Sharded files, Streaming</li>
              </ul>
            </div>
            <div>
              <strong className="text-gray-900 dark:text-gray-100">2. 데이터 증강</strong>
              <ul className="text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>• <strong>Albumentations</strong>: 이미지 증강 (빠름)</li>
                <li>• <strong>imgaug</strong>: 다양한 증강 기법</li>
                <li>• <strong>AutoAugment</strong>: 자동 증강 정책 학습</li>
              </ul>
            </div>
            <div>
              <strong className="text-gray-900 dark:text-gray-100">3. 멀티프로세싱</strong>
              <ul className="text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>• num_workers 최적화 (CPU 코어 수 고려)</li>
                <li>• pin_memory=True (GPU 전송 최적화)</li>
                <li>• prefetch_factor 조정</li>
              </ul>
            </div>
            <div>
              <strong className="text-gray-900 dark:text-gray-100">4. 배치 샘플링</strong>
              <ul className="text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>• Weighted sampling (class imbalance)</li>
                <li>• Batch balancing</li>
                <li>• Dynamic batching (가변 길이)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Hyperparameter Tuning */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Hyperparameter Tuning
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-4 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-2 text-pink-900 dark:text-pink-300">Grid Search</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              모든 조합 탐색
            </p>
            <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 정확하지만 느림</li>
              <li>• 파라미터 적을 때 유용</li>
            </ul>
          </div>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-4 border border-indigo-200 dark:border-indigo-800">
            <h4 className="font-semibold mb-2 text-indigo-900 dark:text-indigo-300">Random Search</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              랜덤 샘플링
            </p>
            <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Grid보다 효율적</li>
              <li>• 중요한 파라미터 발견</li>
            </ul>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">Bayesian Optimization</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              이전 결과 기반 탐색
            </p>
            <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Optuna, Ray Tune</li>
              <li>• 가장 효율적</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. 모델 최적화 & 배포 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          모델 최적화 & 배포
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {/* ONNX */}
          <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
            <h4 className="font-semibold mb-3 text-violet-900 dark:text-violet-300 text-lg">🔄 ONNX (Open Neural Network Exchange)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              프레임워크 간 모델 교환 및 최적화
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• PyTorch → ONNX → TensorFlow</li>
              <li>• ONNX Runtime으로 추론 최적화</li>
              <li>• 크로스 플랫폼 지원</li>
              <li>• 2-4배 속도 향상 가능</li>
            </ul>
          </div>

          {/* TensorRT */}
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-semibold mb-3 text-emerald-900 dark:text-emerald-300 text-lg">⚡ TensorRT (NVIDIA)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              GPU 추론 최적화 엔진
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Layer fusion, Kernel auto-tuning</li>
              <li>• FP16/INT8 quantization</li>
              <li>• NVIDIA GPU에서 최고 성능</li>
              <li>• 5-10배 속도 향상 가능</li>
            </ul>
          </div>

          {/* Quantization */}
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-3 text-orange-900 dark:text-orange-300 text-lg">📉 Quantization</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              FP32 → INT8/INT4로 변환하여 경량화
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Post-training Quantization</strong>: 학습 후 변환</li>
              <li>• <strong>Quantization-aware Training</strong>: 학습 시 고려</li>
              <li>• 모델 크기 75% 감소</li>
              <li>• 추론 속도 2-4배 향상</li>
            </ul>
          </div>

          {/* Pruning */}
          <div className="bg-cyan-50 dark:bg-cyan-900/20 rounded-xl p-6 border border-cyan-200 dark:border-cyan-800">
            <h4 className="font-semibold mb-3 text-cyan-900 dark:text-cyan-300 text-lg">✂️ Pruning</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              불필요한 가중치/뉴런 제거
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Unstructured Pruning</strong>: 개별 가중치 제거</li>
              <li>• <strong>Structured Pruning</strong>: 채널/레이어 제거</li>
              <li>• 50-90% 파라미터 감소 가능</li>
              <li>• Fine-tuning 필요</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. Model Serving */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Model Serving & MLOps
        </h2>

        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800">
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">🚀 배포 방식</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>REST API</strong>: FastAPI, Flask</li>
                <li>• <strong>gRPC</strong>: 고성능 RPC</li>
                <li>• <strong>Batch Inference</strong>: 대량 처리</li>
                <li>• <strong>Edge Deployment</strong>: 디바이스 온보드</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">📊 모니터링</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>Latency</strong>: 응답 시간 추적</li>
                <li>• <strong>Throughput</strong>: 처리량 측정</li>
                <li>• <strong>Model Drift</strong>: 성능 저하 감지</li>
                <li>• <strong>A/B Testing</strong>: 모델 비교</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">🔧 도구</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>TorchServe</strong>: PyTorch 공식</li>
                <li>• <strong>TF Serving</strong>: TensorFlow 공식</li>
                <li>• <strong>Triton</strong>: NVIDIA 멀티 프레임워크</li>
                <li>• <strong>MLflow</strong>: 실험 관리</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 7. 체크리스트 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 프로덕션 배포 체크리스트
        </h2>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl p-8 border border-green-200 dark:border-green-800">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2 text-green-900 dark:text-green-300">✅ 학습 단계</h3>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>□ 충분한 검증 데이터 분리</li>
                <li>□ Early stopping으로 과적합 방지</li>
                <li>□ 체크포인트 자동 저장</li>
                <li>□ TensorBoard 로깅</li>
                <li>□ 재현 가능성 확보 (seed 고정)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-2 text-green-900 dark:text-green-300">✅ 최적화 단계</h3>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>□ ONNX/TensorRT 변환</li>
                <li>□ Quantization 적용</li>
                <li>□ Latency/Throughput 벤치마크</li>
                <li>□ 배치 크기 최적화</li>
                <li>□ Dynamic batching 고려</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-2 text-green-900 dark:text-green-300">✅ 배포 단계</h3>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>□ 컨테이너화 (Docker)</li>
                <li>□ CI/CD 파이프라인 구축</li>
                <li>□ Load balancing</li>
                <li>□ Health check 엔드포인트</li>
                <li>□ Graceful shutdown</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-2 text-green-900 dark:text-green-300">✅ 운영 단계</h3>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>□ 모니터링 대시보드</li>
                <li>□ 알림 시스템 (Slack, PagerDuty)</li>
                <li>□ 모델 버전 관리</li>
                <li>□ Rollback 전략</li>
                <li>□ 정기적 재학습 계획</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '🛠️ 배포 도구',
            icon: 'github' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'TorchServe',
                authors: 'PyTorch Team',
                year: '2023',
                description: 'PyTorch 모델 서빙 프레임워크',
                link: 'https://github.com/pytorch/serve'
              },
              {
                title: 'NVIDIA Triton Inference Server',
                authors: 'NVIDIA',
                year: '2023',
                description: '멀티 프레임워크 추론 서버',
                link: 'https://github.com/triton-inference-server/server'
              },
              {
                title: 'MLflow',
                authors: 'Databricks',
                year: '2023',
                description: 'ML 실험 관리 및 배포',
                link: 'https://mlflow.org/'
              }
            ]
          },
          {
            title: '📚 MLOps 가이드',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Full Stack Deep Learning',
                authors: 'UC Berkeley',
                year: '2023',
                description: 'MLOps 실전 강의',
                link: 'https://fullstackdeeplearning.com/'
              },
              {
                title: 'Made With ML',
                authors: 'Goku Mohandas',
                year: '2023',
                description: 'ML 프로덕션 가이드',
                link: 'https://madewithml.com/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
