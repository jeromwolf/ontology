'use client'

import React from 'react'
import { Rocket, Code, Database, Cloud, TrendingUp, CheckCircle2, AlertTriangle, Award } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter10() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Rocket className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                실전 응용
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Case Studies, Production Deployment, Best Practices
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Rocket className="w-6 h-6 text-emerald-600" />
              실전 최적화 프로젝트
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                이론을 실제 문제에 적용하는 것은 완전히 다른 차원의 도전입니다.
                <strong>문제 정식화</strong>부터 <strong>프로덕션 배포</strong>까지
                실전에서 마주하는 다양한 이슈와 해결 방법을 학습합니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">실전 프로젝트의 단계</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <span><strong>문제 이해 & 정식화</strong>: 비즈니스 문제를 수학적 문제로 변환</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Code className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>구현 & 검증</strong>: 알고리즘 구현 및 소규모 테스트</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Cloud className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>배포 & 모니터링</strong>: 프로덕션 환경 배포 및 지속 관리</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Case Study 1: Supply Chain */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Case Study 1: 공급망 최적화
          </h2>

          <div className="space-y-6">
            {/* Problem */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <h3 className="text-xl font-bold mb-3">비즈니스 문제</h3>
              <p className="text-blue-100">
                전국 10개 창고에서 100개 매장으로 제품을 배송하는 물류 회사.
                <strong>배송 비용 최소화</strong>하면서 <strong>납기 준수</strong> 및
                <strong>재고 최적화</strong> 필요.
              </p>
            </div>

            {/* Formulation */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                문제 정식화
              </h3>

              <div className="space-y-4">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">결정 변수</h4>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-300">
                    <p>xᵢⱼₖ: 창고 i → 매장 j로 제품 k의 배송량</p>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">목적 함수</h4>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <p>minimize: Σᵢⱼₖ cᵢⱼₖ · xᵢⱼₖ (총 배송 비용)</p>
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">제약 조건</h4>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <p>1. Σᵢ xᵢⱼₖ ≥ dⱼₖ (수요 만족)</p>
                    <p>2. Σⱼ xᵢⱼₖ ≤ sᵢₖ (공급 제약)</p>
                    <p>3. xᵢⱼₖ ≥ 0 (비음수)</p>
                    <p>4. 일부 xᵢⱼₖ ∈ ℤ (정수 제약)</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Solution Approach */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                해결 접근법
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>
                    <strong>문제 유형:</strong> Mixed Integer Linear Programming (MILP)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>
                    <strong>솔버 선택:</strong> Gurobi 또는 CPLEX (상용), PuLP (오픈소스)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>
                    <strong>데이터 준비:</strong> 비용 행렬, 수요/공급 데이터, 거리 정보
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>
                    <strong>검증:</strong> 작은 데이터셋으로 테스트, 제약 만족 확인
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>
                    <strong>결과:</strong> 배송 비용 23% 감소, 재고 회전율 15% 개선
                  </div>
                </div>
              </div>

              <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>💡 핵심 교훈:</strong> 실시간 해가 필요하지 않아
                  매일 밤 배치로 실행. 예외 상황은 휴리스틱으로 처리.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Case Study 2: ML Model Optimization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Case Study 2: 추천 시스템 최적화
          </h2>

          <div className="space-y-6">
            {/* Problem */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <h3 className="text-xl font-bold mb-3">비즈니스 문제</h3>
              <p className="text-purple-100">
                E-커머스 플랫폼에서 개인화 추천 시스템 구축.
                <strong>클릭률(CTR) 최대화</strong>하면서 <strong>다양성 유지</strong> 및
                <strong>응답 시간 100ms 이하</strong> 보장 필요.
              </p>
            </div>

            {/* Multi-objective Formulation */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                다목적 최적화 문제로 정식화
              </h3>

              <div className="space-y-4">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">목적 함수</h4>
                  <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <p>1. <strong>Relevance Score 최대화:</strong> f₁(θ) = -Σ predicted_CTR(item, user; θ)</p>
                    <p>2. <strong>Diversity 최대화:</strong> f₂(θ) = -diversity_score(recommendations)</p>
                    <p>3. <strong>Latency 최소화:</strong> f₃(θ) = inference_time(θ)</p>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">제약 조건</h4>
                  <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <p>1. inference_time(θ) ≤ 100ms</p>
                    <p>2. model_size(θ) ≤ 500MB</p>
                    <p>3. diversity_score ≥ 0.3</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Solution */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                해결 전략
              </h3>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="space-y-3">
                  <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3">
                    <p className="font-bold text-slate-800 dark:text-white mb-1">Phase 1: 모델 최적화</p>
                    <p className="text-slate-600 dark:text-slate-400">
                      Bayesian Optimization으로 하이퍼파라미터 튜닝
                      (학습률, 임베딩 차원, 레이어 수)
                    </p>
                  </div>
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                    <p className="font-bold text-slate-800 dark:text-white mb-1">Phase 2: 다양성 보장</p>
                    <p className="text-slate-600 dark:text-slate-400">
                      MMR (Maximal Marginal Relevance) 알고리즘 적용
                    </p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                    <p className="font-bold text-slate-800 dark:text-white mb-1">Phase 3: 속도 최적화</p>
                    <p className="text-slate-600 dark:text-slate-400">
                      모델 압축 (Quantization, Pruning),
                      캐싱 전략 적용
                    </p>
                  </div>
                  <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                    <p className="font-bold text-slate-800 dark:text-white mb-1">Phase 4: A/B 테스트</p>
                    <p className="text-slate-600 dark:text-slate-400">
                      실제 트래픽으로 성능 검증,
                      지속 모니터링
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>🎯 결과:</strong> CTR 18% 증가, 다양성 지수 0.35 유지,
                  평균 응답 시간 85ms 달성
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Implementation Best Practices */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            구현 Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <Code className="w-6 h-6 text-emerald-600" />
                코드 구조
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>모듈화:</strong> 목적 함수, 제약, 솔버 분리</li>
                <li>• <strong>설정 파일:</strong> YAML/JSON으로 파라미터 관리</li>
                <li>• <strong>로깅:</strong> 모든 실행 과정 기록</li>
                <li>• <strong>단위 테스트:</strong> 각 컴포넌트 검증</li>
                <li>• <strong>버전 관리:</strong> Git + 실험 추적 (MLflow 등)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <Database className="w-6 h-6 text-blue-600" />
                데이터 관리
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>검증:</strong> 입력 데이터 무결성 체크</li>
                <li>• <strong>정규화:</strong> 스케일 차이 처리</li>
                <li>• <strong>샘플링:</strong> 대규모 데이터 다루기</li>
                <li>• <strong>캐싱:</strong> 반복 계산 방지</li>
                <li>• <strong>백업:</strong> 중간 결과 저장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-purple-600" />
                성능 최적화
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>프로파일링:</strong> 병목 지점 식별</li>
                <li>• <strong>병렬화:</strong> 멀티프로세싱/GPU 활용</li>
                <li>• <strong>메모리:</strong> 대용량 데이터 처리 전략</li>
                <li>• <strong>조기 종료:</strong> 수렴 기준 설정</li>
                <li>• <strong>근사해:</strong> 정확도-속도 trade-off</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-orange-600" />
                오류 처리
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>실행 불가능:</strong> 제약 완화 또는 경고</li>
                <li>• <strong>수렴 실패:</strong> 재시도 로직</li>
                <li>• <strong>수치 불안정:</strong> 정규화, 스케일링</li>
                <li>• <strong>시간 초과:</strong> 부분 해 반환</li>
                <li>• <strong>예외 처리:</strong> 명확한 에러 메시지</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Production Deployment */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            프로덕션 배포
          </h2>

          <div className="space-y-6">
            {/* Architecture */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                시스템 아키텍처
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>
                    <strong>API 서버:</strong> FastAPI/Flask로 최적화 요청 처리
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>
                    <strong>작업 큐:</strong> Celery/RabbitMQ로 비동기 처리
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>
                    <strong>캐시 레이어:</strong> Redis로 빠른 응답
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>
                    <strong>데이터베이스:</strong> PostgreSQL로 결과 저장
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>
                    <strong>모니터링:</strong> Prometheus + Grafana
                  </div>
                </div>
              </div>
            </div>

            {/* Monitoring */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                모니터링 지표
              </h3>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">성능 지표</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 평균 실행 시간</li>
                    <li>• 수렴률 (성공/실패 비율)</li>
                    <li>• 목적 함수 값 분포</li>
                    <li>• 반복 횟수</li>
                  </ul>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">시스템 지표</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• CPU/메모리 사용량</li>
                    <li>• API 응답 시간</li>
                    <li>• 에러율</li>
                    <li>• 큐 대기 시간</li>
                  </ul>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">비즈니스 지표</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 비용 절감액</li>
                    <li>• 효율성 개선률</li>
                    <li>• 사용자 만족도</li>
                    <li>• ROI</li>
                  </ul>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">품질 지표</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 제약 위반 빈도</li>
                    <li>• 해의 실행 가능성</li>
                    <li>• 안정성 점수</li>
                    <li>• 재현 가능성</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Common Pitfalls */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            흔한 실수와 해결책
          </h2>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-red-200 dark:border-gray-700">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">과도한 정밀도 추구</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <strong>문제:</strong> 실용적으로 충분한 해를 찾은 후에도 계속 최적화
                  </p>
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    <strong>해결:</strong> 비즈니스 요구사항 기반 종료 조건 설정. "충분히 좋은" 해로 만족.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-orange-200 dark:border-gray-700">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-orange-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">스케일링 무시</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <strong>문제:</strong> 변수들의 스케일 차이가 커서 수렴 문제 발생
                  </p>
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    <strong>해결:</strong> 모든 변수를 비슷한 범위로 정규화. Min-max scaling 또는 standardization 적용.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-yellow-200 dark:border-gray-700">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-yellow-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">검증 부족</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <strong>문제:</strong> 찾은 해가 실제로 제약을 만족하는지 검증하지 않음
                  </p>
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    <strong>해결:</strong> 자동화된 검증 로직 구현. 모든 제약 조건 명시적으로 확인.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-blue-200 dark:border-gray-700">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">하이퍼파라미터 기본값 사용</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <strong>문제:</strong> 알고리즘 파라미터를 튜닝하지 않고 기본값 사용
                  </p>
                  <p className="text-sm text-emerald-700 dark:text-emerald-400">
                    <strong>해결:</strong> 문제 특성에 맞게 파라미터 조정. Grid/Random Search로 최적 파라미터 찾기.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tools and Libraries */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            추천 도구 및 라이브러리
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">선형/정수 계획</h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>Gurobi:</strong> 상용, 최고 성능</li>
                <li>• <strong>CPLEX:</strong> IBM, 학계 무료</li>
                <li>• <strong>PuLP:</strong> Python, 오픈소스</li>
                <li>• <strong>OR-Tools:</strong> Google, 무료</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">비선형 최적화</h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>SciPy:</strong> 다양한 최적화 알고리즘</li>
                <li>• <strong>CVXPY:</strong> 볼록 최적화</li>
                <li>• <strong>Pyomo:</strong> 대수 모델링</li>
                <li>• <strong>JuMP:</strong> Julia 기반</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">메타휴리스틱</h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>DEAP:</strong> 진화 알고리즘</li>
                <li>• <strong>pymoo:</strong> 다목적 최적화</li>
                <li>• <strong>Platypus:</strong> NSGA-II 등</li>
                <li>• <strong>pygmo:</strong> 다양한 알고리즘</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">하이퍼파라미터 튜닝</h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• <strong>Optuna:</strong> Bayesian Opt</li>
                <li>• <strong>Ray Tune:</strong> 분산 튜닝</li>
                <li>• <strong>Hyperopt:</strong> TPE 알고리즘</li>
                <li>• <strong>Ax/BoTorch:</strong> Facebook</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-emerald-50 dark:from-blue-900/20 dark:to-emerald-900/20 rounded-xl p-8 border border-emerald-200 dark:border-emerald-800">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
              핵심 요점
            </h2>
            <ul className="space-y-3 text-slate-700 dark:text-slate-300">
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">1.</span>
                <span><strong>문제 정식화</strong>가 성공의 80%입니다. 비즈니스 문제를 정확히 수학 모델로 변환하세요.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>작게 시작</strong>하여 검증하고, 점진적으로 확장하세요.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>모니터링과 로깅</strong>은 프로덕션에서 필수입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span><strong>완벽한 해</strong>보다 <strong>충분히 좋은 해</strong>를 빠르게 찾는 것이 실용적입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>적절한 <strong>도구와 라이브러리</strong>를 활용하여 개발 시간을 단축하세요.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Final Message */}
        <section>
          <div className="bg-gradient-to-r from-emerald-600 to-teal-700 rounded-2xl p-8 text-white shadow-xl">
            <div className="flex items-center gap-4 mb-4">
              <Award className="w-12 h-12" />
              <h2 className="text-2xl font-bold">최적화 이론 학습 완료!</h2>
            </div>
            <p className="text-emerald-100 mb-6">
              축하합니다! 최적화 이론의 기초부터 실전 응용까지 모든 과정을 마쳤습니다.
              이제 여러분은 다양한 실제 문제에 최적화 기법을 적용할 수 있는 능력을 갖추었습니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="bg-white/10 rounded-lg p-4">
                <p className="font-bold mb-2">학습한 내용</p>
                <ul className="space-y-1 text-emerald-100">
                  <li>• 선형/비선형/볼록 최적화</li>
                  <li>• 정수 계획법</li>
                  <li>• 메타휴리스틱</li>
                  <li>• 제약 최적화</li>
                  <li>• 다목적 최적화</li>
                </ul>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <p className="font-bold mb-2">다음 단계</p>
                <ul className="space-y-1 text-emerald-100">
                  <li>• 실제 프로젝트에 적용</li>
                  <li>• 오픈소스 기여</li>
                  <li>• 최신 논문 읽기</li>
                  <li>• 커뮤니티 참여</li>
                  <li>• 지속적인 학습</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 온라인 강의 & 리소스',
              icon: 'web' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'MIT OCW - Optimization Methods',
                  url: 'https://ocw.mit.edu/courses/sloan-school-of-management/15-093j-optimization-methods-fall-2009/',
                  description: 'MIT 최적화 이론 강의 (무료, 동영상 포함)'
                },
                {
                  title: 'Stanford - Convex Optimization',
                  url: 'https://web.stanford.edu/class/ee364a/',
                  description: 'Stephen Boyd 볼록 최적화 강의 (슬라이드, 과제 포함)'
                },
                {
                  title: 'NEOS Server',
                  url: 'https://neos-server.org/',
                  description: '온라인 최적화 솔버 서버 (무료, 다양한 솔버 지원)'
                },
                {
                  title: 'OR-Library',
                  url: 'http://people.brunel.ac.uk/~mastjjb/jeb/info.html',
                  description: '운영연구 문제 데이터셋 라이브러리'
                }
              ]
            },
            {
              title: '📖 핵심 교재',
              icon: 'research' as const,
              color: 'border-indigo-500',
              items: [
                {
                  title: 'Convex Optimization (Boyd & Vandenberghe)',
                  url: 'https://web.stanford.edu/~boyd/cvxbook/',
                  description: '볼록 최적화 바이블 (PDF 무료, 2004)'
                },
                {
                  title: 'Numerical Optimization (Nocedal & Wright)',
                  url: 'https://www.springer.com/gp/book/9780387303031',
                  description: '수치 최적화 알고리즘 교과서 (2nd Edition, 2006)'
                },
                {
                  title: 'Linear Programming (Vanderbei)',
                  url: 'https://www.springer.com/gp/book/9781461476290',
                  description: '선형 계획법 입문서 (5th Edition, 2020)'
                },
                {
                  title: 'Introduction to Operations Research (Hillier & Lieberman)',
                  url: 'https://www.mheducation.com/highered/product/introduction-operations-research-hillier-lieberman/M9781259872990.html',
                  description: '운영연구 전반을 다루는 교과서 (11th Edition, 2021)'
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-emerald-500',
              items: [
                {
                  title: 'CVX / CVXPY',
                  url: 'https://www.cvxpy.org/',
                  description: 'Python 볼록 최적화 모델링 라이브러리 (오픈소스)'
                },
                {
                  title: 'Gurobi Optimizer',
                  url: 'https://www.gurobi.com/',
                  description: '상용 최적화 솔버 (학계 무료, 최고 성능)'
                },
                {
                  title: 'CPLEX',
                  url: 'https://www.ibm.com/products/ilog-cplex-optimization-studio',
                  description: 'IBM 상용 솔버 (학계 무료, 대규모 MILP)'
                },
                {
                  title: 'SciPy optimize',
                  url: 'https://docs.scipy.org/doc/scipy/reference/optimize.html',
                  description: 'Python 과학 계산 라이브러리 (비선형 최적화)'
                },
                {
                  title: 'JuMP (Julia)',
                  url: 'https://jump.dev/',
                  description: 'Julia 수학적 최적화 모델링 언어 (고성능)'
                }
              ]
            }
          ]}
        />
      </div>
    </div>
  )
}
