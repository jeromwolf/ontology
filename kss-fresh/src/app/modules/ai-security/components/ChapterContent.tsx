'use client';

import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { aiSecurityMetadata } from '../metadata';
import AdversarialAttackVisualizer from './AdversarialAttackVisualizer';
import PrivacyPreservingML from './PrivacyPreservingML';
import ModelSecurityAnalyzer from './ModelSecurityAnalyzer';
import ThreatDetectionDashboard from './ThreatDetectionDashboard';
import SecurityAuditTool from './SecurityAuditTool';

interface Props {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: Props) {
  const chapterIndex = aiSecurityMetadata.chapters.findIndex(ch => ch.id === chapterId);
  const chapter = aiSecurityMetadata.chapters[chapterIndex];
  const prevChapter = chapterIndex > 0 ? aiSecurityMetadata.chapters[chapterIndex - 1] : null;
  const nextChapter = chapterIndex < aiSecurityMetadata.chapters.length - 1 ? aiSecurityMetadata.chapters[chapterIndex + 1] : null;

  const renderContent = () => {
    switch (chapterId) {
      case 'fundamentals':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>AI 보안 기초</h1>
            
            <section className="mb-8">
              <h2>1. AI 보안의 중요성</h2>
              <p>
                인공지능 시스템이 우리 생활의 모든 영역에 통합되면서, AI 보안은 점점 더 중요해지고 있습니다.
                의료, 금융, 자율주행, 보안 시스템 등 중요한 의사결정에 AI가 사용되면서,
                이러한 시스템의 보안 취약점은 심각한 결과를 초래할 수 있습니다.
              </p>
              
              <h3>주요 보안 위협</h3>
              <ul>
                <li><strong>적대적 공격(Adversarial Attacks)</strong>: 모델을 속이기 위한 입력 조작</li>
                <li><strong>모델 추출(Model Extraction)</strong>: API를 통한 모델 복제</li>
                <li><strong>데이터 중독(Data Poisoning)</strong>: 학습 데이터 오염</li>
                <li><strong>프라이버시 침해</strong>: 학습 데이터 정보 유출</li>
                <li><strong>백도어 공격</strong>: 숨겨진 악의적 동작 삽입</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>2. AI 시스템의 공격 표면</h2>
              <p>
                AI 시스템은 전통적인 소프트웨어와는 다른 독특한 공격 표면을 가지고 있습니다.
              </p>
              
              <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">학습 단계 공격</h4>
                <ul className="space-y-2">
                  <li>• 데이터 수집 과정에서의 오염</li>
                  <li>• 라벨링 과정 조작</li>
                  <li>• 학습 알고리즘 취약점 악용</li>
                  <li>• 하이퍼파라미터 조작</li>
                </ul>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">추론 단계 공격</h4>
                <ul className="space-y-2">
                  <li>• 적대적 예제 입력</li>
                  <li>• 모델 역공학</li>
                  <li>• 사이드 채널 공격</li>
                  <li>• API 남용</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>3. 보안 원칙과 베스트 프랙티스</h2>
              <p>
                AI 시스템 보안을 위한 기본 원칙들을 이해하고 적용해야 합니다.
              </p>
              
              <h3>Defense in Depth</h3>
              <p>
                여러 계층의 보안 메커니즘을 구현하여 단일 실패 지점을 방지합니다.
              </p>
              
              <h3>Security by Design</h3>
              <p>
                개발 초기 단계부터 보안을 고려하여 설계합니다.
              </p>
              
              <h3>지속적인 모니터링</h3>
              <p>
                배포된 모델의 동작을 지속적으로 모니터링하고 이상 징후를 탐지합니다.
              </p>
            </section>

            <div className="my-8">
              <ThreatDetectionDashboard />
            </div>
          </div>
        );

      case 'adversarial-attacks':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>적대적 공격</h1>
            
            <section className="mb-8">
              <h2>1. 적대적 예제란?</h2>
              <p>
                적대적 예제(Adversarial Examples)는 의도적으로 설계된 입력으로,
                인간에게는 정상적으로 보이지만 AI 모델을 속여 잘못된 예측을 하도록 만듭니다.
              </p>
              
              <h3>적대적 공격의 특징</h3>
              <ul>
                <li><strong>미세한 변화</strong>: 육안으로 구분하기 어려운 작은 노이즈 추가</li>
                <li><strong>전이성(Transferability)</strong>: 한 모델에서 생성된 적대적 예제가 다른 모델에도 효과적</li>
                <li><strong>견고성</strong>: 압축, 크기 조정 등의 변환에도 공격 효과 유지</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>2. 공격 기법 분류</h2>
              
              <h3>White-box 공격</h3>
              <p>모델의 구조와 가중치를 완전히 알고 있는 상황에서의 공격</p>
              <ul>
                <li><strong>FGSM (Fast Gradient Sign Method)</strong>: 가장 간단하고 빠른 공격</li>
                <li><strong>PGD (Projected Gradient Descent)</strong>: 반복적인 최적화를 통한 강력한 공격</li>
                <li><strong>C&W (Carlini & Wagner)</strong>: 최소한의 왜곡으로 강력한 공격</li>
              </ul>
              
              <h3>Black-box 공격</h3>
              <p>모델의 내부 구조를 모르고 API 접근만 가능한 상황</p>
              <ul>
                <li><strong>전이 기반 공격</strong>: 대체 모델에서 생성한 적대적 예제 사용</li>
                <li><strong>쿼리 기반 공격</strong>: 반복적인 쿼리를 통한 공격 최적화</li>
                <li><strong>결정 경계 공격</strong>: 분류 경계 추정을 통한 공격</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>3. 실제 공격 시나리오</h2>
              
              <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">자율주행차 공격</h4>
                <p>
                  정지 신호에 작은 스티커를 붙여 속도 제한 표지판으로 오인식하게 만드는 공격
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">얼굴 인식 회피</h4>
                <p>
                  특수하게 설계된 안경이나 패치를 착용하여 얼굴 인식 시스템을 회피
                </p>
              </div>
            </section>

            <div className="my-8">
              <AdversarialAttackVisualizer />
            </div>
          </div>
        );

      case 'model-security':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>모델 보안</h1>
            
            <section className="mb-8">
              <h2>1. 모델 추출 공격</h2>
              <p>
                모델 추출(Model Extraction)은 공격자가 대상 모델의 기능을 복제하여
                자신만의 대체 모델을 만드는 공격입니다.
              </p>
              
              <h3>공격 방법</h3>
              <ul>
                <li><strong>API 남용</strong>: 대량의 쿼리를 통한 입출력 수집</li>
                <li><strong>모델 역공학</strong>: 수집된 데이터로 모델 구조 추론</li>
                <li><strong>지식 증류</strong>: 수집된 데이터로 새 모델 학습</li>
              </ul>
              
              <h3>위험성</h3>
              <ul>
                <li>지적 재산권 침해</li>
                <li>비즈니스 모델 손상</li>
                <li>추출된 모델을 통한 2차 공격</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>2. 백도어 공격</h2>
              <p>
                백도어 공격은 모델에 숨겨진 악의적 동작을 삽입하는 공격으로,
                특정 트리거가 있을 때만 활성화됩니다.
              </p>
              
              <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">백도어 유형</h4>
                <ul className="space-y-2">
                  <li>• <strong>데이터 중독</strong>: 학습 데이터에 트리거 삽입</li>
                  <li>• <strong>모델 조작</strong>: 사전 학습된 모델에 백도어 삽입</li>
                  <li>• <strong>공급망 공격</strong>: 서드파티 모델/데이터셋 오염</li>
                </ul>
              </div>
              
              <h3>탐지 방법</h3>
              <ul>
                <li>Neural Cleanse: 트리거 역공학</li>
                <li>활성화 패턴 분석</li>
                <li>모델 프루닝을 통한 백도어 제거</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>3. 모델 보호 기법</h2>
              
              <h3>워터마킹</h3>
              <p>
                모델에 고유한 서명을 삽입하여 소유권을 증명하고 불법 복제를 탐지합니다.
              </p>
              
              <h3>모델 암호화</h3>
              <p>
                배포된 모델의 가중치를 암호화하여 역공학을 방지합니다.
              </p>
              
              <h3>API 보안</h3>
              <ul>
                <li>Rate limiting</li>
                <li>쿼리 패턴 모니터링</li>
                <li>출력 난독화</li>
                <li>적응형 응답</li>
              </ul>
            </section>

            <div className="my-8">
              <ModelSecurityAnalyzer />
            </div>
          </div>
        );

      case 'privacy-preserving':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>프라이버시 보호 ML</h1>
            
            <section className="mb-8">
              <h2>1. 차분 프라이버시</h2>
              <p>
                차분 프라이버시(Differential Privacy)는 개별 데이터 포인트의 
                프라이버시를 보호하면서 통계적 분석을 가능하게 하는 수학적 프레임워크입니다.
              </p>
              
              <h3>핵심 개념</h3>
              <ul>
                <li><strong>ε-차분 프라이버시</strong>: 프라이버시 손실의 상한</li>
                <li><strong>노이즈 추가</strong>: Laplace 또는 Gaussian 노이즈</li>
                <li><strong>프라이버시 예산</strong>: 총 프라이버시 손실 관리</li>
              </ul>
              
              <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">DP-SGD</h4>
                <p>
                  차분 프라이버시를 적용한 확률적 경사 하강법:
                </p>
                <ul className="space-y-2 mt-2">
                  <li>• 그래디언트 클리핑</li>
                  <li>• 노이즈 추가</li>
                  <li>• 프라이버시 회계</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>2. 연합 학습</h2>
              <p>
                연합 학습(Federated Learning)은 데이터를 중앙 서버로 수집하지 않고
                분산된 환경에서 모델을 학습하는 방법입니다.
              </p>
              
              <h3>작동 원리</h3>
              <ol>
                <li>중앙 서버가 초기 모델 배포</li>
                <li>각 클라이언트가 로컬 데이터로 학습</li>
                <li>모델 업데이트만 서버로 전송</li>
                <li>서버가 업데이트 집계 및 새 모델 배포</li>
              </ol>
              
              <h3>보안 강화 기법</h3>
              <ul>
                <li><strong>Secure Aggregation</strong>: 암호화된 집계</li>
                <li><strong>동형 암호화</strong>: 암호화된 상태에서 연산</li>
                <li><strong>차분 프라이버시</strong>: 업데이트에 노이즈 추가</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>3. 프라이버시 보호 기법</h2>
              
              <h3>PATE (Private Aggregation of Teacher Ensembles)</h3>
              <p>
                여러 교사 모델의 합의를 통해 프라이버시를 보호하면서 학생 모델을 학습합니다.
              </p>
              
              <h3>Split Learning</h3>
              <p>
                모델을 여러 부분으로 나누어 각 당사자가 일부만 보유하고 학습합니다.
              </p>
              
              <h3>Homomorphic Encryption</h3>
              <p>
                암호화된 데이터에서 직접 연산을 수행하여 프라이버시를 완벽하게 보호합니다.
              </p>
            </section>

            <div className="my-8">
              <PrivacyPreservingML />
            </div>
          </div>
        );

      case 'robustness':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>견고성과 방어</h1>
            
            <section className="mb-8">
              <h2>1. 적대적 학습</h2>
              <p>
                적대적 학습(Adversarial Training)은 학습 과정에서 적대적 예제를
                포함시켜 모델의 견고성을 향상시키는 방법입니다.
              </p>
              
              <h3>기본 원리</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`min_θ E[(x,y)~D] max_δ∈S L(fθ(x+δ), y)

여기서:
- θ: 모델 파라미터
- δ: 적대적 perturbation
- S: 허용된 perturbation 집합
- L: 손실 함수`}
              </pre>
              
              <h3>학습 전략</h3>
              <ul>
                <li><strong>PGD-AT</strong>: 강력한 적대적 예제로 학습</li>
                <li><strong>TRADES</strong>: 정확도와 견고성의 균형</li>
                <li><strong>MART</strong>: 잘못 분류된 예제에 집중</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>2. 방어 메커니즘</h2>
              
              <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">입력 전처리</h4>
                <ul className="space-y-2">
                  <li>• <strong>Feature Squeezing</strong>: 색상 깊이 감소</li>
                  <li>• <strong>JPEG 압축</strong>: 고주파 노이즈 제거</li>
                  <li>• <strong>Spatial Smoothing</strong>: 공간적 필터링</li>
                  <li>• <strong>Pixel Deflection</strong>: 랜덤 픽셀 변환</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">모델 기반 방어</h4>
                <ul className="space-y-2">
                  <li>• <strong>Defensive Distillation</strong>: 부드러운 예측</li>
                  <li>• <strong>Ensemble Methods</strong>: 다중 모델 합의</li>
                  <li>• <strong>Random Smoothing</strong>: 확률적 방어</li>
                  <li>• <strong>Input Gradient Regularization</strong></li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>3. 인증된 방어</h2>
              <p>
                인증된 방어(Certified Defense)는 특정 크기의 perturbation에 대해
                수학적으로 보장된 견고성을 제공합니다.
              </p>
              
              <h3>주요 기법</h3>
              <ul>
                <li>
                  <strong>Randomized Smoothing</strong>
                  <p className="text-gray-600 dark:text-gray-400">
                    노이즈를 추가한 입력의 평균 예측으로 견고성 인증
                  </p>
                </li>
                <li>
                  <strong>Interval Bound Propagation</strong>
                  <p className="text-gray-600 dark:text-gray-400">
                    각 레이어의 출력 범위를 추적하여 견고성 검증
                  </p>
                </li>
                <li>
                  <strong>Convex Relaxation</strong>
                  <p className="text-gray-600 dark:text-gray-400">
                    비선형 활성화 함수의 convex 근사
                  </p>
                </li>
              </ul>
            </section>

            <div className="my-8">
              <SecurityAuditTool />
            </div>
          </div>
        );

      case 'security-testing':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>보안 테스팅</h1>
            
            <section className="mb-8">
              <h2>1. 보안 평가 프레임워크</h2>
              <p>
                AI 시스템의 보안을 체계적으로 평가하기 위한 프레임워크와 도구들을 활용합니다.
              </p>
              
              <h3>평가 영역</h3>
              <ul>
                <li><strong>견고성 평가</strong>: 적대적 예제에 대한 저항성</li>
                <li><strong>프라이버시 평가</strong>: 멤버십 추론, 모델 역전 공격</li>
                <li><strong>공정성 평가</strong>: 편향과 차별 검사</li>
                <li><strong>설명가능성</strong>: 모델 결정의 해석 가능성</li>
              </ul>
              
              <div className="bg-yellow-50 dark:bg-yellow-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">테스팅 도구</h4>
                <ul className="space-y-2">
                  <li>• <strong>CleverHans</strong>: 적대적 예제 생성 및 평가</li>
                  <li>• <strong>Foolbox</strong>: 다양한 공격 기법 구현</li>
                  <li>• <strong>ART (Adversarial Robustness Toolbox)</strong>: IBM의 종합 도구</li>
                  <li>• <strong>TextAttack</strong>: NLP 모델 공격 프레임워크</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>2. 테스트 시나리오</h2>
              
              <h3>White-box 테스팅</h3>
              <p>모델의 내부 구조를 완전히 알고 있는 상황에서의 테스트</p>
              <ul>
                <li>그래디언트 기반 공격 테스트</li>
                <li>모델 파라미터 분석</li>
                <li>활성화 패턴 검사</li>
              </ul>
              
              <h3>Black-box 테스팅</h3>
              <p>API 접근만 가능한 상황에서의 테스트</p>
              <ul>
                <li>쿼리 효율성 평가</li>
                <li>전이 공격 테스트</li>
                <li>API 남용 시뮬레이션</li>
              </ul>
              
              <h3>Gray-box 테스팅</h3>
              <p>부분적인 정보만 알고 있는 상황</p>
              <ul>
                <li>모델 구조 추론</li>
                <li>하이브리드 공격 테스트</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>3. 자동화된 보안 감사</h2>
              <p>
                CI/CD 파이프라인에 통합 가능한 자동화된 보안 테스트를 구현합니다.
              </p>
              
              <h3>테스트 자동화 단계</h3>
              <ol>
                <li>
                  <strong>데이터 검증</strong>
                  <ul>
                    <li>입력 데이터 무결성 검사</li>
                    <li>데이터 중독 탐지</li>
                    <li>이상치 검출</li>
                  </ul>
                </li>
                <li>
                  <strong>모델 검증</strong>
                  <ul>
                    <li>백도어 스캔</li>
                    <li>견고성 벤치마크</li>
                    <li>성능 저하 모니터링</li>
                  </ul>
                </li>
                <li>
                  <strong>런타임 모니터링</strong>
                  <ul>
                    <li>실시간 이상 탐지</li>
                    <li>공격 패턴 인식</li>
                    <li>자동 대응 시스템</li>
                  </ul>
                </li>
              </ol>
            </section>

            <div className="my-8">
              <ThreatDetectionDashboard />
            </div>
          </div>
        );

      case 'deployment-security':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>배포 보안</h1>
            
            <section className="mb-8">
              <h2>1. 보안 아키텍처</h2>
              <p>
                프로덕션 환경에서 AI 시스템을 안전하게 배포하기 위한 아키텍처 설계가 중요합니다.
              </p>
              
              <h3>계층별 보안</h3>
              <div className="space-y-4">
                <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg">
                  <h4 className="font-semibold mb-2">인프라 계층</h4>
                  <ul className="space-y-2">
                    <li>• 네트워크 격리 및 세분화</li>
                    <li>• 안전한 컨테이너 오케스트레이션</li>
                    <li>• 하드웨어 보안 모듈(HSM) 활용</li>
                    <li>• 신뢰할 수 있는 실행 환경(TEE)</li>
                  </ul>
                </div>
                
                <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg">
                  <h4 className="font-semibold mb-2">애플리케이션 계층</h4>
                  <ul className="space-y-2">
                    <li>• API 게이트웨이 및 rate limiting</li>
                    <li>• 입력 검증 및 sanitization</li>
                    <li>• 모델 서빙 보안</li>
                    <li>• 로깅 및 감사 추적</li>
                  </ul>
                </div>
                
                <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg">
                  <h4 className="font-semibold mb-2">데이터 계층</h4>
                  <ul className="space-y-2">
                    <li>• 암호화된 데이터 저장</li>
                    <li>• 안전한 데이터 파이프라인</li>
                    <li>• 접근 제어 및 권한 관리</li>
                    <li>• 데이터 마스킹 및 익명화</li>
                  </ul>
                </div>
              </div>
            </section>

            <section className="mb-8">
              <h2>2. MLOps 보안</h2>
              <p>
                ML 파이프라인의 각 단계에서 보안을 통합하여 안전한 ML 운영을 구현합니다.
              </p>
              
              <h3>보안 파이프라인</h3>
              <ul>
                <li>
                  <strong>코드 보안</strong>
                  <ul className="mt-2 ml-4">
                    <li>• 정적 코드 분석(SAST)</li>
                    <li>• 의존성 취약점 스캔</li>
                    <li>• 코드 서명 및 검증</li>
                  </ul>
                </li>
                <li>
                  <strong>모델 보안</strong>
                  <ul className="mt-2 ml-4">
                    <li>• 모델 버전 관리 및 추적</li>
                    <li>• 모델 무결성 검증</li>
                    <li>• A/B 테스트 보안</li>
                  </ul>
                </li>
                <li>
                  <strong>배포 보안</strong>
                  <ul className="mt-2 ml-4">
                    <li>• 안전한 CI/CD 파이프라인</li>
                    <li>• 자동화된 보안 테스트</li>
                    <li>• 롤백 및 복구 계획</li>
                  </ul>
                </li>
              </ul>
            </section>

            <section className="mb-8">
              <h2>3. 모니터링과 대응</h2>
              
              <h3>실시간 모니터링</h3>
              <p>
                배포된 AI 시스템의 보안 상태를 지속적으로 모니터링합니다.
              </p>
              
              <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">주요 모니터링 지표</h4>
                <ul className="space-y-2">
                  <li>• 비정상적인 입력 패턴</li>
                  <li>• 예측 신뢰도 변화</li>
                  <li>• API 사용 패턴</li>
                  <li>• 시스템 리소스 사용량</li>
                  <li>• 에러율 및 지연 시간</li>
                </ul>
              </div>
              
              <h3>사고 대응 계획</h3>
              <ol>
                <li><strong>탐지</strong>: 자동화된 알림 시스템</li>
                <li><strong>분류</strong>: 위협 수준 평가</li>
                <li><strong>격리</strong>: 영향받은 시스템 격리</li>
                <li><strong>복구</strong>: 안전한 상태로 복원</li>
                <li><strong>분석</strong>: 사후 분석 및 개선</li>
              </ol>
            </section>

            <div className="my-8">
              <ModelSecurityAnalyzer />
            </div>
          </div>
        );

      case 'case-studies':
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h1>사례 연구</h1>
            
            <section className="mb-8">
              <h2>1. 실제 AI 보안 사고</h2>
              <p>
                실제로 발생한 AI 보안 사고들을 분석하여 교훈을 얻고 방어 전략을 수립합니다.
              </p>
              
              <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
                <h3>사례 1: Microsoft Tay 챗봇 (2016)</h3>
                <p><strong>사고 개요:</strong> Twitter에서 운영된 AI 챗봇이 악의적인 사용자들의 조작으로 인해 부적절한 발언을 학습</p>
                <p><strong>원인:</strong></p>
                <ul>
                  <li>• 실시간 학습 시스템의 취약점</li>
                  <li>• 입력 필터링 부재</li>
                  <li>• 악의적 데이터에 대한 방어 메커니즘 부족</li>
                </ul>
                <p><strong>교훈:</strong></p>
                <ul>
                  <li>• 사용자 입력에 대한 엄격한 검증 필요</li>
                  <li>• 온라인 학습 시스템의 위험성 인식</li>
                  <li>• 콘텐츠 모더레이션 시스템 구축</li>
                </ul>
              </div>
              
              <div className="bg-yellow-50 dark:bg-yellow-950/30 p-6 rounded-lg my-4">
                <h3>사례 2: 자율주행차 적대적 공격 (2018)</h3>
                <p><strong>사고 개요:</strong> 연구자들이 도로 표지판에 스티커를 붙여 Tesla Autopilot을 속이는 데 성공</p>
                <p><strong>공격 방법:</strong></p>
                <ul>
                  <li>• 정지 표지판에 작은 스티커 부착</li>
                  <li>• 차선에 테이프로 가짜 표시</li>
                  <li>• 속도 제한 표지판 조작</li>
                </ul>
                <p><strong>대응 방안:</strong></p>
                <ul>
                  <li>• 다중 센서 융합</li>
                  <li>• 컨텍스트 기반 검증</li>
                  <li>• 적대적 학습 적용</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>2. 산업별 보안 사례</h2>
              
              <h3>금융 AI 보안</h3>
              <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
                <p><strong>과제:</strong> 사기 탐지 시스템의 보안</p>
                <ul className="mt-2">
                  <li>• 공격자가 탐지 회피를 위한 패턴 학습</li>
                  <li>• 모델 추출을 통한 우회 방법 개발</li>
                  <li>• 고객 데이터 프라이버시 보호</li>
                </ul>
                <p className="mt-4"><strong>해결책:</strong></p>
                <ul>
                  <li>• 앙상블 모델 사용</li>
                  <li>• 지속적인 모델 업데이트</li>
                  <li>• 연합 학습 도입</li>
                </ul>
              </div>
              
              <h3>의료 AI 보안</h3>
              <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg my-4">
                <p><strong>과제:</strong> 의료 영상 진단 AI의 보안</p>
                <ul className="mt-2">
                  <li>• 적대적 예제로 인한 오진 위험</li>
                  <li>• 환자 데이터 프라이버시</li>
                  <li>• 규제 준수 (HIPAA 등)</li>
                </ul>
                <p className="mt-4"><strong>해결책:</strong></p>
                <ul>
                  <li>• 차분 프라이버시 적용</li>
                  <li>• 설명가능한 AI 도입</li>
                  <li>• 엄격한 접근 제어</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2>3. 미래 전망과 과제</h2>
              
              <h3>신흥 위협</h3>
              <ul>
                <li><strong>생성 AI 악용</strong>: 딥페이크, 합성 콘텐츠</li>
                <li><strong>LLM 보안</strong>: 프롬프트 인젝션, 정보 유출</li>
                <li><strong>양자 컴퓨팅 위협</strong>: 기존 암호화 무력화</li>
              </ul>
              
              <h3>방어 기술 발전</h3>
              <ul>
                <li><strong>Zero Trust AI</strong>: 모든 입력을 의심</li>
                <li><strong>블록체인 기반 모델 검증</strong></li>
                <li><strong>양자 저항 암호화</strong></li>
                <li><strong>자율적 방어 시스템</strong></li>
              </ul>
              
              <div className="bg-purple-50 dark:bg-purple-950/30 p-6 rounded-lg my-4">
                <h4 className="font-semibold mb-2">Best Practices</h4>
                <ol className="space-y-2">
                  <li>1. Security by Design 원칙 적용</li>
                  <li>2. 지속적인 위협 모델링</li>
                  <li>3. 다계층 방어 전략</li>
                  <li>4. 정기적인 보안 감사</li>
                  <li>5. 사고 대응 계획 수립</li>
                  <li>6. 보안 인식 교육</li>
                </ol>
              </div>
            </section>

            <div className="my-8">
              <SecurityAuditTool />
            </div>
          </div>
        );

      default:
        return <div>Chapter not found</div>;
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <Link
          href="/modules/ai-security"
          className="inline-flex items-center text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 mb-4"
        >
          <ChevronLeft className="w-4 h-4 mr-1" />
          목차로 돌아가기
        </Link>
        
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          {chapter.title}
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          {chapter.description}
        </p>
      </div>

      {renderContent()}

      <div className="mt-12 flex justify-between">
        {prevChapter && (
          <Link
            href={`/modules/ai-security/${prevChapter.id}`}
            className="inline-flex items-center px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            이전: {prevChapter.title}
          </Link>
        )}
        
        {nextChapter && (
          <Link
            href={`/modules/ai-security/${nextChapter.id}`}
            className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 ml-auto"
          >
            다음: {nextChapter.title}
            <ChevronRight className="w-4 h-4 ml-1" />
          </Link>
        )}
      </div>
    </div>
  );
}