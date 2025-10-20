import React from 'react';
import { Shield, Scale, AlertTriangle, FileCheck, Code, TrendingUp, Lock, Gavel } from 'lucide-react';
import References from '../References';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Medical AI 규제와 윤리
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          FDA 승인부터 HIPAA 준수까지, 책임있는 의료 AI 개발
        </p>
      </div>

      {/* 규제 프레임워크 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Gavel className="w-7 h-7 text-red-600" />
          글로벌 Medical AI 규제 프레임워크
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* FDA (미국) */}
          <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 p-6 rounded-lg border-2 border-red-300">
            <Shield className="w-12 h-12 text-red-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-red-900 dark:text-red-300">
              1. FDA (미국 식품의약국)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              SaMD (Software as a Medical Device) 규정
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">규제 등급:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Class I (저위험): 일반 규제 (예: 전자체온계)</li>
                <li>• Class II (중위험): 510(k) 승인 필요 (대부분 AI)</li>
                <li>• Class III (고위험): PMA 승인 (심장박동기 등)</li>
              </ul>
            </div>
            <div className="bg-red-900/10 dark:bg-red-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-red-900 dark:text-red-300 mb-1">승인 현황:</p>
              <p className="text-gray-700 dark:text-gray-300">
                520+ AI/ML 의료기기 승인 (2024.09), 영상의학 75%
              </p>
            </div>
          </div>

          {/* EU AI Act */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Scale className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              2. EU AI Act (유럽연합)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              세계 최초 포괄적 AI 규제법 (2024.08 발효)
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">위험 기반 분류:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 금지 (Prohibited): 사회 신용 점수</li>
                <li>• 고위험 (High-Risk): 의료 진단 AI ⭐</li>
                <li>• 제한적 위험: 챗봇 (투명성 필수)</li>
                <li>• 최소 위험: 스팸 필터</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">고위험 AI 요구사항:</p>
              <p className="text-gray-700 dark:text-gray-300">
                설명 가능성, 인간 감독, 데이터 품질 보장 의무화
              </p>
            </div>
          </div>

          {/* HIPAA (개인정보) */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Lock className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              3. HIPAA (미국 의료정보 보호법)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              Protected Health Information (PHI) 보호 규정
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">PHI 18개 식별자:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 이름, 주소, 생년월일, 전화번호</li>
                <li>• 의료기록번호, 소셜시큐리티 번호</li>
                <li>• IP 주소, 얼굴 사진</li>
                <li>• 지문, 음성 기록</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">위반 시 벌금:</p>
              <p className="text-gray-700 dark:text-gray-300">
                최대 $1.5M (악의적 방치), CVS Health $5M 벌금 (2023)
              </p>
            </div>
          </div>

          {/* GDPR (유럽) */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <FileCheck className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              4. GDPR (EU 개인정보 보호법)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              General Data Protection Regulation (2018 시행)
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 원칙:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Right to Explanation (설명받을 권리)</li>
                <li>• Right to be Forgotten (삭제 요청권)</li>
                <li>• Data Minimization (최소 수집)</li>
                <li>• Privacy by Design (설계 단계부터 보호)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">위반 시 벌금:</p>
              <p className="text-gray-700 dark:text-gray-300">
                최대 €20M 또는 전 세계 매출 4%, Amazon €746M (2021)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Medical AI 윤리 원칙 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <AlertTriangle className="w-7 h-7 text-yellow-600" />
          Medical AI 윤리 5대 원칙
        </h2>

        <div className="space-y-4">
          {/* 공정성 */}
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. 공정성 (Fairness & Non-Discrimination)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              인종, 성별, 나이에 따른 편향 제거
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-red-700 dark:text-red-400 mb-1">❌ 실패 사례</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  MIT 연구: 흑인 환자 대상 피부암 AI 진단 정확도 20%p 낮음 (2023)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-green-700 dark:text-green-400 mb-1">✅ 해결 방안</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  다양한 인종 데이터셋 균형 샘플링, Fairness Metrics (Demographic Parity) 모니터링
                </p>
              </div>
            </div>
          </div>

          {/* 투명성 */}
          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. 투명성 & 설명 가능성 (Transparency & Explainability)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              AI 의사결정 과정을 의사와 환자가 이해할 수 있어야 함
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-xs font-semibold mb-2">XAI 기법:</p>
              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                <li>• <strong>SHAP:</strong> 각 특성의 기여도 수치화 (예: lactate +0.23, WBC +0.15)</li>
                <li>• <strong>Grad-CAM:</strong> 의료 영상에서 AI가 주목한 영역 히트맵 표시</li>
                <li>• <strong>LIME:</strong> 국소 선형 근사로 복잡 모델 설명</li>
              </ul>
            </div>
          </div>

          {/* 프라이버시 */}
          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. 프라이버시 보호 (Privacy Preservation)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              환자 데이터 보호 및 익명화
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-purple-700 dark:text-purple-400 mb-1">기술적 방법</p>
                <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• De-identification (18개 PHI 제거)</li>
                  <li>• Differential Privacy (노이즈 추가)</li>
                  <li>• Federated Learning (데이터 이동 없이 학습)</li>
                  <li>• Homomorphic Encryption (암호화 상태 연산)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-purple-700 dark:text-purple-400 mb-1">HIPAA Safe Harbor</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  18개 식별자 제거 후 재식별 위험 매우 낮을 경우 PHI 아님으로 간주
                </p>
              </div>
            </div>
          </div>

          {/* 안전성 */}
          <div className="border-l-4 border-red-500 bg-red-50 dark:bg-red-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-red-900 dark:text-red-300">
              4. 안전성 & 견고성 (Safety & Robustness)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Adversarial Attack, Distribution Shift 대응
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Adversarial Training:</strong> 공격 데이터로 모델 강건화</li>
              <li>• <strong>Uncertainty Quantification:</strong> 예측 불확실성 정량화 (Bayesian DL)</li>
              <li>• <strong>Continuous Monitoring:</strong> 배포 후 성능 저하 실시간 감지</li>
              <li>• <strong>Human-in-the-Loop:</strong> 고위험 결정은 의사 최종 승인</li>
            </ul>
          </div>

          {/* 책임성 */}
          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              5. 책임성 (Accountability)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              AI 오진 시 법적 책임 소재 명확화
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-pink-700 dark:text-pink-400 mb-1">현재 쟁점</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AI 오진 책임: 의사 vs AI 개발사 vs 병원? → 국가별 판례 축적 중
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-pink-700 dark:text-pink-400 mb-1">권장 사항</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AI는 진단 보조 도구로만 사용, 최종 결정은 의사 (FDA 가이드라인)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 체크리스트 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          Medical AI 규제 준수 체크리스트
        </h2>

        <div className="space-y-6">
          {/* FDA 510(k) 체크리스트 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              ✅ FDA 510(k) 승인 준비 (미국 Class II AI 의료기기)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4">
              <pre className="text-sm text-gray-100">
                <code>{`FDA 510(k) Premarket Notification 체크리스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ 1. 의료기기 분류 확인
   - Device Classification Database 검색
   - Predicate Device (기존 승인 제품) 식별
   - Substantial Equivalence 입증 준비

□ 2. 임상 검증 데이터 준비
   - 최소 500명 환자 데이터 (다기관 권장)
   - Sensitivity/Specificity 95% 이상
   - 인종/성별 하위 그룹 분석 (편향 평가)
   - Reader Study (방사선 전문의 vs AI 비교)

□ 3. 소프트웨어 문서화
   - Software Development Lifecycle (SDLC) 명세
   - 학습 데이터셋 출처 및 라벨링 프로세스
   - 모델 아키텍처 및 하이퍼파라미터
   - Validation Set 성능 (독립 데이터)

□ 4. 위험 관리 계획
   - ISO 14971 Medical Device Risk Management
   - Failure Mode and Effects Analysis (FMEA)
   - Cybersecurity Risk Assessment
   - Post-Market Surveillance Plan

□ 5. Labeling & User Manual
   - Indications for Use (적응증)
   - Contraindications (금기사항)
   - Warnings & Precautions
   - User Training Requirements

□ 6. Quality System (QMS)
   - ISO 13485 준수 (의료기기 품질경영시스템)
   - Design Controls (설계 검증/확인)
   - Document Control (버전 관리)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
참고: 제출 비용 $13,000+ (SME), 승인 기간 3-12개월`}</code>
              </pre>
            </div>
          </div>

          {/* HIPAA 준수 체크리스트 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ HIPAA 준수 체크리스트 (PHI 처리 AI 시스템)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4">
              <pre className="text-sm text-gray-100">
                <code>{`HIPAA Compliance 체크리스트 (Medical AI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ 1. PHI 식별 및 제거
   - 18개 식별자 제거: 이름, 주소, 생년월일, MRN 등
   - Safe Harbor vs Expert Determination 방법 선택
   - De-identification 프로세스 문서화

□ 2. 기술적 보호조치 (Technical Safeguards)
   - PHI 저장 시 AES-256 암호화
   - 전송 시 TLS 1.2+ 사용
   - Access Control (Role-Based, MFA 필수)
   - Audit Logs (모든 PHI 접근 기록)

□ 3. 관리적 보호조치 (Administrative Safeguards)
   - Security Risk Assessment 연 1회 실시
   - 직원 HIPAA 교육 (채용 시 + 연 1회)
   - Business Associate Agreement (BAA) 체결
   - Incident Response Plan (유출 시 대응)

□ 4. 물리적 보호조치 (Physical Safeguards)
   - 서버실 출입 통제
   - 워크스테이션 보안 (화면 자동 잠금)
   - 디바이스 분실 시 원격 삭제

□ 5. Breach Notification Rule
   - 500명 이상 유출: HHS + 언론 공시 (60일 이내)
   - 500명 미만: HHS 연 1회 보고
   - 개인 통지: 60일 이내 서면/이메일

□ 6. 클라우드 사용 시 (AWS/Azure/GCP)
   - HIPAA Eligible Services 사용 확인
   - BAA 체결 필수 (AWS BAA, Azure HIPAA BAA)
   - US 리전 사용 권장 (데이터 주권)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
위반 시 벌금: Tier 1 ($100-$50k/건) ~ Tier 4 ($50k/건, 최대 $1.5M/년)`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 규제 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 Medical AI 규제 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. FDA Total Product Lifecycle (TPLC) 접근법
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              AI 소프트웨어 업데이트마다 재승인 불필요 (2024 가이드라인)
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Pre-Determined Change Control Plan:</strong> 사전 승인된 범위 내 업데이트 허용</li>
              <li>• <strong>Real-World Performance Monitoring:</strong> 배포 후 성능 추적 의무</li>
              <li>• <strong>예시:</strong> IDx-DR (당뇨망막병증 AI) - 알고리즘 개선 시 재심사 면제</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. EU AI Act 시행 (2024.08)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              고위험 의료 AI 의무 준수 사항
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>설명 가능성:</strong> SHAP, LIME 등 XAI 기법 필수 탑재</li>
              <li>• <strong>인간 감독:</strong> 고위험 결정은 의사 승인 필수</li>
              <li>• <strong>데이터 품질:</strong> 훈련 데이터 출처, 편향 평가 문서화</li>
              <li>• <strong>벌금:</strong> 최대 €35M 또는 글로벌 매출 7%</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. WHO AI for Health Ethics Framework (2024)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              개발도상국 의료 AI 거버넌스 가이드
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Digital Divide 해소:</strong> 저소득 국가 접근성 보장</li>
              <li>• <strong>Local Context:</strong> 지역별 질병 분포 반영 (말라리아 등)</li>
              <li>• <strong>Data Sovereignty:</strong> 국가별 데이터 주권 존중</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 규제 통계 */}
      <section className="bg-gradient-to-r from-red-600 to-orange-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          Medical AI 규제 현황 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">520+</p>
            <p className="text-sm opacity-90">FDA 승인 AI/ML 의료기기 (2024.09)</p>
            <p className="text-xs mt-2 opacity-75">출처: FDA Digital Health</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">€746M</p>
            <p className="text-sm opacity-90">Amazon GDPR 최대 벌금 (2021)</p>
            <p className="text-xs mt-2 opacity-75">출처: Luxembourg DPA</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$1.5M</p>
            <p className="text-sm opacity-90">HIPAA 최대 연간 벌금</p>
            <p className="text-xs mt-2 opacity-75">출처: HHS OCR</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">75%</p>
            <p className="text-sm opacity-90">FDA 승인 AI 중 영상의학 비율</p>
            <p className="text-xs mt-2 opacity-75">출처: FDA 2024 분석</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 규제 가이드라인 & 문서',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'FDA Software as a Medical Device (SaMD)',
                url: 'https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd',
                description: 'AI 의료기기 승인 절차 및 기준 (510(k), PMA)'
              },
              {
                title: 'EU AI Act Official Text',
                url: 'https://artificialintelligenceact.eu/',
                description: '2024.08 발효, 고위험 의료 AI 규제 (설명 가능성 필수)'
              },
              {
                title: 'HIPAA Privacy Rule',
                url: 'https://www.hhs.gov/hipaa/for-professionals/privacy/index.html',
                description: 'PHI 보호 규정, 18개 식별자, Safe Harbor 방법'
              },
              {
                title: 'WHO Guidance on AI for Health (2024)',
                url: 'https://www.who.int/publications/i/item/9789240029200',
                description: 'AI 의료 윤리 가이드라인 및 거버넌스 프레임워크'
              },
            ]
          },
          {
            title: '🔬 윤리 & 공정성 연구',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Racial Bias in Medical Imaging AI (Nature Medicine 2024)',
                url: 'https://www.nature.com/articles/s41591-024-02789-5',
                description: 'MIT 연구: 흑인 환자 피부암 AI 정확도 20%p 낮음'
              },
              {
                title: 'Explainable AI for Healthcare (JAMA 2024)',
                url: 'https://jamanetwork.com/journals/jama/fullarticle/2812345',
                description: 'SHAP 기반 XAI로 의사 신뢰도 35% 향상'
              },
              {
                title: 'Differential Privacy in Healthcare (NEJM 2024)',
                url: 'https://www.nejm.org/doi/full/10.1056/NEJMsr2315678',
                description: 'ε-차등 프라이버시 적용으로 재식별 위험 0.001% 미만'
              },
              {
                title: 'Accountability in Medical AI (Lancet Digital Health 2024)',
                url: 'https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00023-4',
                description: 'AI 오진 책임 소재: 국가별 판례 분석'
              },
            ]
          },
          {
            title: '🛠️ 규제 준수 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'IBM AI Fairness 360',
                url: 'https://aif360.mybluemix.net/',
                description: '편향 탐지 및 완화 알고리즘 70+ 개 (오픈소스)'
              },
              {
                title: 'Google What-If Tool',
                url: 'https://pair-code.github.io/what-if-tool/',
                description: 'ML 모델 공정성 시각화, Counterfactual 분석'
              },
              {
                title: 'TensorFlow Privacy',
                url: 'https://github.com/tensorflow/privacy',
                description: 'Differential Privacy 구현 라이브러리'
              },
              {
                title: 'SHAP (SHapley Additive exPlanations)',
                url: 'https://github.com/slundberg/shap',
                description: 'XAI 표준 도구, FDA 승인 심사 필수'
              },
              {
                title: 'PyHealth',
                url: 'https://github.com/sunlabuiuc/PyHealth',
                description: 'HIPAA 준수 의료 ML 파이프라인 구축 도구'
              },
            ]
          },
          {
            title: '📖 표준 & 인증',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'ISO 13485 (Medical Device QMS)',
                url: 'https://www.iso.org/standard/59752.html',
                description: '의료기기 품질경영시스템 국제 표준'
              },
              {
                title: 'ISO 14971 (Risk Management)',
                url: 'https://www.iso.org/standard/72704.html',
                description: '의료기기 위험 관리 표준 (FMEA)'
              },
              {
                title: 'IEC 62304 (Medical Device Software)',
                url: 'https://www.iec.ch/standards/iec-62304',
                description: '의료기기 소프트웨어 생명주기 프로세스'
              },
            ]
          },
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span>글로벌 규제: <strong>FDA SaMD (미국), EU AI Act (유럽), HIPAA/GDPR (개인정보)</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span>윤리 5원칙: <strong>공정성, 투명성, 프라이버시, 안전성, 책임성</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong>2024 트렌드</strong>: FDA TPLC (업데이트 간소화), EU AI Act 시행, XAI 의무화</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span>FDA 승인 520+ 개 (75% 영상의학), GDPR 최대 벌금 <strong>€746M</strong> (Amazon)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span>필수 도구: <strong>SHAP (XAI), IBM AI Fairness 360, TensorFlow Privacy</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
