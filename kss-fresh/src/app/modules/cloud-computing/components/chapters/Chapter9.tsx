'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 9: 비용 최적화
export default function Chapter9() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
          클라우드 비용 최적화
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          클라우드의 유연성과 편리함을 유지하면서 불필요한 비용을 줄이는 전략을 학습합니다. 
          실제 비용 사례 분석부터 자동화된 최적화 방법까지 다룹니다.
        </p>
      </section>

      {/* 1. 클라우드 비용 구조 이해 */}
      <section className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 border-l-4 border-emerald-500">
        <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-4 text-2xl">
          1. 클라우드 비용 구조 이해
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">주요 비용 요소</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-emerald-700 dark:text-emerald-300 block mb-2">Compute (컴퓨팅)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• VM/인스턴스 실행 시간</li>
                  <li>• CPU/메모리 스펙</li>
                  <li>• Lambda 실행 시간 & 요청 수</li>
                  <li>• <strong>최적화 포인트</strong>: 유휴 리소스, 과도한 스펙</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-emerald-700 dark:text-emerald-300 block mb-2">Storage (스토리지)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 저장 용량 (GB당 과금)</li>
                  <li>• 데이터 전송 (아웃바운드)</li>
                  <li>• API 요청 수</li>
                  <li>• <strong>최적화 포인트</strong>: 오래된 데이터, 중복 파일</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-emerald-700 dark:text-emerald-300 block mb-2">Network (네트워크)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 데이터 전송 (특히 인터넷 아웃바운드)</li>
                  <li>• 리전 간 전송</li>
                  <li>• VPN, Direct Connect</li>
                  <li>• <strong>최적화 포인트</strong>: 불필요한 전송, CDN 활용</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-emerald-700 dark:text-emerald-300 block mb-2">Database (데이터베이스)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 인스턴스 타입 & 크기</li>
                  <li>• IOPS, 스토리지</li>
                  <li>• 백업, 스냅샷</li>
                  <li>• <strong>최적화 포인트</strong>: 과도한 IOPS, 오래된 백업</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-emerald-100 dark:bg-emerald-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-emerald-900 dark:text-emerald-100 mb-2">비용 급증 주범 (Top 3)</h4>
            <ul className="space-y-1 text-sm text-emerald-800 dark:text-emerald-200">
              <li>1️⃣ <strong>유휴 리소스</strong>: 사용하지 않는 EC2, RDS 인스턴스 (전체 비용의 30~40%)</li>
              <li>2️⃣ <strong>과도한 스펙</strong>: 실제 필요보다 큰 인스턴스 타입</li>
              <li>3️⃣ <strong>데이터 전송 비용</strong>: 인터넷 아웃바운드 ($0.09/GB) 간과</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 예약 인스턴스 & Savings Plans */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          2. 예약 인스턴스 & Savings Plans
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">AWS 비용 절감 옵션</h4>
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Reserved Instances (RI)</strong>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">특징</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>• 1년/3년 약정</li>
                      <li>• <strong>최대 72% 할인</strong></li>
                      <li>• 특정 인스턴스 타입 고정</li>
                      <li>• 결제 옵션: All Upfront, Partial, No Upfront</li>
                    </ul>
                  </div>
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">적합한 경우</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>✓ 안정적인 워크로드</li>
                      <li>✓ 24/7 실행 인스턴스</li>
                      <li>✓ 예측 가능한 사용량</li>
                      <li>✗ 변동성 큰 워크로드는 비효율</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Savings Plans (더 유연함)</strong>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">특징</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>• 시간당 사용량($) 약정</li>
                      <li>• <strong>최대 72% 할인</strong></li>
                      <li>• 인스턴스 타입 변경 가능</li>
                      <li>• Compute / EC2 / SageMaker SP</li>
                    </ul>
                  </div>
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">권장 사용</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>✓ 유연성 필요</li>
                      <li>✓ Lambda, Fargate 포함</li>
                      <li>✓ 미래 워크로드 예측 어려움</li>
                      <li>📊 AWS Cost Explorer 추천 활용</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Spot Instances (최대 90% 할인)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  AWS의 유휴 용량을 경매 방식으로 사용 - 언제든 중단될 수 있음
                </p>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>사용 사례</strong>: 배치 작업, 빅데이터 분석, CI/CD, 렌더링</li>
                  <li>✗ <strong>부적합</strong>: 데이터베이스, 웹 서버 (중단 시 서비스 영향)</li>
                  <li>💡 <strong>팁</strong>: Spot Fleet으로 여러 인스턴스 타입 조합</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Azure & GCP 할인 옵션</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-2">Azure Reserved VM Instances</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 1년/3년 약정 (최대 72% 할인)</li>
                  <li>• Spot VMs (최대 90% 할인)</li>
                  <li>• Azure Hybrid Benefit (기존 Windows 라이선스 활용)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-2">GCP Committed Use Discounts</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 1년/3년 약정 (최대 57% 할인)</li>
                  <li>• Sustained Use Discounts (자동 적용 25%)</li>
                  <li>• Preemptible VMs (최대 80% 할인)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 자동 스케일링 & 스케줄링 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          3. 자동 스케일링 & 스케줄링
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Auto Scaling (자동 확장)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              수요에 따라 리소스를 자동으로 증감하여 <strong>비용 절감 + 성능 유지</strong>
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">AWS Auto Scaling</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Target Tracking: CPU 70% 유지</li>
                  <li>• Step Scaling: 단계별 증가</li>
                  <li>• Scheduled Scaling: 시간대별</li>
                  <li>• Predictive Scaling: ML 기반 예측</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">실전 예시</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>📈 <strong>피크 타임</strong>: 평일 9-18시 10대</li>
                  <li>📉 <strong>야간</strong>: 18-09시 2대</li>
                  <li>💰 <strong>절감</strong>: 월 $3,000 → $1,200 (60%)</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Instance Scheduler (개발/테스트 환경)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>사용하지 않는 시간에 자동 종료</strong> - 개발/테스트 환경에서 특히 효과적
              </p>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-purple-700 dark:text-purple-300 block mb-2">스케줄 예시</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>평일</strong>: 09:00 시작 → 19:00 중지</li>
                    <li>• <strong>주말</strong>: 완전 중지</li>
                    <li>• <strong>결과</strong>: 주 168시간 → 50시간 (70% 절감)</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-purple-700 dark:text-purple-300 block mb-2">구현 방법</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• AWS Instance Scheduler (Lambda 기반)</li>
                    <li>• Azure Automation</li>
                    <li>• GCP Cloud Scheduler</li>
                    <li>• 태그 기반 정책 (예: Environment=Dev)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 스토리지 최적화 */}
      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
        <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4 text-2xl">
          4. 스토리지 최적화
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Lifecycle 정책 (자동 계층화)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                데이터 액세스 패턴에 따라 자동으로 저렴한 스토리지 클래스로 이동
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="bg-orange-100 dark:bg-orange-900/30 px-3 py-1 rounded text-sm font-semibold">0일</div>
                  <div className="flex-1 text-sm text-gray-700 dark:text-gray-300">
                    S3 Standard ($0.023/GB) - 자주 접근
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="bg-orange-100 dark:bg-orange-900/30 px-3 py-1 rounded text-sm font-semibold">30일</div>
                  <div className="flex-1 text-sm text-gray-700 dark:text-gray-300">
                    S3 Standard-IA ($0.0125/GB) - 가끔 접근 (46% 절감)
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="bg-orange-100 dark:bg-orange-900/30 px-3 py-1 rounded text-sm font-semibold">90일</div>
                  <div className="flex-1 text-sm text-gray-700 dark:text-gray-300">
                    S3 Glacier Instant Retrieval ($0.004/GB) - 분기별 접근 (83% 절감)
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="bg-orange-100 dark:bg-orange-900/30 px-3 py-1 rounded text-sm font-semibold">180일</div>
                  <div className="flex-1 text-sm text-gray-700 dark:text-gray-300">
                    S3 Glacier Deep Archive ($0.00099/GB) - 아카이브 (96% 절감)
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">데이터 정리 전략</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">오래된 스냅샷 삭제</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• EBS 스냅샷: 7일 이상 경과 시 자동 삭제</li>
                  <li>• RDS 백업: 최근 3개만 유지</li>
                  <li>• AMI: 사용하지 않는 이미지 정리</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">중복 데이터 제거</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• S3 Intelligent-Tiering (자동 최적화)</li>
                  <li>• 중복 파일 검사 스크립트</li>
                  <li>• 압축 (gzip, brotli)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-900 dark:text-orange-100 mb-2">실전 사례</h4>
            <p className="text-sm text-orange-800 dark:text-orange-200">
              📊 <strong>로그 파일 1TB</strong> (매월 증가) → Lifecycle 정책 적용<br/>
              💰 비용: $23/월 → $1/월 (96% 절감) - 30일 후 Glacier Deep Archive 이동
            </p>
          </div>
        </div>
      </section>

      {/* 5. 모니터링 & 비용 가시성 */}
      <section className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
        <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4 text-2xl">
          5. 모니터링 & 비용 가시성
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">비용 모니터링 도구</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300 block mb-2">AWS Cost Explorer</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 시각화된 비용 분석 (일/월/분기별)</li>
                  <li>• 서비스별, 태그별 그룹화</li>
                  <li>• Savings Plans/RI 추천</li>
                  <li>• 예측 (ML 기반 미래 비용 예측)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300 block mb-2">AWS Budgets (예산 알림)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 월 예산 설정 (예: $5,000)</li>
                  <li>• 임계값 알림 (85%, 100%, 110%)</li>
                  <li>• 자동 조치: 예산 초과 시 인스턴스 중지</li>
                  <li>• SNS, 이메일, Slack 알림</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300 block mb-2">Cost Anomaly Detection (이상 감지)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• ML 기반 비정상 비용 패턴 자동 감지</li>
                  <li>• 예시: EC2 비용이 갑자기 2배 증가 → 즉시 알림</li>
                  <li>• 루트 원인 분석 (어떤 서비스/리소스인지)</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">태깅 전략 (비용 할당)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>체계적인 태그</strong>로 팀별/프로젝트별 비용 추적
              </p>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-red-700 dark:text-red-300 block mb-2">필수 태그</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>Environment</strong>: Production, Dev, Test</li>
                    <li>• <strong>Project</strong>: ProjectA, ProjectB</li>
                    <li>• <strong>CostCenter</strong>: Engineering, Marketing</li>
                    <li>• <strong>Owner</strong>: team-backend, team-frontend</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-red-700 dark:text-red-300 block mb-2">효과</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 팀별 비용 명확히 분리</li>
                    <li>✓ 개발 환경 vs 프로덕션 비용 비교</li>
                    <li>✓ 책임 소재 명확화</li>
                    <li>✓ 비용 최적화 우선순위 설정</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. 서드파티 도구 */}
      <section className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6 border-l-4 border-teal-500">
        <h3 className="font-semibold text-teal-800 dark:text-teal-200 mb-4 text-2xl">
          6. 서드파티 비용 최적화 도구
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">CloudHealth (VMware)</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 멀티 클라우드 비용 통합 관리</li>
              <li>• 자동 RI 추천 및 구매</li>
              <li>• 거버넌스 정책 강제</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">Spot.io</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• Spot Instance 자동 관리</li>
              <li>• 중단 예측 및 자동 전환</li>
              <li>• 최대 90% 비용 절감</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">Cloudability</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• AI 기반 비용 최적화 추천</li>
              <li>• 팀별 차지백 (Chargeback)</li>
              <li>• 상세한 비용 분석 리포트</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">Kubecost (Kubernetes 전용)</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• Pod/Namespace별 비용 추적</li>
              <li>• 실시간 리소스 효율성 분석</li>
              <li>• Right-sizing 추천</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-emerald-100 to-green-100 dark:from-emerald-900/30 dark:to-green-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-emerald-900 dark:text-emerald-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-emerald-800 dark:text-emerald-200">핵심 전략</h4>
            <ul className="space-y-1 text-emerald-700 dark:text-emerald-300">
              <li>✓ 예약 인스턴스/Savings Plans (최대 72% 할인)</li>
              <li>✓ Spot Instances (최대 90% 할인)</li>
              <li>✓ Auto Scaling & Scheduler (유휴 리소스 제거)</li>
              <li>✓ Storage Lifecycle (자동 계층화)</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-emerald-800 dark:text-emerald-200">모니터링</h4>
            <ul className="space-y-1 text-emerald-700 dark:text-emerald-300">
              <li>✓ Cost Explorer: 비용 분석 및 예측</li>
              <li>✓ Budgets: 예산 알림 및 자동 조치</li>
              <li>✓ 태깅: 팀/프로젝트별 비용 추적</li>
              <li>✓ Anomaly Detection: 이상 비용 즉시 감지</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 비용 최적화 가이드',
            icon: 'web',
            color: 'border-emerald-500',
            items: [
              {
                title: 'AWS Cost Optimization',
                link: 'https://aws.amazon.com/pricing/cost-optimization/',
                description: 'AWS 비용 최적화 공식 가이드 및 모범 사례'
              },
              {
                title: 'Azure Cost Management',
                link: 'https://learn.microsoft.com/en-us/azure/cost-management-billing/',
                description: 'Azure 비용 관리 및 청구 문서'
              },
              {
                title: 'GCP Cost Optimization',
                link: 'https://cloud.google.com/architecture/cost-optimization-principles',
                description: 'Google Cloud 비용 최적화 원칙'
              }
            ]
          },
          {
            title: '📖 백서 & 사례 연구',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Well-Architected - Cost Optimization Pillar',
                link: 'https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/welcome.html',
                description: 'AWS 비용 최적화 프레임워크'
              },
              {
                title: 'FinOps Foundation',
                link: 'https://www.finops.org/introduction/what-is-finops/',
                description: '클라우드 재무 관리 방법론'
              }
            ]
          },
          {
            title: '🛠️ 비용 최적화 도구',
            icon: 'web',
            color: 'border-purple-500',
            items: [
              {
                title: 'AWS Cost Explorer',
                link: 'https://aws.amazon.com/aws-cost-management/aws-cost-explorer/',
                description: '비용 분석 및 예측 도구'
              },
              {
                title: 'Spot.io',
                link: 'https://spot.io/',
                description: 'Spot Instance 자동 관리 플랫폼'
              },
              {
                title: 'Kubecost',
                link: 'https://www.kubecost.com/',
                description: 'Kubernetes 비용 모니터링'
              },
              {
                title: 'Infracost',
                link: 'https://www.infracost.io/',
                description: 'Terraform 비용 예측 (IaC 단계)'
              }
            ]
          },
          {
            title: '🎓 학습 리소스',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'AWS Cloud Economics',
                link: 'https://aws.amazon.com/economics/',
                description: 'TCO 계산기 및 ROI 분석'
              },
              {
                title: 'The FinOps Book',
                link: 'https://www.finops.org/resources/finops-book/',
                description: '클라우드 비용 관리 필독서'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
