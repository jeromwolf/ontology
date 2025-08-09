'use client'

import { 
  Shield, Lock, AlertTriangle, Network, Eye
} from 'lucide-react'

export default function Chapter12() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Lock className="w-6 h-6 text-slate-600" />
            IT vs OT 보안 차이점
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">IT (Information Technology) 보안</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• <strong>우선순위:</strong> 기밀성 → 무결성 → 가용성</li>
                <li>• <strong>목표:</strong> 정보 보호, 데이터 유출 방지</li>
                <li>• <strong>다운타임:</strong> 허용 가능 (패치, 재시작)</li>
                <li>• <strong>접근제어:</strong> 사용자 인증, 권한 관리</li>
                <li>• <strong>업데이트:</strong> 정기적 보안 패치</li>
              </ul>
            </div>
            
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">OT (Operational Technology) 보안</h4>
              <ul className="text-sm text-red-700 dark:text-red-400 space-y-1">
                <li>• <strong>우선순위:</strong> 가용성 → 무결성 → 기밀성</li>
                <li>• <strong>목표:</strong> 생산 연속성, 안전 보장</li>
                <li>• <strong>다운타임:</strong> 절대 불허 (24/7 운영)</li>
                <li>• <strong>접근제어:</strong> 물리적 격리, 네트워크 분할</li>
                <li>• <strong>업데이트:</strong> 계획된 정비 시간에만</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-slate-600" />
            주요 사이버 공격 사례
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">Stuxnet (2010)</h4>
              <p className="text-sm text-red-700 dark:text-red-400 mb-2">이란 핵시설 원심분리기 파괴</p>
              <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                <li>• PLC 제어 시스템 해킹</li>
                <li>• 물리적 손상 유발</li>
                <li>• 최초의 OT 타겟 사이버무기</li>
              </ul>
            </div>
            
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border rounded">
              <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">TRITON/TRISIS (2017)</h4>
              <p className="text-sm text-orange-700 dark:text-orange-400 mb-2">사우디 석유화학 플랜트 안전시스템 공격</p>
              <ul className="text-xs text-orange-600 dark:text-orange-400 space-y-1">
                <li>• SIS(Safety Instrumented System) 침투</li>
                <li>• 안전 시스템 무력화 시도</li>
                <li>• 대형 사고 위험 증대</li>
              </ul>
            </div>

            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border rounded">
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">WannaCry (2017)</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-400 mb-2">전 세계 제조업체 동시 피해</p>
              <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-1">
                <li>• 랜섬웨어 대량 감염</li>
                <li>• 자동차, 반도체 공장 중단</li>
                <li>• 패치 지연의 심각성 인식</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-purple-200 dark:border-purple-800">
        <h3 className="text-2xl font-bold text-purple-900 dark:text-purple-200 mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8" />
          IEC 62443 보안 표준 체계
        </h3>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">IEC 62443-1</h4>
            <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">일반 개념</p>
            <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
              <li>• 용어 정의</li>
              <li>• 개념 모델</li>
              <li>• 메트릭스</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">IEC 62443-2</h4>
            <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">정책 및 절차</p>
            <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
              <li>• 보안 프로그램</li>
              <li>• 위험 관리</li>
              <li>• 패치 관리</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">IEC 62443-3</h4>
            <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">시스템 요구사항</p>
            <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
              <li>• 보안 기술</li>
              <li>• 시스템 설계</li>
              <li>• 위험 평가</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">IEC 62443-4</h4>
            <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">컴포넌트 요구사항</p>
            <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
              <li>• 제품 개발</li>
              <li>• 기술 요구사항</li>
              <li>• 인증 기준</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-8 h-8 text-amber-600" />
          네트워크 분할 및 보안 존
        </h3>
        <div className="space-y-6">
          <div className="grid md:grid-cols-5 gap-4">
            {[
              { zone: "Enterprise", level: "SL 1", desc: "비즈니스 네트워크", color: "blue" },
              { zone: "DMZ", level: "SL 2", desc: "완충 구역", color: "yellow" },
              { zone: "Manufacturing", level: "SL 3", desc: "제조 운영", color: "green" },
              { zone: "Control", level: "SL 4", desc: "제어 시스템", color: "orange" },
              { zone: "Safety", level: "SL 4", desc: "안전 시스템", color: "red" }
            ].map((zone, idx) => (
              <div key={idx} className={`p-4 bg-${zone.color}-50 dark:bg-${zone.color}-900/20 border border-${zone.color}-200 dark:border-${zone.color}-800 rounded-lg text-center`}>
                <h4 className={`font-bold text-${zone.color}-800 dark:text-${zone.color}-300 mb-2`}>{zone.zone}</h4>
                <div className={`text-xs bg-${zone.color}-100 dark:bg-${zone.color}-800 text-${zone.color}-700 dark:text-${zone.color}-300 px-2 py-1 rounded mb-2`}>
                  {zone.level}
                </div>
                <p className={`text-xs text-${zone.color}-600 dark:text-${zone.color}-400`}>{zone.desc}</p>
              </div>
            ))}
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">보안 게이트웨이</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 방화벽 (Firewall)</li>
                <li>• 침입 탐지 시스템 (IDS)</li>
                <li>• 침입 방지 시스템 (IPS)</li>
                <li>• 프로토콜 검증</li>
              </ul>
            </div>
            
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">접근 제어</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 다중 인증 (MFA)</li>
                <li>• 역할 기반 접근 (RBAC)</li>
                <li>• 최소 권한 원칙</li>
                <li>• 세션 관리</li>
              </ul>
            </div>
            
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">모니터링</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• SIEM (보안정보관리)</li>
                <li>• 로그 분석</li>
                <li>• 이상 행위 탐지</li>
                <li>• 실시간 알림</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Eye className="w-8 h-8" />
          보안 운영 체계 (SOC)
        </h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-green-800 dark:text-green-200">24/7 모니터링 체계</h4>
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">Level 1 - 기본 모니터링</h5>
                <ul className="text-sm text-green-600 dark:text-green-400 space-y-1">
                  <li>• 이벤트 수집 및 분류</li>
                  <li>• 자동화된 알림</li>
                  <li>• 기본 대응 절차</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">Level 2 - 전문가 분석</h5>
                <ul className="text-sm text-green-600 dark:text-green-400 space-y-1">
                  <li>• 상세 위협 분석</li>
                  <li>• 포렌식 조사</li>
                  <li>• 대응 전략 수립</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">Level 3 - 고급 헌팅</h5>
                <ul className="text-sm text-green-600 dark:text-green-400 space-y-1">
                  <li>• 능동적 위협 탐지</li>
                  <li>• APT 분석</li>
                  <li>• 정책 개선</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-green-800 dark:text-green-200">사고 대응 절차</h4>
            <div className="space-y-3">
              {[
                { phase: "1. 준비", desc: "대응팀 구성, 절차서 준비" },
                { phase: "2. 탐지", desc: "보안 이벤트 발견, 분류" },
                { phase: "3. 분석", desc: "영향도 평가, 원인 분석" },
                { phase: "4. 격리", desc: "확산 차단, 시스템 격리" },
                { phase: "5. 제거", desc: "악성코드 제거, 취약점 패치" },
                { phase: "6. 복구", desc: "시스템 복원, 정상화 확인" },
                { phase: "7. 학습", desc: "사후 분석, 개선점 도출" }
              ].map((step, idx) => (
                <div key={idx} className="flex items-center gap-4 p-3 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                  <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                    {idx + 1}
                  </div>
                  <div>
                    <h5 className="font-semibold text-green-700 dark:text-green-300 text-sm">{step.phase}</h5>
                    <p className="text-xs text-green-600 dark:text-green-400">{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}