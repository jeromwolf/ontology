'use client';

import { 
  Settings, Target, Users, Shield, Code
} from 'lucide-react';

export default function Chapter13() {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-blue-200 dark:border-blue-800">
        <h3 className="text-2xl font-bold text-blue-900 dark:text-blue-200 mb-6 flex items-center gap-3">
          <Target className="w-8 h-8" />
          7단계 구현 방법론
        </h3>
        <div className="grid lg:grid-cols-7 gap-4">
          {[
            { step: "1단계", title: "현황분석", desc: "As-Is 분석", icon: "📊", color: "blue" },
            { step: "2단계", title: "전략수립", desc: "To-Be 비전", icon: "🎯", color: "green" },
            { step: "3단계", title: "시스템설계", desc: "아키텍처", icon: "🏗️", color: "purple" },
            { step: "4단계", title: "구축실행", desc: "개발/구현", icon: "⚙️", color: "orange" },
            { step: "5단계", title: "테스트", desc: "검증/검수", icon: "🧪", color: "red" },
            { step: "6단계", title: "운영이관", desc: "Go-Live", icon: "🚀", color: "indigo" },
            { step: "7단계", title: "지속개선", desc: "최적화", icon: "📈", color: "teal" }
          ].map((phase, idx) => (
            <div key={idx} className="text-center">
              <div className={`w-16 h-16 bg-${phase.color}-500 rounded-full flex items-center justify-center mx-auto mb-3`}>
                <span className="text-2xl">{phase.icon}</span>
              </div>
              <h4 className="font-bold text-gray-900 dark:text-white text-sm mb-1">{phase.step}</h4>
              <h5 className="font-semibold text-gray-800 dark:text-gray-200 text-sm mb-1">{phase.title}</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400">{phase.desc}</p>
              {idx < 6 && (
                <div className="hidden lg:block absolute top-8 right-0 transform translate-x-1/2">
                  <span className="text-gray-300 text-lg">→</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Target className="w-6 h-6 text-slate-600" />
            애자일 vs 워터폴 방법론
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">애자일 방법론</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• 2-4주 스프린트 단위 개발</li>
                <li>• 빠른 프로토타입과 피드백</li>
                <li>• 변화에 유연한 대응</li>
                <li>• 사용자 중심의 개발</li>
              </ul>
              <div className="mt-2 text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-300 p-2 rounded">
                <strong>적합한 경우:</strong> 요구사항 변동이 많은 AI/데이터 분석 프로젝트
              </div>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">워터폴 방법론</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• 순차적 단계별 진행</li>
                <li>• 명확한 문서화</li>
                <li>• 안정적이고 예측 가능</li>
                <li>• 품질 관리 용이</li>
              </ul>
              <div className="mt-2 text-xs bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-300 p-2 rounded">
                <strong>적합한 경우:</strong> 인프라 구축, 네트워크 설치, 하드웨어 도입
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Shield className="w-6 h-6 text-slate-600" />
            위험 관리 매트릭스
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
                <h4 className="font-semibold text-red-800 dark:text-red-300 text-sm">높음/높음</h4>
                <p className="text-xs text-red-600 dark:text-red-400 mt-1">즉시 대응</p>
                <ul className="text-xs text-red-500 dark:text-red-500 mt-2">
                  <li>• 핵심 시스템 장애</li>
                  <li>• 보안 취약점</li>
                </ul>
              </div>
              <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-800">
                <h4 className="font-semibold text-orange-800 dark:text-orange-300 text-sm">높음/낮음</h4>
                <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">예방 조치</p>
                <ul className="text-xs text-orange-500 dark:text-orange-500 mt-2">
                  <li>• 기술 역량 부족</li>
                  <li>• 일정 지연</li>
                </ul>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-800">
                <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 text-sm">낮음/높음</h4>
                <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">모니터링</p>
                <ul className="text-xs text-yellow-500 dark:text-yellow-500 mt-2">
                  <li>• 예산 초과</li>
                  <li>• 성능 이슈</li>
                </ul>
              </div>
              <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800">
                <h4 className="font-semibold text-green-800 dark:text-green-300 text-sm">낮음/낮음</h4>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">수용</p>
                <ul className="text-xs text-green-500 dark:text-green-500 mt-2">
                  <li>• 경미한 기능 오류</li>
                  <li>• UI/UX 개선</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Users className="w-8 h-8" />
          PoC (Proof of Concept) 기획과 실행
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg border border-green-200 dark:border-green-600">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">1단계: 목표 설정</h4>
            <ul className="text-sm text-green-700 dark:text-green-300 space-y-2">
              <li>• 명확한 성공 기준 정의</li>
              <li>• 측정 가능한 KPI 설정</li>
              <li>• 예상 ROI 계산</li>
              <li>• 실패 조건 명시</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg border border-green-200 dark:border-green-600">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">2단계: 범위 한정</h4>
            <ul className="text-sm text-green-700 dark:text-green-300 space-y-2">
              <li>• 핵심 기능만 선별</li>
              <li>• 제한된 데이터셋 사용</li>
              <li>• 6-12주 단기 실행</li>
              <li>• 소규모 팀 구성</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg border border-green-200 dark:border-green-600">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">3단계: 결과 평가</h4>
            <ul className="text-sm text-green-700 dark:text-green-300 space-y-2">
              <li>• 객관적 성과 측정</li>
              <li>• 사용자 피드백 수집</li>
              <li>• 확장 가능성 검토</li>
              <li>• Go/No-Go 의사결정</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Code className="w-8 h-8 text-amber-600" />
          성공 요인 체크리스트
        </h3>
        <div className="grid md:grid-cols-4 gap-6">
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">경영진 의지</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>□ CEO 직접 참여</li>
              <li>□ 충분한 예산 확보</li>
              <li>□ 장기적 비전 공유</li>
              <li>□ 조직 변화 의지</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">기술적 준비</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>□ IT 인프라 현황 파악</li>
              <li>□ 데이터 품질 검증</li>
              <li>□ 보안 체계 수립</li>
              <li>□ 기술 역량 보유</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">조직 역량</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>□ 전담 조직 구성</li>
              <li>□ 교육 프로그램 운영</li>
              <li>□ 변화관리 계획</li>
              <li>□ 외부 파트너 확보</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">운영 체계</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>□ 거버넌스 체계</li>
              <li>□ 성과 측정 시스템</li>
              <li>□ 지속 개선 프로세스</li>
              <li>□ 문제 해결 체계</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}