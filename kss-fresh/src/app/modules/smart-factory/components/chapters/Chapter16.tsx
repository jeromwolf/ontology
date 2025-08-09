'use client'

import { 
  Rocket, Zap, Globe, Leaf, Brain, Eye, TrendingUp, Settings
} from 'lucide-react'

export default function Chapter16() {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-blue-200 dark:border-blue-800">
        <h3 className="text-2xl font-bold text-blue-900 dark:text-blue-200 mb-6 flex items-center gap-3">
          <Zap className="w-8 h-8" />
          5G 기반 초연결 스마트팩토리
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-200 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-4">네트워크 슬라이싱</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-2">
              <li>• <strong>URLLC:</strong> 초저지연 제어 (1ms 이하)</li>
              <li>• <strong>eMBB:</strong> 대용량 데이터 전송</li>
              <li>• <strong>mMTC:</strong> 대규모 IoT 연결</li>
              <li>• 각 용도별 전용 네트워크 할당</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-200 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-4">엣지 컴퓨팅 진화</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-2">
              <li>• MEC (Multi-access Edge Computing)</li>
              <li>• 현장 즉시 데이터 처리</li>
              <li>• AI 추론의 실시간 실행</li>
              <li>• 클라우드 의존성 최소화</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-200 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-4">프라이빗 5G</h4>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-2">
              <li>• 기업 전용 5G 네트워크</li>
              <li>• 완전한 보안과 제어권</li>
              <li>• 맞춤형 서비스 품질</li>
              <li>• 캠퍼스 네트워크 구축</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Eye className="w-6 h-6 text-slate-600" />
            메타버스 팩토리의 등장
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-400 rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">VR/AR 기반 원격 작업</h4>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• 홀로렌즈를 활용한 작업 가이드</li>
                <li>• 원격 전문가 지원 시스템</li>
                <li>• 3D 공간에서의 협업</li>
                <li>• 실시간 작업 상황 공유</li>
              </ul>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">가상 교육 환경</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• 위험 상황 시뮬레이션 훈련</li>
                <li>• 고가 장비 가상 실습</li>
                <li>• 개인 맞춤형 학습 경로</li>
                <li>• 글로벌 팀 동시 교육</li>
              </ul>
            </div>

            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-400 rounded">
              <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">디지털 협업</h4>
              <ul className="text-sm text-orange-700 dark:text-orange-400 space-y-1">
                <li>• 가상 회의실에서 3D 데이터 검토</li>
                <li>• 실시간 설계 변경 공유</li>
                <li>• 다국적 팀 동시 작업</li>
                <li>• 시공간 제약 없는 협업</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Leaf className="w-6 h-6 text-slate-600" />
            탄소중립과 그린 팩토리
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">에너지 효율화</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• AI 기반 에너지 최적화</li>
                <li>• 스마트 그리드 연동</li>
                <li>• 재생에너지 통합 관리</li>
                <li>• 에너지 저장 시스템(ESS)</li>
              </ul>
            </div>
            
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">순환경제</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• 폐기물 제로 생산</li>
                <li>• 부산물 재활용 최적화</li>
                <li>• 제품 생명주기 연장</li>
                <li>• 지속가능한 원료 사용</li>
              </ul>
            </div>

            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">ESG 경영</h4>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• 탄소발자국 실시간 추적</li>
                <li>• 공급망 지속가능성 관리</li>
                <li>• 환경 영향 평가 자동화</li>
                <li>• ESG 지표 실시간 모니터링</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-8 rounded-xl border border-purple-200 dark:border-purple-800">
        <h3 className="text-2xl font-bold text-purple-900 dark:text-purple-200 mb-6 flex items-center gap-3">
          <Brain className="w-8 h-8" />
          AI의 진화와 생성형 AI
        </h3>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">GPT 기반 지능형 어시스턴트</h4>
            <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
              <li>• 자연어 기반 설비 제어</li>
              <li>• 자동 보고서 생성</li>
              <li>• 기술 문서 자동 작성</li>
              <li>• 지능형 장애 진단</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">이미지 생성 AI</h4>
            <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
              <li>• 제품 디자인 자동 생성</li>
              <li>• 3D 모델 자동 생성</li>
              <li>• 가상 프로토타입 제작</li>
              <li>• 패키징 디자인 최적화</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">코드 생성 AI</h4>
            <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
              <li>• PLC 프로그램 자동 생성</li>
              <li>• HMI 화면 자동 구성</li>
              <li>• 테스트 케이스 자동 작성</li>
              <li>• 버그 자동 수정</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">멀티모달 AI</h4>
            <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
              <li>• 음성+이미지 통합 인식</li>
              <li>• 다감각 품질 검사</li>
              <li>• 상황 인지형 제어</li>
              <li>• 복합 데이터 분석</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/20 dark:to-slate-900/20 p-8 rounded-xl border border-gray-200 dark:border-gray-800">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-200 mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8" />
          완전 자율 팩토리 (Lights-Out Factory)
        </h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200">기술적 요소</h4>
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-gray-800/30 rounded-lg border border-gray-200 dark:border-gray-600">
                <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">완전 자율 로봇</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">AGI 기반 상황 판단과 의사결정</p>
              </div>
              
              <div className="p-4 bg-white dark:bg-gray-800/30 rounded-lg border border-gray-200 dark:border-gray-600">
                <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">자가 치유 시스템</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">장애 자동 감지, 진단, 복구</p>
              </div>
              
              <div className="p-4 bg-white dark:bg-gray-800/30 rounded-lg border border-gray-200 dark:border-gray-600">
                <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">적응형 생산라인</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">수요 변화에 실시간 재구성</p>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200">현실적 한계</h4>
            <div className="space-y-3">
              <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-700">
                <h5 className="font-semibold text-yellow-700 dark:text-yellow-300 text-sm mb-1">예외 상황 대응</h5>
                <p className="text-xs text-yellow-600 dark:text-yellow-400">예상치 못한 상황 처리의 한계</p>
              </div>
              
              <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-700">
                <h5 className="font-semibold text-red-700 dark:text-red-300 text-sm mb-1">안전성 우려</h5>
                <p className="text-xs text-red-600 dark:text-red-400">무인 운영 시 안전 보장 문제</p>
              </div>
              
              <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-700">
                <h5 className="font-semibold text-orange-700 dark:text-orange-300 text-sm mb-1">높은 투자 비용</h5>
                <p className="text-xs text-orange-600 dark:text-orange-400">완전 자동화 구축 비용</p>
              </div>
              
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
                <h5 className="font-semibold text-blue-700 dark:text-blue-300 text-sm mb-1">고용 문제</h5>
                <p className="text-xs text-blue-600 dark:text-blue-400">일자리 대체에 따른 사회적 이슈</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-amber-600" />
          2030년 스마트팩토리 비전
        </h3>
        <div className="space-y-6">
          <div className="grid md:grid-cols-5 gap-4">
            {[
              { year: "2024", desc: "5G 상용화", progress: 100, color: "blue" },
              { year: "2025", desc: "메타버스 도입", progress: 75, color: "green" },
              { year: "2027", desc: "AI 통합 완료", progress: 50, color: "purple" },
              { year: "2029", desc: "탄소중립 달성", progress: 25, color: "emerald" },
              { year: "2030", desc: "완전 자율화", progress: 10, color: "orange" }
            ].map((milestone, idx) => (
              <div key={idx} className="text-center">
                <div className={`w-20 h-20 bg-${milestone.color}-100 dark:bg-${milestone.color}-900/30 rounded-full flex items-center justify-center mx-auto mb-3 border-4 border-${milestone.color}-500`}>
                  <span className={`text-${milestone.color}-700 dark:text-${milestone.color}-300 font-bold text-sm`}>{milestone.year}</span>
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white text-sm mb-2">{milestone.desc}</h4>
                <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2`}>
                  <div className={`bg-${milestone.color}-500 h-2 rounded-full`} style={{width: `${milestone.progress}%`}}></div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{milestone.progress}%</p>
              </div>
            ))}
          </div>
          
          <div className="mt-8 p-6 bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800">
            <h4 className="text-xl font-bold text-indigo-900 dark:text-indigo-200 mb-4 flex items-center gap-3">
              <Settings className="w-6 h-6" />
              실습: 2030년 스마트팩토리 비전 수립
            </h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-white dark:bg-indigo-800/30 rounded-lg border border-indigo-200 dark:border-indigo-600">
                <h5 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">1단계: 현황 분석</h5>
                <ul className="text-sm text-indigo-600 dark:text-indigo-400 space-y-1">
                  <li>• 현재 디지털 성숙도 평가</li>
                  <li>• 경쟁사 벤치마킹</li>
                  <li>• 기술 트렌드 분석</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-indigo-800/30 rounded-lg border border-indigo-200 dark:border-indigo-600">
                <h5 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">2단계: 미래 시나리오</h5>
                <ul className="text-sm text-indigo-600 dark:text-indigo-400 space-y-1">
                  <li>• 기술 융합 시나리오</li>
                  <li>• 시장 변화 예측</li>
                  <li>• 리스크 요인 식별</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-indigo-800/30 rounded-lg border border-indigo-200 dark:border-indigo-600">
                <h5 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">3단계: 로드맵 수립</h5>
                <ul className="text-sm text-indigo-600 dark:text-indigo-400 space-y-1">
                  <li>• 단계별 실행 계획</li>
                  <li>• 투자 우선순위</li>
                  <li>• 성과 측정 지표</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}