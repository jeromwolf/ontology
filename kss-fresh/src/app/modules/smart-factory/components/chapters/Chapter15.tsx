'use client';

import {
  Users, Heart, BookOpen, Award, TrendingUp
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter15() {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-blue-200 dark:border-blue-800">
        <h3 className="text-2xl font-bold text-blue-900 dark:text-blue-200 mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8" />
          Kotter의 8단계 변화관리 모델
        </h3>
        <div className="grid md:grid-cols-4 gap-4">
          {[
            { step: 1, title: "위기감 조성", desc: "변화 필요성 인식", icon: "🚨" },
            { step: 2, title: "추진 연합체", desc: "변화 리더십 구축", icon: "🤝" },
            { step: 3, title: "비전 수립", desc: "명확한 미래상", icon: "🎯" },
            { step: 4, title: "비전 소통", desc: "전사 공유 확산", icon: "📢" },
            { step: 5, title: "권한 위임", desc: "실행 권한 부여", icon: "⚡" },
            { step: 6, title: "단기 성과", desc: "빠른 승리 창출", icon: "🏆" },
            { step: 7, title: "성과 가속화", desc: "지속적 개선", icon: "🚀" },
            { step: 8, title: "문화 정착", desc: "새로운 문화", icon: "🌱" }
          ].map((phase, idx) => (
            <div key={idx} className="text-center p-4 bg-white dark:bg-blue-800/30 rounded-lg border border-blue-200 dark:border-blue-600">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-2xl">{phase.icon}</span>
              </div>
              <h4 className="font-bold text-blue-800 dark:text-blue-200 text-sm mb-2">Step {phase.step}</h4>
              <h5 className="font-semibold text-blue-700 dark:text-blue-300 text-sm mb-2">{phase.title}</h5>
              <p className="text-xs text-blue-600 dark:text-blue-400">{phase.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Heart className="w-6 h-6 text-slate-600" />
            저항 요인 분석 & 대응 전략
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">일자리 불안</h4>
              <p className="text-sm text-red-700 dark:text-red-400 mb-2">자동화로 인한 고용 감소 우려</p>
              <div className="text-xs bg-red-100 dark:bg-red-800 text-red-800 dark:text-red-300 p-2 rounded">
                <strong>대응방안:</strong> 재교육을 통한 직무 전환, 고부가가치 업무로 배치
              </div>
            </div>
            
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-400 rounded">
              <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">기술 두려움</h4>
              <p className="text-sm text-orange-700 dark:text-orange-400 mb-2">새로운 기술에 대한 막연한 불안</p>
              <div className="text-xs bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-300 p-2 rounded">
                <strong>대응방안:</strong> 단계적 학습, 멘토링 시스템, 성공 경험 공유
              </div>
            </div>

            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 rounded">
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">업무 방식 변화</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-400 mb-2">기존 프로세스에 대한 강한 고착화</p>
              <div className="text-xs bg-yellow-100 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-300 p-2 rounded">
                <strong>대응방안:</strong> 점진적 변화, 충분한 적응 기간, 피드백 수렴
              </div>
            </div>

            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">세대 간 격차</h4>
              <p className="text-sm text-blue-700 dark:text-blue-400 mb-2">디지털 네이티브와 기존 세대 차이</p>
              <div className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-300 p-2 rounded">
                <strong>대응방안:</strong> 세대별 맞춤 교육, 상호 멘토링, 팀 빌딩
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-slate-600" />
            세대별 교육 전략
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">베이비부머 (1946-1964)</h4>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• 체계적이고 순차적인 학습</li>
                <li>• 풍부한 경험 활용</li>
                <li>• 소그룹 집중 교육</li>
                <li>• 충분한 실습 시간</li>
              </ul>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">X세대 (1965-1980)</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• 독립적 학습 선호</li>
                <li>• 실무 중심 교육</li>
                <li>• 온라인 학습 병행</li>
                <li>• 리더십 역할 부여</li>
              </ul>
            </div>

            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">밀레니얼 (1981-1996)</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• 기술 친화적 환경</li>
                <li>• 협업 중심 학습</li>
                <li>• 즉시 피드백 제공</li>
                <li>• 성장 기회 명시</li>
              </ul>
            </div>

            <div className="p-4 bg-teal-50 dark:bg-teal-900/20 border rounded">
              <h4 className="font-semibold text-teal-800 dark:text-teal-300 mb-2">Z세대 (1997~)</h4>
              <ul className="text-sm text-teal-700 dark:text-teal-400 space-y-1">
                <li>• 모바일 우선 학습</li>
                <li>• 마이크로러닝</li>
                <li>• 게임화 요소</li>
                <li>• 소셜 학습</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Award className="w-8 h-8" />
          디지털 역량 매트릭스
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-green-200 dark:border-green-700">
                <th className="text-left py-3 px-4 font-semibold text-green-900 dark:text-green-200">역량 영역</th>
                <th className="text-center py-3 px-4 font-semibold text-green-900 dark:text-green-200">초급 (Level 1)</th>
                <th className="text-center py-3 px-4 font-semibold text-green-900 dark:text-green-200">중급 (Level 2)</th>
                <th className="text-center py-3 px-4 font-semibold text-green-900 dark:text-green-200">고급 (Level 3)</th>
                <th className="text-center py-3 px-4 font-semibold text-green-900 dark:text-green-200">전문가 (Level 4)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-green-200 dark:divide-green-700">
              <tr>
                <td className="py-3 px-4 text-green-900 dark:text-green-200 font-medium">디지털 리터러시</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">기본 SW 활용</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">데이터 분석</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">시각화, 대시보드</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">AI/ML 이해</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-green-900 dark:text-green-200 font-medium">프로세스 혁신</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">현황 파악</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">개선점 도출</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">최적화 설계</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">자동화 구현</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-green-900 dark:text-green-200 font-medium">문제 해결</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">문제 인식</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">원인 분석</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">해결책 구현</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">예방적 개선</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-green-900 dark:text-green-200 font-medium">협업</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">팀원 협력</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">부서간 협업</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">프로젝트 리딩</td>
                <td className="py-3 px-4 text-center text-green-700 dark:text-green-300">전사 혁신 주도</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-amber-600" />
          단계별 교육 프로그램
        </h3>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">1</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">인식 개선</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 스마트팩토리 개념</li>
              <li>• 성공 사례 공유</li>
              <li>• 변화 필요성</li>
              <li>• 비전 공유</li>
            </ul>
            <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs text-blue-600 dark:text-blue-400">
              4시간 워크숍
            </div>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">2</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">기초 교육</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 디지털 도구 활용</li>
              <li>• 데이터 이해</li>
              <li>• 기본 분석</li>
              <li>• 보안 인식</li>
            </ul>
            <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/20 rounded text-xs text-green-600 dark:text-green-400">
              2주 과정
            </div>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">3</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">실무 적용</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 시스템 활용</li>
              <li>• 프로세스 개선</li>
              <li>• 문제 해결</li>
              <li>• 프로젝트 참여</li>
            </ul>
            <div className="mt-3 p-2 bg-purple-50 dark:bg-purple-900/20 rounded text-xs text-purple-600 dark:text-purple-400">
              3개월 OJT
            </div>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">4</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">리더 양성</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 혁신 리더십</li>
              <li>• 변화 관리</li>
              <li>• 멘토링</li>
              <li>• 전략 수립</li>
            </ul>
            <div className="mt-3 p-2 bg-orange-50 dark:bg-orange-900/20 rounded text-xs text-orange-600 dark:text-orange-400">
              6개월 과정
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-8 rounded-xl border border-purple-200 dark:border-purple-800">
        <h3 className="text-2xl font-bold text-purple-900 dark:text-purple-200 mb-6 flex items-center gap-3">
          <Heart className="w-8 h-8" />
          문화 혁신 전략
        </h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-purple-800 dark:text-purple-200">기존 문화 → 새로운 문화</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <span className="text-sm text-purple-700 dark:text-purple-300">완벽 추구</span>
                <span className="text-purple-500">→</span>
                <span className="text-sm text-purple-700 dark:text-purple-300">빠른 학습</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <span className="text-sm text-purple-700 dark:text-purple-300">개인 성과</span>
                <span className="text-purple-500">→</span>
                <span className="text-sm text-purple-700 dark:text-purple-300">팀 협업</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <span className="text-sm text-purple-700 dark:text-purple-300">경험 중심</span>
                <span className="text-purple-500">→</span>
                <span className="text-sm text-purple-700 dark:text-purple-300">데이터 중심</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <span className="text-sm text-purple-700 dark:text-purple-300">위계적 소통</span>
                <span className="text-purple-500">→</span>
                <span className="text-sm text-purple-700 dark:text-purple-300">수평적 소통</span>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-purple-800 dark:text-purple-200">실행 방안</h4>
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">실패 허용 문화</h5>
                <p className="text-sm text-purple-600 dark:text-purple-400">빠른 실패, 빠른 학습을 통한 혁신 가속화</p>
              </div>
              
              <div className="p-4 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">데이터 기반 의사결정</h5>
                <p className="text-sm text-purple-600 dark:text-purple-400">직관보다는 객관적 데이터에 기반한 판단</p>
              </div>
              
              <div className="p-4 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">지속적 학습</h5>
                <p className="text-sm text-purple-600 dark:text-purple-400">개인과 조직의 지속적인 역량 개발</p>
              </div>
              
              <div className="p-4 bg-white dark:bg-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-600">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">개방형 혁신</h5>
                <p className="text-sm text-purple-600 dark:text-purple-400">외부와의 협업을 통한 혁신 생태계 구축</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 변화관리 프레임워크 & 모델',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'Kotter\'s 8-Step Change Model',
                url: 'https://www.kotterinc.com/methodology/8-steps/',
                description: 'John Kotter의 8단계 변화관리 모델 - 위기감 조성부터 문화 정착까지 체계적 접근법.'
              },
              {
                title: 'Prosci ADKAR Model',
                url: 'https://www.prosci.com/methodology/adkar',
                description: 'Prosci의 ADKAR 모델 - Awareness, Desire, Knowledge, Ability, Reinforcement 5단계 개인 변화 관리.'
              },
              {
                title: 'McKinsey - Leading Digital Transformation',
                url: 'https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/the-organization-blog/leading-a-digital-transformation',
                description: 'McKinsey의 디지털 전환 리더십 가이드 - 조직 변화 관리 전략과 성공 요인.'
              },
              {
                title: 'Lewin\'s Change Management Model',
                url: 'https://www.mindtools.com/pages/article/newPPM_94.htm',
                description: 'Kurt Lewin의 Unfreeze-Change-Refreeze 모델 - 고전적 변화관리 3단계 이론.'
              },
              {
                title: 'WEF - Upskilling for Industry 4.0',
                url: 'https://www.weforum.org/reports/upskilling-for-shared-prosperity',
                description: 'World Economic Forum - 산업 4.0 시대를 위한 업스킬링 및 리스킬링 전략 보고서.'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Resistance to Digital Transformation in Manufacturing',
                url: 'https://www.sciencedirect.com/science/article/pii/S0166361519304445',
                description: 'ScienceDirect - 제조업 디지털 전환에 대한 저항 요인 분석과 극복 전략 연구.'
              },
              {
                title: 'Generational Differences in Digital Learning',
                url: 'https://www.emerald.com/insight/content/doi/10.1108/JWL-02-2020-0033/full/html',
                description: 'Emerald Journal - 세대별 디지털 학습 선호도와 효과적 교육 방법론 연구.'
              },
              {
                title: 'Culture Change in Smart Manufacturing',
                url: 'https://ieeexplore.ieee.org/document/9184522',
                description: 'IEEE - 스마트 제조 환경에서의 조직 문화 변화 관리 사례 연구.'
              },
              {
                title: 'Digital Competency Framework for Manufacturing',
                url: 'https://www.sciencedirect.com/science/article/pii/S0166361520301883',
                description: 'ScienceDirect - 제조업 종사자를 위한 디지털 역량 프레임워크 개발 연구.'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프로그램',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: '한국산업인력공단 - 스마트팩토리 교육',
                url: 'https://www.hrdkorea.or.kr/',
                description: '한국산업인력공단 스마트팩토리 교육 프로그램 - 정부 지원 직업훈련 과정.'
              },
              {
                title: '산업통상자원부 - 스마트공장 확산 사업',
                url: 'https://www.smart-factory.kr/',
                description: '스마트공장추진단 - 중소기업 스마트팩토리 구축 지원 및 교육 프로그램.'
              },
              {
                title: 'LinkedIn Learning - Digital Transformation',
                url: 'https://www.linkedin.com/learning/topics/digital-transformation',
                description: 'LinkedIn Learning - 디지털 전환 관련 온라인 강좌 및 학습 경로.'
              },
              {
                title: 'Coursera - Industry 4.0 & Smart Factory',
                url: 'https://www.coursera.org/courses?query=industry%204.0',
                description: 'Coursera - 산업 4.0 및 스마트팩토리 관련 MOOC 강의 플랫폼.'
              },
              {
                title: 'Udemy - Change Management Courses',
                url: 'https://www.udemy.com/topic/change-management/',
                description: 'Udemy - 변화관리 실무 교육 과정 - ADKAR, Kotter 모델 등 실습 중심.'
              }
            ]
          }
        ]}
      />
    </div>
  );
}