'use client';

import React from 'react';
import { Settings, Activity } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-8 border border-blue-200 dark:border-blue-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
            <Settings className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">DevOps 문화와 철학</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          DevOps의 핵심 개념과 문화 변화, 그리고 현대적인 소프트웨어 개발 도구체인을 이해합니다.
        </p>
      </div>

      <h2>📚 DevOps란 무엇인가?</h2>
      <p>
        DevOps는 <strong>Development(개발)</strong>와 <strong>Operations(운영)</strong>을 합친 용어로, 
        소프트웨어 개발과 IT 운영 간의 벽을 허무는 문화적, 기술적 움직임입니다.
      </p>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 my-6">
        <h3 className="text-yellow-800 dark:text-yellow-300 mt-0 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          핵심 원칙
        </h3>
        <ul className="space-y-2 text-yellow-700 dark:text-yellow-300">
          <li><strong>협업(Collaboration):</strong> 개발팀과 운영팀의 긴밀한 협력</li>
          <li><strong>자동화(Automation):</strong> 반복적인 작업의 자동화</li>
          <li><strong>지속적 통합/배포(CI/CD):</strong> 빈번하고 안정적인 배포</li>
          <li><strong>모니터링(Monitoring):</strong> 실시간 시스템 관찰과 개선</li>
          <li><strong>피드백(Feedback):</strong> 빠른 피드백 루프 구축</li>
        </ul>
      </div>

      <h2>🔄 전통적 방식 vs DevOps</h2>
      <div className="grid md:grid-cols-2 gap-6 my-6">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <h3 className="text-red-800 dark:text-red-300 mt-0">전통적 방식</h3>
          <ul className="text-red-700 dark:text-red-300 space-y-1">
            <li>• 개발팀과 운영팀의 분리</li>
            <li>• 긴 개발 사이클 (몇 개월)</li>
            <li>• 수동 배포 및 테스트</li>
            <li>• 문제 발생 시 비난 문화</li>
            <li>• 사일로(Silo) 조직 구조</li>
          </ul>
        </div>
        
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
          <h3 className="text-green-800 dark:text-green-300 mt-0">DevOps 방식</h3>
          <ul className="text-green-700 dark:text-green-300 space-y-1">
            <li>• 통합된 크로스 펑셔널 팀</li>
            <li>• 짧은 반복 주기 (몇 시간/일)</li>
            <li>• 자동화된 파이프라인</li>
            <li>• 공동 책임과 투명성</li>
            <li>• 협업 중심 문화</li>
          </ul>
        </div>
      </div>

      <h2>🛠️ DevOps 도구체인 생태계</h2>
      <p>DevOps를 구현하기 위한 다양한 도구들이 존재하며, 각각 특정 단계에서 중요한 역할을 합니다.</p>

      <div className="grid md:grid-cols-3 gap-4 my-8">
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h4 className="text-blue-800 dark:text-blue-300 m-0">개발</h4>
          </div>
          <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
            <li>• Git, GitHub, GitLab</li>
            <li>• VSCode, IntelliJ</li>
            <li>• Slack, Jira</li>
          </ul>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-green-600 dark:text-green-400" />
            <h4 className="text-green-800 dark:text-green-300 m-0">CI/CD</h4>
          </div>
          <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
            <li>• Jenkins, GitHub Actions</li>
            <li>• CircleCI, GitLab CI</li>
            <li>• ArgoCD, Flux</li>
          </ul>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <h4 className="text-purple-800 dark:text-purple-300 m-0">컨테이너화</h4>
          </div>
          <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-1">
            <li>• Docker, Podman</li>
            <li>• Kubernetes</li>
            <li>• Helm, Kustomize</li>
          </ul>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            <h4 className="text-orange-800 dark:text-orange-300 m-0">인프라</h4>
          </div>
          <ul className="text-sm text-orange-700 dark:text-orange-300 space-y-1">
            <li>• Terraform, Ansible</li>
            <li>• AWS, Azure, GCP</li>
            <li>• Pulumi</li>
          </ul>
        </div>

        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-red-600 dark:text-red-400" />
            <h4 className="text-red-800 dark:text-red-300 m-0">모니터링</h4>
          </div>
          <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
            <li>• Prometheus, Grafana</li>
            <li>• ELK Stack</li>
            <li>• DataDog, New Relic</li>
          </ul>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900/20 border border-gray-200 dark:border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <h4 className="text-gray-800 dark:text-gray-300 m-0">보안</h4>
          </div>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• HashiCorp Vault</li>
            <li>• SonarQube, Snyk</li>
            <li>• OWASP ZAP</li>
          </ul>
        </div>
      </div>

      <h2>🌟 DevOps 문화 구축하기</h2>
      <p>기술적인 도구 도입보다 중요한 것은 조직 문화의 변화입니다.</p>

      <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-6 my-6">
        <h3 className="text-indigo-800 dark:text-indigo-300 mt-0">문화 변화 단계</h3>
        <ol className="text-indigo-700 dark:text-indigo-300 space-y-3">
          <li><strong>1. 공동 목표 설정:</strong> 개발과 운영이 같은 비즈니스 목표를 공유</li>
          <li><strong>2. 소통 채널 구축:</strong> 정기적인 미팅, 공유 도구 활용</li>
          <li><strong>3. 실패에 대한 관점 변화:</strong> 비난 대신 학습의 기회로</li>
          <li><strong>4. 작은 성공 경험:</strong> 간단한 자동화부터 시작</li>
          <li><strong>5. 지속적 개선:</strong> 회고와 개선의 반복</li>
        </ol>
      </div>

      <h2>📊 DevOps 성숙도 평가</h2>
      <p>조직의 DevOps 성숙도를 평가하는 주요 지표들입니다.</p>

      <div className="overflow-x-auto my-6">
        <table className="w-full border-collapse border border-gray-300 dark:border-gray-600">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-800">
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">영역</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">초급</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">중급</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">고급</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">배포 빈도</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">월 1회</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">주 1회</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">일 여러 번</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-800/50">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">배포 시간</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">몇 시간</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">몇 십분</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">몇 분</td>
            </tr>
            <tr>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">장애 복구</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">며칠</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">몇 시간</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">몇 분</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-800/50">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">변경 실패율</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">&gt; 30%</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">10-30%</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">&lt; 10%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-6 my-8">
        <h3 className="text-emerald-800 dark:text-emerald-300 mt-0 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          실습: DevOps 체크리스트
        </h3>
        <p className="text-emerald-700 dark:text-emerald-300 mb-4">
          여러분의 조직에서 다음 항목들 중 몇 개나 실현되고 있는지 체크해보세요.
        </p>
        <div className="space-y-2">
          {[
            "개발팀과 운영팀이 정기적으로 소통한다",
            "코드 변경 시 자동으로 테스트가 실행된다", 
            "배포가 자동화되어 있다",
            "시스템 상태를 실시간으로 모니터링한다",
            "장애 발생 시 롤백이 자동화되어 있다",
            "인프라가 코드로 관리된다",
            "보안 검사가 파이프라인에 통합되어 있다",
            "팀원들이 DevOps 도구를 능숙하게 사용한다"
          ].map((item, index) => (
            <label key={index} className="flex items-center gap-3 text-emerald-700 dark:text-emerald-300">
              <input type="checkbox" className="w-4 h-4 text-emerald-600 border-emerald-300 rounded focus:ring-emerald-500" />
              <span>{item}</span>
            </label>
          ))}
        </div>
      </div>

      <h2>🎯 다음 단계</h2>
      <p>
        DevOps 문화와 철학을 이해했다면, 이제 실제 도구들을 활용해 컨테이너화와 오케스트레이션을 시작할 시간입니다. 
        다음 챕터에서는 Docker의 기초부터 시작해보겠습니다.
      </p>
    </div>
  )
}