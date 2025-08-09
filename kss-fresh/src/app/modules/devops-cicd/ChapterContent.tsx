import React from 'react'
import { 
  Container, 
  Server, 
  GitBranch, 
  Shield, 
  Monitor, 
  Settings,
  Code,
  Database,
  Cloud,
  Terminal,
  Layers,
  Lock,
  Activity,
  Cpu,
  Lightbulb
} from 'lucide-react'

interface ChapterContentProps {
  chapterId: string
}

const ChapterContent: React.FC<ChapterContentProps> = ({ chapterId }) => {
  switch (chapterId) {
    case 'devops-culture':
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
                <Code className="w-5 h-5 text-blue-600 dark:text-blue-400" />
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
                <GitBranch className="w-5 h-5 text-green-600 dark:text-green-400" />
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
                <Container className="w-5 h-5 text-purple-600 dark:text-purple-400" />
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
                <Cloud className="w-5 h-5 text-orange-600 dark:text-orange-400" />
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
                <Monitor className="w-5 h-5 text-red-600 dark:text-red-400" />
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
                <Shield className="w-5 h-5 text-gray-600 dark:text-gray-400" />
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

    case 'docker-fundamentals':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl p-8 mb-8 border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                <Container className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Docker 기초와 컨테이너화</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              컨테이너의 개념부터 Docker 실무 활용까지, 현대적 애플리케이션 배포의 핵심을 학습합니다.
            </p>
          </div>

          <h2>🚀 컨테이너란 무엇인가?</h2>
          <p>
            컨테이너는 애플리케이션과 그 실행에 필요한 모든 것들을 하나의 패키지로 묶은 경량화된 가상화 기술입니다. 
            "한 번 빌드하면 어디서든 실행"이라는 철학을 구현합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-6 my-8">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
              <h3 className="text-red-800 dark:text-red-300 mt-0 flex items-center gap-2">
                <Server className="w-5 h-5" />
                가상 머신 (VM)
              </h3>
              <ul className="text-red-700 dark:text-red-300 space-y-2">
                <li>• <strong>무겁다:</strong> 전체 OS 포함</li>
                <li>• <strong>느린 시작:</strong> 몇 분 소요</li>
                <li>• <strong>높은 리소스 사용:</strong> GB 단위 메모리</li>
                <li>• <strong>완전한 격리:</strong> 하드웨어 수준</li>
                <li>• <strong>적은 밀도:</strong> 호스트당 적은 VM 수</li>
              </ul>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-blue-800 dark:text-blue-300 mt-0 flex items-center gap-2">
                <Container className="w-5 h-5" />
                컨테이너
              </h3>
              <ul className="text-blue-700 dark:text-blue-300 space-y-2">
                <li>• <strong>경량:</strong> OS 커널 공유</li>
                <li>• <strong>빠른 시작:</strong> 몇 초 내</li>
                <li>• <strong>낮은 리소스 사용:</strong> MB 단위 메모리</li>
                <li>• <strong>프로세스 수준 격리:</strong> 네임스페이스</li>
                <li>• <strong>높은 밀도:</strong> 호스트당 많은 컨테이너</li>
              </ul>
            </div>
          </div>

          <h2>🐳 Docker 아키텍처</h2>
          <p>Docker는 클라이언트-서버 아키텍처로 구성되어 있습니다.</p>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 my-6">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Terminal className="w-8 h-8 text-white" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Docker Client</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">사용자 명령어 인터페이스</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Server className="w-8 h-8 text-white" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Docker Daemon</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">컨테이너 관리 엔진</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Database className="w-8 h-8 text-white" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Docker Registry</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">이미지 저장소</p>
              </div>
            </div>
          </div>

          <h2>📦 핵심 개념들</h2>
          
          <div className="space-y-6 my-8">
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-blue-800 dark:text-blue-300 mt-0 flex items-center gap-2">
                <Layers className="w-5 h-5" />
                이미지 (Image)
              </h3>
              <p className="text-blue-700 dark:text-blue-300 mb-3">
                애플리케이션과 실행 환경을 포함한 읽기 전용 템플릿. 레이어 구조로 구성됨.
              </p>
              <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-3">
                <code className="text-sm text-blue-800 dark:text-blue-300">
                  docker pull nginx:latest<br/>
                  docker images
                </code>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <h3 className="text-green-800 dark:text-green-300 mt-0 flex items-center gap-2">
                <Container className="w-5 h-5" />
                컨테이너 (Container)
              </h3>
              <p className="text-green-700 dark:text-green-300 mb-3">
                이미지로부터 생성된 실행 중인 인스턴스. 쓰기 가능한 레이어를 추가로 가짐.
              </p>
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-3">
                <code className="text-sm text-green-800 dark:text-green-300">
                  docker run -d --name web nginx:latest<br/>
                  docker ps
                </code>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6">
              <h3 className="text-purple-800 dark:text-purple-300 mt-0 flex items-center gap-2">
                <Code className="w-5 h-5" />
                Dockerfile
              </h3>
              <p className="text-purple-700 dark:text-purple-300 mb-3">
                이미지를 빌드하기 위한 명령어들을 포함한 텍스트 파일.
              </p>
              <div className="bg-purple-100 dark:bg-purple-900/30 rounded p-3">
                <code className="text-sm text-purple-800 dark:text-purple-300">
                  docker build -t myapp:v1 .<br/>
                  docker history myapp:v1
                </code>
              </div>
            </div>
          </div>

          <h2>⚡ 기본 Docker 명령어</h2>
          <p>실무에서 가장 많이 사용하는 Docker 명령어들을 익혀보겠습니다.</p>

          <div className="overflow-x-auto my-6">
            <table className="w-full border-collapse border border-gray-300 dark:border-gray-600">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-800">
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">명령어</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">설명</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">예시</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker run</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">컨테이너 생성 및 실행</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker run -p 80:80 nginx</code></td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker ps</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">실행 중인 컨테이너 목록</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker ps -a</code></td>
                </tr>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker images</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">로컬 이미지 목록</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker images --filter dangling=true</code></td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker exec</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">실행 중인 컨테이너에서 명령 실행</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker exec -it web bash</code></td>
                </tr>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker logs</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">컨테이너 로그 확인</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker logs -f web</code></td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker stop</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">컨테이너 중지</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>docker stop web</code></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h2>📄 첫 번째 Dockerfile 작성</h2>
          <p>간단한 Node.js 웹 애플리케이션을 컨테이너화해보겠습니다.</p>

          <div className="bg-gray-900 rounded-lg p-6 my-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-300 text-sm font-medium">Dockerfile</span>
              <button className="text-blue-400 hover:text-blue-300 text-sm">복사</button>
            </div>
            <pre className="text-gray-100 text-sm overflow-x-auto">
              <code>{`# Base image 선택
FROM node:18-alpine

# Working directory 설정
WORKDIR /app

# Package files 복사
COPY package*.json ./

# Dependencies 설치
RUN npm ci --only=production

# Application code 복사
COPY . .

# Non-root user 생성 (보안)
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

# Port 노출
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:3000/health || exit 1

# 시작 명령어
CMD ["npm", "start"]`}</code>
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 my-6">
            <h3 className="text-yellow-800 dark:text-yellow-300 mt-0 flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Dockerfile 작성 모범 사례
            </h3>
            <ul className="text-yellow-700 dark:text-yellow-300 space-y-2">
              <li>• <strong>최소한의 base image:</strong> alpine 버전 사용</li>
              <li>• <strong>레이어 최적화:</strong> 관련 명령어 그룹화</li>
              <li>• <strong>캐시 활용:</strong> 자주 변경되는 파일을 뒤로</li>
              <li>• <strong>보안:</strong> non-root 사용자로 실행</li>
              <li>• <strong>health check:</strong> 컨테이너 상태 모니터링</li>
            </ul>
          </div>

          <h2>🔧 빌드 및 실행</h2>
          <p>작성한 Dockerfile을 이용해 이미지를 빌드하고 컨테이너를 실행해보겠습니다.</p>

          <div className="space-y-4 my-6">
            <div className="bg-gray-100 dark:bg-gray-800 rounded p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">1. 이미지 빌드</p>
              <code className="text-blue-600 dark:text-blue-400">docker build -t myapp:v1.0 .</code>
            </div>

            <div className="bg-gray-100 dark:bg-gray-800 rounded p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">2. 컨테이너 실행</p>
              <code className="text-blue-600 dark:text-blue-400">docker run -d -p 3000:3000 --name myapp myapp:v1.0</code>
            </div>

            <div className="bg-gray-100 dark:bg-gray-800 rounded p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">3. 로그 확인</p>
              <code className="text-blue-600 dark:text-blue-400">docker logs -f myapp</code>
            </div>

            <div className="bg-gray-100 dark:bg-gray-800 rounded p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">4. 컨테이너 내부 접속</p>
              <code className="text-blue-600 dark:text-blue-400">docker exec -it myapp sh</code>
            </div>
          </div>

          <h2>🔍 이미지 분석 및 최적화</h2>
          <p>빌드된 이미지를 분석하고 크기를 최적화하는 방법을 알아보겠습니다.</p>

          <div className="grid md:grid-cols-2 gap-6 my-8">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
              <h4 className="text-red-800 dark:text-red-300 mt-0">비효율적인 Dockerfile</h4>
              <div className="bg-red-100 dark:bg-red-900/30 rounded p-3 text-sm">
                <code className="text-red-800 dark:text-red-300">
                  FROM node:18<br/>
                  COPY . .<br/>
                  RUN npm install<br/>
                  RUN npm run build<br/>
                  CMD ["npm", "start"]
                </code>
              </div>
              <ul className="text-red-700 dark:text-red-300 text-sm mt-3 space-y-1">
                <li>• 큰 base image</li>
                <li>• 개발 의존성 포함</li>
                <li>• 캐시 비효율</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <h4 className="text-green-800 dark:text-green-300 mt-0">최적화된 Dockerfile</h4>
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-3 text-sm">
                <code className="text-green-800 dark:text-green-300">
                  FROM node:18-alpine<br/>
                  WORKDIR /app<br/>
                  COPY package*.json ./<br/>
                  RUN npm ci --only=production<br/>
                  COPY . .<br/>
                  CMD ["npm", "start"]
                </code>
              </div>
              <ul className="text-green-700 dark:text-green-300 text-sm mt-3 space-y-1">
                <li>• Alpine base (작은 크기)</li>
                <li>• 프로덕션 의존성만</li>
                <li>• 레이어 캐싱 최적화</li>
              </ul>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-6 my-8">
            <h3 className="text-emerald-800 dark:text-emerald-300 mt-0 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              실습: 이미지 분석하기
            </h3>
            <p className="text-emerald-700 dark:text-emerald-300 mb-4">
              다음 명령어들을 사용해서 여러분의 이미지를 분석해보세요.
            </p>
            <div className="space-y-3">
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded p-3">
                <code className="text-emerald-800 dark:text-emerald-300 text-sm">
                  # 이미지 히스토리 확인<br/>
                  docker history myapp:v1.0<br/><br/>
                  # 이미지 상세 정보<br/>
                  docker inspect myapp:v1.0<br/><br/>
                  # 이미지 크기 비교<br/>
                  docker images | grep myapp
                </code>
              </div>
            </div>
          </div>

          <h2>🌐 포트 매핑과 네트워킹</h2>
          <p>컨테이너의 네트워킹 개념과 포트 매핑을 이해해보겠습니다.</p>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 my-6">
            <h4 className="text-blue-800 dark:text-blue-300 mt-0">네트워킹 모드</h4>
            <div className="grid gap-4">
              <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-3">
                <h5 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Bridge (기본값)</h5>
                <code className="text-sm text-blue-700 dark:text-blue-400">docker run -p 8080:80 nginx</code>
                <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">호스트 포트 8080을 컨테이너 포트 80에 연결</p>
              </div>
              <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-3">
                <h5 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Host</h5>
                <code className="text-sm text-blue-700 dark:text-blue-400">docker run --network=host nginx</code>
                <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">호스트의 네트워크 인터페이스 직접 사용</p>
              </div>
              <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-3">
                <h5 className="font-medium text-blue-800 dark:text-blue-300 mb-2">None</h5>
                <code className="text-sm text-blue-700 dark:text-blue-400">docker run --network=none alpine</code>
                <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">네트워크 연결 없음 (완전 격리)</p>
              </div>
            </div>
          </div>

          <h2>🎯 다음 단계</h2>
          <p>
            Docker 기초를 마스터했다면, 이제 더 고급 기능들을 배울 차례입니다. 
            다음 챕터에서는 Docker Compose를 활용한 멀티컨테이너 관리와 네트워킹, 볼륨 등을 다룹니다.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 my-6">
            <h4 className="text-gray-900 dark:text-white mb-4">예습할 내용</h4>
            <ul className="text-gray-700 dark:text-gray-300 space-y-2">
              <li>• Docker Compose로 멀티컨테이너 관리</li>
              <li>• 컨테이너 간 네트워킹</li>
              <li>• 데이터 영속성을 위한 볼륨 사용</li>
              <li>• 멀티스테이지 빌드 최적화</li>
              <li>• 컨테이너 레지스트리 활용</li>
            </ul>
          </div>
        </div>
      )

    case 'docker-advanced':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-8 border border-cyan-200 dark:border-cyan-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-cyan-500 rounded-xl flex items-center justify-center">
                <Layers className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Docker 고급 기법</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              Docker Compose, 네트워킹, 볼륨, 그리고 이미지 최적화까지 실무에서 필요한 고급 기술들을 마스터합니다.
            </p>
          </div>

          <h2>🎼 Docker Compose 소개</h2>
          <p>
            Docker Compose는 멀티컨테이너 애플리케이션을 정의하고 실행하기 위한 도구입니다. 
            YAML 파일로 서비스들을 정의하고, 단일 명령으로 전체 애플리케이션을 관리할 수 있습니다.
          </p>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 my-6">
            <h3 className="text-blue-800 dark:text-blue-300 mt-0">Compose의 장점</h3>
            <ul className="text-blue-700 dark:text-blue-300 space-y-2">
              <li>• <strong>단순한 구성:</strong> YAML 파일로 모든 서비스 정의</li>
              <li>• <strong>일관된 환경:</strong> 개발부터 프로덕션까지 동일한 구성</li>
              <li>• <strong>서비스 오케스트레이션:</strong> 서비스 간 의존성 관리</li>
              <li>• <strong>스케일링:</strong> 서비스 인스턴스 개수 조절</li>
              <li>• <strong>네트워킹:</strong> 자동 서비스 디스커버리</li>
            </ul>
          </div>

          <h2>📄 Docker Compose 파일 작성</h2>
          <p>웹 애플리케이션, 데이터베이스, 캐시를 포함한 풀스택 애플리케이션을 예제로 작성해보겠습니다.</p>

          <div className="bg-gray-900 rounded-lg p-6 my-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-300 text-sm font-medium">docker-compose.yml</span>
              <button className="text-blue-400 hover:text-blue-300 text-sm">복사</button>
            </div>
            <pre className="text-gray-100 text-sm overflow-x-auto">
              <code>{`version: '3.8'

services:
  # 웹 애플리케이션
  web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
      - REDIS_URL=redis://cache:6379
    depends_on:
      - db
      - cache
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL 데이터베이스
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis 캐시
  cache:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Nginx 리버스 프록시
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge`}</code>
            </pre>
          </div>

          <h2>🔧 Compose 명령어</h2>
          <p>Docker Compose를 관리하는 주요 명령어들을 익혀보겠습니다.</p>

          <div className="overflow-x-auto my-6">
            <table className="w-full border-collapse border border-gray-300 dark:border-gray-600">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-800">
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">명령어</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">설명</th>
                  <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">옵션</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>up</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">서비스 시작</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>-d</code> (백그라운드), <code>--build</code> (재빌드)</td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>down</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">서비스 중지 및 삭제</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>-v</code> (볼륨 삭제), <code>--rmi all</code> (이미지 삭제)</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>ps</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">실행 중인 서비스 확인</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>-a</code> (모든 컨테이너)</td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>logs</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">서비스 로그 확인</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>-f</code> (실시간), <code>--tail=100</code> (마지막 100줄)</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>exec</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">서비스에서 명령 실행</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>-it</code> (대화형 터미널)</td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-800/50">
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>scale</code></td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">서비스 인스턴스 개수 조절</td>
                  <td className="border border-gray-300 dark:border-gray-600 px-4 py-3"><code>web=3</code> (web 서비스 3개)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h2>🌐 Docker 네트워킹</h2>
          <p>컨테이너 간 통신과 외부 네트워크 연결을 위한 Docker 네트워킹을 알아보겠습니다.</p>

          <div className="grid md:grid-cols-2 gap-6 my-8">
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-blue-800 dark:text-blue-300 mt-0">기본 네트워크 드라이버</h3>
              <div className="space-y-3">
                <div>
                  <h4 className="font-medium text-blue-700 dark:text-blue-300">Bridge (기본값)</h4>
                  <p className="text-sm text-blue-600 dark:text-blue-400">단일 호스트의 컨테이너 연결</p>
                </div>
                <div>
                  <h4 className="font-medium text-blue-700 dark:text-blue-300">Host</h4>
                  <p className="text-sm text-blue-600 dark:text-blue-400">호스트 네트워크 직접 사용</p>
                </div>
                <div>
                  <h4 className="font-medium text-blue-700 dark:text-blue-300">Overlay</h4>
                  <p className="text-sm text-blue-600 dark:text-blue-400">여러 호스트 간 컨테이너 연결</p>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <h3 className="text-green-800 dark:text-green-300 mt-0">네트워크 관리</h3>
              <div className="space-y-2 text-sm">
                <code className="block bg-green-100 dark:bg-green-900/30 p-2 rounded text-green-800 dark:text-green-300">
                  docker network create mynet
                </code>
                <code className="block bg-green-100 dark:bg-green-900/30 p-2 rounded text-green-800 dark:text-green-300">
                  docker network ls
                </code>
                <code className="block bg-green-100 dark:bg-green-900/30 p-2 rounded text-green-800 dark:text-green-300">
                  docker network inspect mynet
                </code>
                <code className="block bg-green-100 dark:bg-green-900/30 p-2 rounded text-green-800 dark:text-green-300">
                  docker run --network=mynet nginx
                </code>
              </div>
            </div>
          </div>

          <h2>💾 Docker 볼륨</h2>
          <p>컨테이너의 데이터 영속성을 보장하기 위한 볼륨 관리 방법을 학습합니다.</p>

          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6 my-6">
            <h3 className="text-purple-800 dark:text-purple-300 mt-0">볼륨 타입</h3>
            <div className="grid gap-4 mt-4">
              <div className="bg-purple-100 dark:bg-purple-900/30 rounded p-4">
                <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-2">1. Named Volume</h4>
                <code className="text-sm text-purple-700 dark:text-purple-400">docker volume create mydata</code>
                <p className="text-sm text-purple-700 dark:text-purple-400 mt-1">Docker가 관리하는 볼륨, 데이터 공유에 적합</p>
              </div>
              
              <div className="bg-purple-100 dark:bg-purple-900/30 rounded p-4">
                <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-2">2. Bind Mount</h4>
                <code className="text-sm text-purple-700 dark:text-purple-400">-v /host/path:/container/path</code>
                <p className="text-sm text-purple-700 dark:text-purple-400 mt-1">호스트 경로 직접 마운트, 개발 시 유용</p>
              </div>
              
              <div className="bg-purple-100 dark:bg-purple-900/30 rounded p-4">
                <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-2">3. tmpfs Mount</h4>
                <code className="text-sm text-purple-700 dark:text-purple-400">--tmpfs /tmp</code>
                <p className="text-sm text-purple-700 dark:text-purple-400 mt-1">메모리에 임시 파일 시스템 생성</p>
              </div>
            </div>
          </div>

          <h2>🏗️ 멀티스테이지 빌드</h2>
          <p>이미지 크기를 최소화하고 보안을 강화하는 멀티스테이지 빌드 기법을 배워봅시다.</p>

          <div className="bg-gray-900 rounded-lg p-6 my-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-300 text-sm font-medium">Dockerfile (Multi-stage)</span>
              <button className="text-blue-400 hover:text-blue-300 text-sm">복사</button>
            </div>
            <pre className="text-gray-100 text-sm overflow-x-auto">
              <code>{`# Build stage
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

COPY . .
RUN npm run build

# Runtime stage
FROM node:18-alpine AS runtime

# 보안을 위한 non-root 사용자
RUN addgroup -g 1001 -S nodejs \\
    && adduser -S nextjs -u 1001

WORKDIR /app

# 필요한 파일만 복사
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node healthcheck.js || exit 1

CMD ["node", "dist/server.js"]`}</code>
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6 my-6">
            <h3 className="text-green-800 dark:text-green-300 mt-0">멀티스테이지 빌드 장점</h3>
            <ul className="text-green-700 dark:text-green-300 space-y-2">
              <li>• <strong>작은 이미지 크기:</strong> 런타임에 필요한 파일만 포함</li>
              <li>• <strong>보안 강화:</strong> 빌드 도구와 소스코드 제외</li>
              <li>• <strong>캐시 최적화:</strong> 스테이지별 캐시 레이어</li>
              <li>• <strong>빌드 환경 분리:</strong> 개발/빌드/런타임 환경 구분</li>
            </ul>
          </div>

          <h2>📦 컨테이너 레지스트리</h2>
          <p>이미지를 저장하고 배포하기 위한 컨테이너 레지스트리 활용법을 알아봅시다.</p>

          <div className="grid md:grid-cols-3 gap-4 my-8">
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">Docker Hub</h4>
              <div className="space-y-2 text-sm">
                <code className="block bg-blue-100 dark:bg-blue-900/30 p-2 rounded text-blue-700 dark:text-blue-300">
                  docker push myapp:v1.0
                </code>
                <p className="text-blue-600 dark:text-blue-400">가장 인기 있는 퍼블릭 레지스트리</p>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
              <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-3">AWS ECR</h4>
              <div className="space-y-2 text-sm">
                <code className="block bg-orange-100 dark:bg-orange-900/30 p-2 rounded text-orange-700 dark:text-orange-300">
                  aws ecr get-login-password
                </code>
                <p className="text-orange-600 dark:text-orange-400">AWS의 완전 관리형 레지스트리</p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/20 border border-gray-200 dark:border-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-300 mb-3">Harbor</h4>
              <div className="space-y-2 text-sm">
                <code className="block bg-gray-100 dark:bg-gray-800 p-2 rounded text-gray-700 dark:text-gray-300">
                  docker push harbor.io/myapp
                </code>
                <p className="text-gray-600 dark:text-gray-400">엔터프라이즈 기능이 풍부한 오픈소스</p>
              </div>
            </div>
          </div>

          <h2>🛡️ 보안 모범 사례</h2>
          <p>컨테이너 보안을 위한 핵심 원칙들을 알아보겠습니다.</p>

          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 my-6">
            <h3 className="text-red-800 dark:text-red-300 mt-0 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              보안 체크리스트
            </h3>
            <div className="space-y-3 mt-4">
              {[
                "최소 권한 원칙: root 사용자 사용 금지",
                "신뢰할 수 있는 base image 사용",
                "정기적인 이미지 업데이트 및 취약점 스캔",
                "불필요한 패키지 및 파일 제거",
                "시크릿 정보를 이미지에 포함하지 않기",
                "컨테이너 리소스 제한 설정",
                "읽기 전용 루트 파일 시스템 사용",
                "네트워크 액세스 제한"
              ].map((item, index) => (
                <label key={index} className="flex items-start gap-3 text-red-700 dark:text-red-300">
                  <input type="checkbox" className="w-4 h-4 mt-1 text-red-600 border-red-300 rounded focus:ring-red-500" />
                  <span className="text-sm">{item}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-6 my-8">
            <h3 className="text-emerald-800 dark:text-emerald-300 mt-0 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              실습: 풀스택 앱 구축하기
            </h3>
            <p className="text-emerald-700 dark:text-emerald-300 mb-4">
              다음과 같은 구성으로 완전한 웹 애플리케이션을 구축해보세요.
            </p>
            <div className="grid gap-4">
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded p-4">
                <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">1. 프론트엔드 (React)</h4>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">nginx로 정적 파일 서빙, 멀티스테이지 빌드 적용</p>
              </div>
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded p-4">
                <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">2. 백엔드 API (Node.js/Python)</h4>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">REST API 서버, 환경 변수로 설정 관리</p>
              </div>
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded p-4">
                <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">3. 데이터베이스 (PostgreSQL)</h4>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">데이터 볼륨 마운트, 초기화 스크립트 포함</p>
              </div>
              <div className="bg-emerald-100 dark:bg-emerald-900/30 rounded p-4">
                <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">4. 캐시 (Redis)</h4>
                <p className="text-sm text-emerald-700 dark:text-emerald-300">세션 저장소 또는 API 캐시로 활용</p>
              </div>
            </div>
          </div>

          <h2>🎯 다음 단계</h2>
          <p>
            Docker 고급 기법을 마스터했다면, 이제 Kubernetes를 통한 컨테이너 오케스트레이션으로 넘어갈 차례입니다. 
            다음 챕터에서는 Kubernetes의 기본 개념과 주요 리소스들을 학습합니다.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 my-6">
            <h4 className="text-gray-900 dark:text-white mb-4">준비해야 할 것들</h4>
            <ul className="text-gray-700 dark:text-gray-300 space-y-2">
              <li>• Kubernetes 클러스터 (minikube, kind, 또는 클라우드)</li>
              <li>• kubectl CLI 도구</li>
              <li>• 컨테이너 레지스트리 계정</li>
              <li>• YAML 기본 문법 이해</li>
              <li>• 이전 챕터에서 만든 컨테이너 이미지들</li>
            </ul>
          </div>
        </div>
      )

    // ... 나머지 챕터들도 동일한 패턴으로 구현
    case 'kubernetes-basics':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 mb-8 border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 기초</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              Kubernetes 아키텍처와 핵심 오브젝트들을 이해하고, kubectl로 클러스터를 관리하는 방법을 학습합니다.
            </p>
          </div>

          <p>
            이 챕터에서는 Kubernetes의 기본 개념과 아키텍처, 주요 오브젝트들에 대해 학습합니다.
            실습을 통해 Pod, Service, Deployment를 만들고 관리하는 방법을 익히겠습니다.
          </p>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 my-6">
            <h3 className="text-blue-800 dark:text-blue-300 mt-0">주요 학습 내용</h3>
            <ul className="text-blue-700 dark:text-blue-300">
              <li>• Kubernetes 아키텍처 이해</li>
              <li>• Pod, Service, Deployment 개념</li>
              <li>• kubectl 기본 명령어</li>
              <li>• YAML 매니페스트 작성</li>
            </ul>
          </div>

          <p className="text-gray-600 dark:text-gray-400">
            상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
          </p>
        </div>
      )

    case 'kubernetes-advanced':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
                <Server className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 운영</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              Ingress, ConfigMap, Secret, 스케일링 등 Kubernetes 클러스터 운영에 필요한 고급 기능들을 학습합니다.
            </p>
          </div>

          <p>
            이 챕터에서는 프로덕션 환경에서 Kubernetes를 운영하기 위한 고급 기능들을 다룹니다.
          </p>

          <p className="text-gray-600 dark:text-gray-400">
            상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
          </p>
        </div>
      )

    case 'cicd-pipelines':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-8 border border-green-200 dark:border-green-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
                <GitBranch className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">CI/CD 파이프라인 구축</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              GitHub Actions와 Jenkins를 활용하여 자동화된 빌드, 테스트, 배포 파이프라인을 구축합니다.
            </p>
          </div>

          <p>
            이 챕터에서는 CI/CD의 개념부터 실제 파이프라인 구축까지 실습합니다.
          </p>

          <p className="text-gray-600 dark:text-gray-400">
            상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
          </p>
        </div>
      )

    case 'gitops-deployment':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-indigo-50 to-cyan-50 dark:from-indigo-900/20 dark:to-cyan-900/20 rounded-2xl p-8 mb-8 border border-indigo-200 dark:border-indigo-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-indigo-500 rounded-xl flex items-center justify-center">
                <GitBranch className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">GitOps와 배포 전략</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              GitOps 원칙에 따른 선언적 배포와 Blue-Green, Canary, Rolling Update 등 다양한 배포 전략을 학습합니다.
            </p>
          </div>

          <p>
            이 챕터에서는 GitOps 개념과 ArgoCD 사용법, 그리고 다양한 배포 전략들을 실습합니다.
          </p>

          <p className="text-gray-600 dark:text-gray-400">
            상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
          </p>
        </div>
      )

    case 'monitoring-security':
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl p-8 mb-8 border border-red-200 dark:border-red-800">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-red-500 rounded-xl flex items-center justify-center">
                <Monitor className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">모니터링, 로깅, 보안</h1>
            </div>
            <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
              Prometheus, Grafana, ELK Stack을 활용한 모니터링과 로깅, 그리고 컨테이너 보안 모범 사례를 학습합니다.
            </p>
          </div>

          <p>
            이 챕터에서는 프로덕션 시스템의 모니터링, 로깅, 보안에 대해 학습합니다.
          </p>

          <p className="text-gray-600 dark:text-gray-400">
            상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
          </p>
        </div>
      )

    default:
      return (
        <div className="prose prose-lg max-w-none dark:prose-invert">
          <h1>챕터를 찾을 수 없습니다</h1>
          <p>요청하신 챕터가 존재하지 않습니다.</p>
        </div>
      )
  }
}

export default ChapterContent