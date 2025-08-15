'use client';

import React from 'react';
import { Container, Server, Terminal, Database, Layers, Code, Lightbulb, Activity } from 'lucide-react';

export default function Chapter2() {
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
}