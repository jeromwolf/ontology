'use client'

import React from 'react'
import { Layers, Shield, Activity } from 'lucide-react'

export default function Chapter3() {
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
}