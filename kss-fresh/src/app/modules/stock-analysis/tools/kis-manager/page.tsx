/**
 * KIS API 토큰 관리 페이지
 * 개발자와 관리자를 위한 토큰 상태 모니터링 및 관리 도구
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Shield, Server, Database, Activity, AlertCircle, CheckCircle, ExternalLink } from 'lucide-react';
import KISTokenStatus from '@/components/charts/ProChart/KISTokenStatus';

export default function KISManagerPage() {
  const [activeTab, setActiveTab] = useState<'status' | 'config' | 'test' | 'docs'>('status');

  const tabs = [
    { id: 'status', name: '토큰 상태', icon: Shield },
    { id: 'config', name: '설정', icon: Server },
    { id: 'test', name: '테스트', icon: Activity },
    { id: 'docs', name: '문서', icon: Database },
  ];

  return (
    <div className=\"min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white\">
      {/* Header */}
      <div className=\"border-b border-gray-700 bg-gray-900/50 backdrop-blur-xl\">
        <div className=\"max-w-7xl mx-auto px-4 py-4\">
          <div className=\"flex items-center justify-between\">
            <div className=\"flex items-center gap-4\">
              <Link href=\"/modules/stock-analysis/tools\" className=\"text-gray-400 hover:text-white transition-colors\">
                <ArrowLeft className=\"w-5 h-5\" />
              </Link>
              <div>
                <h1 className=\"text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent\">
                  KIS API 관리자
                </h1>
                <p className=\"text-gray-400 text-sm mt-1\">
                  한국투자증권 API 토큰 및 연결 상태 관리
                </p>
              </div>
            </div>
            
            <div className=\"flex items-center gap-2 px-3 py-1.5 bg-green-900/30 border border-green-500/30 rounded-lg\">
              <CheckCircle className=\"w-4 h-4 text-green-400\" />
              <span className=\"text-sm text-green-400\">관리자 모드</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className=\"border-b border-gray-700 bg-gray-900/30\">
        <div className=\"max-w-7xl mx-auto px-4\">
          <div className=\"flex space-x-8\">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-4 border-b-2 font-medium text-sm transition-colors \${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-gray-400 hover:text-gray-300'
                  }\`}
                >
                  <Icon className=\"w-4 h-4\" />
                  {tab.name}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className=\"max-w-7xl mx-auto px-4 py-8\">
        {activeTab === 'status' && (
          <div className=\"space-y-6\">
            <div>
              <h2 className=\"text-xl font-bold mb-2\">토큰 상태 모니터링</h2>
              <p className=\"text-gray-400 mb-6\">
                KIS API 토큰의 현재 상태와 유효성을 실시간으로 모니터링합니다.
              </p>
            </div>
            
            <div className=\"grid lg:grid-cols-2 gap-6\">
              <KISTokenStatus />
              
              <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6 space-y-4\">
                <h3 className=\"text-lg font-semibold flex items-center gap-2\">
                  <Activity className=\"w-5 h-5 text-blue-400\" />
                  토큰 관리 가이드
                </h3>
                
                <div className=\"space-y-4\">
                  <div className=\"bg-blue-900/20 border border-blue-500/30 rounded-lg p-4\">
                    <h4 className=\"font-medium text-blue-200 mb-2\">✅ 정상 상태</h4>
                    <ul className=\"text-sm text-blue-300 space-y-1\">
                      <li>• 토큰이 유효하고 24시간 이내에 생성됨</li>
                      <li>• API 연결이 정상적으로 작동</li>
                      <li>• 실시간 데이터 수신 가능</li>
                    </ul>
                  </div>
                  
                  <div className=\"bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-4\">
                    <h4 className=\"font-medium text-yellow-200 mb-2\">⚠️ 갱신 필요</h4>
                    <ul className=\"text-sm text-yellow-300 space-y-1\">
                      <li>• 토큰이 24시간 이상 경과</li>
                      <li>• 토큰 만료 시간이 임박</li>
                      <li>• '토큰 갱신' 버튼을 클릭하여 새로 발급</li>
                    </ul>
                  </div>
                  
                  <div className=\"bg-red-900/20 border border-red-500/30 rounded-lg p-4\">
                    <h4 className=\"font-medium text-red-200 mb-2\">❌ 오류 상태</h4>
                    <ul className=\"text-sm text-red-300 space-y-1\">
                      <li>• 환경변수 설정 확인 필요</li>
                      <li>• KIS API 키가 유효하지 않음</li>
                      <li>• 네트워크 연결 문제</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'config' && (
          <div className=\"space-y-6\">
            <div>
              <h2 className=\"text-xl font-bold mb-2\">환경 설정</h2>
              <p className=\"text-gray-400 mb-6\">
                KIS API 연동을 위한 환경변수와 설정을 관리합니다.
              </p>
            </div>

            <div className=\"grid lg:grid-cols-2 gap-6\">
              <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6\">
                <h3 className=\"text-lg font-semibold mb-4\">필수 환경변수</h3>
                
                <div className=\"space-y-4\">
                  <div>
                    <label className=\"block text-sm font-medium text-gray-300 mb-2\">
                      NEXT_PUBLIC_KIS_APP_KEY
                    </label>
                    <div className=\"flex items-center gap-2\">
                      <input
                        type=\"password\"
                        placeholder=\"KIS API 앱 키\"
                        className=\"flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500\"
                        defaultValue={process.env.NEXT_PUBLIC_KIS_APP_KEY || ''}
                        readOnly
                      />
                      <div className={`w-3 h-3 rounded-full \${
                        process.env.NEXT_PUBLIC_KIS_APP_KEY ? 'bg-green-400' : 'bg-red-400'
                      }\`} />
                    </div>
                  </div>

                  <div>
                    <label className=\"block text-sm font-medium text-gray-300 mb-2\">
                      NEXT_PUBLIC_KIS_APP_SECRET
                    </label>
                    <div className=\"flex items-center gap-2\">
                      <input
                        type=\"password\"
                        placeholder=\"KIS API 앱 시크릿\"
                        className=\"flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500\"
                        defaultValue={process.env.NEXT_PUBLIC_KIS_APP_SECRET || ''}
                        readOnly
                      />
                      <div className={`w-3 h-3 rounded-full \${
                        process.env.NEXT_PUBLIC_KIS_APP_SECRET ? 'bg-green-400' : 'bg-red-400'
                      }\`} />
                    </div>
                  </div>
                </div>

                <div className=\"mt-6 p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg\">
                  <div className=\"flex items-start gap-2\">
                    <AlertCircle className=\"w-4 h-4 text-yellow-400 mt-0.5\" />
                    <div>
                      <p className=\"text-sm text-yellow-200 font-medium\">보안 주의사항</p>
                      <p className=\"text-xs text-yellow-300 mt-1\">
                        환경변수는 .env.local 파일에 저장되며, Git에 커밋되지 않습니다.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6\">
                <h3 className=\"text-lg font-semibold mb-4\">설정 가이드</h3>
                
                <div className=\"space-y-4\">
                  <div className=\"border border-gray-600 rounded-lg p-4\">
                    <h4 className=\"font-medium mb-2\">1. KIS API 키 발급</h4>
                    <p className=\"text-sm text-gray-400 mb-3\">
                      한국투자증권 OpenAPI 포털에서 API 키를 발급받으세요.
                    </p>
                    <a
                      href=\"https://apiportal.koreainvestment.com\"
                      target=\"_blank\"
                      rel=\"noopener noreferrer\"
                      className=\"inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm\"
                    >
                      <ExternalLink className=\"w-4 h-4\" />
                      KIS OpenAPI 포털 바로가기
                    </a>
                  </div>

                  <div className=\"border border-gray-600 rounded-lg p-4\">
                    <h4 className=\"font-medium mb-2\">2. 환경변수 설정</h4>
                    <p className=\"text-sm text-gray-400 mb-3\">
                      프로젝트 루트에 .env.local 파일을 생성하고 키를 입력하세요.
                    </p>
                    <div className=\"bg-gray-900 p-3 rounded text-xs font-mono\">
                      <div>NEXT_PUBLIC_KIS_APP_KEY=your_app_key_here</div>
                      <div>NEXT_PUBLIC_KIS_APP_SECRET=your_app_secret_here</div>
                    </div>
                  </div>

                  <div className=\"border border-gray-600 rounded-lg p-4\">
                    <h4 className=\"font-medium mb-2\">3. 서버 재시작</h4>
                    <p className=\"text-sm text-gray-400\">
                      환경변수 변경 후 개발 서버를 재시작하세요.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'test' && (
          <div className=\"space-y-6\">
            <div>
              <h2 className=\"text-xl font-bold mb-2\">API 연결 테스트</h2>
              <p className=\"text-gray-400 mb-6\">
                KIS API 연결 상태와 데이터 수신을 테스트합니다.
              </p>
            </div>

            <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6\">
              <div className=\"text-center py-12\">
                <Activity className=\"w-16 h-16 mx-auto mb-4 text-blue-400\" />
                <h3 className=\"text-xl font-bold mb-2\">API 테스트 도구</h3>
                <p className=\"text-gray-400 mb-6\">
                  이 기능은 개발 중입니다. 곧 제공될 예정입니다.
                </p>
                <div className=\"flex justify-center gap-4\">
                  <Link
                    href=\"/modules/stock-analysis/tools/pro-trading-chart\"
                    className=\"px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors\"
                  >
                    차트에서 실제 테스트
                  </Link>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'docs' && (
          <div className=\"space-y-6\">
            <div>
              <h2 className=\"text-xl font-bold mb-2\">API 문서</h2>
              <p className=\"text-gray-400 mb-6\">
                KIS API 사용 방법과 참고 자료입니다.
              </p>
            </div>

            <div className=\"grid lg:grid-cols-2 gap-6\">
              <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6\">
                <h3 className=\"text-lg font-semibold mb-4\">공식 문서</h3>
                
                <div className=\"space-y-3\">
                  <a
                    href=\"https://apiportal.koreainvestment.com/apiservice/apiservice-domestic-stock\"
                    target=\"_blank\"
                    rel=\"noopener noreferrer\"
                    className=\"flex items-center justify-between p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors\"
                  >
                    <div>
                      <div className=\"font-medium\">국내주식 API</div>
                      <div className=\"text-sm text-gray-400\">주식 시세, 차트 데이터</div>
                    </div>
                    <ExternalLink className=\"w-4 h-4 text-gray-400\" />
                  </a>

                  <a
                    href=\"https://apiportal.koreainvestment.com/apiservice/oauth2\"
                    target=\"_blank\"
                    rel=\"noopener noreferrer\"
                    className=\"flex items-center justify-between p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors\"
                  >
                    <div>
                      <div className=\"font-medium\">OAuth2 인증</div>
                      <div className=\"text-sm text-gray-400\">토큰 발급 및 갱신</div>
                    </div>
                    <ExternalLink className=\"w-4 h-4 text-gray-400\" />
                  </a>

                  <a
                    href=\"https://apiportal.koreainvestment.com/intro\"
                    target=\"_blank\"
                    rel=\"noopener noreferrer\"
                    className=\"flex items-center justify-between p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors\"
                  >
                    <div>
                      <div className=\"font-medium\">시작 가이드</div>
                      <div className=\"text-sm text-gray-400\">API 사용 방법</div>
                    </div>
                    <ExternalLink className=\"w-4 h-4 text-gray-400\" />
                  </a>
                </div>
              </div>

              <div className=\"bg-gray-800/50 border border-gray-700 rounded-lg p-6\">
                <h3 className=\"text-lg font-semibold mb-4\">내부 구현</h3>
                
                <div className=\"space-y-3\">
                  <div className=\"p-3 bg-gray-700 rounded-lg\">
                    <div className=\"font-medium mb-1\">KISTokenManager</div>
                    <div className=\"text-sm text-gray-400\">
                      토큰 자동 관리 및 갱신
                    </div>
                    <div className=\"text-xs text-blue-400 mt-1\">
                      /src/lib/auth/kis-token-manager.ts
                    </div>
                  </div>

                  <div className=\"p-3 bg-gray-700 rounded-lg\">
                    <div className=\"font-medium mb-1\">KISApiService</div>
                    <div className=\"text-sm text-gray-400\">
                      주식 데이터 조회 서비스
                    </div>
                    <div className=\"text-xs text-blue-400 mt-1\">
                      /src/lib/services/kis-api-service.ts
                    </div>
                  </div>

                  <div className=\"p-3 bg-gray-700 rounded-lg\">
                    <div className=\"font-medium mb-1\">KISTokenStatus</div>
                    <div className=\"text-sm text-gray-400\">
                      토큰 상태 모니터링 UI
                    </div>
                    <div className=\"text-xs text-blue-400 mt-1\">
                      /src/components/charts/ProChart/KISTokenStatus.tsx
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}