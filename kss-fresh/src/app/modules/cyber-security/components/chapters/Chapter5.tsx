import React from 'react';
import { Shield, Lock, Eye, AlertTriangle, Code, Network } from 'lucide-react';
import References from '../References';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          제로트러스트 보안 (Zero Trust Security)
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          NIST SP 800-207 기반 제로트러스트 아키텍처 실전 구현
        </p>
      </div>

      {/* 2024-2025 제로트러스트 도입 트렌드 */}
      <section className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <AlertTriangle className="w-7 h-7" />
          2024-2025 제로트러스트 도입 트렌드
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">📈 글로벌 도입률 급증</h3>
            <p className="text-sm mb-2">
              2024년 기준 72%의 기업이 제로트러스트 전략 채택 중 (2021년 대비 +45%)
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Cybersecurity Insiders Zero Trust Report 2024
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🏢 정부 규제 강화</h3>
            <p className="text-sm mb-2">
              미국 연방정부 EO 14028: 2024년까지 모든 연방 기관 제로트러스트 의무화
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: U.S. Executive Order 14028 (2021)
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">☁️ 클라우드 제로트러스트</h3>
            <p className="text-sm mb-2">
              Cloudflare, Zscaler, Palo Alto 등 ZTNA 솔루션 시장 연 35% 성장
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              시장 규모: $27.4B (2024) → $60.7B (2028)
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🤖 AI 기반 제로트러스트</h3>
            <p className="text-sm mb-2">
              머신러닝 기반 행동 분석, 위험 기반 인증 (Risk-Based Authentication)
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              사용자 행동 분석(UEBA) 통합
            </p>
          </div>
        </div>
      </section>

      {/* NIST SP 800-207 제로트러스트 아키텍처 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-7 h-7 text-indigo-600" />
          NIST SP 800-207 제로트러스트 아키텍처
        </h2>

        <div className="mb-6 bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border-l-4 border-indigo-500">
          <h3 className="font-bold text-indigo-900 dark:text-indigo-300 mb-2">📘 NIST 정의</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            "리소스 보호를 위해 네트워크 위치에 대한 암묵적 신뢰를 제거하고, 모든 액세스 요청을 검증하는 사이버 보안 패러다임"
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300">
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">
              ❌ 전통적 경계 기반 보안 (Perimeter Security)
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-red-600">✗</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>암묵적 신뢰</strong>: 내부 네트워크 = 신뢰
                </p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-600">✗</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>수평 이동 가능</strong>: 한 번 침투 시 전체 네트워크 접근
                </p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-600">✗</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>VPN = 전체 액세스</strong>: VPN 연결만으로 모든 리소스 접근
                </p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ 제로트러스트 아키텍처
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>명시적 검증</strong>: 모든 요청마다 검증
                </p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>Micro-Segmentation</strong>: 리소스별 격리
                </p>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>JIT/JEA</strong>: Just-In-Time, Just-Enough-Access
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-blue-900 dark:text-blue-300 mb-2">🏗️ NIST 제로트러스트 7가지 핵심 원칙</h4>
          <div className="grid md:grid-cols-2 gap-2 text-sm">
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>1. Never Trust, Always Verify</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">네트워크 위치 무관 검증</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>2. Least Privilege Access</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">최소 권한 원칙</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>3. Assume Breach</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">침해 전제 설계</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>4. Verify Explicitly</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">모든 데이터 포인트 검증</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>5. Micro-Segmentation</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">세밀한 네트워크 분할</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>6. Continuous Monitoring</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">실시간 모니터링</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded col-span-2">
              <strong>7. Context-Aware Policy Enforcement</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">컨텍스트 기반 정책 (시간, 위치, 디바이스, 행동)</p>
            </div>
          </div>
        </div>
      </section>

      {/* Cloudflare Zero Trust 실전 구현 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-orange-600" />
          Cloudflare Zero Trust 실전 구현
        </h2>

        <div className="mb-6 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
          <h3 className="font-bold text-orange-900 dark:text-orange-300 mb-2">🌐 Cloudflare Zero Trust</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
            글로벌 에지 네트워크를 활용한 ZTNA 솔루션 (무료 플랜 50명까지)
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            사용 기업: Shopify, Canva, Discord 등 / 310개 도시 CDN 활용
          </p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
          <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
            ✅ Cloudflare Access Policy 설정 (Terraform)
          </h3>
          <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`# 1. Cloudflare Access Application 생성
resource "cloudflare_access_application" "internal_app" {
  zone_id          = var.cloudflare_zone_id
  name             = "Internal Dashboard"
  domain           = "dashboard.example.com"
  type             = "self_hosted"
  session_duration = "24h"

  # CORS 설정
  cors_headers {
    allowed_origins = ["https://example.com"]
    allow_all_headers = true
    allow_credentials = true
  }
}

# 2. Access Policy (다중 조건)
resource "cloudflare_access_policy" "dashboard_policy" {
  application_id = cloudflare_access_application.internal_app.id
  zone_id        = var.cloudflare_zone_id
  name           = "Allow Engineering Team"
  precedence     = 1
  decision       = "allow"

  # 조건 1: Okta SSO 인증
  include {
    okta {
      name                  = var.okta_name
      identity_provider_id  = var.okta_provider_id
    }
  }

  # 조건 2: 이메일 도메인 제한
  include {
    email_domain {
      domain = "example.com"
    }
  }

  # 조건 3: 국가 제한
  include {
    geo {
      country_code = ["US", "KR", "JP"]
    }
  }

  # 제외 조건: 특정 이메일
  exclude {
    email {
      email = "contractor@external.com"
    }
  }

  # 추가 요구사항: MFA 필수
  require {
    mfa = true
  }
}

# 3. Device Posture Check (디바이스 검증)
resource "cloudflare_device_posture_rule" "corporate_device" {
  account_id = var.cloudflare_account_id
  name       = "Corporate Device Check"
  type       = "os_version"

  match {
    platform = "mac"
  }

  input {
    version         = "13.0"
    operator        = ">="
    os_distro_name  = "macOS"
  }
}`}
          </pre>
        </div>

        <div className="mt-4 grid md:grid-cols-3 gap-3">
          <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded-lg">
            <h4 className="font-bold text-orange-900 dark:text-orange-300 mb-2 text-sm">📱 Cloudflare WARP</h4>
            <p className="text-xs text-gray-700 dark:text-gray-300">
              클라이언트 앱 (무료) - VPN 없이 Zero Trust 네트워크 접근
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
            <h4 className="font-bold text-blue-900 dark:text-blue-300 mb-2 text-sm">🔐 Cloudflare Gateway</h4>
            <p className="text-xs text-gray-700 dark:text-gray-300">
              DNS 필터링 + 보안 웹 게이트웨이 (Secure Web Gateway)
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
            <h4 className="font-bold text-purple-900 dark:text-purple-300 mb-2 text-sm">🏗️ Cloudflare Tunnel</h4>
            <p className="text-xs text-gray-700 dark:text-gray-300">
              인바운드 포트 개방 없이 애플리케이션 노출 (cloudflared)
            </p>
          </div>
        </div>
      </section>

      {/* ZTNA 주요 솔루션 비교 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Network className="w-7 h-7 text-purple-600" />
          ZTNA 주요 솔루션 비교
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-purple-50 dark:bg-purple-900/20">
              <tr>
                <th className="px-4 py-3 text-left font-bold text-purple-900 dark:text-purple-300">솔루션</th>
                <th className="px-4 py-3 text-left font-bold text-purple-900 dark:text-purple-300">특징</th>
                <th className="px-4 py-3 text-left font-bold text-purple-900 dark:text-purple-300">가격</th>
                <th className="px-4 py-3 text-left font-bold text-purple-900 dark:text-purple-300">적합한 기업</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr className="bg-white dark:bg-gray-800">
                <td className="px-4 py-3 font-semibold">Cloudflare Zero Trust</td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  310개 도시 글로벌 네트워크<br />
                  <span className="text-xs">무료 플랜, 쉬운 설정</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  $7/user/월<br />
                  <span className="text-xs text-green-600">(50명까지 무료)</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">스타트업, 중소기업</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-900/50">
                <td className="px-4 py-3 font-semibold">Zscaler ZPA</td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  150개국 데이터센터<br />
                  <span className="text-xs">업계 선두, 엔터프라이즈</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  문의 필요<br />
                  <span className="text-xs">(최소 $15/user/월 추정)</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">대기업, 금융권</td>
              </tr>
              <tr className="bg-white dark:bg-gray-800">
                <td className="px-4 py-3 font-semibold">Palo Alto Prisma Access</td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  SASE 통합 플랫폼<br />
                  <span className="text-xs">ZTNA + SWG + CASB</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  문의 필요<br />
                  <span className="text-xs">(고가, 엔터프라이즈)</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">대기업, 글로벌</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-900/50">
                <td className="px-4 py-3 font-semibold">Tailscale</td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  WireGuard 기반<br />
                  <span className="text-xs">개발자 친화적, P2P</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  $6/user/월<br />
                  <span className="text-xs text-green-600">(100 디바이스 무료)</span>
                </td>
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">개발팀, 스타트업</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border-l-4 border-yellow-500">
          <h4 className="font-bold text-yellow-900 dark:text-yellow-300 mb-2">💡 선택 가이드</h4>
          <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
            <div>
              <p className="font-semibold mb-1">Cloudflare/Tailscale</p>
              <ul className="ml-4 space-y-1 text-xs">
                <li>• 예산 제한적</li>
                <li>• 빠른 구축 필요</li>
                <li>• IT 팀 소규모</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold mb-1">Zscaler</p>
              <ul className="ml-4 space-y-1 text-xs">
                <li>• 글로벌 지사 多</li>
                <li>• 높은 트래픽</li>
                <li>• 금융/의료 규제</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold mb-1">Palo Alto Prisma</p>
              <ul className="ml-4 space-y-1 text-xs">
                <li>• SASE 통합 필요</li>
                <li>• 기존 Palo Alto 고객</li>
                <li>• 최고 수준 보안</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 제로트러스트 프레임워크',
            icon: 'web' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'NIST SP 800-207 - Zero Trust Architecture',
                url: 'https://csrc.nist.gov/publications/detail/sp/800-207/final',
                description: '미국 표준기술연구소 공식 제로트러스트 아키텍처 정의 (2020)'
              },
              {
                title: 'CISA Zero Trust Maturity Model',
                url: 'https://www.cisa.gov/zero-trust-maturity-model',
                description: '미국 사이버안보국(CISA)의 제로트러스트 성숙도 모델 v2.0'
              },
              {
                title: 'Google BeyondCorp Papers',
                url: 'https://cloud.google.com/beyondcorp',
                description: 'Google의 제로트러스트 실전 구현 사례 (2011-2019 논문 시리즈)'
              },
              {
                title: 'Microsoft Zero Trust Deployment Guide',
                url: 'https://learn.microsoft.com/en-us/security/zero-trust/',
                description: 'Azure AD, Conditional Access 기반 제로트러스트 구현 가이드'
              }
            ]
          },
          {
            title: '🔧 ZTNA 솔루션',
            icon: 'tools' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Cloudflare Zero Trust',
                url: 'https://developers.cloudflare.com/cloudflare-one/',
                description: '무료 플랜 제공 / 50명까지 무료 / Terraform Provider 지원'
              },
              {
                title: 'Tailscale',
                url: 'https://tailscale.com/',
                description: 'WireGuard 기반 / 개발자 친화적 / 100 디바이스 무료'
              },
              {
                title: 'Zscaler Private Access (ZPA)',
                url: 'https://www.zscaler.com/products/zscaler-private-access',
                description: '업계 1위 ZTNA 솔루션 / 엔터프라이즈급'
              },
              {
                title: 'Palo Alto Prisma Access',
                url: 'https://www.paloaltonetworks.com/sase/access',
                description: 'SASE 통합 플랫폼 / ZTNA + SWG + CASB'
              }
            ]
          },
          {
            title: '📖 제로트러스트 구현 리소스',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Terraform Cloudflare Provider - Zero Trust',
                url: 'https://registry.terraform.io/providers/cloudflare/cloudflare/latest/docs/resources/access_application',
                description: 'Cloudflare Access 인프라 코드화 (IaC) 공식 문서'
              },
              {
                title: 'NIST Zero Trust Architecture Use Cases',
                url: 'https://www.nccoe.nist.gov/projects/implementing-zero-trust-architecture',
                description: '제로트러스트 실제 구현 사례 및 가이드'
              },
              {
                title: 'Gartner Magic Quadrant for ZTNA',
                url: 'https://www.gartner.com/en/documents/4018023',
                description: 'ZTNA 벤더 비교 및 평가 보고서 (연간 업데이트)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
