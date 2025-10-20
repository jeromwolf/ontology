import React from 'react';
import { Network, Shield, AlertCircle, Lock, TrendingUp, Code, ExternalLink } from 'lucide-react';

// References 컴포넌트
interface ReferenceItem {
  title: string;
  url: string;
  description: string;
}

interface ReferenceSection {
  title: string;
  icon: 'web' | 'research' | 'tools';
  color: string;
  items: ReferenceItem[];
}

function References({ sections }: { sections: ReferenceSection[] }) {
  const getIcon = (type: string) => {
    switch (type) {
      case 'web': return <ExternalLink className="w-4 h-4" />;
      case 'research': return <TrendingUp className="w-4 h-4" />;
      case 'tools': return <Code className="w-4 h-4" />;
      default: return null;
    }
  };

  return (
    <section className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
        📚 References & Further Reading
      </h2>

      <div className="space-y-6">
        {sections.map((section, idx) => (
          <div key={idx} className={`border-l-4 ${section.color} bg-white dark:bg-gray-800 rounded-r-lg p-4`}>
            <h3 className="font-bold text-lg mb-3 text-gray-900 dark:text-white flex items-center gap-2">
              {getIcon(section.icon)}
              {section.title}
            </h3>
            <ul className="space-y-2">
              {section.items.map((item, i) => (
                <li key={i} className="text-sm">
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                  >
                    {item.title}
                  </a>
                  <p className="text-gray-600 dark:text-gray-400 text-xs mt-1">{item.description}</p>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          네트워크 보안
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          네트워크 공격 유형과 방어 기법을 학습합니다
        </p>
      </div>

      {/* 2024-2025 최신 네트워크 위협 트렌드 */}
      <section className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6 shadow-lg border-l-4 border-red-500">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-red-600" />
          2024-2025 최신 네트워크 위협 트렌드
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold text-lg mb-2 text-red-900 dark:text-red-300">
              🔥 DDoS 공격의 진화
            </h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>17.2 million requests/sec</strong> - 역대 최대 HTTP DDoS 공격 (Cloudflare, 2024)</li>
              <li>• <strong>Mirai 봇넷 변종</strong> - IoT 기기 3,900만 대 감염 (2025)</li>
              <li>• <strong>Application-layer DDoS</strong> - L7 공격 150% 증가 (Imperva, 2024)</li>
              <li>• <strong>Ransom DDoS (RDDoS)</strong> - 비트코인 요구 공격 급증</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold text-lg mb-2 text-orange-900 dark:text-orange-300">
              🌐 5G/6G 네트워크 취약점
            </h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>Diameter/GTP 프로토콜 공격</strong> - 코어 네트워크 침투</li>
              <li>• <strong>Network Slicing 악용</strong> - 가상 네트워크 격리 우회</li>
              <li>• <strong>eSIM 하이재킹</strong> - 원격 SIM 프로파일 탈취</li>
              <li>• <strong>Edge Computing 공격면</strong> - MEC 서버 노출</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              🔐 VPN 취약점 악용
            </h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>CVE-2024-21887</strong> - Ivanti Connect Secure RCE (CVSS 9.1)</li>
              <li>• <strong>Pulse Secure 제로데이</strong> - APT29 그룹 악용 사례</li>
              <li>• <strong>VPN 크리덴셜 탈취</strong> - 다크웹 거래 200% 증가 (2024)</li>
              <li>• <strong>SSL VPN MitM</strong> - TLS 인터셉션 공격</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              📡 DNS over HTTPS (DoH) 악용
            </h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>C2 통신 우회</strong> - 방화벽/IDS 탐지 회피</li>
              <li>• <strong>Exfiltration via DoH</strong> - 데이터 유출 채널</li>
              <li>• <strong>악성코드 DoH 활용</strong> - 75% 이상 탐지 우회 (Cisco, 2024)</li>
              <li>• <strong>Split-Horizon DNS 공격</strong> - 내부/외부 DNS 불일치 악용</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 bg-yellow-100 dark:bg-yellow-900/30 p-4 rounded-lg border-l-4 border-yellow-500">
          <p className="text-sm text-gray-800 dark:text-gray-200">
            <strong>📊 통계:</strong> 2024년 기업의 <strong>87%</strong>가 네트워크 기반 공격 경험 (Fortinet, 2024).
            평균 네트워크 침해 탐지 시간 <strong>212일</strong> (IBM X-Force, 2024).
          </p>
        </div>
      </section>

      {/* OSI 7계층별 보안 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Network className="w-7 h-7 text-blue-600" />
          OSI 7계층별 보안 위협
        </h2>

        <div className="space-y-3">
          {[
            { layer: '7. 응용 계층', threats: 'SQL Injection, XSS, CSRF, 파일 업로드 취약점' },
            { layer: '6. 표현 계층', threats: 'SSL/TLS 취약점, 암호화 공격' },
            { layer: '5. 세션 계층', threats: '세션 하이재킹, Man-in-the-Middle' },
            { layer: '4. 전송 계층', threats: 'SYN Flooding, Port Scanning' },
            { layer: '3. 네트워크 계층', threats: 'IP Spoofing, ICMP Flooding, Routing 공격' },
            { layer: '2. 데이터링크 계층', threats: 'ARP Spoofing, MAC Flooding' },
            { layer: '1. 물리 계층', threats: '도청, 케이블 절단, 전파 방해' },
          ].map((item, idx) => (
            <div key={idx} className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-bold text-blue-900 dark:text-blue-300">{item.layer}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{item.threats}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 방화벽 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-7 h-7 text-green-600" />
          방화벽 (Firewall) - 실전 설정
        </h2>

        <div className="space-y-6">
          {/* 1. Linux iptables 실전 예제 */}
          <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border-2 border-green-400">
            <h3 className="font-bold text-xl mb-3 text-green-900 dark:text-green-300 flex items-center gap-2">
              <Code className="w-5 h-5" />
              1. Linux iptables - 패킷 필터링
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              가장 널리 사용되는 리눅스 방화벽 설정 (Netfilter 기반)
            </p>

            <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
              <pre className="text-green-400 text-xs font-mono">
{`# 1. 기본 정책: 모든 INPUT 차단, OUTPUT 허용
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 2. Loopback 허용 (필수)
iptables -A INPUT -i lo -j ACCEPT

# 3. 이미 연결된 세션 허용 (Stateful)
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# 4. SSH (22) - 특정 IP만 허용
iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 22 -j ACCEPT

# 5. HTTP/HTTPS (80/443) - 전체 허용
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT

# 6. ICMP Ping 제한 (초당 1개)
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT

# 7. SYN Flood 방어
iptables -A INPUT -p tcp --syn -m limit --limit 10/s -j ACCEPT

# 8. Port Scan 차단 (nmap 방어)
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP

# 9. 로깅 (차단된 패킷)
iptables -A INPUT -j LOG --log-prefix "IPTABLES-DROPPED: "

# 10. 설정 저장 (재부팅 후에도 유지)
iptables-save > /etc/iptables/rules.v4`}
              </pre>
            </div>

            <div className="mt-3 bg-yellow-100 dark:bg-yellow-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>⚠️ 주의:</strong> SSH 규칙을 잘못 설정하면 원격 서버 접속이 차단될 수 있습니다.
              반드시 콘솔 접근 가능한 환경에서 테스트하세요.
            </div>
          </div>

          {/* 2. pfSense 웹 방화벽 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border-2 border-blue-400">
            <h3 className="font-bold text-xl mb-3 text-blue-900 dark:text-blue-300 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              2. pfSense - 엔터프라이즈급 방화벽
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              GUI 기반 오픈소스 방화벽 (FreeBSD 기반, Fortune 500 기업 사용)
            </p>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-blue-800 dark:text-blue-400">주요 기능</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>Stateful Firewall</strong> - 연결 추적</li>
                  <li>✓ <strong>NAT/PAT</strong> - 포트 포워딩, 1:1 NAT</li>
                  <li>✓ <strong>VPN</strong> - IPsec, OpenVPN, WireGuard</li>
                  <li>✓ <strong>Traffic Shaping</strong> - QoS, 대역폭 제어</li>
                  <li>✓ <strong>IDS/IPS</strong> - Snort/Suricata 통합</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-blue-800 dark:text-blue-400">실전 사용 예시</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• DMZ 구성 (웹서버 격리)</li>
                  <li>• VPN 게이트웨이 (Site-to-Site)</li>
                  <li>• 멀티 WAN (이중화, 로드밸런싱)</li>
                  <li>• Guest WiFi 격리</li>
                  <li>• GeoIP 기반 차단</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-xs text-gray-700 dark:text-gray-300">
                <strong>🔧 설정 예시:</strong> Firewall → Rules → WAN → Add
                <br/>→ Action: Block, Protocol: TCP, Source: Any, Destination Port: 23 (Telnet 차단)
              </p>
            </div>
          </div>

          {/* 3. Next-Generation Firewall (NGFW) */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border-2 border-purple-400">
            <h3 className="font-bold text-xl mb-3 text-purple-900 dark:text-purple-300">
              3. NGFW - 차세대 방화벽
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              애플리케이션 계층 인식 + Deep Packet Inspection (DPI)
            </p>

            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Palo Alto Networks</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• App-ID (5,000+ 앱 인식)</li>
                  <li>• User-ID (사용자별 정책)</li>
                  <li>• WildFire (샌드박싱)</li>
                  <li>• SSL 복호화 (TLS 1.3)</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Cisco Firepower</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Snort 3 엔진 내장</li>
                  <li>• Talos 위협 인텔리전스</li>
                  <li>• URL 필터링 (80개 카테고리)</li>
                  <li>• Threat Grid 통합</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Fortinet FortiGate</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• FortiASIC (하드웨어 가속)</li>
                  <li>• SD-WAN 기능 내장</li>
                  <li>• FortiGuard Labs 위협 DB</li>
                  <li>• Security Fabric 통합</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 bg-indigo-100 dark:bg-indigo-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>💡 NGFW vs 전통 방화벽:</strong> 전통 방화벽은 포트 기반 차단만 가능하지만,
              NGFW는 "Facebook 메신저는 허용, 파일 전송은 차단" 같은 세밀한 정책 적용이 가능합니다.
            </div>
          </div>

          {/* 4. 클라우드 네이티브 방화벽 */}
          <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border-2 border-orange-400">
            <h3 className="font-bold text-xl mb-3 text-orange-900 dark:text-orange-300">
              4. 클라우드 방화벽 (AWS/Azure/GCP)
            </h3>

            <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
              <pre className="text-orange-400 text-xs font-mono">
{`# AWS Security Group (Terraform)
resource "aws_security_group" "web_server" {
  name        = "web-server-sg"
  description = "Allow HTTP/HTTPS inbound"

  # HTTPS 허용
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow HTTPS from internet"
  }

  # SSH - VPN IP만 허용
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.1.0/24"]
    description = "Allow SSH from VPN subnet"
  }

  # 모든 Outbound 허용
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "WebServerSG"
    Environment = "Production"
  }
}`}
              </pre>
            </div>

            <div className="mt-3 grid md:grid-cols-3 gap-2 text-xs">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-orange-700 dark:text-orange-400">AWS</strong>
                <p className="text-gray-600 dark:text-gray-400">Security Groups + Network ACL</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-400">Azure</strong>
                <p className="text-gray-600 dark:text-gray-400">NSG (Network Security Group)</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-red-700 dark:text-red-400">GCP</strong>
                <p className="text-gray-600 dark:text-gray-400">VPC Firewall Rules</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* IDS/IPS */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <AlertCircle className="w-7 h-7 text-red-600" />
          침입 탐지/차단 시스템 (IDS/IPS)
        </h2>

        <div className="space-y-6">
          {/* 1. Snort - 오픈소스 IDS */}
          <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border-2 border-orange-400">
            <h3 className="text-xl font-bold mb-3 text-orange-900 dark:text-orange-300 flex items-center gap-2">
              <Code className="w-5 h-5" />
              1. Snort - 실전 룰 작성
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              세계에서 가장 널리 사용되는 오픈소스 IDS (Cisco 소유, 600만+ 다운로드)
            </p>

            <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto mb-3">
              <pre className="text-orange-400 text-xs font-mono">
{`# Snort Rule 기본 구조
# [액션] [프로토콜] [소스IP] [소스포트] [방향] [목적지IP] [목적지포트] (룰 옵션)

# 1. SQL Injection 탐지
alert tcp any any -> $HOME_NET 80 (
  msg:"SQL Injection Attempt - UNION SELECT";
  flow:to_server,established;
  content:"UNION"; nocase;
  content:"SELECT"; nocase; distance:0;
  classtype:web-application-attack;
  sid:1000001; rev:1;
)

# 2. XSS 공격 탐지
alert tcp any any -> $HOME_NET 80 (
  msg:"XSS Attack - Script Tag Detected";
  flow:to_server,established;
  content:"<script"; nocase; http_uri;
  pcre:"/<script[^>]*>/i";
  classtype:web-application-attack;
  sid:1000002; rev:1;
)

# 3. Port Scan 탐지 (SYN Scan)
alert tcp any any -> $HOME_NET any (
  msg:"Possible SYN Port Scan";
  flags:S;
  detection_filter:track by_src, count 20, seconds 60;
  classtype:attempted-recon;
  sid:1000003; rev:1;
)

# 4. SSH Brute Force 탐지
alert tcp any any -> $HOME_NET 22 (
  msg:"SSH Brute Force Attempt";
  flow:to_server,established;
  content:"SSH-"; depth:4;
  detection_filter:track by_src, count 5, seconds 60;
  classtype:attempted-admin;
  sid:1000004; rev:1;
)

# 5. Command Injection 탐지
alert tcp any any -> $HOME_NET 80 (
  msg:"Command Injection - bash/sh execution";
  flow:to_server,established;
  content:"/bin/"; nocase; http_uri;
  pcre:"/(bash|sh|cmd|powershell)/i";
  classtype:web-application-attack;
  sid:1000005; rev:1;
)

# 6. DNS Tunneling 탐지 (과도한 TXT 레코드)
alert udp any any -> any 53 (
  msg:"Possible DNS Tunneling - Large TXT Query";
  content:"|00 10|"; offset:2; depth:2;
  dsize:>100;
  classtype:policy-violation;
  sid:1000006; rev:1;
)

# 7. Cryptocurrency Mining (Coinhive) 탐지
alert tcp any any -> any any (
  msg:"Cryptocurrency Mining Script Detected";
  flow:to_client,established;
  content:"coinhive.min.js"; nocase; http_uri;
  classtype:trojan-activity;
  sid:1000007; rev:1;
)`}
              </pre>
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-orange-800 dark:text-orange-400">Snort 3 신기능</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• <strong>Multi-threading</strong> - CPU 코어당 분산 처리</li>
                  <li>• <strong>Lua 스크립팅</strong> - 커스텀 탐지 로직</li>
                  <li>• <strong>Hyperscan 엔진</strong> - 정규식 가속</li>
                  <li>• <strong>AppID</strong> - 5,000+ 애플리케이션 탐지</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-orange-800 dark:text-orange-400">룰 옵션 설명</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">msg</code> - 경고 메시지</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">flow</code> - TCP 세션 상태</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">content</code> - 패턴 매칭</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">pcre</code> - 정규식</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">sid</code> - 룰 고유 ID</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 2. Suricata - 차세대 IDS/IPS */}
          <div className="bg-red-50 dark:bg-red-900/20 p-5 rounded-lg border-2 border-red-400">
            <h3 className="text-xl font-bold mb-3 text-red-900 dark:text-red-300">
              2. Suricata - 멀티스레드 IPS
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              OISF 재단의 고성능 IDS/IPS (Snort 룰 호환 + 추가 기능)
            </p>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-red-800 dark:text-red-400">핵심 기능</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>멀티코어 확장성</strong> - 100Gbps+ 처리 가능</li>
                  <li>✓ <strong>File Extraction</strong> - 악성파일 자동 추출</li>
                  <li>✓ <strong>TLS/JA3 지문</strong> - 암호화 트래픽 분석</li>
                  <li>✓ <strong>EVE JSON 로그</strong> - SIEM 통합 용이</li>
                  <li>✓ <strong>Lua Output</strong> - 커스텀 로깅</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-red-800 dark:text-red-400">실전 사용 사례</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• <strong>SELKS</strong> - Suricata + ELK 통합 플랫폼</li>
                  <li>• <strong>Security Onion</strong> - NSM 올인원 솔루션</li>
                  <li>• <strong>pfSense IDS</strong> - 오픈소스 방화벽 통합</li>
                  <li>• <strong>Cloud IDS</strong> - AWS VPC, Azure vNet</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 bg-gray-900 p-3 rounded-lg">
              <p className="text-xs font-mono text-red-400">
                # Suricata YAML 설정 예시<br/>
                af-packet:<br/>
                &nbsp;&nbsp;- interface: eth0<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;threads: 4<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;cluster-id: 99<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;defrag: yes<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;use-mmap: yes
              </p>
            </div>
          </div>

          {/* 3. Zeek (Bro) - 네트워크 분석 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border-2 border-blue-400">
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              3. Zeek (구 Bro) - 네트워크 보안 모니터 (NSM)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Berkeley 연구소 개발, 시그니처가 아닌 행위 기반 탐지
            </p>

            <div className="grid md:grid-cols-3 gap-2 text-xs">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-400">프로토콜 분석</strong>
                <p className="text-gray-600 dark:text-gray-400">HTTP, DNS, FTP, SSH, SSL 등 50+ 프로토콜</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-400">로그 생성</strong>
                <p className="text-gray-600 dark:text-gray-400">conn.log, dns.log, http.log 등 구조화 로그</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-400">스크립팅</strong>
                <p className="text-gray-600 dark:text-gray-400">Zeek 언어로 커스텀 탐지 로직</p>
              </div>
            </div>

            <div className="mt-3 bg-green-100 dark:bg-green-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>💡 Zeek vs Snort:</strong> Snort는 알려진 공격 탐지 (시그니처),
              Zeek는 이상 행위 탐지 (네트워크 전체 가시성). 실무에서는 두 도구를 함께 사용.
            </div>
          </div>

          {/* 4. 엔터프라이즈 IPS */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border-2 border-purple-400">
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              4. 엔터프라이즈 IPS 솔루션
            </h3>

            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Cisco Firepower NGIPS</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Talos 위협 인텔리전스</li>
                  <li>• AMP (Advanced Malware Protection)</li>
                  <li>• 100Gbps+ 처리량</li>
                  <li>• 가격: $50K+ (어플라이언스)</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Palo Alto Threat Prevention</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• WildFire 샌드박싱</li>
                  <li>• DNS Sinkhole</li>
                  <li>• Inline ML 모델</li>
                  <li>• 가격: $40K+ (라이선스)</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-purple-800 dark:text-purple-400">Fortinet FortiIPS</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• FortiASIC 하드웨어 가속</li>
                  <li>• FortiGuard Labs DB</li>
                  <li>• 10,000+ 시그니처</li>
                  <li>• 가격: $30K+ (번들)</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 bg-yellow-100 dark:bg-yellow-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>📊 시장 점유율 (2024):</strong> Cisco 28%, Palo Alto 22%, Fortinet 18%,
              Check Point 12%, 기타 20% (Gartner Magic Quadrant)
            </div>
          </div>
        </div>
      </section>

      {/* VPN */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Lock className="w-7 h-7 text-indigo-600" />
          VPN (Virtual Private Network) - 실전 구축
        </h2>

        <div className="space-y-6">
          {/* 1. WireGuard - 차세대 VPN */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-5 rounded-lg border-2 border-indigo-400">
            <h3 className="font-bold text-xl mb-3 text-indigo-900 dark:text-indigo-300 flex items-center gap-2">
              <Code className="w-5 h-5" />
              1. WireGuard - 차세대 VPN (Linux 커널 내장)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              기존 VPN 대비 10배 빠른 속도, 4,000줄의 코드로 감사 용이 (OpenVPN 100,000줄)
            </p>

            <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
              <pre className="text-indigo-400 text-xs font-mono">
{`# 서버 설정 (/etc/wireguard/wg0.conf)
[Interface]
Address = 10.0.0.1/24
ListenPort = 51820
PrivateKey = <서버 비공개키>
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT

# 클라이언트 1 (Alice)
[Peer]
PublicKey = <Alice 공개키>
AllowedIPs = 10.0.0.2/32

# 클라이언트 2 (Bob)
[Peer]
PublicKey = <Bob 공개키>
AllowedIPs = 10.0.0.3/32

# 서버 실행
wg-quick up wg0
wg show  # 상태 확인

# 클라이언트 설정 (/etc/wireguard/wg0-client.conf)
[Interface]
Address = 10.0.0.2/24
PrivateKey = <클라이언트 비공개키>
DNS = 1.1.1.1

[Peer]
PublicKey = <서버 공개키>
Endpoint = vpn.example.com:51820
AllowedIPs = 0.0.0.0/0  # 모든 트래픽 라우팅 (Full Tunnel)
PersistentKeepalive = 25`}
              </pre>
            </div>

            <div className="mt-3 bg-green-100 dark:bg-green-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>✅ WireGuard 장점:</strong> ChaCha20 암호화, Curve25519 키 교환,
              NAT 자동 통과, 모바일 네트워크 전환 시 재연결 불필요 (Roaming Support)
            </div>
          </div>

          {/* 2. IPsec Site-to-Site */}
          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border-2 border-blue-400">
            <h3 className="font-bold text-xl mb-3 text-blue-900 dark:text-blue-300">
              2. IPsec Site-to-Site VPN (strongSwan)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              본사(HQ) - 지사(Branch) 네트워크 연결 (L3 IPsec Tunnel)
            </p>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-blue-800 dark:text-blue-400">본사 (HQ) - 200.1.1.1</h4>
                <div className="text-xs font-mono text-gray-700 dark:text-gray-300">
                  <p>Local Network: 192.168.10.0/24</p>
                  <p>Remote Network: 192.168.20.0/24</p>
                  <p>Tunnel Mode: IKEv2</p>
                  <p>Encryption: AES-256-GCM</p>
                  <p>Authentication: PSK</p>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-blue-800 dark:text-blue-400">지사 (Branch) - 200.2.2.2</h4>
                <div className="text-xs font-mono text-gray-700 dark:text-gray-300">
                  <p>Local Network: 192.168.20.0/24</p>
                  <p>Remote Network: 192.168.10.0/24</p>
                  <p>Tunnel Mode: IKEv2</p>
                  <p>Encryption: AES-256-GCM</p>
                  <p>Authentication: PSK</p>
                </div>
              </div>
            </div>

            <div className="mt-3 bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-xs text-gray-700 dark:text-gray-300">
                <strong>🔧 설정 예시 (ipsec.conf):</strong><br/>
                conn hq-to-branch<br/>
                &nbsp;&nbsp;left=200.1.1.1<br/>
                &nbsp;&nbsp;leftsubnet=192.168.10.0/24<br/>
                &nbsp;&nbsp;right=200.2.2.2<br/>
                &nbsp;&nbsp;rightsubnet=192.168.20.0/24<br/>
                &nbsp;&nbsp;ike=aes256-sha256-modp2048!<br/>
                &nbsp;&nbsp;esp=aes256gcm128-modp2048!<br/>
                &nbsp;&nbsp;keyexchange=ikev2<br/>
                &nbsp;&nbsp;auto=start
              </p>
            </div>
          </div>

          {/* 3. OpenVPN */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border-2 border-purple-400">
            <h3 className="font-bold text-xl mb-3 text-purple-900 dark:text-purple-300">
              3. OpenVPN - Remote Access VPN
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              SSL/TLS 기반, 방화벽 우회 용이 (TCP 443 사용 가능)
            </p>

            <div className="grid md:grid-cols-3 gap-3 text-xs">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold mb-2 text-purple-800 dark:text-purple-400">인증 방식</h4>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• <strong>인증서 (X.509)</strong> - EasyRSA, 가장 안전</li>
                  <li>• <strong>ID/PW</strong> - LDAP/AD 연동</li>
                  <li>• <strong>OTP</strong> - Google Authenticator</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold mb-2 text-purple-800 dark:text-purple-400">주요 설정</h4>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">dev tun</code> - Layer 3 VPN</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">proto udp</code> - 프로토콜</li>
                  <li>• <code className="bg-gray-200 dark:bg-gray-700 px-1">cipher AES-256-GCM</code> - 암호화</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold mb-2 text-purple-800 dark:text-purple-400">클라이언트</h4>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Windows - OpenVPN GUI</li>
                  <li>• macOS - Tunnelblick</li>
                  <li>• Android/iOS - OpenVPN Connect</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 bg-yellow-100 dark:bg-yellow-900/30 p-3 rounded text-xs text-gray-800 dark:text-gray-200">
              <strong>📊 사용률:</strong> WireGuard (신규 배포 60%), OpenVPN (레거시 30%), IPsec (엔터프라이즈 10%)
            </div>
          </div>

          {/* 4. 엔터프라이즈 VPN 솔루션 */}
          <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border-2 border-orange-400">
            <h3 className="font-bold text-xl mb-3 text-orange-900 dark:text-orange-300">
              4. 엔터프라이즈 VPN 솔루션
            </h3>

            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-orange-800 dark:text-orange-400">Cisco AnyConnect</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• SSL/IPsec 듀얼 모드</li>
                  <li>• ASA/FTD 통합</li>
                  <li>• Posture 검사 (백신, 패치)</li>
                  <li>• 가격: $150/user/year</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-orange-800 dark:text-orange-400">Palo Alto GlobalProtect</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• HIP (Host Information Profile)</li>
                  <li>• 위협 방지 통합</li>
                  <li>• IPv6 지원</li>
                  <li>• 가격: $120/user/year</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-bold text-sm mb-2 text-orange-800 dark:text-orange-400">Fortinet FortiClient</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• SSL-VPN + IPsec</li>
                  <li>• 엔드포인트 보안 통합</li>
                  <li>• Zero Trust 지원</li>
                  <li>• 가격: $90/user/year</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 문서 & 가이드',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'NIST SP 800-41 Rev. 1 - Guidelines on Firewalls and Firewall Policy',
                url: 'https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final',
                description: '미국 표준기술연구소(NIST)의 방화벽 정책 가이드라인 (2009, 실무 표준)'
              },
              {
                title: 'Snort 3 User Manual',
                url: 'https://docs.snort.org/',
                description: 'Snort 3.x 공식 문서 - 룰 작성, 성능 튜닝, 배포 가이드'
              },
              {
                title: 'Suricata Documentation',
                url: 'https://suricata.readthedocs.io/',
                description: 'Suricata 공식 문서 - EVE JSON, Lua 스크립팅, 멀티 스레딩'
              },
              {
                title: 'pfSense Official Documentation',
                url: 'https://docs.netgate.com/pfsense/',
                description: 'pfSense 완전 가이드 - NAT, VPN, IDS/IPS 통합 설정'
              },
              {
                title: 'WireGuard Whitepaper',
                url: 'https://www.wireguard.com/papers/wireguard.pdf',
                description: 'Jason A. Donenfeld의 WireGuard 프로토콜 논문 (2017)'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'DDoS Attacks in 2024: Trends and Mitigation - Cloudflare Research',
                url: 'https://blog.cloudflare.com/ddos-threat-report-2024-q1',
                description: '2024년 DDoS 공격 트렌드 분석 - 17.2M req/sec 최대 공격 기록'
              },
              {
                title: '5G Security Architecture (3GPP TS 33.501)',
                url: 'https://www.3gpp.org/ftp/Specs/archive/33_series/33.501/',
                description: '5G 네트워크 보안 아키텍처 표준 - SEAF, AUSF, ARPF 인증'
              },
              {
                title: 'DNS over HTTPS (DoH) - RFC 8484',
                url: 'https://datatracker.ietf.org/doc/html/rfc8484',
                description: 'IETF 표준 DoH 프로토콜 - 프라이버시 및 보안 고려사항'
              },
              {
                title: 'Zeek (Bro) Network Security Monitor - Academic Papers',
                url: 'https://zeek.org/documentation/',
                description: 'Berkeley 연구소의 네트워크 보안 모니터링 프레임워크 연구'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Security Onion - Free NSM Platform',
                url: 'https://securityonion.net/',
                description: 'Suricata, Zeek, Elasticsearch 통합 플랫폼 (Ubuntu 기반)'
              },
              {
                title: 'Wireshark User Guide',
                url: 'https://www.wireshark.org/docs/wsug_html_chunked/',
                description: '패킷 분석 필수 도구 - 필터 문법, 프로토콜 디코딩'
              },
              {
                title: 'tcpdump Manual & Examples',
                url: 'https://www.tcpdump.org/manpages/tcpdump.1.html',
                description: '커맨드라인 패킷 캡처 - BPF 필터, pcap 포맷'
              },
              {
                title: 'iptables Tutorial 1.2.2',
                url: 'https://www.frozentux.net/iptables-tutorial/iptables-tutorial.html',
                description: 'iptables 완전 가이드 - 체인, 타겟, 매칭 룰 (Frozentux.net)'
              },
              {
                title: 'WireGuard Setup Guide (DigitalOcean)',
                url: 'https://www.digitalocean.com/community/tutorials/how-to-set-up-wireguard-on-ubuntu-20-04',
                description: 'Ubuntu 20.04 WireGuard 설치 및 설정 실습 가이드'
              },
              {
                title: 'Cisco Talos Intelligence',
                url: 'https://talosintelligence.com/',
                description: 'Cisco의 위협 인텔리전스 - 무료 Snort 룰, IOC 제공'
              }
            ]
          }
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          📌 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• <strong>네트워크 보안 3대 핵심:</strong> 방화벽 (경계 방어) + IDS/IPS (침입 탐지/차단) + VPN (안전한 통신)</li>
          <li>• <strong>2024-2025 트렌드:</strong> DDoS 진화, 5G 취약점, VPN 제로데이, DoH 악용</li>
          <li>• <strong>실전 도구:</strong> iptables/pfSense (방화벽), Snort/Suricata (IDS/IPS), WireGuard/OpenVPN (VPN)</li>
          <li>• <strong>심층 방어 (Defense in Depth):</strong> OSI 7계층 각 계층별 다층 보안 적용 필수</li>
          <li>• <strong>엔터프라이즈 전환:</strong> NGFW (Palo Alto, Cisco) + 통합 위협 인텔리전스 (Talos, WildFire)</li>
        </ul>
      </section>
    </div>
  );
}
