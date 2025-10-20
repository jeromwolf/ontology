import React from 'react';
import { Activity, Eye, Database, Shield, Terminal, TrendingUp, Search } from 'lucide-react';
import References from '../References';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* Header - 2024-2025 SOC Trends */}
      <section className="bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl p-8 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <TrendingUp className="w-8 h-8" />
          <h2 className="text-3xl font-bold">2024-2025 보안 운영 트렌드</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">87%</div>
            <div className="text-sm opacity-90">기업의 SIEM 도입률 (Gartner 2024)</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">3.2분</div>
            <div className="text-sm opacity-90">평균 알림 분석 시간 (AI 활용시)</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">$650M</div>
            <div className="text-sm opacity-90">SIEM 시장 규모 (2024)</div>
          </div>
        </div>
        <div className="mt-6 p-4 bg-white/10 backdrop-blur-sm rounded-lg">
          <p className="text-lg">
            <strong>핵심:</strong> SIEM + SOAR + XDR 통합이 차세대 SOC의 표준입니다.
            AI/ML 기반 자동화와 Threat Intelligence 연동이 필수적입니다.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Eye className="w-7 h-7 text-blue-600" />
          SOC (Security Operations Center)
        </h2>

        <div className="space-y-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg">
            <h3 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-3">
              SOC의 핵심 역할
            </h3>
            <div className="grid md:grid-cols-2 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-blue-700 dark:text-blue-400 mb-1">24/7 모니터링</p>
                <p className="text-gray-600 dark:text-gray-400">실시간 보안 이벤트 감시</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-blue-700 dark:text-blue-400 mb-1">위협 탐지</p>
                <p className="text-gray-600 dark:text-gray-400">이상 행위 및 공격 탐지</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-blue-700 dark:text-blue-400 mb-1">사고 대응</p>
                <p className="text-gray-600 dark:text-gray-400">신속한 대응 및 조치</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-blue-700 dark:text-blue-400 mb-1">보고 및 분석</p>
                <p className="text-gray-600 dark:text-gray-400">정기 보안 리포트 생성</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SIEM Queries Section */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <div className="flex items-center gap-3 mb-6">
          <Database className="w-8 h-8 text-purple-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            실전 SIEM 쿼리 (2024 업데이트)
          </h2>
        </div>

        {/* Splunk Queries */}
        <div className="mb-8">
          <h3 className="text-2xl font-bold mb-4 text-orange-600 dark:text-orange-400 flex items-center gap-2">
            <Terminal className="w-6 h-6" />
            Splunk SPL (Search Processing Language)
          </h3>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500 p-4">
              <p className="font-bold text-orange-800 dark:text-orange-300 mb-2">1. 브루트포스 공격 탐지 (최근 15분)</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`index=auth sourcetype=linux_secure "Failed password"
| stats count by src_ip, user
| where count > 5
| eval severity="HIGH"
| table _time, src_ip, user, count, severity`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                ✅ 15분 내 5회 이상 실패한 로그인 시도를 탐지합니다.
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500 p-4">
              <p className="font-bold text-orange-800 dark:text-orange-300 mb-2">2. 의심스러운 PowerShell 실행</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`index=windows sourcetype=WinEventLog:Security EventCode=4688
| search NewProcessName="*powershell.exe*"
| search CommandLine="*-EncodedCommand*" OR CommandLine="*-Exec Bypass*"
| stats count by ComputerName, User, CommandLine
| where count > 0`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                🚨 인코딩된 PowerShell 명령이나 실행 정책 우회를 탐지합니다.
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500 p-4">
              <p className="font-bold text-orange-800 dark:text-orange-300 mb-2">3. 대용량 데이터 유출 탐지</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`index=network sourcetype=firewall action=allowed
| stats sum(bytes_out) as total_bytes by src_ip, dest_ip
| where total_bytes > 1073741824
| eval total_GB = round(total_bytes/1024/1024/1024, 2)
| table _time, src_ip, dest_ip, total_GB
| sort - total_GB`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                📊 1GB 이상의 아웃바운드 트래픽을 탐지합니다.
              </p>
            </div>
          </div>
        </div>

        {/* Elastic EQL Queries */}
        <div className="mb-8">
          <h3 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400 flex items-center gap-2">
            <Search className="w-6 h-6" />
            Elastic EQL (Event Query Language)
          </h3>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4">
              <p className="font-bold text-blue-800 dark:text-blue-300 mb-2">1. 랜섬웨어 파일 암호화 패턴 탐지</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`sequence by host.id with maxspan=5m
  [file where event.type == "creation" and
   file.extension in ("encrypted", "locked", "crypt")]
  [file where event.type == "creation" and
   file.name : "*README*.txt"]`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                🔐 5분 내 암호화 파일 + 랜섬 노트 생성을 시퀀스로 탐지합니다.
              </p>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4">
              <p className="font-bold text-blue-800 dark:text-blue-300 mb-2">2. 프로세스 인젝션 공격 탐지</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`process where event.type == "start" and
  process.name : ("rundll32.exe", "regsvr32.exe", "mshta.exe") and
  process.args : ("*http*", "*.js", "*.vbs", "*.ps1")
| head 50`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                ⚠️ LOLBins(Living Off the Land Binaries)를 악용한 공격을 탐지합니다.
              </p>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4">
              <p className="font-bold text-blue-800 dark:text-blue-300 mb-2">3. 크리덴셜 덤핑 탐지 (Mimikatz)</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`process where process.name : "lsass.exe" and
  process.pe.original_file_name : ("mimikatz.exe", "procdump*.exe") or
  process.command_line : ("*sekurlsa*", "*lsadump*")
| tail 100`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                🛡️ Mimikatz 및 LSASS 메모리 덤프 시도를 탐지합니다.
              </p>
            </div>
          </div>
        </div>

        {/* Microsoft Sentinel KQL Queries */}
        <div className="mb-8">
          <h3 className="text-2xl font-bold mb-4 text-sky-600 dark:text-sky-400 flex items-center gap-2">
            <Shield className="w-6 h-6" />
            Microsoft Sentinel KQL (Kusto Query Language)
          </h3>

          <div className="space-y-6">
            <div className="bg-sky-50 dark:bg-sky-900/20 border-l-4 border-sky-500 p-4">
              <p className="font-bold text-sky-800 dark:text-sky-300 mb-2">1. Azure AD 의심스러운 로그인</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`SigninLogs
| where TimeGenerated > ago(1h)
| where RiskLevelDuringSignIn in ("high", "medium")
| where ResultType != 0
| project TimeGenerated, UserPrincipalName, IPAddress,
  Location, RiskLevelDuringSignIn, ResultDescription
| sort by TimeGenerated desc`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                🔍 위험도가 높은 실패한 로그인 시도를 1시간 단위로 모니터링합니다.
              </p>
            </div>

            <div className="bg-sky-50 dark:bg-sky-900/20 border-l-4 border-sky-500 p-4">
              <p className="font-bold text-sky-800 dark:text-sky-300 mb-2">2. Office 365 대량 파일 다운로드</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`OfficeActivity
| where TimeGenerated > ago(15m)
| where Operation == "FileDownloaded"
| summarize FileCount = count() by UserId, ClientIP
| where FileCount > 50
| extend Severity = "High"
| project TimeGenerated, UserId, ClientIP, FileCount, Severity`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                📁 15분 내 50개 이상 파일 다운로드를 탐지합니다 (데이터 유출 의심).
              </p>
            </div>

            <div className="bg-sky-50 dark:bg-sky-900/20 border-l-4 border-sky-500 p-4">
              <p className="font-bold text-sky-800 dark:text-sky-300 mb-2">3. VM에서의 암호화폐 채굴 탐지</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`AzureDiagnostics
| where ResourceType == "VIRTUALMACHINES"
| where Category == "NetworkSecurityGroupFlowEvent"
| where DestinationPort in (3333, 4444, 5555, 7777, 9999)
| extend IsCryptoMiningPort = true
| summarize count() by SourceIP, DestinationIP, DestinationPort
| where count_ > 10`}
              </pre>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                ⛏️ 암호화폐 채굴 풀에서 사용하는 포트로의 연결을 탐지합니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Activity className="w-7 h-7 text-orange-600" />
          보안 메트릭 (KPI)
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            { metric: 'MTTD (Mean Time To Detect)', value: '평균 탐지 시간' },
            { metric: 'MTTR (Mean Time To Respond)', value: '평균 대응 시간' },
            { metric: 'False Positive Rate', value: '오탐률' },
            { metric: 'Vulnerability Patch Time', value: '평균 패치 시간' },
            { metric: 'Security Incident Count', value: '보안 사고 건수' },
            { metric: 'Compliance Rate', value: '규정 준수율' },
          ].map((item, idx) => (
            <div key={idx} className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h3 className="font-bold text-orange-900 dark:text-orange-300 mb-1">{item.metric}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{item.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* SOC Automation & Best Practices */}
      <section className="bg-gradient-to-br from-emerald-100 to-teal-100 dark:from-emerald-900/30 dark:to-teal-900/30 rounded-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <Activity className="w-8 h-8 text-emerald-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            SOC 자동화 & 베스트 프랙티스
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
              SOAR 플레이북 예시
            </h3>
            <div className="space-y-3 text-sm">
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-3 rounded">
                <p className="font-bold text-emerald-800 dark:text-emerald-300">1. 자동 격리</p>
                <p className="text-gray-600 dark:text-gray-400">멀웨어 탐지 시 호스트 자동 네트워크 격리</p>
              </div>
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-3 rounded">
                <p className="font-bold text-emerald-800 dark:text-emerald-300">2. 티켓 생성</p>
                <p className="text-gray-600 dark:text-gray-400">Jira/ServiceNow 자동 티켓 생성</p>
              </div>
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-3 rounded">
                <p className="font-bold text-emerald-800 dark:text-emerald-300">3. Threat Intel 연동</p>
                <p className="text-gray-600 dark:text-gray-400">VirusTotal/AlienVault OTX IOC 자동 조회</p>
              </div>
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-3 rounded">
                <p className="font-bold text-emerald-800 dark:text-emerald-300">4. 보고서 자동 생성</p>
                <p className="text-gray-600 dark:text-gray-400">주간/월간 보안 리포트 자동화</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-teal-700 dark:text-teal-400">
              SOC 성숙도 모델 (CMMI)
            </h3>
            <div className="space-y-3 text-sm">
              <div className="bg-teal-50 dark:bg-teal-900/20 p-3 rounded">
                <p className="font-bold text-teal-800 dark:text-teal-300">Level 1: 초기 (Ad-hoc)</p>
                <p className="text-gray-600 dark:text-gray-400">수동 모니터링, 표준화 미흡</p>
              </div>
              <div className="bg-teal-50 dark:bg-teal-900/20 p-3 rounded">
                <p className="font-bold text-teal-800 dark:text-teal-300">Level 2: 관리형 (Managed)</p>
                <p className="text-gray-600 dark:text-gray-400">SIEM 도입, 기본 룰셋</p>
              </div>
              <div className="bg-teal-50 dark:bg-teal-900/20 p-3 rounded">
                <p className="font-bold text-teal-800 dark:text-teal-300">Level 3: 정의형 (Defined)</p>
                <p className="text-gray-600 dark:text-gray-400">표준 프로세스, 플레이북 문서화</p>
              </div>
              <div className="bg-teal-50 dark:bg-teal-900/20 p-3 rounded">
                <p className="font-bold text-teal-800 dark:text-teal-300">Level 4-5: 최적화 (Optimized)</p>
                <p className="text-gray-600 dark:text-gray-400">AI/ML 자동화, 지속적 개선</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4 text-purple-700 dark:text-purple-400">
            2024 SOC KPI 벤치마크
          </h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
              <div className="text-2xl font-bold text-purple-700 dark:text-purple-400 mb-1">5.8분</div>
              <p className="text-gray-600 dark:text-gray-400">평균 MTTD (Mean Time To Detect)</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
              <div className="text-2xl font-bold text-purple-700 dark:text-purple-400 mb-1">27분</div>
              <p className="text-gray-600 dark:text-gray-400">평균 MTTR (Mean Time To Respond)</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
              <div className="text-2xl font-bold text-purple-700 dark:text-purple-400 mb-1">12%</div>
              <p className="text-gray-600 dark:text-gray-400">평균 False Positive Rate (오탐률)</p>
            </div>
          </div>
        </div>
      </section>

      {/* References Section */}
      <References
        sections={[
          {
            title: '📚 SIEM 공식 문서',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Splunk Enterprise Security Documentation',
                url: 'https://docs.splunk.com/Documentation/ES/latest',
                description: 'Splunk ES 공식 문서. SPL 쿼리 문법 및 상관분석 룰 작성 가이드.'
              },
              {
                title: 'Elastic Security Solution',
                url: 'https://www.elastic.co/guide/en/security/current/index.html',
                description: 'Elastic SIEM 및 EQL(Event Query Language) 공식 가이드.'
              },
              {
                title: 'Microsoft Sentinel Documentation',
                url: 'https://learn.microsoft.com/en-us/azure/sentinel/',
                description: 'Azure Sentinel KQL(Kusto Query Language) 완벽 가이드 및 플레이북.'
              },
              {
                title: 'Gartner SIEM Magic Quadrant 2024',
                url: 'https://www.gartner.com/en/documents/siem-magic-quadrant',
                description: 'SIEM 솔루션 시장 분석 및 벤더 비교 (Gartner 평가).'
              }
            ]
          },
          {
            title: '🔬 SOC 운영 가이드',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'NIST SP 800-137 - Information Security Continuous Monitoring',
                url: 'https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-137.pdf',
                description: 'NIST의 지속적 모니터링 가이드라인. SOC 운영 표준.'
              },
              {
                title: 'SANS SOC Survey 2024',
                url: 'https://www.sans.org/white-papers/soc-survey/',
                description: 'SOC 운영 실태 조사 및 베스트 프랙티스 (SANS 연례 보고서).'
              },
              {
                title: 'MITRE ATT&CK Detection Guide',
                url: 'https://attack.mitre.org/resources/attack-data-sources/',
                description: 'MITRE 프레임워크 기반 탐지 전략 및 데이터 소스 매핑.'
              }
            ]
          },
          {
            title: '🛠️ SOAR & 자동화 도구',
            icon: 'tools' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'Palo Alto Cortex XSOAR',
                url: 'https://www.paloaltonetworks.com/cortex/xsoar',
                description: 'SOAR 플랫폼 선두주자. 3000+ 통합 앱 지원.'
              },
              {
                title: 'IBM QRadar SOAR',
                url: 'https://www.ibm.com/security/security-orchestration-automation-response',
                description: 'IBM의 SOAR 솔루션 (구 Resilient). 자동화 플레이북 제공.'
              },
              {
                title: 'TheHive Project',
                url: 'https://thehive-project.org/',
                description: '오픈소스 보안 사고 대응 플랫폼. MISP 연동 지원.'
              },
              {
                title: 'Shuffle SOAR',
                url: 'https://shuffler.io/',
                description: '오픈소스 SOAR 플랫폼. Workflow 자동화 및 API 통합.'
              }
            ]
          },
          {
            title: '📊 Threat Intelligence',
            icon: 'web' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'AlienVault OTX (Open Threat Exchange)',
                url: 'https://otx.alienvault.com/',
                description: '오픈소스 위협 인텔리전스 플랫폼. 무료 IOC 공유.'
              },
              {
                title: 'MISP (Malware Information Sharing Platform)',
                url: 'https://www.misp-project.org/',
                description: '멀웨어 정보 공유 플랫폼. STIX/TAXII 표준 지원.'
              },
              {
                title: 'VirusTotal',
                url: 'https://www.virustotal.com/gui/',
                description: '파일/URL/IP 위협 분석 서비스. 70+ AV 엔진 통합.'
              }
            ]
          },
          {
            title: '🎓 교육 & 인증',
            icon: 'docs' as const,
            color: 'border-sky-500',
            items: [
              {
                title: 'SANS SEC450: Blue Team Fundamentals',
                url: 'https://www.sans.org/cyber-security-courses/blue-team-fundamentals-security-operations-analysis/',
                description: 'SOC 애널리스트 필수 과정. SIEM 쿼리 실습 포함.'
              },
              {
                title: 'Splunk Certified Power User',
                url: 'https://www.splunk.com/en_us/training/certification-track/splunk-power-user.html',
                description: 'Splunk 공식 인증. SPL 고급 쿼리 능력 검증.'
              },
              {
                title: 'GIAC Certified SOC Analyst (GCSA)',
                url: 'https://www.giac.org/certifications/certified-soc-analyst-gcsa/',
                description: 'SOC 운영 전문 인증. GIAC 공식 자격증.'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
