import React from 'react';
import { AlertTriangle, Clock, FileText, Shield, Terminal, Folder, Users, TrendingUp } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section className="bg-gradient-to-r from-red-600 to-orange-600 text-white rounded-xl p-8 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <TrendingUp className="w-8 h-8" />
          <h2 className="text-3xl font-bold">2024-2025 사고 대응 트렌드</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">207일</div>
            <div className="text-sm opacity-90">평균 침해 탐지 시간 (IBM 2024)</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">73일</div>
            <div className="text-sm opacity-90">평균 대응 완료 시간</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="text-4xl font-bold mb-2">$4.88M</div>
            <div className="text-sm opacity-90">평균 침해 비용</div>
          </div>
        </div>
        <div className="mt-6 p-4 bg-white/10 backdrop-blur-sm rounded-lg">
          <p className="text-lg">
            빠른 탐지와 신속한 대응이 피해를 최소화합니다.
            NIST 표준 프레임워크와 실전 플레이북을 활용한 체계적 접근이 필수입니다.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            NIST SP 800-61 Rev.2 사고 대응 프레임워크
          </h2>
        </div>

        <div className="space-y-6">
          <div className="border-l-4 border-blue-500 bg-gradient-to-r from-blue-50 to-transparent dark:from-blue-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold text-lg">1</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Preparation (준비)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">IR 팀 구성 및 역할 정의</p>
              <p className="ml-4">- CISO, IR Manager, Forensics Analyst, Threat Hunter</p>
              <p className="ml-4">- 24/7 On-call 체계 구축</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">도구 및 인프라 준비</p>
              <p className="ml-4">- SIEM (Splunk, Elastic, Sentinel)</p>
              <p className="ml-4">- EDR (CrowdStrike, SentinelOne)</p>
              <p className="ml-4">- Forensics Tools (Volatility, FTK, Autopsy)</p>
            </div>
          </div>

          <div className="border-l-4 border-yellow-500 bg-gradient-to-r from-yellow-50 to-transparent dark:from-yellow-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-yellow-500 text-white flex items-center justify-center font-bold text-lg">2</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Detection & Analysis (탐지 및 분석)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">탐지 소스</p>
              <p className="ml-4">- SIEM alerts, IDS/IPS, EDR, Threat Intel</p>
              <p className="ml-4">- User reports, System logs, Network traffic</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">초기 분석</p>
              <p className="ml-4">- IOC 추출 (IP, Domain, Hash, Registry)</p>
              <p className="ml-4">- Timeline 구성</p>
              <p className="ml-4">- 영향 범위 평가 (Scope Assessment)</p>
            </div>
          </div>

          <div className="border-l-4 border-orange-500 bg-gradient-to-r from-orange-50 to-transparent dark:from-orange-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-orange-500 text-white flex items-center justify-center font-bold text-lg">3</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Containment (격리)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">단기 격리 (Short-term Containment)</p>
              <p className="ml-4">- 네트워크 격리 (iptables DROP, VLAN 분리)</p>
              <p className="ml-4">- 계정 비활성화 (Active Directory 잠금)</p>
              <p className="ml-4">- 시스템 격리 (호스트 방화벽)</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">증거 보존</p>
              <p className="ml-4">- 메모리 덤프 (winpmem, LiME)</p>
              <p className="ml-4">- 디스크 이미지 (FTK Imager)</p>
              <p className="ml-4">- 로그 수집 (Syslog, Windows Event)</p>
            </div>
          </div>

          <div className="border-l-4 border-red-500 bg-gradient-to-r from-red-50 to-transparent dark:from-red-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-500 text-white flex items-center justify-center font-bold text-lg">4</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Eradication (근절)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">멀웨어 제거</p>
              <p className="ml-4">- 악성 프로세스 종료</p>
              <p className="ml-4">- 레지스트리 정리</p>
              <p className="ml-4">- 지속성 메커니즘 제거 (Scheduled Task, Service)</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">계정 정리</p>
              <p className="ml-4">- 침해된 계정 비밀번호 재설정</p>
              <p className="ml-4">- 불필요한 계정 삭제</p>
              <p className="ml-4">- 권한 재검토 (Least Privilege)</p>
            </div>
          </div>

          <div className="border-l-4 border-green-500 bg-gradient-to-r from-green-50 to-transparent dark:from-green-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-green-500 text-white flex items-center justify-center font-bold text-lg">5</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Recovery (복구)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">복구 절차</p>
              <p className="ml-4">- 백업에서 시스템 복원</p>
              <p className="ml-4">- 단계적 서비스 재개 (Phased Approach)</p>
              <p className="ml-4">- 모니터링 강화 (재발 탐지)</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">복구 검증</p>
              <p className="ml-4">- 보안 스캔 (Vulnerability Scanner)</p>
              <p className="ml-4">- 침투 테스트 (재침해 여부 확인)</p>
              <p className="ml-4">- 비즈니스 연속성 확인</p>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 bg-gradient-to-r from-purple-50 to-transparent dark:from-purple-900/20 p-6 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold text-lg">6</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Post-Incident Activity (사후 활동)</h3>
            </div>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200">사고 보고서 작성</p>
              <p className="ml-4">- Timeline, Root Cause, Impact Analysis</p>
              <p className="ml-4">- 대응 활동 및 비용 산출</p>
              <p className="ml-4">- 교훈 (Lessons Learned)</p>
              <p className="font-bold text-lg text-gray-800 dark:text-gray-200 mt-4">프로세스 개선</p>
              <p className="ml-4">- IR 플레이북 업데이트</p>
              <p className="ml-4">- 탐지 규칙 추가 (SIEM Correlation Rule)</p>
              <p className="ml-4">- 교육 및 훈련 (Tabletop Exercise)</p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-br from-red-100 to-orange-100 dark:from-red-900/30 dark:to-orange-900/30 rounded-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <AlertTriangle className="w-8 h-8 text-red-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            랜섬웨어 대응 플레이북
          </h2>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
            <FileText className="w-6 h-6 text-blue-600" />
            실제 사례: LockBit 3.0 랜섬웨어 공격 (2024년 3월)
          </h3>
          <div className="space-y-3 text-gray-700 dark:text-gray-300">
            <p><strong>공격 벡터:</strong> VPN 취약점 (CVE-2023-XXXX) 악용</p>
            <p><strong>최초 침투:</strong> 2024-03-15 02:34 AM</p>
            <p><strong>탐지 시각:</strong> 2024-03-15 09:12 AM (6시간 38분 후)</p>
            <p><strong>암호화 범위:</strong> 245GB (파일서버 3대, DB 백업 1대)</p>
            <p><strong>요구 금액:</strong> 50 BTC (약 $2.73M)</p>
            <p><strong>복구 방법:</strong> 오프라인 백업 복원 (협상 거부)</p>
            <p><strong>총 다운타임:</strong> 18시간</p>
            <p><strong>교훈:</strong> VPN MFA 미적용, 백업 격리 미흡</p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
            <Terminal className="w-6 h-6 text-green-600" />
            골든 타임: 첫 15분 대응 절차
          </h3>

          <div className="space-y-4">
            <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4">
              <p className="font-bold text-red-700 dark:text-red-400 mb-2">즉시 실행 (1-5분)</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono">
{`# 1. 네트워크 즉시 차단 (Linux)
sudo iptables -A INPUT -j DROP
sudo iptables -A OUTPUT -j DROP

# 2. 백업 시스템 격리
sudo systemctl stop veeamservice
sudo umount /mnt/backup

# 3. Windows 방화벽 전체 활성화
netsh advfirewall set allprofiles state on
netsh advfirewall firewall add rule name="Block All" dir=out action=block`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-bold text-yellow-700 dark:text-yellow-400 mb-2">분석 및 증거 수집 (5-10분)</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono">
{`# 4. 악성 프로세스 식별
ps aux | grep -i "encrypt"
Get-Process | Where-Object ProcessName -match "encrypt"

# 5. 메모리 덤프 (증거 보존)
sudo ./winpmem-3.3.rc3.exe -o memory.raw
sudo insmod lime.ko "path=/tmp/mem.lime format=lime"

# 6. 네트워크 연결 확인
netstat -anp | grep ESTABLISHED
Get-NetTCPConnection | Where State -eq "Established"`}
              </pre>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4">
              <p className="font-bold text-blue-700 dark:text-blue-400 mb-2">격리 및 보고 (10-15분)</p>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono">
{`# 7. Active Directory 계정 즉시 잠금
Disable-ADAccount -Identity compromised_user
Set-ADUser -Identity all_users -ChangePasswordAtLogon $true

# 8. 랜섬 노트 수집 및 IOC 추출
find / -name "*README*.txt" -o -name "*DECRYPT*.txt"
Get-ChildItem -Recurse -Filter "*README*.txt"

# 9. CISO 및 법무팀 즉시 보고
echo "Incident ID: INC-2024-$(date +%s)" > /var/log/incident.log`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <div className="flex items-center gap-3 mb-6">
          <Folder className="w-8 h-8 text-purple-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            디지털 포렌식 도구
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-purple-800 dark:text-purple-300">디스크 포렌식</h3>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-purple-700 dark:text-purple-400">FTK Imager (AccessData)</p>
                <p className="text-gray-600 dark:text-gray-400">무료 디스크 이미징 도구. 법정 증거로 인정.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-purple-700 dark:text-purple-400">Autopsy (Sleuth Kit)</p>
                <p className="text-gray-600 dark:text-gray-400">오픈소스 디지털 포렌식 플랫폼. GUI 기반.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-purple-700 dark:text-purple-400">EnCase Forensic</p>
                <p className="text-gray-600 dark:text-gray-400">엔터프라이즈급 포렌식 솔루션. ($3,594/년)</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-blue-800 dark:text-blue-300">메모리 포렌식</h3>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-blue-700 dark:text-blue-400">Volatility Framework</p>
                <p className="text-gray-600 dark:text-gray-400">메모리 분석 표준 도구. Windows/Linux/Mac 지원.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-blue-700 dark:text-blue-400">Rekall Memory Forensics</p>
                <p className="text-gray-600 dark:text-gray-400">Google의 메모리 분석 프레임워크.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-blue-700 dark:text-blue-400">Redline (FireEye)</p>
                <p className="text-gray-600 dark:text-gray-400">무료 메모리 및 파일 시스템 분석 도구.</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-green-800 dark:text-green-300">네트워크 포렌식</h3>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-green-700 dark:text-green-400">Wireshark</p>
                <p className="text-gray-600 dark:text-gray-400">패킷 분석 표준 도구. PCAP 파일 분석.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-green-700 dark:text-green-400">NetworkMiner</p>
                <p className="text-gray-600 dark:text-gray-400">네트워크 포렌식 분석 도구. 파일 추출 기능.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-green-700 dark:text-green-400">Zeek (Bro IDS)</p>
                <p className="text-gray-600 dark:text-gray-400">네트워크 보안 모니터링 프레임워크.</p>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4 text-orange-800 dark:text-orange-300">악성코드 분석</h3>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-orange-700 dark:text-orange-400">IDA Pro / Ghidra</p>
                <p className="text-gray-600 dark:text-gray-400">리버스 엔지니어링 도구. Ghidra는 NSA 무료 버전.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-orange-700 dark:text-orange-400">PEStudio</p>
                <p className="text-gray-600 dark:text-gray-400">PE 파일 정적 분석 도구.</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-bold text-orange-700 dark:text-orange-400">Cuckoo Sandbox</p>
                <p className="text-gray-600 dark:text-gray-400">자동화된 멀웨어 동적 분석 시스템.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4 text-indigo-800 dark:text-indigo-300">
            Volatility 메모리 포렌식 명령어
          </h3>
          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono">
{`# 프로세스 목록 확인
volatility -f memory.raw --profile=Win10x64_19041 pslist

# 숨겨진 프로세스 탐지
volatility -f memory.raw --profile=Win10x64_19041 psscan

# 네트워크 연결 확인
volatility -f memory.raw --profile=Win10x64_19041 netscan

# 악성 코드 인젝션 탐지
volatility -f memory.raw --profile=Win10x64_19041 malfind

# 실행 명령어 확인
volatility -f memory.raw --profile=Win10x64_19041 cmdline

# DLL 목록 확인
volatility -f memory.raw --profile=Win10x64_19041 dlllist`}
          </pre>
        </div>
      </section>

      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <Clock className="w-8 h-8 text-indigo-600" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            IR 핵심 메트릭 (KPI)
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">MTTD</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Mean Time To Detect (평균 탐지 시간)</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">207일</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              업계 평균 (IBM 2024). 목표: 24시간 이내
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-2">MTTR</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Mean Time To Respond (평균 대응 시간)</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">73일</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              업계 평균. 목표: 48시간 이내
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-2">MTRC</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Mean Time To Contain (평균 격리 시간)</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">16일</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              업계 평균. 목표: 1시간 이내
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-green-600 dark:text-green-400 mb-2">MTTE</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Mean Time To Eradicate (평균 근절 시간)</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">28일</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              업계 평균. 목표: 24시간 이내
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-2">MTTREC</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Mean Time To Recover (평균 복구 시간)</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">23일</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              업계 평균. 목표: 72시간 이내
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-2">Breach Cost</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">침해사고당 평균 비용</p>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">$4.88M</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              IBM 2024 조사. 전년 대비 10% 증가
            </p>
          </div>
        </div>

        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
            KPI 개선 전략
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-green-600 font-bold mt-1">✓</span>
              <span><strong>자동화:</strong> SOAR 플랫폼 도입으로 MTTR 70% 단축</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600 font-bold mt-1">✓</span>
              <span><strong>AI/ML:</strong> 이상 탐지 모델로 MTTD 80% 개선</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600 font-bold mt-1">✓</span>
              <span><strong>Threat Intel:</strong> IOC 피드 연동으로 조기 탐지</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600 font-bold mt-1">✓</span>
              <span><strong>훈련:</strong> 정기적 Tabletop Exercise로 대응 속도 향상</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 IR 표준 및 가이드라인',
            icon: 'docs' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'NIST SP 800-61 Rev.2 - Computer Security Incident Handling Guide',
                url: 'https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf',
                description: '사고 대응의 국제 표준. 6단계 프레임워크를 상세히 설명합니다.'
              },
              {
                title: 'SANS Incident Handler Handbook',
                url: 'https://www.sans.org/white-papers/33901/',
                description: 'SANS 연구소의 실전 IR 핸드북. 체크리스트 및 템플릿 포함.'
              },
              {
                title: 'CISA Ransomware Guide',
                url: 'https://www.cisa.gov/stopransomware/ransomware-guide',
                description: '미국 사이버보안청의 랜섬웨어 대응 가이드 (2024년 업데이트)'
              }
            ]
          },
          {
            title: '🔬 침해사고 분석 도구',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Volatility Framework',
                url: 'https://www.volatilityfoundation.org/',
                description: '메모리 포렌식 표준 도구. Windows/Linux/Mac 지원.'
              },
              {
                title: 'FTK Imager',
                url: 'https://www.exterro.com/ftk-imager',
                description: 'AccessData의 무료 디스크 이미징 도구. 법정 증거로 인정.'
              },
              {
                title: 'Autopsy Digital Forensics',
                url: 'https://www.autopsy.com/',
                description: '오픈소스 디지털 포렌식 플랫폼. GUI 기반 분석.'
              },
              {
                title: 'Wireshark',
                url: 'https://www.wireshark.org/',
                description: '네트워크 트래픽 분석 도구. PCAP 파일 분석 필수.'
              }
            ]
          },
          {
            title: '📖 최신 연구 및 보고서',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'IBM Cost of a Data Breach Report 2024',
                url: 'https://www.ibm.com/security/data-breach',
                description: '글로벌 침해사고 비용 및 트렌드 분석. 평균 $4.88M 피해.'
              },
              {
                title: 'Verizon DBIR 2024',
                url: 'https://www.verizon.com/business/resources/reports/dbir/',
                description: 'Verizon의 연례 침해사고 보고서. 공격 패턴 통계 제공.'
              },
              {
                title: 'MITRE ATT&CK Framework',
                url: 'https://attack.mitre.org/',
                description: '공격 기술 및 전술 분류 체계. IR 분석에 필수.'
              }
            ]
          },
          {
            title: '🛠️ 실전 IR 플레이북',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'AWS Security Incident Response Guide',
                url: 'https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/',
                description: 'AWS 클라우드 환경에서의 IR 절차. Lambda 기반 자동화.'
              },
              {
                title: 'Microsoft Azure Security Response',
                url: 'https://learn.microsoft.com/en-us/security/operations/incident-response-overview',
                description: 'Azure Sentinel 기반 IR 워크플로우. Playbook 템플릿 제공.'
              },
              {
                title: 'Google Cloud Incident Response',
                url: 'https://cloud.google.com/security/incident-response',
                description: 'GCP Chronicle 활용 IR. Security Command Center 통합.'
              }
            ]
          },
          {
            title: '📊 IR 메트릭 및 벤치마크',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Ponemon Institute - Cyber Resilience Report',
                url: 'https://www.ponemon.org/',
                description: 'IR 성숙도 평가 및 업계 벤치마크. MTTD/MTTR 통계.'
              },
              {
                title: 'ENISA Threat Landscape',
                url: 'https://www.enisa.europa.eu/topics/threat-risk-management/threats-and-trends',
                description: 'EU 사이버보안청의 위협 동향. 랜섬웨어 트렌드 분석.'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
