import React from 'react';
import { Shield, Lock, AlertTriangle, Eye, FileWarning, Zap, Code, TrendingUp, Globe } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          사이버 보안 기초
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          현대 사이버 보안의 핵심 개념과 실무 적용 방법을 학습합니다
        </p>
      </div>

      {/* CIA 삼원칙 - 실무 예제 포함 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-7 h-7 text-red-600" />
          CIA 삼원칙 (정보보안 3대 원칙)
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          {/* 기밀성 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Lock className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              기밀성 (Confidentiality)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              허가된 사용자만 정보에 접근할 수 있도록 보호
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">구현 방법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• AES-256 암호화</li>
                <li>• RBAC (Role-Based Access Control)</li>
                <li>• OAuth 2.0 / OpenID Connect</li>
                <li>• Zero Knowledge Encryption</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">실무 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                금융권의 개인정보 암호화, 의료 데이터 HIPAA 준수
              </p>
            </div>
          </div>

          {/* 무결성 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <FileWarning className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              무결성 (Integrity)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              정보가 위조되거나 변조되지 않도록 보호
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">구현 방법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• SHA-256 해시 함수</li>
                <li>• HMAC 메시지 인증</li>
                <li>• 디지털 서명 (RSA, ECDSA)</li>
                <li>• Blockchain 기반 검증</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">실무 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                소프트웨어 배포 시 체크섬 검증, Git commit 서명
              </p>
            </div>
          </div>

          {/* 가용성 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Zap className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              가용성 (Availability)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              필요할 때 언제든지 정보에 접근할 수 있도록 보장
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">구현 방법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• CDN (CloudFlare, Akamai)</li>
                <li>• Load Balancing (HAProxy, Nginx)</li>
                <li>• Auto-Scaling & Failover</li>
                <li>• DDoS Mitigation (Arbor, Radware)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">실무 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                AWS Multi-AZ 배포, 99.99% SLA 보장 시스템
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실제 코드 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실무 코드 예제
        </h2>

        <div className="space-y-6">
          {/* AES 암호화 예제 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-indigo-900 dark:text-indigo-300">
              1. AES-256 암호화 (Python)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def encrypt_data(plaintext: bytes, key: bytes) -> tuple:
    """AES-256-GCM 암호화"""
    iv = os.urandom(12)  # 96-bit IV for GCM
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return (iv, ciphertext, encryptor.tag)

# 사용 예시
key = os.urandom(32)  # 256-bit key
data = b"Sensitive customer data"
iv, encrypted, tag = encrypt_data(data, key)`}</code>
              </pre>
            </div>
          </div>

          {/* SHA-256 해시 예제 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. SHA-256 무결성 검증 (Node.js)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`const crypto = require('crypto');

function verifyFileIntegrity(fileBuffer, expectedHash) {
  // SHA-256 해시 계산
  const hash = crypto
    .createHash('sha256')
    .update(fileBuffer)
    .digest('hex');

  // 무결성 검증
  if (hash === expectedHash) {
    console.log('✓ File integrity verified');
    return true;
  } else {
    console.error('✗ File has been tampered!');
    return false;
  }
}

// 다운로드한 파일 검증
const downloadedFile = fs.readFileSync('app.zip');
const publishedHash = '5d41402abc4b2a76b9719d911017c592';
verifyFileIntegrity(downloadedFile, publishedHash);`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 사이버 위협 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 최신 사이버 위협 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-red-500 bg-red-50 dark:bg-red-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-red-900 dark:text-red-300 flex items-center gap-2">
              1. AI 기반 사이버 공격 급증 🤖
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              ChatGPT 등 생성형 AI를 악용한 정교한 피싱 이메일 및 멀웨어 코드 자동 생성
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-sm font-semibold mb-2 text-red-800 dark:text-red-300">실제 사례:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• WormGPT: 악성 피싱 이메일 자동 생성 도구 (2024.01)</li>
                <li>• FraudGPT: 제로데이 취약점 자동 탐색 AI (2024.03)</li>
                <li>• Deepfake Voice: CEO 음성 위조 사기 증가 (평균 피해액 $243,000)</li>
              </ul>
            </div>
          </div>

          <div className="border-l-4 border-orange-500 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-orange-900 dark:text-orange-300">
              2. 랜섬웨어 2.0 - 이중 갈취 공격
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              데이터 암호화 + 유출 협박을 결합한 진화된 랜섬웨어
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-orange-700 dark:text-orange-400 mb-1">LockBit 3.0</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  2024년 상반기 피해액 $910M, 2,300+ 기업 공격
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-orange-700 dark:text-orange-400 mb-1">BlackCat (ALPHV)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  RaaS(Ransomware-as-a-Service) 모델, 60% 수익 공유
                </p>
              </div>
            </div>
          </div>

          <div className="border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-yellow-900 dark:text-yellow-300">
              3. 공급망 공격 (Supply Chain Attack)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              신뢰받는 소프트웨어/라이브러리를 통한 대규모 침투
            </p>
            <ul className="text-sm space-y-2 text-gray-600 dark:text-gray-400 ml-4">
              <li>
                <span className="font-semibold">• SolarWinds (2020) 교훈:</span> 18,000+ 조직 감염, 9개월간 미탐지
              </li>
              <li>
                <span className="font-semibold">• Log4Shell (CVE-2021-44228):</span> 전 세계 93% 기업 영향
              </li>
              <li>
                <span className="font-semibold">• XZ Utils 백도어 (2024.03):</span> SSH 접근 권한 탈취 시도
              </li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              4. 클라우드 네이티브 공격
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              Kubernetes, Docker, Serverless 환경을 표적으로 한 공격
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-sm font-semibold mb-2">주요 공격 벡터:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 잘못 설정된 S3 버킷 (평균 250TB 데이터 유출/건)</li>
                <li>• Kubernetes API Server 노출 (30,000+ 취약 인스턴스)</li>
                <li>• AWS Lambda 함수 인젝션 공격</li>
                <li>• Container Escape (runC CVE-2019-5736)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 글로벌 사이버 보안 통계 */}
      <section className="bg-gradient-to-r from-red-600 to-orange-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Globe className="w-7 h-7" />
          2024-2025 글로벌 사이버 보안 현황
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$10.5조</p>
            <p className="text-sm opacity-90">2025 예상 사이버 범죄 피해액</p>
            <p className="text-xs mt-2 opacity-75">출처: Cybersecurity Ventures, 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">72%</p>
            <p className="text-sm opacity-90">기업의 랜섬웨어 공격 경험률</p>
            <p className="text-xs mt-2 opacity-75">출처: Sophos State of Ransomware 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">39초</p>
            <p className="text-sm opacity-90">평균 사이버 공격 발생 주기</p>
            <p className="text-xs mt-2 opacity-75">출처: University of Maryland, 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">204일</p>
            <p className="text-sm opacity-90">평균 침해 사고 탐지 소요 시간 (MTTD)</p>
            <p className="text-xs mt-2 opacity-75">출처: IBM Cost of Data Breach 2024</p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">$4.88M</p>
            <p className="text-sm">평균 데이터 유출 비용</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">3.5M</p>
            <p className="text-sm">2025 예상 사이버보안 인력 부족 수</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">45%</p>
            <p className="text-sm">제로데이 공격 증가율 (YoY)</p>
          </div>
        </div>
      </section>

      {/* 현대 보안 모범 사례 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
          현업 보안 전문가의 필수 실천 사항
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">개인 보안</h3>
            <div className="space-y-3">
              {[
                { rule: '패스워드 매니저 사용', detail: '1Password, Bitwarden, KeePass' },
                { rule: 'Hardware Security Key', detail: 'YubiKey, Google Titan Key (FIDO2)' },
                { rule: 'VPN 상시 사용', detail: 'WireGuard, OpenVPN, Tailscale' },
                { rule: '이메일 보안', detail: 'ProtonMail, SPF/DKIM/DMARC 설정' },
                { rule: 'Endpoint Protection', detail: 'CrowdStrike, SentinelOne, Microsoft Defender' },
              ].map((item, idx) => (
                <div key={idx} className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-l-4 border-blue-500">
                  <span className="flex-shrink-0 w-7 h-7 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                    {idx + 1}
                  </span>
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">{item.rule}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-bold text-lg mb-3 text-purple-900 dark:text-purple-300">조직 보안</h3>
            <div className="space-y-3">
              {[
                { rule: 'Zero Trust Architecture', detail: 'Never Trust, Always Verify' },
                { rule: 'SIEM 구축', detail: 'Splunk, Elastic Security, QRadar' },
                { rule: 'Vulnerability Management', detail: 'Tenable, Qualys, Rapid7' },
                { rule: 'Security Awareness Training', detail: 'KnowBe4, Proofpoint (월 1회 필수)' },
                { rule: 'Incident Response Plan', detail: 'NIST CSF, SANS IR Framework' },
              ].map((item, idx) => (
                <div key={idx} className="flex items-start gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-l-4 border-purple-500">
                  <span className="flex-shrink-0 w-7 h-7 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                    {idx + 1}
                  </span>
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">{item.rule}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <section className="bg-gradient-to-r from-indigo-900 to-purple-900 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6">📚 References & Further Reading</h2>

        <div className="space-y-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold mb-2">📖 필수 문서 및 프레임워크</h3>
            <ul className="space-y-2 text-sm">
              <li>• <a href="https://www.nist.gov/cyberframework" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">NIST Cybersecurity Framework 2.0</a> - 미국 표준기술연구소</li>
              <li>• <a href="https://owasp.org/www-project-top-ten/" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">OWASP Top 10 (2021)</a> - 웹 애플리케이션 보안 취약점</li>
              <li>• <a href="https://www.cisecurity.org/controls" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">CIS Controls v8</a> - 사이버 보안 필수 통제항목</li>
              <li>• <a href="https://www.mitre.org/attack" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">MITRE ATT&CK Framework</a> - 공격 전술 및 기법 분류</li>
            </ul>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold mb-2">📊 업계 리포트 (2024-2025)</h3>
            <ul className="space-y-2 text-sm">
              <li>• IBM Cost of a Data Breach Report 2024</li>
              <li>• Verizon Data Breach Investigations Report (DBIR) 2024</li>
              <li>• Mandiant M-Trends 2024</li>
              <li>• Sophos State of Ransomware 2024</li>
            </ul>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold mb-2">🛠️ 실습 리소스</h3>
            <ul className="space-y-2 text-sm">
              <li>• <a href="https://www.hackthebox.com" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">HackTheBox</a> - 윤리적 해킹 실습 플랫폼</li>
              <li>• <a href="https://tryhackme.com" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">TryHackMe</a> - 초급~고급 사이버 보안 교육</li>
              <li>• <a href="https://portswigger.net/web-security" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">PortSwigger Web Security Academy</a> - 무료 웹 보안 교육</li>
              <li>• <a href="https://www.kali.org" className="text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer">Kali Linux</a> - 침투 테스트 전용 OS</li>
            </ul>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold mb-2">🎓 자격증 로드맵</h3>
            <ul className="space-y-2 text-sm">
              <li>• <strong>입문</strong>: CompTIA Security+, CEH (Certified Ethical Hacker)</li>
              <li>• <strong>중급</strong>: CISSP, CISM, OSCP (Offensive Security Certified Professional)</li>
              <li>• <strong>고급</strong>: GIAC (GPEN, GCIH, GXPN), OSCE, OSEE</li>
              <li>• <strong>클라우드</strong>: AWS Certified Security Specialty, CCSP</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong>CIA 삼원칙</strong>은 정보보안의 근간이며, 실무에서는 AES-256, SHA-256, Load Balancing 등으로 구현</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong>2024-2025 주요 위협</strong>: AI 기반 공격, 랜섬웨어 2.0, 공급망 공격, 클라우드 네이티브 공격</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong>평균 데이터 유출 비용 $4.88M</strong>, 탐지까지 204일 소요 - 예방이 최선의 방어</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong>Zero Trust Architecture</strong> 채택, SIEM 구축, 정기적 보안 교육이 조직 보안의 핵심</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span>NIST CSF, OWASP, MITRE ATT&CK 프레임워크를 활용한 체계적 보안 관리 필요</span>
          </li>
        </ul>
      </section>
    </div>
  );
}
