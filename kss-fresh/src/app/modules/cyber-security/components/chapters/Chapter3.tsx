import React from 'react';
import { Code, Bug, Shield, AlertTriangle, Lock, FileCode } from 'lucide-react';
import References from '../References';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          애플리케이션 보안
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          OWASP Top 10과 시큐어 코딩 기법을 실전 예제로 학습합니다
        </p>
      </div>

      {/* 2024-2025 애플리케이션 보안 위협 트렌드 */}
      <section className="bg-gradient-to-r from-red-600 to-orange-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <AlertTriangle className="w-7 h-7" />
          2024-2025 애플리케이션 보안 위협 트렌드
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🚨 LLM/AI 애플리케이션 공격 급증</h3>
            <p className="text-sm mb-2">
              Prompt Injection, Model Poisoning 등 새로운 공격 유형 등장
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: OWASP Top 10 for LLM Applications (2024)
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🔐 Supply Chain 공격 심화</h3>
            <p className="text-sm mb-2">
              npm, PyPI 등 패키지 저장소를 통한 악성 코드 유포 증가 (+350%)
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Sonatype 2024 State of Software Supply Chain
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">📱 API 보안 취약점 증가</h3>
            <p className="text-sm mb-2">
              REST/GraphQL API의 인증/권한 취약점으로 데이터 유출 증가
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Salt Security API Security Report 2024
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">💰 비즈니스 로직 취약점 악용</h3>
            <p className="text-sm mb-2">
              쿠폰/포인트 시스템 등 비즈니스 로직 결함을 통한 금전 탈취
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              평균 피해액: $4.2M/건 (Verizon DBIR 2024)
            </p>
          </div>
        </div>
      </section>

      {/* OWASP Top 10 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Bug className="w-7 h-7 text-red-600" />
          OWASP Top 10 (2023)
        </h2>

        <div className="space-y-3">
          {[
            { rank: 1, name: 'Broken Access Control', desc: '접근 제어 취약점' },
            { rank: 2, name: 'Cryptographic Failures', desc: '암호화 실패' },
            { rank: 3, name: 'Injection', desc: 'SQL/OS 명령 주입' },
            { rank: 4, name: 'Insecure Design', desc: '불안전한 설계' },
            { rank: 5, name: 'Security Misconfiguration', desc: '보안 설정 오류' },
            { rank: 6, name: 'Vulnerable Components', desc: '취약한 컴포넌트' },
            { rank: 7, name: 'Authentication Failures', desc: '인증 실패' },
            { rank: 8, name: 'Data Integrity Failures', desc: '데이터 무결성 실패' },
            { rank: 9, name: 'Logging Failures', desc: '로깅 실패' },
            { rank: 10, name: 'SSRF', desc: '서버측 요청 위조' },
          ].map((item) => (
            <div key={item.rank} className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-4 rounded-lg border-l-4 border-red-500">
              <div className="flex items-center gap-3">
                <span className="flex-shrink-0 w-10 h-10 bg-red-600 text-white rounded-full flex items-center justify-center font-bold">
                  {item.rank}
                </span>
                <div>
                  <p className="font-bold text-red-900 dark:text-red-300">{item.name}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{item.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* #1 Broken Access Control - 실전 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Lock className="w-7 h-7 text-red-600" />
          #1 Broken Access Control (접근 제어 취약점)
        </h2>

        <div className="mb-6 bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-l-4 border-red-500">
          <h3 className="font-bold text-red-900 dark:text-red-300 mb-2">🎯 실제 공격 시나리오</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            사용자가 URL을 직접 조작하여 다른 사용자의 데이터에 무단 접근 (IDOR - Insecure Direct Object Reference)
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300">
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">
              ❌ 취약한 코드 (Node.js/Express)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`// ❌ 인증만 하고 권한 검증 없음
app.get('/api/users/:userId/profile',
  authenticateToken,
  async (req, res) => {
    const profile = await db.query(
      'SELECT * FROM profiles WHERE user_id = ?',
      [req.params.userId]
    );
    res.json(profile);
  }
);

// 🚨 공격: GET /api/users/123/profile
// 결과: 로그인만 하면 누구나 userId=123 프로필 조회 가능`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ 안전한 코드 (권한 검증 추가)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`// ✅ 본인 확인 또는 관리자 권한 검증
app.get('/api/users/:userId/profile',
  authenticateToken,
  async (req, res) => {
    const requestedUserId = parseInt(req.params.userId);
    const currentUserId = req.user.id;
    const isAdmin = req.user.role === 'admin';

    // 본인이거나 관리자만 접근 가능
    if (requestedUserId !== currentUserId && !isAdmin) {
      return res.status(403).json({
        error: 'Access denied'
      });
    }

    const profile = await db.query(
      'SELECT * FROM profiles WHERE user_id = ?',
      [requestedUserId]
    );
    res.json(profile);
  }
);`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-blue-900 dark:text-blue-300 mb-2">🛡️ 추가 방어 기법</h4>
          <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300 ml-4">
            <li>• UUID 사용: 순차 ID 대신 예측 불가능한 UUID 사용</li>
            <li>• RBAC/ABAC: Role-Based 또는 Attribute-Based Access Control</li>
            <li>• Rate Limiting: 무차별 대입 공격 방지</li>
            <li>• Audit Logging: 모든 접근 시도 기록</li>
          </ul>
        </div>
      </section>

      {/* #3 Injection - SQL Injection 실전 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-purple-600" />
          #3 Injection - SQL Injection 실전 방어
        </h2>

        <div className="mb-6 bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-l-4 border-red-500">
          <h3 className="font-bold text-red-900 dark:text-red-300 mb-2">🎯 2024년 실제 사례</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
            MOVEit Transfer 취약점(CVE-2023-34362): SQL Injection으로 2,000개 이상 기업 데이터 유출
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            피해 기업: Shell, BBC, British Airways 등 / 피해자: 6천만 명 이상
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300">
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">
              ❌ 취약한 코드 (Python/Flask)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`# ❌ 문자열 연결 방식 (매우 위험!)
@app.route('/search')
def search():
    keyword = request.args.get('q')

    query = f"""
        SELECT * FROM products
        WHERE name LIKE '%{keyword}%'
    """

    results = db.execute(query)
    return jsonify(results)

# 🚨 공격: /search?q='; DROP TABLE products; --
# 결과: products 테이블 전체 삭제!`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ 안전한 코드 (Parameterized Query)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`# ✅ Parameterized Query + 입력값 검증
from sqlalchemy import text

@app.route('/search')
def search():
    keyword = request.args.get('q', '')

    # 1. 입력값 검증 (화이트리스트)
    if not re.match(r'^[a-zA-Z0-9\\s]+$', keyword):
        return jsonify({'error': 'Invalid input'}), 400

    # 2. Parameterized Query (바인딩)
    query = text("""
        SELECT * FROM products
        WHERE name LIKE :keyword
    """)

    # 3. 길이 제한 적용
    safe_keyword = f'%{keyword[:50]}%'

    results = db.execute(
        query,
        {'keyword': safe_keyword}
    ).fetchall()

    return jsonify([dict(r) for r in results])`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-purple-900 dark:text-purple-300 mb-2">🛡️ 다층 방어 전략</h4>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div>
              <p className="font-semibold text-purple-800 dark:text-purple-200 mb-1">코드 레벨</p>
              <ul className="text-gray-700 dark:text-gray-300 ml-4 space-y-1">
                <li>• ORM 사용 (SQLAlchemy, Prisma)</li>
                <li>• Prepared Statements</li>
                <li>• 입력값 화이트리스트 검증</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold text-purple-800 dark:text-purple-200 mb-1">인프라 레벨</p>
              <ul className="text-gray-700 dark:text-gray-300 ml-4 space-y-1">
                <li>• WAF 규칙 (ModSecurity, Cloudflare)</li>
                <li>• DB 최소 권한 계정 사용</li>
                <li>• 에러 메시지 마스킹</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* XSS (Cross-Site Scripting) 실전 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-orange-600" />
          XSS (Cross-Site Scripting) 방어
        </h2>

        <div className="mb-6 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
          <h3 className="font-bold text-orange-900 dark:text-orange-300 mb-2">🎯 공격 유형</h3>
          <div className="grid md:grid-cols-3 gap-2 text-sm text-gray-700 dark:text-gray-300">
            <div>• <strong>Reflected XSS</strong>: URL 파라미터 반사</div>
            <div>• <strong>Stored XSS</strong>: DB 저장 후 실행</div>
            <div>• <strong>DOM-based XSS</strong>: 클라이언트 측 실행</div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300">
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">
              ❌ 취약한 코드 (React)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`// ❌ dangerouslySetInnerHTML 사용
function Comment({ content }) {
  return (
    <div
      dangerouslySetInnerHTML={{
        __html: content
      }}
    />
  );
}

// 🚨 공격: content = "<img src=x onerror='alert(1)'>"
// 결과: 악성 스크립트 실행`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ 안전한 코드 (DOMPurify)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`// ✅ DOMPurify로 새니타이징
import DOMPurify from 'dompurify';

function Comment({ content }) {
  const sanitized = DOMPurify.sanitize(content, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong'],
    ALLOWED_ATTR: []
  });

  return (
    <div
      dangerouslySetInnerHTML={{
        __html: sanitized
      }}
    />
  );
}

// ✅ 또는 React 기본 이스케이핑 활용
function CommentSafe({ content }) {
  return <div>{content}</div>;  // 자동 이스케이핑
}`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-orange-900 dark:text-orange-300 mb-2">🛡️ Content Security Policy (CSP)</h4>
          <pre className="bg-gray-900 text-green-400 p-3 rounded text-sm overflow-x-auto font-mono">
{`<!-- HTTP 헤더 또는 meta 태그로 설정 -->
Content-Security-Policy:
  default-src 'self';
  script-src 'self' 'nonce-{random}' https://cdn.example.com;
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  object-src 'none';
  base-uri 'self';`}
          </pre>
        </div>
      </section>

      {/* SAST/DAST 도구 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <FileCode className="w-7 h-7 text-indigo-600" />
          SAST/DAST 보안 테스팅 도구
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border-2 border-indigo-300">
            <h3 className="font-bold text-lg mb-3 text-indigo-900 dark:text-indigo-300">
              🔍 SAST (Static Application Security Testing)
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              소스 코드를 분석하여 취약점을 찾는 화이트박스 테스팅
            </p>

            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-indigo-700 dark:text-indigo-300">SonarQube</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">오픈소스 / 30개 언어 지원 / CI/CD 통합</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-indigo-700 dark:text-indigo-300">Semgrep</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">오픈소스 / 빠른 스캔 / 커스텀 룰 작성</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-indigo-700 dark:text-indigo-300">Checkmarx</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">기업용 / OWASP Top 10 커버리지 99%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-indigo-700 dark:text-indigo-300">Snyk Code</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">개발자 친화적 / IDE 통합 / 실시간 피드백</p>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-2 border-purple-300">
            <h3 className="font-bold text-lg mb-3 text-purple-900 dark:text-purple-300">
              🌐 DAST (Dynamic Application Security Testing)
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              실행 중인 애플리케이션을 공격하여 취약점을 찾는 블랙박스 테스팅
            </p>

            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">OWASP ZAP</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">오픈소스 / 무료 / 초보자 친화적</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Burp Suite</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">업계 표준 / 펜테스터 필수 도구 / $399/년</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Acunetix</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">7,000+ 취약점 DB / 자동화 스캔</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Nuclei</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">오픈소스 / 템플릿 기반 / 5,000+ 템플릿</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
          <h4 className="font-bold text-green-900 dark:text-green-300 mb-2">💡 DevSecOps 통합 예제</h4>
          <pre className="bg-gray-900 text-green-400 p-3 rounded text-sm overflow-x-auto font-mono">
{`# GitHub Actions CI/CD 파이프라인
name: Security Scan

on: [push, pull_request]

jobs:
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: "p/owasp-top-ten"

  dast:
    runs-on: ubuntu-latest
    steps:
      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: 'https://staging.example.com'`}
          </pre>
        </div>
      </section>

      {/* 시큐어 코딩 원칙 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-7 h-7 text-blue-600" />
          시큐어 코딩 7대 원칙
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            { title: '입력값 검증', desc: '모든 사용자 입력을 신뢰하지 말고 검증', example: 'regex, 화이트리스트, 길이 제한' },
            { title: '최소 권한 원칙', desc: '필요한 최소한의 권한만 부여', example: 'RBAC, Least Privilege' },
            { title: '심층 방어', desc: '다층적 보안 메커니즘 적용', example: 'WAF + 코드 검증 + DB 권한' },
            { title: '안전한 기본값', desc: '보안적으로 안전한 기본 설정 사용', example: 'HTTPS, Secure Headers' },
            { title: '실패 시 안전', desc: '에러 발생 시에도 안전한 상태 유지', example: 'Fail-Safe, 에러 마스킹' },
            { title: '중요 데이터 보호', desc: '민감 정보는 암호화 저장', example: 'AES-256, bcrypt, Vault' },
            { title: '보안 업데이트', desc: '정기적인 보안 패치 적용', example: 'Dependabot, Renovate' },
          ].map((principle, idx) => (
            <div key={idx} className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h3 className="font-bold text-blue-900 dark:text-blue-300 mb-2">
                {idx + 1}. {principle.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{principle.desc}</p>
              <p className="text-xs text-blue-700 dark:text-blue-300">예: {principle.example}</p>
            </div>
          ))}
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 OWASP 공식 리소스',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'OWASP Top 10 - 2021',
                url: 'https://owasp.org/www-project-top-ten/',
                description: '웹 애플리케이션 보안 위협 톱 10 (3년마다 업데이트)'
              },
              {
                title: 'OWASP ASVS (Application Security Verification Standard)',
                url: 'https://owasp.org/www-project-application-security-verification-standard/',
                description: '애플리케이션 보안 검증 표준 - 4개 레벨 체계'
              },
              {
                title: 'OWASP Cheat Sheet Series',
                url: 'https://cheatsheetseries.owasp.org/',
                description: '150개 이상의 보안 주제별 실전 가이드'
              },
              {
                title: 'OWASP Top 10 for LLM Applications (2024)',
                url: 'https://genai.owasp.org/',
                description: 'LLM/AI 애플리케이션 특화 보안 가이드 (최신)'
              }
            ]
          },
          {
            title: '🔧 SAST/DAST 도구',
            icon: 'tools' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Semgrep',
                url: 'https://semgrep.dev/',
                description: '오픈소스 SAST 도구 - 30개 언어, 커스텀 룰 작성 가능'
              },
              {
                title: 'SonarQube Community Edition',
                url: 'https://www.sonarsource.com/products/sonarqube/',
                description: '무료 코드 품질 및 보안 분석 플랫폼'
              },
              {
                title: 'OWASP ZAP (Zed Attack Proxy)',
                url: 'https://www.zaproxy.org/',
                description: '오픈소스 DAST 도구 - 웹 취약점 스캐너'
              },
              {
                title: 'Snyk Open Source',
                url: 'https://snyk.io/',
                description: '의존성 취약점 스캔 - IDE/CI/CD 통합 (무료 플랜 有)'
              },
              {
                title: 'Nuclei Templates',
                url: 'https://github.com/projectdiscovery/nuclei-templates',
                description: '5,000개 이상의 취약점 탐지 템플릿'
              }
            ]
          },
          {
            title: '📖 보안 개발 가이드',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'CWE/SANS Top 25 Most Dangerous Software Errors',
                url: 'https://cwe.mitre.org/top25/',
                description: '업계가 선정한 가장 위험한 소프트웨어 취약점 25개'
              },
              {
                title: 'NIST Secure Software Development Framework (SSDF)',
                url: 'https://csrc.nist.gov/publications/detail/sp/800-218/final',
                description: '미국 표준기술연구소의 보안 개발 프레임워크'
              },
              {
                title: 'Mozilla Web Security Guidelines',
                url: 'https://infosec.mozilla.org/guidelines/web_security',
                description: 'Mozilla의 웹 보안 가이드라인 (CSP, HTTPS, Cookies 등)'
              },
              {
                title: 'PortSwigger Web Security Academy',
                url: 'https://portswigger.net/web-security',
                description: 'Burp Suite 제작사의 무료 웹 보안 교육 (실습 포함)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
