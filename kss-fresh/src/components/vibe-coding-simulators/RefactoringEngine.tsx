'use client'

import React, { useState } from 'react'
import { RefreshCw, Code2, TrendingDown, Zap, ArrowRight, Copy, Check } from 'lucide-react'

type RefactoringStrategy =
  | 'extract-function'
  | 'rename'
  | 'inline'
  | 'remove-duplicate'
  | 'simplify-conditional'
  | 'optimize-loop'

interface RefactoringResult {
  before: string
  after: string
  strategy: RefactoringStrategy
  metrics: {
    linesReduced: number
    complexityBefore: number
    complexityAfter: number
    readabilityScore: number
  }
  explanation: string
}

export default function RefactoringEngine() {
  const [legacyCode, setLegacyCode] = useState('')
  const [selectedStrategy, setSelectedStrategy] = useState<RefactoringStrategy>('extract-function')
  const [result, setResult] = useState<RefactoringResult | null>(null)
  const [isRefactoring, setIsRefactoring] = useState(false)
  const [copied, setCopied] = useState(false)

  const strategies: { value: RefactoringStrategy; label: string; icon: string }[] = [
    { value: 'extract-function', label: '함수 추출', icon: '📦' },
    { value: 'rename', label: '변수명 개선', icon: '✏️' },
    { value: 'inline', label: '인라인 단순화', icon: '➡️' },
    { value: 'remove-duplicate', label: '중복 제거', icon: '🔄' },
    { value: 'simplify-conditional', label: '조건문 단순화', icon: '🔀' },
    { value: 'optimize-loop', label: '루프 최적화', icon: '⚡' }
  ]

  const exampleCodes: { title: string; code: string; strategy: RefactoringStrategy }[] = [
    {
      title: '함수 추출',
      strategy: 'extract-function',
      code: `def process_order(order):
    # 가격 계산
    total = 0
    for item in order['items']:
        total += item['price'] * item['quantity']

    # 할인 적용
    if order['customer_type'] == 'premium':
        total *= 0.9
    elif order['customer_type'] == 'vip':
        total *= 0.8

    # 배송비 추가
    if total < 50:
        total += 5

    return total`
    },
    {
      title: '변수명 개선',
      strategy: 'rename',
      code: `def calc(x, y, z):
    a = x * y
    b = a + z
    c = b / 2
    return c

def proc_data(d):
    r = []
    for i in d:
        t = i * 2
        r.append(t)
    return r`
    },
    {
      title: '중복 제거',
      strategy: 'remove-duplicate',
      code: `def send_email_to_user(user):
    subject = "Welcome"
    message = f"Hello {user.name}"
    send_email(user.email, subject, message)
    log(f"Email sent to {user.email}")

def send_email_to_admin(admin):
    subject = "Welcome"
    message = f"Hello {admin.name}"
    send_email(admin.email, subject, message)
    log(f"Email sent to {admin.email}")`
    }
  ]

  const refactorCode = (code: string, strategy: RefactoringStrategy): RefactoringResult => {
    let after = code
    let explanation = ''
    let linesReduced = 0
    let complexityBefore = 10
    let complexityAfter = 5
    let readabilityScore = 75

    switch (strategy) {
      case 'extract-function':
        after = `def calculate_subtotal(items):
    """아이템 목록의 소계를 계산합니다."""
    return sum(item['price'] * item['quantity'] for item in items)

def apply_customer_discount(total, customer_type):
    """고객 등급에 따른 할인을 적용합니다."""
    discounts = {'premium': 0.9, 'vip': 0.8}
    return total * discounts.get(customer_type, 1.0)

def add_shipping_fee(total, threshold=50, fee=5):
    """배송비를 추가합니다."""
    return total + fee if total < threshold else total

def process_order(order):
    """주문을 처리하고 최종 금액을 반환합니다."""
    total = calculate_subtotal(order['items'])
    total = apply_customer_discount(total, order['customer_type'])
    total = add_shipping_fee(total)
    return total`
        explanation = '긴 함수를 여러 개의 작은 함수로 분리했습니다. 각 함수는 단일 책임을 가지며, 재사용 가능하고 테스트하기 쉽습니다.'
        linesReduced = -2
        complexityBefore = 12
        complexityAfter = 4
        readabilityScore = 92
        break

      case 'rename':
        after = `def calculate_average(price, quantity, discount):
    """평균 가격을 계산합니다."""
    subtotal = price * quantity
    total = subtotal + discount
    average = total / 2
    return average

def process_data(data_list):
    """데이터를 처리하고 두 배로 만듭니다."""
    result = []
    for item in data_list:
        doubled_value = item * 2
        result.append(doubled_value)
    return result`
        explanation = '의미 없는 변수명(x, y, a, b, c)을 명확한 이름으로 변경했습니다. 코드의 의도가 명확해졌습니다.'
        linesReduced = 0
        complexityBefore = 8
        complexityAfter = 6
        readabilityScore = 88
        break

      case 'inline':
        after = `def calculate_average(price, quantity, discount):
    """평균 가격을 계산합니다."""
    return ((price * quantity) + discount) / 2

def process_data(data_list):
    """데이터를 처리하고 두 배로 만듭니다."""
    return [item * 2 for item in data_list]`
        explanation = '불필요한 중간 변수를 제거하고 표현식을 인라인화했습니다. List comprehension을 활용하여 더 pythonic한 코드가 되었습니다.'
        linesReduced = 6
        complexityBefore = 8
        complexityAfter = 3
        readabilityScore = 85
        break

      case 'remove-duplicate':
        after = `def send_welcome_email(recipient, recipient_type="user"):
    """환영 이메일을 발송합니다.

    Args:
        recipient: 이메일을 받을 사용자 객체
        recipient_type: 수신자 유형 (user 또는 admin)
    """
    subject = "Welcome"
    message = f"Hello {recipient.name}"
    send_email(recipient.email, subject, message)
    log(f"Email sent to {recipient.email} ({recipient_type})")

def send_email_to_user(user):
    send_welcome_email(user, "user")

def send_email_to_admin(admin):
    send_welcome_email(admin, "admin")`
        explanation = '중복된 코드를 공통 함수로 추출했습니다. DRY (Don\'t Repeat Yourself) 원칙을 따릅니다.'
        linesReduced = 4
        complexityBefore = 10
        complexityAfter = 5
        readabilityScore = 90
        break

      case 'simplify-conditional':
        after = `def get_user_status(age, is_verified, has_premium):
    """사용자 상태를 반환합니다."""
    if age < 18:
        return "minor"

    if not is_verified:
        return "unverified"

    return "premium" if has_premium else "standard"

def calculate_discount(user_type):
    """할인율을 계산합니다."""
    discounts = {
        "vip": 0.2,
        "premium": 0.1,
        "standard": 0.05
    }
    return discounts.get(user_type, 0.0)`
        explanation = '중첩된 조건문을 early return과 guard clauses로 단순화했습니다. Dictionary lookup을 활용하여 if-elif 체인을 제거했습니다.'
        linesReduced = 3
        complexityBefore = 15
        complexityAfter = 6
        readabilityScore = 87
        break

      case 'optimize-loop':
        after = `def find_duplicates(arr):
    """배열에서 중복된 요소를 찾습니다. O(n) 시간 복잡도"""
    seen = set()
    duplicates = set()

    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def filter_and_transform(data, condition):
    """조건을 만족하는 데이터를 필터링하고 변환합니다."""
    return [transform(item) for item in data if condition(item)]`
        explanation = '중첩 루프(O(n²))를 Set 자료구조 활용(O(n))으로 최적화했습니다. List comprehension으로 코드를 간결하게 만들었습니다.'
        linesReduced = 5
        complexityBefore = 20
        complexityAfter = 8
        readabilityScore = 84
        break
    }

    return {
      before: code,
      after,
      strategy,
      metrics: {
        linesReduced,
        complexityBefore,
        complexityAfter,
        readabilityScore
      },
      explanation
    }
  }

  const handleRefactor = () => {
    if (legacyCode.trim().length === 0) return

    setIsRefactoring(true)
    setTimeout(() => {
      const refactored = refactorCode(legacyCode, selectedStrategy)
      setResult(refactored)
      setIsRefactoring(false)
    }, 1500)
  }

  const loadExample = (example: typeof exampleCodes[0]) => {
    setLegacyCode(example.code)
    setSelectedStrategy(example.strategy)
    setResult(null)
  }

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getComplexityColor = (complexity: number) => {
    if (complexity <= 5) return 'text-green-600 dark:text-green-400'
    if (complexity <= 10) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <RefreshCw className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              AI 리팩토링 엔진
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            레거시 코드를 현대적이고 유지보수 가능한 코드로 변환
          </p>
        </div>

        {/* Strategy Selection */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">리팩토링 전략:</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {strategies.map((strategy) => (
              <button
                key={strategy.value}
                onClick={() => setSelectedStrategy(strategy.value)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  selectedStrategy === strategy.value
                    ? 'border-purple-600 bg-purple-50 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:border-purple-400'
                }`}
              >
                <div className="text-2xl mb-1">{strategy.icon}</div>
                <div className="text-xs font-medium">{strategy.label}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Example Buttons */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">예제 코드:</h3>
          <div className="flex flex-wrap gap-3">
            {exampleCodes.map((example, index) => (
              <button
                key={index}
                onClick={() => loadExample(example)}
                className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg border border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900 transition-colors text-sm"
              >
                {example.title}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Before */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
                <Code2 className="w-5 h-5 text-red-600" />
                레거시 코드
              </h2>
              <button
                onClick={handleRefactor}
                disabled={legacyCode.trim().length === 0 || isRefactoring}
                className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isRefactoring ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    변환 중...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4" />
                    리팩토링
                  </>
                )}
              </button>
            </div>

            <textarea
              value={legacyCode}
              onChange={(e) => setLegacyCode(e.target.value)}
              placeholder="리팩토링할 코드를 입력하세요..."
              className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-mono text-sm rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-red-500 resize-none"
              spellCheck={false}
            />
          </div>

          {/* After */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-green-600" />
                리팩토링된 코드
              </h2>
              {result && (
                <button
                  onClick={() => handleCopy(result.after)}
                  className="p-2 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
                  title="복사"
                >
                  {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                </button>
              )}
            </div>

            {result ? (
              <div className="h-96 p-4 bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800 rounded-lg overflow-y-auto">
                <pre className="text-gray-800 dark:text-gray-200 font-mono text-sm whitespace-pre-wrap">
                  {result.after}
                </pre>
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-300 dark:border-gray-700">
                <div className="text-center text-gray-400 dark:text-gray-600">
                  <ArrowRight className="w-12 h-12 mx-auto mb-2" />
                  <p>리팩토링 결과가 여기에 표시됩니다</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Metrics */}
        {result && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Metrics Cards */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-6">개선 메트릭</h2>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">줄 수 변화</div>
                  <div className={`text-3xl font-bold ${result.metrics.linesReduced > 0 ? 'text-green-600' : 'text-blue-600'}`}>
                    {result.metrics.linesReduced > 0 ? '-' : '+'}{Math.abs(result.metrics.linesReduced)}
                  </div>
                </div>

                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">가독성 점수</div>
                  <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                    {result.metrics.readabilityScore}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">복잡도 (이전)</span>
                    <span className={`text-lg font-bold ${getComplexityColor(result.metrics.complexityBefore)}`}>
                      {result.metrics.complexityBefore}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-red-500"
                      style={{ width: `${Math.min(result.metrics.complexityBefore * 5, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-center">
                  <TrendingDown className="w-6 h-6 text-green-600" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">복잡도 (이후)</span>
                    <span className={`text-lg font-bold ${getComplexityColor(result.metrics.complexityAfter)}`}>
                      {result.metrics.complexityAfter}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-green-500"
                      style={{ width: `${Math.min(result.metrics.complexityAfter * 5, 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl text-white">
                <div className="text-sm mb-1">복잡도 감소율</div>
                <div className="text-4xl font-bold">
                  {Math.round(((result.metrics.complexityBefore - result.metrics.complexityAfter) / result.metrics.complexityBefore) * 100)}%
                </div>
              </div>
            </div>

            {/* Explanation */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4">💡 개선 내용</h2>
              <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
                {result.explanation}
              </p>

              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <h3 className="font-semibold text-gray-800 dark:text-white mb-2">리팩토링 원칙</h3>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>✓ 단일 책임 원칙 (SRP)</li>
                  <li>✓ DRY (Don't Repeat Yourself)</li>
                  <li>✓ 명확한 변수명 사용</li>
                  <li>✓ 복잡도 최소화</li>
                  <li>✓ 가독성 향상</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
