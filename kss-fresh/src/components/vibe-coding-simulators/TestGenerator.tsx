'use client'

import React, { useState } from 'react'
import { FlaskConical, CheckCircle, XCircle, Play, Copy, Check } from 'lucide-react'

type TestFramework = 'jest' | 'pytest' | 'junit'

interface TestCase {
  name: string
  code: string
  type: 'normal' | 'edge' | 'error'
}

interface TestResult {
  framework: TestFramework
  testCases: TestCase[]
  coverage: number
  edgeCases: string[]
  setupCode: string
}

export default function TestGenerator() {
  const [functionCode, setFunctionCode] = useState('')
  const [framework, setFramework] = useState<TestFramework>('pytest')
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [copied, setCopied] = useState(false)

  const exampleFunctions = [
    {
      title: 'Python 함수',
      framework: 'pytest' as TestFramework,
      code: `def calculate_discount(price, discount_percent):
    """가격에 할인율을 적용합니다.

    Args:
        price: 원가
        discount_percent: 할인율 (0-100)

    Returns:
        할인된 가격
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")

    return price * (1 - discount_percent / 100)`
    },
    {
      title: 'JavaScript 함수',
      framework: 'jest' as TestFramework,
      code: `function validateEmail(email) {
  if (typeof email !== 'string') {
    throw new TypeError('Email must be a string');
  }

  const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return emailRegex.test(email);
}`
    },
    {
      title: 'Java 메서드',
      framework: 'junit' as TestFramework,
      code: `public int findMax(int[] numbers) {
    if (numbers == null || numbers.length == 0) {
        throw new IllegalArgumentException("Array cannot be null or empty");
    }

    int max = numbers[0];
    for (int num : numbers) {
        if (num > max) {
            max = num;
        }
    }
    return max;
}`
    }
  ]

  const generateTests = (code: string, selectedFramework: TestFramework): TestResult => {
    let testCases: TestCase[] = []
    let setupCode = ''
    let coverage = 0
    let edgeCases: string[] = []

    if (selectedFramework === 'pytest') {
      setupCode = `import pytest\nfrom your_module import calculate_discount\n`

      testCases = [
        {
          name: 'test_normal_discount',
          type: 'normal',
          code: `def test_normal_discount():
    """정상적인 할인 계산을 테스트합니다."""
    assert calculate_discount(100, 10) == 90
    assert calculate_discount(50, 20) == 40
    assert calculate_discount(200, 50) == 100`
        },
        {
          name: 'test_zero_discount',
          type: 'edge',
          code: `def test_zero_discount():
    """할인율이 0%일 때를 테스트합니다."""
    assert calculate_discount(100, 0) == 100`
        },
        {
          name: 'test_full_discount',
          type: 'edge',
          code: `def test_full_discount():
    """할인율이 100%일 때를 테스트합니다."""
    assert calculate_discount(100, 100) == 0`
        },
        {
          name: 'test_negative_price',
          type: 'error',
          code: `def test_negative_price():
    """음수 가격에 대한 예외를 테스트합니다."""
    with pytest.raises(ValueError, match="Price cannot be negative"):
        calculate_discount(-10, 10)`
        },
        {
          name: 'test_invalid_discount',
          type: 'error',
          code: `def test_invalid_discount():
    """유효하지 않은 할인율에 대한 예외를 테스트합니다."""
    with pytest.raises(ValueError, match="Discount must be between 0 and 100"):
        calculate_discount(100, 150)
    with pytest.raises(ValueError):
        calculate_discount(100, -5)`
        },
        {
          name: 'test_zero_price',
          type: 'edge',
          code: `def test_zero_price():
    """가격이 0일 때를 테스트합니다."""
    assert calculate_discount(0, 10) == 0`
        },
        {
          name: 'test_floating_point',
          type: 'edge',
          code: `def test_floating_point():
    """소수점 가격과 할인율을 테스트합니다."""
    assert abs(calculate_discount(99.99, 10) - 89.991) < 0.01`
        }
      ]

      edgeCases = [
        '음수 가격 입력',
        '0원 가격',
        '할인율 0%',
        '할인율 100%',
        '할인율 범위 초과 (음수, 100 이상)',
        '소수점 가격',
        '매우 큰 가격 값'
      ]

      coverage = 95
    } else if (selectedFramework === 'jest') {
      setupCode = `const { validateEmail } = require('./your-module');\n`

      testCases = [
        {
          name: 'should validate correct email formats',
          type: 'normal',
          code: `test('should validate correct email formats', () => {
  expect(validateEmail('user@example.com')).toBe(true);
  expect(validateEmail('test.user@domain.co.kr')).toBe(true);
  expect(validateEmail('name+tag@company.org')).toBe(true);
});`
        },
        {
          name: 'should reject invalid email formats',
          type: 'normal',
          code: `test('should reject invalid email formats', () => {
  expect(validateEmail('invalid')).toBe(false);
  expect(validateEmail('missing@domain')).toBe(false);
  expect(validateEmail('@nodomain.com')).toBe(false);
  expect(validateEmail('spaces in@email.com')).toBe(false);
});`
        },
        {
          name: 'should handle empty string',
          type: 'edge',
          code: `test('should handle empty string', () => {
  expect(validateEmail('')).toBe(false);
});`
        },
        {
          name: 'should throw error for non-string input',
          type: 'error',
          code: `test('should throw error for non-string input', () => {
  expect(() => validateEmail(null)).toThrow(TypeError);
  expect(() => validateEmail(undefined)).toThrow(TypeError);
  expect(() => validateEmail(123)).toThrow(TypeError);
  expect(() => validateEmail({})).toThrow(TypeError);
});`
        },
        {
          name: 'should handle special characters',
          type: 'edge',
          code: `test('should handle special characters', () => {
  expect(validateEmail('user!@example.com')).toBe(false);
  expect(validateEmail('user#@example.com')).toBe(false);
});`
        }
      ]

      edgeCases = [
        '빈 문자열',
        'null/undefined 입력',
        '숫자/객체 타입 입력',
        '특수문자 포함',
        '공백 포함',
        '@가 없는 경우',
        '도메인 없는 경우'
      ]

      coverage = 92
    } else if (selectedFramework === 'junit') {
      setupCode = `import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;
\npublic class YourClassTest {\n    private YourClass instance;\n
    @Before
    public void setUp() {
        instance = new YourClass();
    }\n`

      testCases = [
        {
          name: 'testFindMaxNormalCase',
          type: 'normal',
          code: `    @Test
    public void testFindMaxNormalCase() {
        int[] numbers = {3, 7, 2, 9, 1};
        assertEquals(9, instance.findMax(numbers));
    }`
        },
        {
          name: 'testFindMaxSingleElement',
          type: 'edge',
          code: `    @Test
    public void testFindMaxSingleElement() {
        int[] numbers = {42};
        assertEquals(42, instance.findMax(numbers));
    }`
        },
        {
          name: 'testFindMaxAllNegative',
          type: 'edge',
          code: `    @Test
    public void testFindMaxAllNegative() {
        int[] numbers = {-5, -2, -10, -1};
        assertEquals(-1, instance.findMax(numbers));
    }`
        },
        {
          name: 'testFindMaxNullArray',
          type: 'error',
          code: `    @Test(expected = IllegalArgumentException.class)
    public void testFindMaxNullArray() {
        instance.findMax(null);
    }`
        },
        {
          name: 'testFindMaxEmptyArray',
          type: 'error',
          code: `    @Test(expected = IllegalArgumentException.class)
    public void testFindMaxEmptyArray() {
        int[] numbers = {};
        instance.findMax(numbers);
    }`
        },
        {
          name: 'testFindMaxDuplicateMax',
          type: 'edge',
          code: `    @Test
    public void testFindMaxDuplicateMax() {
        int[] numbers = {5, 9, 3, 9, 1};
        assertEquals(9, instance.findMax(numbers));
    }`
        }
      ]

      edgeCases = [
        'null 배열',
        '빈 배열',
        '단일 요소',
        '모든 음수',
        '최대값 중복',
        'Integer.MAX_VALUE 포함',
        'Integer.MIN_VALUE 포함'
      ]

      coverage = 88
    }

    return {
      framework: selectedFramework,
      testCases,
      coverage,
      edgeCases,
      setupCode
    }
  }

  const handleGenerate = () => {
    if (functionCode.trim().length === 0) return

    setIsGenerating(true)
    setTimeout(() => {
      const result = generateTests(functionCode, framework)
      setTestResult(result)
      setIsGenerating(false)
    }, 1500)
  }

  const loadExample = (example: typeof exampleFunctions[0]) => {
    setFunctionCode(example.code)
    setFramework(example.framework)
    setTestResult(null)
  }

  const handleCopy = () => {
    if (!testResult) return

    const fullCode = testResult.setupCode + '\n' + testResult.testCases.map(tc => tc.code).join('\n\n')
    navigator.clipboard.writeText(fullCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getTestTypeColor = (type: TestCase['type']) => {
    switch (type) {
      case 'normal':
        return 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
      case 'edge':
        return 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300'
      case 'error':
        return 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
    }
  }

  const getTestTypeLabel = (type: TestCase['type']) => {
    switch (type) {
      case 'normal':
        return '정상'
      case 'edge':
        return '경계값'
      case 'error':
        return '예외'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <FlaskConical className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              AI 테스트 생성기
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            함수 코드로부터 완전한 테스트 코드 자동 생성
          </p>
        </div>

        {/* Framework Selection */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">테스트 프레임워크:</h3>
          <div className="flex gap-3">
            {(['jest', 'pytest', 'junit'] as TestFramework[]).map((fw) => (
              <button
                key={fw}
                onClick={() => setFramework(fw)}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  framework === fw
                    ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:border-purple-400'
                }`}
              >
                {fw.charAt(0).toUpperCase() + fw.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Example Buttons */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">예제 함수:</h3>
          <div className="flex flex-wrap gap-3">
            {exampleFunctions.map((example, index) => (
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

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white">함수 코드</h2>
              <button
                onClick={handleGenerate}
                disabled={functionCode.trim().length === 0 || isGenerating}
                className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    테스트 생성
                  </>
                )}
              </button>
            </div>

            <textarea
              value={functionCode}
              onChange={(e) => setFunctionCode(e.target.value)}
              placeholder="테스트할 함수 코드를 입력하세요..."
              className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-mono text-sm rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
              spellCheck={false}
            />
          </div>

          {/* Output */}
          <div>
            {testResult ? (
              <div className="space-y-6">
                {/* Coverage Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold text-gray-800 dark:text-white">커버리지 리포트</h2>
                    <button
                      onClick={handleCopy}
                      className="p-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                      title="전체 코드 복사"
                    >
                      {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                    </button>
                  </div>

                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-700 dark:text-gray-300">테스트 커버리지</span>
                      <span className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                        {testResult.coverage}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className="h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-600 transition-all duration-1000"
                        style={{ width: `${testResult.coverage}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="text-xl font-bold text-green-600 dark:text-green-400">
                        {testResult.testCases.filter(tc => tc.type === 'normal').length}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">정상 케이스</div>
                    </div>
                    <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                      <div className="text-xl font-bold text-yellow-600 dark:text-yellow-400">
                        {testResult.testCases.filter(tc => tc.type === 'edge').length}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">경계값</div>
                    </div>
                    <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="text-xl font-bold text-red-600 dark:text-red-400">
                        {testResult.testCases.filter(tc => tc.type === 'error').length}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">예외 처리</div>
                    </div>
                  </div>
                </div>

                {/* Test Cases */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 max-h-[600px] overflow-y-auto">
                  <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">
                    생성된 테스트 ({testResult.testCases.length})
                  </h3>

                  {/* Setup Code */}
                  <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Setup:</div>
                    <pre className="text-xs text-gray-800 dark:text-gray-200 font-mono overflow-x-auto">
                      {testResult.setupCode}
                    </pre>
                  </div>

                  {/* Test Cases */}
                  <div className="space-y-4">
                    {testResult.testCases.map((testCase, index) => (
                      <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                        <div className="p-3 bg-gray-50 dark:bg-gray-900 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-600" />
                            <span className="font-mono text-sm text-gray-800 dark:text-gray-200">
                              {testCase.name}
                            </span>
                          </div>
                          <span className={`text-xs px-2 py-1 rounded ${getTestTypeColor(testCase.type)}`}>
                            {getTestTypeLabel(testCase.type)}
                          </span>
                        </div>
                        <pre className="p-4 bg-gray-900 text-green-400 text-xs font-mono overflow-x-auto">
                          {testCase.code}
                        </pre>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Edge Cases Suggestions */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">💡 테스트해야 할 엣지 케이스</h3>
                  <ul className="space-y-2">
                    {testResult.edgeCases.map((edgeCase, index) => (
                      <li key={index} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <span className="text-purple-600 dark:text-purple-400">•</span>
                        {edgeCase}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 h-full flex items-center justify-center">
                <div className="text-center text-gray-400 dark:text-gray-600">
                  <FlaskConical className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-lg">함수 코드를 입력하고<br />테스트 생성 버튼을 클릭하세요</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
