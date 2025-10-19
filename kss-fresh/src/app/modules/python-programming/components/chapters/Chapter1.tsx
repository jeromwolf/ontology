'use client';

import { BookOpen, Code2, Terminal, Lightbulb, CheckCircle2, Play } from 'lucide-react';
import Link from 'next/link';

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-6">
          <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Python 기초와 문법
          </h2>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            Python은 간결하고 읽기 쉬운 문법으로 프로그래밍 입문자에게 가장 적합한 언어입니다.
            이 챕터에서는 Python의 기본 문법과 개발 환경 설정부터 시작하여,
            프로그래밍의 기초를 탄탄히 다질 수 있습니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. Python 설치와 개발 환경 설정하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Python 공식 사이트에서 다운로드하고 IDE 설정까지
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-indigo-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 기본 문법과 데이터 타입 이해하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              숫자, 문자열, 불리언 등 핵심 데이터 타입 마스터
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-purple-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. 변수와 연산자 완전 정복하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              변수 선언부터 산술, 비교, 논리 연산자까지
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-pink-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. Python REPL 활용하여 실습하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              대화형 인터프리터로 즉시 코드 실행 및 테스트
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: Python 설치 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Terminal className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          1. Python 설치와 환경 설정
        </h3>

        <div className="space-y-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-10 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Python 다운로드
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              공식 사이트 <a href="https://www.python.org" target="_blank" rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline">python.org</a>에서
              최신 버전(Python 3.12+)을 다운로드하세요.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <div className="text-gray-600 dark:text-gray-400 mb-2"># Windows</div>
              <div className="text-gray-900 dark:text-white mb-4">
                Python-3.12.0.exe 실행 후 "Add Python to PATH" 체크
              </div>

              <div className="text-gray-600 dark:text-gray-400 mb-2"># macOS</div>
              <div className="text-gray-900 dark:text-white mb-4">
                brew install python3
              </div>

              <div className="text-gray-600 dark:text-gray-400 mb-2"># Linux (Ubuntu/Debian)</div>
              <div className="text-gray-900 dark:text-white">
                sudo apt update && sudo apt install python3 python3-pip
              </div>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: 버전 확인하기
                </h5>
                <p className="text-sm text-amber-800 dark:text-amber-300 mb-3">
                  터미널에서 Python이 제대로 설치되었는지 확인하세요:
                </p>
                <code className="block bg-amber-100 dark:bg-amber-900/50 text-amber-900 dark:text-amber-200 p-3 rounded text-sm">
                  python3 --version
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 기본 문법 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          2. 기본 문법과 데이터 타입
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Hello, World! - 첫 번째 프로그램
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 가장 간단한 Python 프로그램
print("Hello, World!")

# 출력:
# Hello, World!`}
              </pre>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mt-3">
              <code className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-1 rounded">
                print()
              </code> 함수는 화면에 텍스트를 출력합니다. Python에서 가장 자주 사용하는 함수입니다.
            </p>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              기본 데이터 타입
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500"></span>
                  숫자 (Numbers)
                </h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div className="text-gray-700 dark:text-gray-300">
                    integer = 42      # 정수<br/>
                    floating = 3.14   # 실수<br/>
                    complex_num = 1+2j # 복소수
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                  문자열 (Strings)
                </h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div className="text-gray-700 dark:text-gray-300">
                    text1 = "Hello"<br/>
                    text2 = 'World'<br/>
                    multi = """여러 줄<br/>
                    문자열"""
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                  불리언 (Boolean)
                </h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div className="text-gray-700 dark:text-gray-300">
                    is_active = True<br/>
                    is_complete = False<br/>
                    # True, False는 대문자!
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-gray-500"></span>
                  None (없음)
                </h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 font-mono text-sm">
                  <div className="text-gray-700 dark:text-gray-300">
                    empty = None<br/>
                    # 값이 없음을 나타냄<br/>
                    # null, nil과 유사
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 변수와 연산자 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3. 변수와 연산자
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              변수 선언과 할당
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# Python은 변수 선언 시 타입을 명시하지 않습니다 (동적 타이핑)
name = "Kelly"          # 문자열
age = 25                # 정수
height = 175.5          # 실수
is_student = True       # 불리언

# 여러 변수 동시 할당
x, y, z = 1, 2, 3
a = b = c = 0          # 같은 값으로 초기화

# 값 교환 (swap)
x, y = y, x            # Python의 강력한 기능!

print(f"이름: {name}, 나이: {age}")
# 출력: 이름: Kelly, 나이: 25`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              연산자
            </h4>

            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-3">산술 연산자</h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="grid grid-cols-2 gap-3 text-gray-700 dark:text-gray-300">
                    <div>+ (더하기): 10 + 3 = 13</div>
                    <div>- (빼기): 10 - 3 = 7</div>
                    <div>* (곱하기): 10 * 3 = 30</div>
                    <div>/ (나누기): 10 / 3 = 3.333...</div>
                    <div>// (정수 나누기): 10 // 3 = 3</div>
                    <div>% (나머지): 10 % 3 = 1</div>
                    <div>** (거듭제곱): 10 ** 3 = 1000</div>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-3">비교 연산자</h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="grid grid-cols-2 gap-3 text-gray-700 dark:text-gray-300">
                    <div>== (같음): 5 == 5 → True</div>
                    <div>!= (다름): 5 != 3 → True</div>
                    <div>&gt; (크다): 5 &gt; 3 → True</div>
                    <div>&lt; (작다): 5 &lt; 3 → False</div>
                    <div>&gt;= (크거나 같음): 5 &gt;= 5 → True</div>
                    <div>&lt;= (작거나 같음): 3 &lt;= 5 → True</div>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-3">논리 연산자</h5>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
                  <pre className="text-gray-700 dark:text-gray-300">
{`x = True
y = False

and (그리고): x and y → False
or  (또는):   x or y  → True
not (부정):   not x   → False`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: Python REPL */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Play className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          4. Python REPL 실습
        </h3>

        <div className="space-y-4">
          <p className="text-gray-700 dark:text-gray-300">
            REPL(Read-Eval-Print Loop)은 Python 코드를 즉시 실행하고 결과를 확인할 수 있는
            대화형 환경입니다. 터미널에서 <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">python3</code>를
            입력하면 시작됩니다.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`$ python3
Python 3.12.0 (main, Oct  2 2023, 10:00:00)
>>> print("Hello, REPL!")
Hello, REPL!

>>> 2 + 2
4

>>> name = "Python"
>>> f"I love {name}!"
'I love Python!'

>>> exit()  # REPL 종료`}
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Terminal className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                  실습: Python REPL 시뮬레이터
                </h5>
                <p className="text-sm text-blue-800 dark:text-blue-300 mb-4">
                  브라우저에서 바로 Python 코드를 실행해보세요!
                  실시간 피드백과 함께 안전하게 실습할 수 있습니다.
                </p>
                <Link
                  href="/modules/python-programming/simulators/python-repl"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                  REPL 시뮬레이터 실행
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Practice Exercises */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          연습 문제
        </h3>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 1: 변수와 출력
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              자신의 이름, 나이, 좋아하는 프로그래밍 언어를 변수에 저장하고,
              f-string을 사용하여 출력하세요.
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <div className="text-gray-600 dark:text-gray-400 mb-2"># 예시 출력:</div>
              <div className="text-gray-900 dark:text-white">
                안녕하세요! 저는 Kelly이고, 25살입니다. Python을 좋아합니다!
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 2: 계산기 만들기
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              두 개의 숫자를 변수에 저장하고, 사칙연산(+, -, *, /)의 결과를 출력하세요.
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <pre className="text-gray-700 dark:text-gray-300">
{`a = 15
b = 4

print(f"{a} + {b} = {a + b}")
print(f"{a} - {b} = {a - b}")
print(f"{a} * {b} = {a * b}")
print(f"{a} / {b} = {a / b}")`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 3: 변수 교환
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              두 변수의 값을 교환하는 코드를 작성하세요. (임시 변수 없이)
            </p>
          </div>
        </div>
      </section>

      {/* Summary */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          요약
        </h3>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                Python은 설치가 쉽고, 공식 사이트에서 무료로 다운로드할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                기본 데이터 타입에는 숫자, 문자열, 불리언, None이 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                변수는 동적 타이핑으로 자동으로 타입이 결정됩니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                산술, 비교, 논리 연산자를 사용하여 다양한 계산과 비교를 수행할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                Python REPL은 코드를 즉시 실행하고 결과를 확인하는 강력한 도구입니다
              </span>
            </li>
          </ul>
        </div>
      </section>

      {/* Next Steps */}
      <section className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          다음 단계
        </h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          기본 문법을 익혔다면, 이제 Python의 강력한 컬렉션(리스트, 딕셔너리, 집합)을
          배울 차례입니다. Chapter 2에서 데이터를 효율적으로 관리하는 방법을 학습하세요!
        </p>
        <Link
          href="/modules/python-programming/data-types-collections"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 2: 데이터 타입과 컬렉션 →
        </Link>
      </section>
    </div>
  );
}
