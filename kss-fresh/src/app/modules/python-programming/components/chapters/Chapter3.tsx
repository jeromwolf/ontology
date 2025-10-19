'use client';

import { BookOpen, Code2, Lightbulb, Play, CheckCircle2, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Header */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            함수와 모듈
          </h2>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            함수는 재사용 가능한 코드 블록을 만드는 핵심 도구입니다. 이 챕터에서는 함수 정의, 매개변수,
            반환값부터 모듈 시스템까지 Python의 코드 재사용 메커니즘을 완전히 마스터합니다.
          </p>
        </div>
      </section>

      {/* Section 1: 함수 기초 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          1. 함수 정의와 호출
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">기본 함수 정의</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 함수 정의 기본 구조
def greet(name):
    """사용자에게 인사하는 함수"""
    return f"안녕하세요, {name}님!"

# 함수 호출
message = greet("Kelly")
print(message)  # 출력: 안녕하세요, Kelly님!

# 독스트링 확인
print(greet.__doc__)  # 출력: 사용자에게 인사하는 함수`}
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-1" />
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">함수 작성 원칙</h5>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                    <span><strong>단일 책임:</strong> 하나의 함수는 하나의 작업만 수행</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                    <span><strong>명확한 이름:</strong> 함수의 동작을 정확히 표현하는 이름 사용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                    <span><strong>독스트링 작성:</strong> 함수의 목적과 사용법 문서화</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">반환값이 없는 함수</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`def print_info(name, age):
    """사용자 정보를 출력하는 함수 (반환값 없음)"""
    print(f"이름: {name}")
    print(f"나이: {age}")
    # return 문이 없으면 자동으로 None 반환

result = print_info("Alice", 25)
print(result)  # 출력: None

# 명시적으로 None 반환
def do_nothing():
    return None
    # 또는 그냥 return`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 2: 매개변수와 인자 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          2. 매개변수와 인자의 모든 것
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">기본 매개변수 (Default Parameters)</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`def create_user(name, age=18, city="Seoul"):
    """기본값이 있는 매개변수"""
    return {
        "name": name,
        "age": age,
        "city": city
    }

# 다양한 호출 방법
user1 = create_user("Alice")
print(user1)  # {'name': 'Alice', 'age': 18, 'city': 'Seoul'}

user2 = create_user("Bob", 25)
print(user2)  # {'name': 'Bob', 'age': 25, 'city': 'Seoul'}

user3 = create_user("Charlie", 30, "Busan")
print(user3)  # {'name': 'Charlie', 'age': 30, 'city': 'Busan'}`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">키워드 인자 (Keyword Arguments)</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 순서 무관하게 인자 전달 가능
user = create_user(city="Daegu", name="David", age=28)
print(user)  # {'name': 'David', 'age': 28, 'city': 'Daegu'}

# 위치 인자와 키워드 인자 혼합
user = create_user("Eve", city="Incheon")
print(user)  # {'name': 'Eve', 'age': 18, 'city': 'Incheon'}`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">가변 인자 (*args, **kwargs)</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# *args: 가변 개수의 위치 인자
def sum_all(*numbers):
    """임의 개수의 숫자를 합산"""
    total = 0
    for num in numbers:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs: 가변 개수의 키워드 인자
def print_info(**info):
    """임의 개수의 키워드 인자 출력"""
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Seoul")
# 출력:
# name: Alice
# age: 25
# city: Seoul

# 혼합 사용
def complex_function(required, *args, default="value", **kwargs):
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Default: {default}")
    print(f"Kwargs: {kwargs}")

complex_function(1, 2, 3, default="custom", key1="val1", key2="val2")`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 3: Lambda 함수 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          3. Lambda 함수 (익명 함수)
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 일반 함수
def add(x, y):
    return x + y

# Lambda 함수로 동일한 기능
add_lambda = lambda x, y: x + y

print(add(3, 5))         # 8
print(add_lambda(3, 5))  # 8

# Lambda 실전 활용 - 정렬
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]

# score 기준 정렬
sorted_students = sorted(students, key=lambda s: s["score"], reverse=True)
print(sorted_students)
# [{'name': 'Bob', 'score': 92}, {'name': 'Alice', 'score': 85}, ...]

# Lambda with map, filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-1" />
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">Lambda 사용 가이드</h5>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400 mt-1">✓</span>
                    <span>간단한 한 줄 함수에만 사용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-600 dark:text-yellow-400 mt-1">✓</span>
                    <span>map, filter, sorted 등의 인자로 활용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 dark:text-red-400 mt-1">✗</span>
                    <span>복잡한 로직은 일반 함수 사용 권장</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 모듈과 패키지 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          4. 모듈과 패키지로 코드 재사용하기
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">모듈 만들기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <p className="text-gray-700 dark:text-gray-300 mb-4 text-sm">
              <strong>math_utils.py</strong> 파일 생성:
            </p>
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`"""수학 유틸리티 함수 모음"""

def add(a, b):
    """두 수를 더함"""
    return a + b

def multiply(a, b):
    """두 수를 곱함"""
    return a * b

def power(base, exponent):
    """거듭제곱 계산"""
    return base ** exponent

# 모듈 레벨 변수
PI = 3.14159

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 실행되는 코드
    print("math_utils 모듈 테스트")
    print(f"2 + 3 = {add(2, 3)}")
    print(f"2 * 3 = {multiply(2, 3)}")`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">모듈 import 방법</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 방법 1: 전체 모듈 import
import math_utils
result = math_utils.add(5, 3)
print(result)  # 8
print(math_utils.PI)  # 3.14159

# 방법 2: 특정 함수만 import
from math_utils import add, multiply
result = add(5, 3)
print(result)  # 8

# 방법 3: 별칭 사용
import math_utils as mu
result = mu.multiply(4, 5)
print(result)  # 20

# 방법 4: 모든 함수 import (권장하지 않음)
from math_utils import *
result = power(2, 3)
print(result)  # 8`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">패키지 구조</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 디렉토리 구조
mypackage/
    __init__.py
    math/
        __init__.py
        basic.py
        advanced.py
    string/
        __init__.py
        utils.py

# __init__.py 파일 (mypackage/__init__.py)
"""My Package 초기화"""
from .math import basic, advanced
from .string import utils

__version__ = "1.0.0"

# 사용 예시
from mypackage.math import basic
result = basic.add(5, 3)

# 또는
from mypackage import math
result = math.basic.add(5, 3)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 5: 표준 라이브러리 활용 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          5. 주요 표준 라이브러리 모듈
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# math 모듈
import math
print(math.sqrt(16))      # 4.0
print(math.pi)            # 3.141592653589793
print(math.ceil(4.3))     # 5
print(math.floor(4.7))    # 4

# random 모듈
import random
print(random.randint(1, 10))        # 1~10 사이 정수
print(random.choice(['a', 'b', 'c']))  # 랜덤 선택
random.shuffle([1, 2, 3, 4, 5])     # 리스트 섞기

# datetime 모듈
from datetime import datetime, timedelta
now = datetime.now()
print(now)  # 현재 시간
tomorrow = now + timedelta(days=1)
print(tomorrow)

# os 모듈 (운영체제 기능)
import os
print(os.getcwd())  # 현재 작업 디렉토리
# os.mkdir('new_folder')  # 디렉토리 생성
# os.listdir('.')  # 파일 목록

# json 모듈
import json
data = {"name": "Alice", "age": 25}
json_str = json.dumps(data)  # Python -> JSON
print(json_str)  # '{"name": "Alice", "age": 25}'
parsed = json.loads(json_str)  # JSON -> Python
print(parsed)  # {'name': 'Alice', 'age': 25}`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 6: 실전 예제 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Play className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          6. 실전 프로젝트: 계산기 모듈
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <p className="text-gray-700 dark:text-gray-300 mb-4 text-sm">
              <strong>calculator.py</strong> - 완전한 계산기 모듈:
            </p>
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`"""강력한 계산기 모듈"""

def add(*numbers):
    """임의 개수의 숫자 합계"""
    return sum(numbers)

def subtract(a, b):
    """두 수의 차"""
    return a - b

def multiply(*numbers):
    """임의 개수의 숫자 곱"""
    result = 1
    for num in numbers:
        result *= num
    return result

def divide(a, b):
    """나눗셈 (에러 처리 포함)"""
    try:
        return a / b
    except ZeroDivisionError:
        return "0으로 나눌 수 없습니다"

def power(base, exponent=2):
    """거듭제곱 (기본값: 제곱)"""
    return base ** exponent

def calculate_average(*numbers):
    """평균 계산"""
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

# 사용 예시
if __name__ == "__main__":
    print("=== 계산기 테스트 ===")
    print(f"덧셈: {add(1, 2, 3, 4, 5)}")  # 15
    print(f"뺄셈: {subtract(10, 3)}")      # 7
    print(f"곱셈: {multiply(2, 3, 4)}")    # 24
    print(f"나눗셈: {divide(10, 2)}")      # 5.0
    print(f"거듭제곱: {power(2, 3)}")      # 8
    print(f"평균: {calculate_average(10, 20, 30)}")  # 20.0`}
            </pre>
          </div>
        </div>
      </section>

      {/* Simulator Link */}
      <section>
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-8 text-white">
          <div className="flex items-center gap-3 mb-4">
            <Play className="w-8 h-8" />
            <h3 className="text-2xl font-bold">인터랙티브 시뮬레이터</h3>
          </div>
          <p className="mb-6 text-blue-100">
            함수 실행 과정을 시각적으로 이해하고 직접 코드를 실행해보세요.
          </p>
          <Link
            href="/modules/python-programming/simulators/function-tracer"
            className="inline-flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
          >
            함수 실행 추적기 체험하기
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Key Takeaways */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
          핵심 요약
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <ul className="space-y-4">
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">1</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">함수는 재사용 가능한 코드 블록</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">def 키워드로 정의하고, 독스트링으로 문서화하며, return으로 값을 반환합니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">2</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">매개변수 유형 완벽 이해</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">기본값, 키워드 인자, *args, **kwargs를 활용하여 유연한 함수를 작성할 수 있습니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">3</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">모듈로 코드 조직화</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">관련된 함수들을 .py 파일로 묶어 import하여 재사용성을 극대화합니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">4</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">표준 라이브러리 활용</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">math, random, datetime, os, json 등 Python 내장 모듈을 적극 활용합니다.</p>
              </div>
            </li>
          </ul>
        </div>
      </section>

      {/* Next Steps */}
      <section>
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">다음 단계</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            함수와 모듈을 마스터했다면, 이제 파일 입출력으로 실제 데이터를 다루는 방법을 배워봅시다!
          </p>
          <Link
            href="/modules/python-programming/file-io"
            className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold hover:gap-3 transition-all"
          >
            Chapter 4: 파일 입출력과 데이터 처리
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}
