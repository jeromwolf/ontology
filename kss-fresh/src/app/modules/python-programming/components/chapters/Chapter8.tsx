'use client';

import { AtSign, CheckCircle2, Code2, Lightbulb, Play, Zap } from 'lucide-react';
import Link from 'next/link';

export default function Chapter8() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <AtSign className="w-6 h-6 text-violet-600 dark:text-violet-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            데코레이터와 제너레이터
          </h2>
        </div>

        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            데코레이터와 제너레이터는 Python의 고급 기능으로 코드의 재사용성과 효율성을 극대화합니다.
            데코레이터는 함수를 수정하지 않고 기능을 추가하고, 제너레이터는 메모리 효율적으로 대량 데이터를 처리합니다.
            프로페셔널 Python 개발자의 필수 스킬입니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-violet-600 dark:text-violet-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-violet-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. 함수 데코레이터의 원리 이해하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              @ 문법과 클로저를 활용한 함수 래핑
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-purple-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 커스텀 데코레이터 작성 마스터
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              로깅, 타이밍, 캐싱 등 실전 데코레이터 구현
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-fuchsia-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. 제너레이터 함수로 메모리 효율 향상
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              yield 키워드로 지연 평가 구현
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-pink-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. yield와 이터레이터 완전 정복
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              커스텀 이터레이터 & 양방향 제너레이터
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: 데코레이터 기초 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <AtSign className="w-6 h-6 text-violet-600 dark:text-violet-400" />
          1. 데코레이터의 원리와 기본 문법
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              데코레이터란?
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              데코레이터는 함수를 인자로 받아 새로운 함수를 반환하는 고차 함수입니다.
              원본 함수를 수정하지 않고 기능을 추가할 수 있습니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 기본 데코레이터
def my_decorator(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

# 수동 적용
def say_hello():
    print("Hello!")

say_hello = my_decorator(say_hello)
say_hello()
# 출력:
# 함수 실행 전
# Hello!
# 함수 실행 후

# @ 문법 사용
@my_decorator
def say_goodbye():
    print("Goodbye!")

say_goodbye()  # 동일한 결과`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              인자가 있는 함수 데코레이팅
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`def logger(func):
    def wrapper(*args, **kwargs):
        print(f"함수 호출: {func.__name__}")
        print(f"인자: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"반환값: {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

@logger
def greet(name, greeting="안녕"):
    return f"{greeting}, {name}!"

print(add(3, 5))
# 함수 호출: add
# 인자: (3, 5), {}
# 반환값: 8
# 8

print(greet("Kelly", greeting="Hello"))
# 함수 호출: greet
# 인자: ('Kelly',), {'greeting': 'Hello'}
# 반환값: Hello, Kelly!
# Hello, Kelly!`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 실전 데코레이터 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-violet-600 dark:text-violet-400" />
          2. 실전 커스텀 데코레이터
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              타이밍 데코레이터
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import time
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 실행 시간: {end - start:.4f}초")
        return result
    return wrapper

@timing
def slow_function():
    time.sleep(1)
    return "완료"

result = slow_function()
# slow_function 실행 시간: 1.0012초`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              재시도 데코레이터
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"시도 {attempt + 1} 실패: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_api():
    import random
    if random.random() < 0.7:
        raise ConnectionError("네트워크 오류")
    return "성공"`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              검증 데코레이터
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`def validate_positive(func):
    @wraps(func)
    def wrapper(x):
        if x <= 0:
            raise ValueError("양수만 허용됩니다")
        return func(x)
    return wrapper

@validate_positive
def square_root(x):
    return x ** 0.5

print(square_root(16))  # 4.0
# print(square_root(-1))  # ValueError!`}
              </pre>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: 클래스 데코레이터
                </h5>
                <div className="bg-amber-100 dark:bg-amber-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-amber-900 dark:text-amber-200">
{`class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"호출 {self.count}번째")
        return self.func(*args, **kwargs)

@CountCalls
def process():
    print("처리 중...")

process()  # 호출 1번째
process()  # 호출 2번째`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 제너레이터 기초 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Zap className="w-6 h-6 text-violet-600 dark:text-violet-400" />
          3. 제너레이터와 yield
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              제너레이터 기본 개념
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              제너레이터는 이터레이터를 생성하는 간단한 방법입니다.
              yield로 값을 하나씩 반환하여 메모리를 효율적으로 사용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 일반 함수 (모든 값을 메모리에 저장)
def get_numbers_list(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

nums = get_numbers_list(1000000)  # 메모리 많이 사용!

# 제너레이터 (값을 하나씩 생성)
def get_numbers_gen(n):
    for i in range(n):
        yield i ** 2

nums_gen = get_numbers_gen(1000000)  # 메모리 적게 사용!

# 사용
for num in nums_gen:
    if num > 100:
        break
    print(num)  # 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100

# next()로 수동 호출
gen = get_numbers_gen(5)
print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 4
# print(next(gen))  # StopIteration!`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              제너레이터 표현식
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 리스트 컴프리헨션 (모든 값 생성)
squares_list = [x**2 for x in range(1000000)]

# 제너레이터 표현식 (지연 평가)
squares_gen = (x**2 for x in range(1000000))

# 메모리 비교
import sys
print(sys.getsizeof(squares_list))  # ~8MB
print(sys.getsizeof(squares_gen))   # ~128 bytes

# 제너레이터 체이닝
numbers = range(10)
squares = (x**2 for x in numbers)
evens = (x for x in squares if x % 2 == 0)
print(list(evens))  # [0, 4, 16, 36, 64]`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              실전 제너레이터 활용
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 파일 읽기 (메모리 효율적)
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# 무한 시퀀스 생성
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 처음 10개 피보나치 수
fib = fibonacci()
first_ten = [next(fib) for _ in range(10)]
print(first_ten)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# 배치 처리
def batch_data(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # 남은 데이터
        yield batch

data = range(25)
for batch in batch_data(data, 10):
    print(batch)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# [20, 21, 22, 23, 24]`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 고급 제너레이터 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4. 양방향 제너레이터와 이터레이터
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              send()로 양방향 통신
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`def echo_generator():
    value = None
    while True:
        value = yield value
        if value is not None:
            value = value.upper()

gen = echo_generator()
next(gen)  # 제너레이터 준비

print(gen.send("hello"))  # HELLO
print(gen.send("world"))  # WORLD`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              커스텀 이터레이터 클래스
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# 사용
for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          핵심 요약
        </h3>

        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-violet-600 dark:text-violet-400 mt-0.5 flex-shrink-0" />
              <span>
                데코레이터는 @ 문법으로 함수를 수정하지 않고 기능을 추가할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-violet-600 dark:text-violet-400 mt-0.5 flex-shrink-0" />
              <span>
                @wraps를 사용하여 메타데이터를 보존하고, 인자가 있는 데코레이터를 작성할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-violet-600 dark:text-violet-400 mt-0.5 flex-shrink-0" />
              <span>
                제너레이터는 yield로 값을 지연 생성하여 메모리를 효율적으로 사용합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-violet-600 dark:text-violet-400 mt-0.5 flex-shrink-0" />
              <span>
                제너레이터 표현식, send(), 커스텀 이터레이터로 고급 패턴을 구현할 수 있습니다
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
          데코레이터와 제너레이터를 마스터했다면, 이제 Python의 최신 기능인 비동기 프로그래밍을 학습할 차례입니다.
          Chapter 9에서 async/await로 고성능 비동기 코드를 작성하세요!
        </p>
        <Link
          href="/modules/python-programming/async-programming"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 9: 비동기 프로그래밍 →
        </Link>
      </section>
    </div>
  );
}
