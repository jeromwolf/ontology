'use client';

import { Calendar, CheckCircle2, Code2, Database, FolderTree, Lightbulb, Package, Play } from 'lucide-react';
import Link from 'next/link';

export default function Chapter7() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <Package className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Python 표준 라이브러리
          </h2>
        </div>

        <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            Python의 표준 라이브러리는 "배터리 포함(Batteries Included)" 철학을 실현합니다.
            설치 없이 바로 사용 가능한 강력한 모듈들로 날짜, 파일, 자료구조, 시스템 제어 등을 쉽게 다룰 수 있습니다.
            실무에서 가장 많이 사용하는 핵심 모듈을 마스터합니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-emerald-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. datetime으로 날짜와 시간 다루기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              날짜 계산, 포맷 변환, 타임존 처리 완벽 정복
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-teal-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. collections 모듈의 강력한 자료구조
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Counter, defaultdict, deque, namedtuple 활용
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-cyan-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. itertools와 functools 실전 활용
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              이터레이터 조작과 함수형 프로그래밍 도구
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. os와 sys 모듈로 시스템 제어하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              파일 시스템, 환경 변수, 프로세스 관리
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: datetime 모듈 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Calendar className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          1. datetime으로 날짜와 시간 다루기
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              날짜와 시간 기본 연산
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from datetime import datetime, date, time, timedelta

# 현재 날짜와 시간
now = datetime.now()
today = date.today()

print(now)    # 2025-01-10 15:30:45.123456
print(today)  # 2025-01-10

# 특정 날짜 생성
birthday = date(1990, 5, 15)
meeting = datetime(2025, 12, 25, 14, 30)

# 날짜 차이 계산
age_days = (today - birthday).days
print(f"생일로부터 {age_days}일 경과")

# timedelta로 날짜 연산
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
past_hour = now - timedelta(hours=1)

print(f"내일: {tomorrow}")
print(f"다음 주: {next_week}")

# 날짜 비교
if today < date(2025, 12, 31):
    print("올해가 아직 남았습니다")

# 요일 확인 (0=월요일, 6=일요일)
weekday = today.weekday()
days = ['월', '화', '수', '목', '금', '토', '일']
print(f"오늘은 {days[weekday]}요일입니다")`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              날짜 포맷 변환 (strftime/strptime)
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 날짜 → 문자열 (strftime)
now = datetime.now()
formatted = now.strftime("%Y년 %m월 %d일 %H시 %M분")
print(formatted)  # 2025년 01월 10일 15시 30분

iso_format = now.strftime("%Y-%m-%d")
print(iso_format)  # 2025-01-10

# 주요 포맷 코드
formats = {
    "%Y": "연도 4자리 (2025)",
    "%m": "월 2자리 (01-12)",
    "%d": "일 2자리 (01-31)",
    "%H": "시간 24시간제 (00-23)",
    "%I": "시간 12시간제 (01-12)",
    "%M": "분 (00-59)",
    "%S": "초 (00-59)",
    "%A": "요일 전체 (Monday)",
    "%B": "월 전체 (January)",
    "%p": "AM/PM"
}

# 문자열 → 날짜 (strptime)
date_string = "2025-12-25 14:30:00"
parsed = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
print(parsed)

# 다양한 포맷 파싱
formats_input = [
    ("2025/01/10", "%Y/%m/%d"),
    ("Jan 10, 2025", "%b %d, %Y"),
    ("10-01-2025 15:30", "%d-%m-%Y %H:%M")
]

for date_str, format_str in formats_input:
    parsed = datetime.strptime(date_str, format_str)
    print(f"{date_str} → {parsed}")`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                  Pro Tip: 타임존 처리 (pytz)
                </h5>
                <div className="bg-blue-100 dark:bg-blue-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-blue-900 dark:text-blue-200">
{`from datetime import datetime
import pytz

# UTC 시간
utc_now = datetime.now(pytz.UTC)

# 한국 시간으로 변환
kst = pytz.timezone('Asia/Seoul')
korea_time = utc_now.astimezone(kst)

print(f"UTC: {utc_now}")
print(f"한국: {korea_time}")`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: collections 모듈 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Database className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          2. collections 모듈의 강력한 자료구조
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Counter: 요소 개수 세기
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from collections import Counter

# 리스트 요소 개수
votes = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counter = Counter(votes)

print(counter)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(counter['apple'])  # 3
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# 문자열 문자 빈도
text = "hello world"
char_count = Counter(text)
print(char_count)  # Counter({'l': 3, 'o': 2, ...})

# Counter 연산
c1 = Counter(['a', 'b', 'c', 'a'])
c2 = Counter(['a', 'b', 'd'])

print(c1 + c2)  # Counter({'a': 3, 'b': 2, 'c': 1, 'd': 1})
print(c1 - c2)  # Counter({'a': 1, 'c': 1})
print(c1 & c2)  # Counter({'a': 1, 'b': 1}) (교집합)

# 실전 예제: 단어 빈도 분석
text = """
Python is a high-level programming language.
Python is known for its simplicity.
Python is widely used in data science.
"""
words = text.lower().split()
word_freq = Counter(words)
print(word_freq.most_common(3))  # [('python', 3), ('is', 3), ...]`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              defaultdict: 기본값이 있는 딕셔너리
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from collections import defaultdict

# 일반 딕셔너리의 문제
normal_dict = {}
# normal_dict['key'] += 1  # KeyError!

# defaultdict 사용
count_dict = defaultdict(int)  # 기본값 0
count_dict['apple'] += 1
count_dict['banana'] += 1
count_dict['apple'] += 1
print(count_dict)  # defaultdict(<class 'int'>, {'apple': 2, 'banana': 1})

# 리스트를 기본값으로
group_dict = defaultdict(list)
group_dict['fruits'].append('apple')
group_dict['fruits'].append('banana')
group_dict['vegetables'].append('carrot')
print(group_dict)
# defaultdict(<class 'list'>, {'fruits': ['apple', 'banana'],
#                               'vegetables': ['carrot']})

# 실전 예제: 학생 성적 그룹화
students = [
    ('Kelly', 'A'),
    ('John', 'B'),
    ('Alice', 'A'),
    ('Bob', 'C'),
    ('Eve', 'B')
]

grade_groups = defaultdict(list)
for name, grade in students:
    grade_groups[grade].append(name)

for grade, names in sorted(grade_groups.items()):
    print(f"{grade}등급: {', '.join(names)}")`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              deque: 양방향 큐
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from collections import deque

# deque 생성
queue = deque([1, 2, 3])

# 양쪽에서 추가/제거 (O(1) 시간복잡도!)
queue.append(4)      # 오른쪽 추가: [1, 2, 3, 4]
queue.appendleft(0)  # 왼쪽 추가: [0, 1, 2, 3, 4]

right = queue.pop()        # 오른쪽 제거: 4
left = queue.popleft()     # 왼쪽 제거: 0

print(queue)  # deque([1, 2, 3])

# 회전
queue.rotate(1)   # 오른쪽으로 1칸: [3, 1, 2]
queue.rotate(-1)  # 왼쪽으로 1칸: [1, 2, 3]

# 최대 길이 제한
limited = deque(maxlen=3)
for i in range(5):
    limited.append(i)
    print(limited)
# 최종: deque([2, 3, 4], maxlen=3) (0, 1은 자동 제거)

# 실전 예제: 최근 N개 이력 관리
class RecentHistory:
    def __init__(self, maxsize=5):
        self.history = deque(maxlen=maxsize)

    def add(self, item):
        self.history.append(item)

    def get_recent(self):
        return list(self.history)

history = RecentHistory(3)
for i in range(5):
    history.add(f"action_{i}")
print(history.get_recent())  # ['action_2', 'action_3', 'action_4']`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              namedtuple: 이름 있는 튜플
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from collections import namedtuple

# namedtuple 정의
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', ['name', 'age', 'city'])

# 생성
p = Point(10, 20)
person = Person('Kelly', 25, 'Seoul')

# 접근 (인덱스 또는 이름)
print(p.x, p.y)        # 10 20
print(p[0], p[1])      # 10 20
print(person.name)     # Kelly
print(person.age)      # 25

# 불변 (immutable)
# p.x = 30  # AttributeError!

# 딕셔너리로 변환
person_dict = person._asdict()
print(person_dict)  # {'name': 'Kelly', 'age': 25, 'city': 'Seoul'}

# 값 교체 (새 인스턴스 생성)
new_person = person._replace(age=26)
print(new_person)  # Person(name='Kelly', age=26, city='Seoul')

# 실전 예제: CSV 데이터 구조화
Employee = namedtuple('Employee', ['id', 'name', 'department', 'salary'])

employees = [
    Employee(1, 'Alice', 'IT', 60000),
    Employee(2, 'Bob', 'HR', 55000),
    Employee(3, 'Charlie', 'IT', 65000)
]

# 부서별 평균 연봉
from collections import defaultdict
dept_salaries = defaultdict(list)
for emp in employees:
    dept_salaries[emp.department].append(emp.salary)

for dept, salaries in dept_salaries.items():
    avg = sum(salaries) / len(salaries)
    print(f"{dept}: {avg:,.0f}원")`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: itertools & functools */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          3. itertools와 functools 실전 활용
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              itertools: 강력한 이터레이터 도구
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from itertools import (
    count, cycle, repeat,
    chain, combinations, permutations,
    product, groupby, islice
)

# 무한 이터레이터
counter = count(start=1, step=2)  # 1, 3, 5, 7, ...
print([next(counter) for _ in range(5)])  # [1, 3, 5, 7, 9]

cycler = cycle(['A', 'B', 'C'])  # A, B, C, A, B, C, ...
print([next(cycler) for _ in range(7)])  # ['A', 'B', 'C', 'A', 'B', 'C', 'A']

# 조합과 순열
items = ['A', 'B', 'C']
print(list(combinations(items, 2)))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]
print(list(permutations(items, 2)))  # [('A', 'B'), ('A', 'C'), ('B', 'A'), ...]

# 카르테시안 곱
print(list(product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# 체인 (여러 이터러블 연결)
chained = chain([1, 2], [3, 4], [5])
print(list(chained))  # [1, 2, 3, 4, 5]

# 슬라이싱
data = count()
first_ten = list(islice(data, 10))  # [0, 1, 2, ..., 9]

# groupby: 연속된 그룹화
data = [
    {'name': 'Alice', 'dept': 'IT'},
    {'name': 'Bob', 'dept': 'IT'},
    {'name': 'Charlie', 'dept': 'HR'},
    {'name': 'David', 'dept': 'HR'}
]

# 부서별 그룹화 (정렬 필수!)
from operator import itemgetter
data_sorted = sorted(data, key=itemgetter('dept'))

for dept, group in groupby(data_sorted, key=itemgetter('dept')):
    members = [person['name'] for person in group]
    print(f"{dept}: {', '.join(members)}")`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              functools: 함수형 프로그래밍 도구
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from functools import reduce, partial, lru_cache, wraps

# reduce: 누적 연산
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# partial: 부분 적용 함수
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27

# lru_cache: 메모이제이션 (캐싱)
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # 빠르게 계산!
print(fibonacci.cache_info())  # 캐시 통계

# wraps: 데코레이터 메타데이터 보존
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """인사 함수"""
    return f"Hello, {name}"

print(greet.__name__)  # greet (보존됨!)
print(greet.__doc__)   # 인사 함수 (보존됨!)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: os & sys 모듈 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <FolderTree className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
          4. os와 sys 모듈로 시스템 제어하기
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              os 모듈: 파일 시스템 제어
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import os

# 현재 작업 디렉토리
cwd = os.getcwd()
print(f"현재 디렉토리: {cwd}")

# 디렉토리 변경
# os.chdir('/path/to/directory')

# 디렉토리 생성/삭제
os.makedirs('data/logs', exist_ok=True)  # 중간 디렉토리도 생성
# os.rmdir('data/logs')  # 빈 디렉토리만 삭제
# os.removedirs('data/logs')  # 재귀적 삭제

# 파일/디렉토리 목록
files = os.listdir('.')
print(f"파일 수: {len(files)}")

# 파일 존재 확인
if os.path.exists('data.txt'):
    print("파일 존재")

# 경로 조작
path = os.path.join('data', 'logs', 'app.log')
print(path)  # data/logs/app.log (OS에 맞게 자동 변환)

dirname = os.path.dirname(path)   # data/logs
basename = os.path.basename(path) # app.log
name, ext = os.path.splitext(basename)  # ('app', '.log')

# 파일 정보
if os.path.exists('data.txt'):
    size = os.path.getsize('data.txt')
    mtime = os.path.getmtime('data.txt')
    print(f"크기: {size} bytes")

# 환경 변수
home = os.environ.get('HOME', '/default/home')
os.environ['MY_VAR'] = 'value'

# 명령 실행
# os.system('ls -la')  # 위험! 보안 주의
# subprocess 모듈 사용 권장

# 실전 예제: 파일 검색
def find_files(directory, extension):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                result.append(os.path.join(root, file))
    return result

# python_files = find_files('.', '.py')`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              sys 모듈: 시스템 설정 및 프로세스
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import sys

# 명령행 인자
print(f"스크립트 이름: {sys.argv[0]}")
# python script.py arg1 arg2
# sys.argv = ['script.py', 'arg1', 'arg2']

# Python 버전 및 경로
print(f"Python 버전: {sys.version}")
print(f"실행 경로: {sys.executable}")

# 모듈 검색 경로
print(f"모듈 경로: {sys.path}")
sys.path.append('/custom/module/path')

# 표준 입출력
sys.stdout.write("Hello\\n")
sys.stderr.write("Error!\\n")

# 프로그램 종료
# sys.exit(0)  # 정상 종료
# sys.exit(1)  # 에러 종료

# 플랫폼 정보
print(f"플랫폼: {sys.platform}")  # linux, darwin, win32
print(f"최대 정수: {sys.maxsize}")

# 재귀 한계
print(f"재귀 한계: {sys.getrecursionlimit()}")
# sys.setrecursionlimit(2000)  # 주의해서 사용

# 실전 예제: CLI 도구
def main():
    if len(sys.argv) < 2:
        print("사용법: python script.py <command>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    if command == 'start':
        print("서비스 시작...")
    elif command == 'stop':
        print("서비스 중지...")
    else:
        print(f"알 수 없는 명령: {command}", file=sys.stderr)
        sys.exit(1)

# if __name__ == '__main__':
#     main()`}
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

        <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
              <span>
                datetime으로 날짜 연산, 포맷 변환, 타임존 처리를 자유롭게 다룰 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
              <span>
                collections의 Counter, defaultdict, deque, namedtuple로 효율적인 자료구조를 활용합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
              <span>
                itertools와 functools로 함수형 프로그래밍 스타일의 코드를 작성할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
              <span>
                os와 sys 모듈로 파일 시스템, 환경 변수, 프로세스를 제어합니다
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
          표준 라이브러리를 마스터했다면, 이제 Python의 고급 기능인 데코레이터와 제너레이터를 학습할 차례입니다.
          Chapter 8에서 함수를 강화하는 데코레이터와 메모리 효율적인 제너레이터를 익히세요!
        </p>
        <Link
          href="/modules/python-programming/decorators-generators"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 8: 데코레이터와 제너레이터 →
        </Link>
      </section>
    </div>
  );
}
