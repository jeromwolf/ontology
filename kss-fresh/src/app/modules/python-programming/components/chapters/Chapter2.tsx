'use client';

import { BookOpen, Box, Grid, Database, Lightbulb, CheckCircle2, Play, Layers } from 'lucide-react';
import Link from 'next/link';

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-6">
          <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            데이터 타입과 컬렉션
          </h2>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            Python의 컬렉션(리스트, 튜플, 딕셔너리, 집합)은 여러 데이터를 효율적으로 관리하는 강력한 도구입니다.
            각 컬렉션의 특징과 사용법을 익혀 데이터 처리의 달인이 되어보세요.
            실전에서 가장 많이 사용되는 필수 자료구조를 완벽하게 마스터할 수 있습니다.
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
              1. 리스트, 튜플, 집합 완전 정복하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              순서형 컬렉션의 특징과 차이점 이해하기
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-indigo-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 딕셔너리로 데이터 관리하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Key-Value 쌍으로 효율적인 데이터 구조 설계
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-purple-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. 타입 변환의 원리와 실전 활용
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              자료형 간 변환과 형변환 마스터하기
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-pink-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. 컬렉션 연산과 메서드 마스터하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              슬라이싱, 정렬, 검색 등 고급 기법 활용
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: 리스트 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Box className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          1. 리스트 (List) - 동적 배열
        </h3>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              리스트 생성과 기본 연산
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              리스트는 순서가 있고 변경 가능한(mutable) 컬렉션입니다. 대괄호 <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">[]</code>를 사용하여 생성합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 리스트 생성
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # 여러 타입 혼합 가능

# 빈 리스트
empty_list = []
empty_list2 = list()

# 인덱싱과 슬라이싱
print(fruits[0])      # 'apple' (첫 번째 요소)
print(fruits[-1])     # 'cherry' (마지막 요소)
print(fruits[0:2])    # ['apple', 'banana'] (슬라이싱)
print(fruits[::-1])   # ['cherry', 'banana', 'apple'] (역순)

# 리스트 수정
fruits[1] = "blueberry"  # 요소 변경
fruits.append("orange")  # 끝에 추가
fruits.insert(1, "kiwi") # 특정 위치에 삽입
fruits.remove("apple")   # 값으로 제거
popped = fruits.pop()    # 마지막 요소 제거 및 반환`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              리스트 메서드와 고급 기능
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">정렬과 역순</h5>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  <div>numbers.sort()        # 오름차순</div>
                  <div>numbers.sort(reverse=True) # 내림차순</div>
                  <div>numbers.reverse()     # 역순 변경</div>
                  <div>sorted(numbers)       # 새 리스트 반환</div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">검색과 개수</h5>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  <div>fruits.index("apple") # 인덱스 찾기</div>
                  <div>fruits.count("apple") # 개수 세기</div>
                  <div>"apple" in fruits     # 포함 여부</div>
                  <div>len(fruits)           # 길이</div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">리스트 연산</h5>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  <div>[1,2] + [3,4]  # [1,2,3,4]</div>
                  <div>[1,2] * 3      # [1,2,1,2,1,2]</div>
                  <div>list1.extend(list2) # 합치기</div>
                  <div>list1.clear()  # 모두 제거</div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">리스트 컴프리헨션</h5>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  <div>[x**2 for x in range(5)]</div>
                  <div># [0, 1, 4, 9, 16]</div>
                  <div className="mt-2">[x for x in nums if x%2==0]</div>
                  <div># 짝수만 필터링</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: 리스트 vs 튜플 선택 기준
                </h5>
                <p className="text-sm text-amber-800 dark:text-amber-300">
                  데이터를 수정해야 한다면 리스트를, 변경하지 않을 데이터라면 튜플을 사용하세요.
                  튜플이 메모리 효율과 성능 면에서 약간 더 유리합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 튜플과 집합 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Layers className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          2. 튜플(Tuple)과 집합(Set)
        </h3>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              튜플 - 불변(Immutable) 시퀀스
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              튜플은 생성 후 수정할 수 없는 컬렉션입니다. 소괄호 <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">()</code>를 사용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 튜플 생성
coordinates = (3, 4)
person = ("Kelly", 25, "Engineer")
single = (42,)  # 요소가 하나일 때 콤마 필수!

# 튜플 언패킹 (Unpacking)
x, y = coordinates
name, age, job = person
print(f"{name}는 {age}살 {job}입니다")

# 튜플 활용 - 여러 값 반환
def get_stats():
    return (100, 85, 92)  # 여러 값 동시 반환

min_val, avg_val, max_val = get_stats()

# 튜플은 딕셔너리 키로 사용 가능 (리스트는 불가)
locations = {
    (0, 0): "origin",
    (1, 1): "diagonal"
}`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              집합(Set) - 중복 없는 컬렉션
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              집합은 순서가 없고 중복을 허용하지 않는 컬렉션입니다. 중괄호 <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">{'{}'}</code>를 사용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 집합 생성
numbers = {1, 2, 3, 4, 5}
unique = {1, 2, 2, 3, 3, 3}  # {1, 2, 3} - 중복 자동 제거
empty_set = set()  # 빈 집합 ({}는 빈 딕셔너리!)

# 집합 연산
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(set1 | set2)  # {1, 2, 3, 4, 5, 6} 합집합
print(set1 & set2)  # {3, 4} 교집합
print(set1 - set2)  # {1, 2} 차집합
print(set1 ^ set2)  # {1, 2, 5, 6} 대칭 차집합

# 집합 메서드
set1.add(10)        # 요소 추가
set1.remove(1)      # 요소 제거 (없으면 에러)
set1.discard(999)   # 요소 제거 (없어도 에러 없음)
set1.clear()        # 모든 요소 제거

# 실용 예제: 리스트에서 중복 제거
numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique_numbers = list(set(numbers))  # [1, 2, 3, 4, 5]`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 딕셔너리 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Database className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          3. 딕셔너리(Dictionary) - Key-Value 저장소
        </h3>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              딕셔너리 기본 사용법
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              딕셔너리는 키(key)와 값(value)의 쌍으로 데이터를 저장하는 가장 강력한 자료구조입니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 딕셔너리 생성
student = {
    "name": "Kelly",
    "age": 25,
    "major": "Computer Science",
    "gpa": 3.9
}

# 다양한 생성 방법
dict1 = dict(name="Kelly", age=25)
dict2 = dict([("name", "Kelly"), ("age", 25)])

# 값 접근
print(student["name"])           # "Kelly"
print(student.get("age"))        # 25
print(student.get("grade", "N/A")) # 없으면 기본값 반환

# 값 수정 및 추가
student["age"] = 26              # 수정
student["email"] = "k@mail.com"  # 추가

# 값 제거
del student["gpa"]               # 키-값 쌍 제거
email = student.pop("email")     # 제거하고 값 반환
student.clear()                  # 모든 항목 제거`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              딕셔너리 순회와 고급 기법
            </h4>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 딕셔너리 순회
scores = {"math": 90, "english": 85, "science": 92}

# 키만 순회
for subject in scores:
    print(subject)

# 키-값 순회
for subject, score in scores.items():
    print(f"{subject}: {score}점")

# 값만 순회
for score in scores.values():
    print(score)

# 딕셔너리 컴프리헨션
squared = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

filtered = {k: v for k, v in scores.items() if v >= 90}
# {"math": 90, "science": 92}

# 딕셔너리 병합 (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = dict1 | dict2  # {"a": 1, "b": 2, "c": 3, "d": 4}

# 중첩 딕셔너리
students = {
    "student1": {"name": "Kelly", "age": 25},
    "student2": {"name": "John", "age": 23}
}
print(students["student1"]["name"])  # "Kelly"`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 타입 변환과 실전 활용 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Grid className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          4. 타입 변환과 실전 활용
        </h3>

        <div className="space-y-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              컬렉션 간 타입 변환
            </h4>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 리스트 → 다른 타입
my_list = [1, 2, 3, 2, 1]
my_tuple = tuple(my_list)    # (1, 2, 3, 2, 1)
my_set = set(my_list)        # {1, 2, 3} - 중복 제거!

# 문자열 → 리스트
text = "Hello"
chars = list(text)           # ['H', 'e', 'l', 'l', 'o']
words = "a b c".split()      # ['a', 'b', 'c']

# 리스트 → 문자열
chars = ['P', 'y', 't', 'h', 'o', 'n']
word = ''.join(chars)        # "Python"
sentence = ' '.join(['Hello', 'World'])  # "Hello World"

# 딕셔너리 → 리스트
scores = {"math": 90, "english": 85}
keys = list(scores.keys())           # ["math", "english"]
values = list(scores.values())       # [90, 85]
items = list(scores.items())         # [("math", 90), ("english", 85)]

# 리스트 쌍 → 딕셔너리
pairs = [("a", 1), ("b", 2), ("c", 3)]
dictionary = dict(pairs)  # {"a": 1, "b": 2, "c": 3}`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              실전 활용 예제
            </h4>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-900 dark:text-white">
{`# 예제 1: 단어 빈도수 계산
text = "python is fun python is powerful"
words = text.split()
word_count = {}

for word in words:
    word_count[word] = word_count.get(word, 0) + 1

print(word_count)  # {'python': 2, 'is': 2, 'fun': 1, 'powerful': 1}

# 예제 2: 리스트에서 최댓값, 최솟값 찾기
numbers = [45, 23, 67, 12, 89, 34]
print(f"최댓값: {max(numbers)}")
print(f"최솟값: {min(numbers)}")
print(f"합계: {sum(numbers)}")
print(f"평균: {sum(numbers) / len(numbers)}")

# 예제 3: 두 리스트를 딕셔너리로 결합
keys = ["name", "age", "city"]
values = ["Kelly", 25, "Seoul"]
profile = dict(zip(keys, values))
# {"name": "Kelly", "age": 25, "city": "Seoul"}

# 예제 4: 중첩 리스트 평탄화
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6]`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Play className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                  실습: 컬렉션 시각화 도구
                </h5>
                <p className="text-sm text-blue-800 dark:text-blue-300 mb-4">
                  리스트, 딕셔너리, 집합의 모든 연산을 인터랙티브하게 실습해보세요!
                  각 컬렉션의 동작을 시각적으로 확인할 수 있습니다.
                </p>
                <Link
                  href="/modules/python-programming/simulators/collection-visualizer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                  컬렉션 시각화 도구 실행
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
              문제 1: 리스트 필터링
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              주어진 숫자 리스트에서 짝수만 추출하여 새로운 리스트를 만드세요.
              (리스트 컴프리헨션 사용)
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <div className="text-gray-600 dark:text-gray-400 mb-2"># 시작 코드</div>
              <div className="text-gray-900 dark:text-white">
                numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]<br/>
                # 여기에 코드 작성
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 2: 딕셔너리 활용
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              학생들의 이름과 점수가 담긴 딕셔너리에서 평균 점수 이상인 학생들만 출력하세요.
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <pre className="text-gray-700 dark:text-gray-300">
{`students = {
    "Alice": 85,
    "Bob": 70,
    "Charlie": 95,
    "David": 60,
    "Eve": 90
}
# 평균 점수를 계산하고, 평균 이상인 학생 출력`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 3: 집합 연산
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              두 개의 리스트가 주어졌을 때, 교집합, 합집합, 차집합을 각각 구하세요.
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-sm">
              <pre className="text-gray-700 dark:text-gray-300">
{`list1 = [1, 2, 3, 4, 5, 6]
list2 = [4, 5, 6, 7, 8, 9]

# 집합으로 변환 후 연산 수행`}
              </pre>
            </div>
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
                <strong>리스트</strong>는 순서가 있고 변경 가능한 컬렉션으로, 가장 많이 사용됩니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong>튜플</strong>은 불변 컬렉션으로, 안전한 데이터 저장과 함수 반환값에 유용합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong>집합</strong>은 중복을 허용하지 않으며, 집합 연산(합집합, 교집합 등)에 강력합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                <strong>딕셔너리</strong>는 Key-Value 쌍으로 데이터를 저장하는 가장 강력한 자료구조입니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <span>
                컬렉션 간 타입 변환을 활용하면 데이터 처리가 훨씬 유연해집니다
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
          컬렉션을 활용한 데이터 관리를 익혔다면, 이제 함수를 사용하여 코드를 재사용 가능하게
          만들어봅시다. Chapter 3에서 함수와 모듈의 모든 것을 배워보세요!
        </p>
        <Link
          href="/modules/python-programming/functions-modules"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 3: 함수와 모듈 →
        </Link>
      </section>
    </div>
  );
}
