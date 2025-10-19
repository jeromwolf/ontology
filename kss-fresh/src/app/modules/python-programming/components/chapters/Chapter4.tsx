'use client';

import { BookOpen, Code2, Lightbulb, Play, CheckCircle2, ArrowRight, FileText, Database } from 'lucide-react';
import Link from 'next/link';

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Header */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            파일 입출력과 데이터 처리
          </h2>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            실제 데이터는 파일에 저장되어 있습니다. 이 챕터에서는 텍스트 파일부터 CSV, JSON까지
            다양한 파일 형식을 다루는 방법과 안전한 파일 처리 기법을 완전히 마스터합니다.
          </p>
        </div>
      </section>

      {/* Section 1: 텍스트 파일 읽기/쓰기 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <FileText className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          1. 텍스트 파일 읽기와 쓰기
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">파일 열기와 닫기 (open/close)</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 기본 파일 읽기
file = open('example.txt', 'r')  # 읽기 모드로 열기
content = file.read()             # 전체 내용 읽기
file.close()                      # 반드시 닫아야 함!
print(content)

# with 문을 사용한 안전한 파일 처리 (권장)
with open('example.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)
# with 블록을 벗어나면 자동으로 파일이 닫힘

# 파일 모드
# 'r'  - 읽기 (기본값)
# 'w'  - 쓰기 (기존 내용 삭제)
# 'a'  - 추가 (기존 내용 뒤에 추가)
# 'r+' - 읽기+쓰기
# 'b'  - 바이너리 모드 (예: 'rb', 'wb')`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">다양한 읽기 방법</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 방법 1: read() - 전체 내용을 문자열로
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

# 방법 2: readline() - 한 줄씩 읽기
with open('example.txt', 'r') as file:
    line = file.readline()  # 첫 번째 줄
    print(line)
    line = file.readline()  # 두 번째 줄
    print(line)

# 방법 3: readlines() - 모든 줄을 리스트로
with open('example.txt', 'r') as file:
    lines = file.readlines()  # ['line1\\n', 'line2\\n', ...]
    for line in lines:
        print(line.strip())  # strip()으로 공백 제거

# 방법 4: 반복문으로 한 줄씩 (메모리 효율적, 권장)
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">파일 쓰기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`# 쓰기 모드 - 기존 파일 덮어쓰기
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write("첫 번째 줄\\n")
    file.write("두 번째 줄\\n")

    # 리스트를 파일에 쓰기
    lines = ["세 번째 줄\\n", "네 번째 줄\\n"]
    file.writelines(lines)

# 추가 모드 - 기존 내용 뒤에 추가
with open('output.txt', 'a', encoding='utf-8') as file:
    file.write("추가된 줄\\n")

# 실전 예제: 로그 파일 작성
import datetime

def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('app.log', 'a', encoding='utf-8') as file:
        file.write(f"[{timestamp}] {message}\\n")

write_log("Application started")
write_log("User logged in")`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 2: CSV 파일 다루기 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Database className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          2. CSV 파일 다루기
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-1" />
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">CSV란?</h5>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  Comma-Separated Values의 약자로, 콤마로 구분된 값들을 저장하는 텍스트 파일입니다.
                  Excel, 데이터베이스 등과 호환성이 좋아 데이터 교환 형식으로 널리 사용됩니다.
                </p>
              </div>
            </div>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">CSV 읽기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import csv

# CSV 파일 읽기
with open('students.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # 헤더 읽기
    header = next(csv_reader)
    print(f"헤더: {header}")  # ['이름', '나이', '점수']

    # 데이터 행 읽기
    for row in csv_reader:
        print(row)  # ['Alice', '20', '85']

# DictReader로 딕셔너리 형태로 읽기 (권장)
with open('students.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        print(f"이름: {row['이름']}, 나이: {row['나이']}, 점수: {row['점수']}")
        # 딕셔너리 형태: {'이름': 'Alice', '나이': '20', '점수': '85'}

# 예시: students.csv
# 이름,나이,점수
# Alice,20,85
# Bob,22,90
# Charlie,21,78`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">CSV 쓰기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import csv

# CSV 파일 쓰기
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)

    # 헤더 쓰기
    csv_writer.writerow(['이름', '나이', '점수'])

    # 데이터 행 쓰기
    csv_writer.writerow(['Alice', 20, 85])
    csv_writer.writerow(['Bob', 22, 90])

    # 여러 행 한 번에 쓰기
    rows = [
        ['Charlie', 21, 78],
        ['David', 23, 92]
    ]
    csv_writer.writerows(rows)

# DictWriter로 딕셔너리 형태로 쓰기
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['이름', '나이', '점수']
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)

    csv_writer.writeheader()  # 헤더 자동 생성

    csv_writer.writerow({'이름': 'Alice', '나이': 20, '점수': 85})
    csv_writer.writerow({'이름': 'Bob', '나이': 22, '점수': 90})`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 3: JSON 파일 다루기 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          3. JSON 파일 다루기
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-1" />
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">JSON이란?</h5>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  JavaScript Object Notation의 약자로, 구조화된 데이터를 표현하는 텍스트 기반 형식입니다.
                  웹 API, 설정 파일 등에서 널리 사용되며, Python의 딕셔너리/리스트 구조와 매우 유사합니다.
                </p>
              </div>
            </div>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">JSON 읽기와 쓰기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import json

# JSON 파일 읽기
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)  # JSON -> Python 객체
    print(data)  # 딕셔너리 또는 리스트

# JSON 파일 쓰기
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding", "music"],
    "address": {
        "city": "Seoul",
        "country": "Korea"
    }
}

with open('output.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
    # ensure_ascii=False: 한글 그대로 저장
    # indent=2: 가독성 위해 들여쓰기

# JSON 문자열로 변환/파싱
json_string = json.dumps(data, ensure_ascii=False, indent=2)
print(json_string)

parsed_data = json.loads(json_string)
print(parsed_data)`}
            </pre>
          </div>

          <h4 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">복잡한 JSON 다루기</h4>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import json

# 중첩된 JSON 데이터
data = {
    "users": [
        {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "skills": ["Python", "JavaScript"]
        },
        {
            "id": 2,
            "name": "Bob",
            "email": "bob@example.com",
            "skills": ["Java", "C++"]
        }
    ],
    "metadata": {
        "version": "1.0",
        "created_at": "2025-01-10"
    }
}

# JSON 파일로 저장
with open('users.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

# JSON 파일 읽기
with open('users.json', 'r', encoding='utf-8') as file:
    loaded_data = json.load(file)

    # 데이터 접근
    for user in loaded_data['users']:
        print(f"ID: {user['id']}")
        print(f"이름: {user['name']}")
        print(f"스킬: {', '.join(user['skills'])}")
        print("---")`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 4: 파일 예외 처리 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          4. 안전한 파일 처리 (예외 처리)
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import os

# 파일 존재 여부 확인
def safe_read_file(filename):
    """안전하게 파일 읽기"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        return None
    except PermissionError:
        print(f"오류: '{filename}' 파일에 접근 권한이 없습니다.")
        return None
    except UnicodeDecodeError:
        print(f"오류: '{filename}' 파일의 인코딩을 확인하세요.")
        return None
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        return None

# 사용 예시
content = safe_read_file('example.txt')
if content:
    print(content)

# os 모듈을 활용한 파일 체크
def process_file(filename):
    """파일 처리 전 체크"""
    # 파일 존재 확인
    if not os.path.exists(filename):
        print(f"파일이 존재하지 않습니다: {filename}")
        return

    # 파일인지 확인 (디렉토리 아님)
    if not os.path.isfile(filename):
        print(f"파일이 아닙니다: {filename}")
        return

    # 파일 크기 확인 (너무 큰 파일 방지)
    file_size = os.path.getsize(filename)
    if file_size > 10 * 1024 * 1024:  # 10MB
        print(f"파일이 너무 큽니다: {file_size} bytes")
        return

    # 안전하게 처리
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        print(f"파일 읽기 성공: {len(content)} 문자")

# 디렉토리 생성 (없으면)
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 안전하게 파일 쓰기
output_path = os.path.join(output_dir, 'result.txt')
with open(output_path, 'w', encoding='utf-8') as file:
    file.write("데이터 저장 완료")`}
            </pre>
          </div>
        </div>
      </section>

      {/* Section 5: 실전 프로젝트 */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center gap-2">
          <Play className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          5. 실전 프로젝트: 학생 성적 관리 시스템
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`import csv
import json
import os

class StudentManager:
    """학생 성적 관리 시스템"""

    def __init__(self, data_dir='student_data'):
        self.data_dir = data_dir
        # 데이터 디렉토리 생성
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.csv_file = os.path.join(data_dir, 'students.csv')
        self.json_file = os.path.join(data_dir, 'summary.json')

    def add_student(self, name, age, score):
        """학생 추가 (CSV에 저장)"""
        file_exists = os.path.exists(self.csv_file)

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 파일이 없으면 헤더 추가
            if not file_exists:
                writer.writerow(['이름', '나이', '점수'])

            writer.writerow([name, age, score])

        print(f"학생 추가 완료: {name}")

    def get_all_students(self):
        """모든 학생 조회"""
        if not os.path.exists(self.csv_file):
            return []

        students = []
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append({
                    '이름': row['이름'],
                    '나이': int(row['나이']),
                    '점수': float(row['점수'])
                })

        return students

    def calculate_summary(self):
        """통계 계산 및 JSON 저장"""
        students = self.get_all_students()

        if not students:
            print("학생 데이터가 없습니다.")
            return

        scores = [s['점수'] for s in students]

        summary = {
            '총_학생_수': len(students),
            '평균_점수': sum(scores) / len(scores),
            '최고_점수': max(scores),
            '최저_점수': min(scores),
            '학생_목록': students
        }

        # JSON으로 저장
        with open(self.json_file, 'w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"통계 저장 완료: {self.json_file}")
        return summary

    def print_report(self):
        """리포트 출력"""
        summary = self.calculate_summary()

        if summary:
            print("\\n=== 학생 성적 리포트 ===")
            print(f"총 학생 수: {summary['총_학생_수']}명")
            print(f"평균 점수: {summary['평균_점수']:.2f}점")
            print(f"최고 점수: {summary['최고_점수']}점")
            print(f"최저 점수: {summary['최저_점수']}점")
            print("\\n학생 목록:")
            for student in summary['학생_목록']:
                print(f"  {student['이름']} ({student['나이']}세): {student['점수']}점")

# 사용 예시
manager = StudentManager()

# 학생 추가
manager.add_student("Alice", 20, 85.5)
manager.add_student("Bob", 22, 92.0)
manager.add_student("Charlie", 21, 78.5)

# 리포트 출력
manager.print_report()`}
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
            안전한 환경에서 파일 입출력을 직접 실습해보세요.
          </p>
          <Link
            href="/modules/python-programming/simulators/file-io-playground"
            className="inline-flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
          >
            파일 입출력 플레이그라운드 체험하기
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
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">with 문으로 안전하게 파일 처리</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">자동으로 파일을 닫아 리소스 누수를 방지하고, 예외 상황에서도 안전합니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">2</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">CSV는 표 형식 데이터에 최적</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">DictReader/DictWriter를 사용하면 딕셔너리 형태로 쉽게 다룰 수 있습니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">3</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">JSON은 구조화된 데이터에 최적</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Python 객체와 JSON 간 변환이 쉽고, 웹 API와 호환성이 좋습니다.</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-green-600 dark:text-green-400">4</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-1">항상 예외 처리를 추가</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">FileNotFoundError, PermissionError 등을 적절히 처리하여 안정성을 확보합니다.</p>
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
            파일 처리를 마스터했다면, 이제 객체지향 프로그래밍으로 코드를 더 구조화해봅시다!
          </p>
          <Link
            href="/modules/python-programming/oop-basics"
            className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 font-semibold hover:gap-3 transition-all"
          >
            Chapter 5: 객체지향 프로그래밍
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}
