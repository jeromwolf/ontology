'use client';

import { AlertTriangle, Shield, Code2, CheckCircle2, Lightbulb, FileWarning, Play, Bug } from 'lucide-react';
import Link from 'next/link';

export default function Chapter6() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <Shield className="w-6 h-6 text-red-600 dark:text-red-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            예외 처리
          </h2>
        </div>

        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            예외 처리는 프로그램이 예상치 못한 상황에 대응하여 안정적으로 동작하도록 만드는 핵심 기술입니다.
            사용자 입력 오류, 파일 없음, 네트워크 장애 등 다양한 예외 상황을 우아하게 처리하는 방법을 학습합니다.
            프로덕션 레벨 코드에서 반드시 필요한 기술입니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-red-600 dark:text-red-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-red-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. try-except로 안전한 코드 작성
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              예외 발생 시 프로그램 중단 방지 및 안전 처리
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-orange-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 커스텀 예외 클래스 만들기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              도메인 특화 예외로 명확한 에러 처리
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-yellow-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. finally와 else 절 완벽 활용
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              예외 유무와 관계없이 필요한 처리 수행
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-pink-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. 에러 로깅과 디버깅 전략
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              실전 운영 환경에서 에러 추적 및 분석
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: try-except 기초 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
          1. try-except로 안전한 코드 작성
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              예외 처리의 필요성
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              예외(Exception)는 프로그램 실행 중 발생하는 오류입니다.
              처리하지 않으면 프로그램이 종료되지만, try-except로 안전하게 처리할 수 있습니다.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-5 border border-red-200 dark:border-red-800">
                <h5 className="font-semibold text-red-900 dark:text-red-200 mb-3">예외 처리 없음</h5>
                <div className="bg-white dark:bg-gray-900 rounded p-4 font-mono text-sm">
                  <pre className="text-gray-700 dark:text-gray-300">
{`number = int(input("숫자 입력: "))
# 사용자가 "abc" 입력
# ValueError: invalid literal
# 프로그램 중단!`}
                  </pre>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-5 border border-green-200 dark:border-green-800">
                <h5 className="font-semibold text-green-900 dark:text-green-200 mb-3">예외 처리 적용</h5>
                <div className="bg-white dark:bg-gray-900 rounded p-4 font-mono text-sm">
                  <pre className="text-gray-700 dark:text-gray-300">
{`try:
    number = int(input("숫자 입력: "))
except ValueError:
    print("숫자만 입력하세요!")
    number = 0
# 계속 실행됨`}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              주요 내장 예외 타입
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">ValueError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      잘못된 값 (예: int(&quot;abc&quot;))
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">TypeError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      잘못된 타입 (예: &quot;3&quot; + 3)
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">IndexError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      잘못된 인덱스 (예: list[100])
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">KeyError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      딕셔너리에 없는 키
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">FileNotFoundError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      파일을 찾을 수 없음
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded p-4">
                    <code className="text-red-600 dark:text-red-400 font-semibold">ZeroDivisionError</code>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      0으로 나누기 (예: 5 / 0)
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              다중 예외 처리
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 여러 예외 타입 각각 처리
try:
    data = {'name': 'Kelly', 'age': 25}
    age = int(data['grade'])  # KeyError 또는 ValueError 발생 가능
except KeyError as e:
    print(f"키가 존재하지 않습니다: {e}")
except ValueError as e:
    print(f"숫자 변환 실패: {e}")
except Exception as e:
    print(f"알 수 없는 오류: {e}")

# 여러 예외를 동일하게 처리
try:
    result = 10 / 0
except (ValueError, TypeError, ZeroDivisionError) as e:
    print(f"계산 오류: {e}")`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: finally와 else */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-red-600 dark:text-red-400" />
          2. finally와 else 절 완벽 활용
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              finally: 항상 실행되는 코드
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              finally 블록은 예외 발생 여부와 관계없이 반드시 실행됩니다.
              리소스 정리(파일 닫기, DB 연결 해제)에 주로 사용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 파일 처리 예제
file = None
try:
    file = open('data.txt', 'r')
    content = file.read()
    process_data(content)  # 이 함수에서 예외 발생 가능
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"처리 중 오류: {e}")
finally:
    if file:
        file.close()  # 예외 발생 여부와 관계없이 항상 실행
        print("파일 닫기 완료")

# 더 간단한 방법: with 문 (컨텍스트 매니저)
try:
    with open('data.txt', 'r') as file:
        content = file.read()
        process_data(content)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
# with 문이 자동으로 파일을 닫아줍니다!`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              else: 예외가 없을 때만 실행
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 사용자 입력 처리
def get_positive_number():
    try:
        num = int(input("양수를 입력하세요: "))
        if num <= 0:
            raise ValueError("양수가 아닙니다")
    except ValueError as e:
        print(f"오류: {e}")
        return None
    else:
        print("유효한 입력입니다")
        return num
    finally:
        print("입력 처리 완료")

# 전체 구조
try:
    # 예외가 발생할 수 있는 코드
    risky_operation()
except SpecificError:
    # 특정 예외 처리
    handle_error()
else:
    # 예외가 없을 때만 실행
    success_operation()
finally:
    # 항상 실행 (리소스 정리)
    cleanup()`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 커스텀 예외 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3. 커스텀 예외 클래스 만들기
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              도메인 특화 예외 설계
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              프로젝트에 맞는 명확한 예외를 정의하면 에러 처리가 더 직관적이고 유지보수가 쉬워집니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 기본 커스텀 예외
class InvalidAgeError(ValueError):
    """나이 값이 유효하지 않을 때 발생"""
    pass

class InsufficientBalanceError(Exception):
    """잔액 부족 시 발생"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        message = f"잔액 부족: {balance}원 (필요: {amount}원)"
        super().__init__(message)

# 사용 예제
class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientBalanceError(self.balance, amount)
        self.balance -= amount
        return self.balance

# 실행
try:
    account = BankAccount(10000)
    account.withdraw(15000)
except InsufficientBalanceError as e:
    print(f"출금 실패: {e}")
    print(f"현재 잔액: {e.balance}원")`}
              </pre>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: 예외 계층 구조 설계
                </h5>
                <div className="bg-amber-100 dark:bg-amber-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-amber-900 dark:text-amber-200">
{`# 프로젝트 전체 기본 예외
class MyAppError(Exception):
    """모든 애플리케이션 예외의 기본"""
    pass

# 카테고리별 예외
class DatabaseError(MyAppError):
    pass

class NetworkError(MyAppError):
    pass`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 로깅과 디버깅 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Bug className="w-6 h-6 text-red-600 dark:text-red-400" />
          4. 에러 로깅과 디버깅 전략
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              logging 모듈 활용
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 사용 예제
def divide(a, b):
    try:
        logger.info(f"나누기 연산: {a} / {b}")
        result = a / b
    except ZeroDivisionError:
        logger.error("0으로 나누기 시도", exc_info=True)
        raise
    else:
        logger.info(f"결과: {result}")
        return result`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              실전 예외 처리 패턴
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 패턴 1: Retry 로직
from time import sleep

def fetch_data_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"시도 {attempt + 1} 실패: {e}")
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)  # 지수 백오프

# 패턴 2: Fallback 값
def get_config(key, default=None):
    try:
        with open('config.json') as f:
            return json.load(f)[key]
    except (FileNotFoundError, KeyError):
        return default`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Simulator Link */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
          <div className="flex items-start gap-3">
            <FileWarning className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
            <div>
              <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                실습: 예외 처리 시뮬레이터
              </h5>
              <p className="text-sm text-blue-800 dark:text-blue-300 mb-4">
                다양한 예외 상황을 직접 만들어보고 처리 방법을 연습하세요.
                실시간 피드백으로 안전한 코드 작성 패턴을 익힐 수 있습니다.
              </p>
              <Link
                href="/modules/python-programming/simulators/exception-simulator"
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                예외 처리 시뮬레이터 실행
              </Link>
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
              문제 1: 안전한 숫자 입력 함수
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              사용자로부터 정수를 입력받되, 잘못된 입력 시 다시 입력받는 함수를 작성하세요.
              최대 3회까지 시도 가능하며, 실패 시 None을 반환합니다.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 2: 파일 처리 with 로깅
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              파일을 읽어서 줄 수를 반환하는 함수를 작성하세요.
              파일이 없으면 0을 반환하고, 다른 예외 발생 시 로깅 후 예외를 다시 발생시킵니다.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              문제 3: 커스텀 예외로 검증 로직 만들기
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              사용자 등록 시 이메일 형식을 검증하는 함수를 작성하세요.
              InvalidEmailError 커스텀 예외를 정의하고, 잘못된 형식 시 발생시킵니다.
            </p>
          </div>
        </div>
      </section>

      {/* Summary */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          요약
        </h3>

        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <span>
                try-except로 예외를 안전하게 처리하여 프로그램 중단을 방지할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <span>
                finally는 항상 실행되어 리소스 정리에 사용하고, else는 예외 없을 때만 실행됩니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <span>
                커스텀 예외 클래스로 도메인 특화 에러 처리가 가능하며 코드 가독성이 향상됩니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <span>
                logging과 traceback으로 체계적인 에러 추적 및 디버깅이 가능합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <span>
                Retry, Fallback 패턴으로 안정적인 프로덕션 코드를 작성할 수 있습니다
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
          예외 처리를 마스터했다면, 이제 Python의 강력한 표준 라이브러리를 탐험할 차례입니다.
          Chapter 7에서 datetime, collections, itertools 등 실무에서 자주 사용하는 모듈을 학습하세요!
        </p>
        <Link
          href="/modules/python-programming/standard-library"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 7: Python 표준 라이브러리 →
        </Link>
      </section>
    </div>
  );
}
