'use client';

import { CheckCircle2, Clock, Lightbulb, Zap } from 'lucide-react';
import Link from 'next/link';

export default function Chapter9() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <Zap className="w-6 h-6 text-sky-600 dark:text-sky-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            비동기 프로그래밍
          </h2>
        </div>

        <div className="bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/20 dark:to-blue-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            비동기 프로그래밍은 I/O 작업(파일, 네트워크, DB)의 대기 시간 동안 다른 작업을 수행하여 성능을 극대화합니다.
            Python의 async/await 문법과 asyncio 모듈로 고성능 비동기 코드를 작성할 수 있습니다.
            웹 API, 크롤러, 실시간 애플리케이션의 필수 기술입니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-sky-600 dark:text-sky-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-sky-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. async/await 문법 완벽 이해
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              코루틴 정의, 실행, await 키워드 마스터
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. asyncio로 비동기 작업 다루기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              이벤트 루프, Task, gather 활용
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-cyan-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. 동시성 작업 처리와 성능 최적화
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              병렬 API 호출, 타임아웃, 에러 처리
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-indigo-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. 실전 비동기 패턴 구현하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              aiohttp, 웹 크롤러, 실시간 데이터 처리
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: async/await 기초 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Zap className="w-6 h-6 text-sky-600 dark:text-sky-400" />
          1. async/await 기본 문법
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              동기 vs 비동기 비교
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import time
import asyncio

# 동기 코드 (순차 실행)
def fetch_data_sync(id):
    print(f"데이터 {id} 요청 시작")
    time.sleep(2)  # API 호출 시뮬레이션
    print(f"데이터 {id} 완료")
    return f"data_{id}"

start = time.time()
result1 = fetch_data_sync(1)
result2 = fetch_data_sync(2)
result3 = fetch_data_sync(3)
print(f"총 시간: {time.time() - start:.2f}초")  # ~6초

# 비동기 코드 (병렬 실행)
async def fetch_data_async(id):
    print(f"데이터 {id} 요청 시작")
    await asyncio.sleep(2)  # 비동기 대기
    print(f"데이터 {id} 완료")
    return f"data_{id}"

async def main():
    start = time.time()
    results = await asyncio.gather(
        fetch_data_async(1),
        fetch_data_async(2),
        fetch_data_async(3)
    )
    print(f"총 시간: {time.time() - start:.2f}초")  # ~2초!
    return results

# asyncio.run(main())  # Python 3.7+`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              코루틴 정의와 실행
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 코루틴 정의
async def greet(name):
    await asyncio.sleep(1)
    return f"안녕, {name}!"

# 실행 방법 1: asyncio.run()
result = asyncio.run(greet("Kelly"))
print(result)  # 안녕, Kelly!

# 실행 방법 2: await 사용
async def main():
    result = await greet("John")
    print(result)

asyncio.run(main())`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: asyncio 핵심 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Clock className="w-6 h-6 text-sky-600 dark:text-sky-400" />
          2. asyncio 핵심 기능
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              asyncio.gather() - 병렬 실행
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`async def task1():
    await asyncio.sleep(2)
    return "작업1 완료"

async def task2():
    await asyncio.sleep(1)
    return "작업2 완료"

async def task3():
    await asyncio.sleep(3)
    return "작업3 완료"

# 모든 작업 동시 실행
async def main():
    results = await asyncio.gather(
        task1(),
        task2(),
        task3()
    )
    print(results)  # ['작업1 완료', '작업2 완료', '작업3 완료']
    # 총 시간: ~3초 (가장 긴 task3 기준)

asyncio.run(main())`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Task와 타임아웃
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# Task 생성
async def main():
    task = asyncio.create_task(fetch_data_async(1))
    await asyncio.sleep(1)
    print("다른 작업 진행...")
    result = await task

# 타임아웃
async def slow_operation():
    await asyncio.sleep(5)
    return "완료"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("시간 초과!")`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 실전 패턴 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3. 실전 비동기 패턴
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              비동기 웹 크롤러
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import aiohttp

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"오류: {url} - {e}")
        return None

async def crawl_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# 사용
urls = ["https://example.com/page1", "https://example.com/page2"]
results = asyncio.run(crawl_urls(urls))`}
              </pre>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: Semaphore로 동시 작업 수 제한
                </h5>
                <div className="bg-amber-100 dark:bg-amber-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-amber-900 dark:text-amber-200">
{`sem = asyncio.Semaphore(5)  # 최대 5개 동시 실행

async def limited_task(url):
    async with sem:
        return await fetch_url(url)`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          핵심 요약
        </h3>

        <div className="bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-sky-600 dark:text-sky-400 mt-0.5 flex-shrink-0" />
              <span>
                async/await로 I/O 대기 시간 동안 다른 작업을 수행하여 성능을 극대화합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-sky-600 dark:text-sky-400 mt-0.5 flex-shrink-0" />
              <span>
                asyncio.gather()로 여러 코루틴을 동시에 실행하고, wait_for()로 타임아웃을 설정합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-sky-600 dark:text-sky-400 mt-0.5 flex-shrink-0" />
              <span>
                Task 객체로 백그라운드 작업을 제어하고, Queue로 생산자-소비자 패턴을 구현합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-sky-600 dark:text-sky-400 mt-0.5 flex-shrink-0" />
              <span>
                aiohttp로 고성능 웹 크롤러를 만들고, Semaphore로 동시 작업 수를 제한합니다
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
          비동기 프로그래밍을 마스터했다면, 이제 Python 개발의 모범 사례와 실전 배포 전략을 학습할 차례입니다.
          Chapter 10에서 PEP 8, 가상 환경, 패키지 관리, 프로덕션 배포를 완벽하게 마스터하세요!
        </p>
        <Link
          href="/modules/python-programming/best-practices"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 10: 모범 사례와 실전 배포 →
        </Link>
      </section>
    </div>
  );
}
