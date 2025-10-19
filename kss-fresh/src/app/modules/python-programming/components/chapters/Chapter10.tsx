'use client';

import { Award, CheckCircle2, Lightbulb, Package, Rocket, Shield } from 'lucide-react';

export default function Chapter10() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <Award className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            모범 사례와 실전 배포
          </h2>
        </div>

        <div className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            프로페셔널 Python 개발자로 성장하기 위한 필수 지식입니다.
            PEP 8 스타일 가이드, 가상 환경 관리, 패키지 배포, 프로덕션 배포 전략을 마스터하여
            견고하고 유지보수 가능한 고품질 코드를 작성하세요.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-amber-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. PEP 8 스타일 가이드 준수하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              일관된 코드 스타일, 네이밍 규칙, 포맷팅
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-yellow-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 가상 환경으로 프로젝트 관리
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              venv, pipenv, poetry 활용한 환경 분리
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-orange-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. pip와 패키지 관리 완전 정복
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              requirements.txt, pyproject.toml 관리
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-red-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. 프로덕션 레벨 코드 작성 기법
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              로깅, 환경 변수, 보안, CI/CD 적용
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: PEP 8 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Shield className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          1. PEP 8 스타일 가이드
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              네이밍 규칙
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 변수, 함수: snake_case
user_name = "Kelly"
def calculate_total(items):
    pass

# 클래스: PascalCase
class UserAccount:
    pass

# 상수: UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100
API_KEY = "secret"

# Private: 앞에 언더스코어
def _internal_method():
    pass

class MyClass:
    def __init__(self):
        self._private_var = 42  # Protected
        self.__very_private = 99  # Private`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              코드 포맷팅
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 한 줄 최대 79자 (PEP 8 권장)
# 긴 문자열
long_text = (
    "이것은 매우 긴 문자열입니다. "
    "여러 줄로 나누어 작성하면 "
    "가독성이 향상됩니다."
)

# 함수 인자가 많을 때
def complex_function(
    param1, param2, param3,
    param4, param5, param6
):
    pass

# 리스트/딕셔너리
data = {
    'name': 'Kelly',
    'age': 25,
    'city': 'Seoul',
}

# 연산자 주변 공백
x = 1 + 2  # Good
y=1+2      # Bad

# Import 순서: 표준 → 서드파티 → 로컬
import os
import sys

import requests
import numpy as np

from myapp import utils`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                  Pro Tip: Black & Flake8 자동 포맷팅
                </h5>
                <div className="bg-blue-100 dark:bg-blue-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-blue-900 dark:text-blue-200">
{`# Black 설치 및 실행
pip install black
black myfile.py

# Flake8 (린트)
pip install flake8
flake8 myfile.py`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 가상 환경 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Package className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          2. 가상 환경 관리
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              venv 사용법
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 가상 환경 생성
python -m venv venv

# 활성화
# Windows
venv\\Scripts\\activate
# Mac/Linux
source venv/bin/activate

# 패키지 설치
pip install requests pandas

# requirements.txt 생성
pip freeze > requirements.txt

# requirements.txt로 설치
pip install -r requirements.txt

# 비활성화
deactivate`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              pyproject.toml (Modern)
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`[project]
name = "myproject"
version = "1.0.0"
dependencies = [
    "requests>=2.28.0",
    "pandas>=1.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
]

# 설치
pip install .
pip install .[dev]`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 프로덕션 코드 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Rocket className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          3. 프로덕션 레벨 코드 작성
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              로깅 설정
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`import logging

# 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 사용
logger.debug("디버그 메시지")
logger.info("정보 메시지")
logger.warning("경고 메시지")
logger.error("에러 메시지")
logger.critical("심각한 에러")`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              환경 변수 관리
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# .env 파일
DATABASE_URL=postgresql://localhost/mydb
API_KEY=secret_key_here
DEBUG=True

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')
DEBUG = os.getenv('DEBUG', 'False') == 'True'

# 절대 커밋하지 말 것: .gitignore
# .env
# *.log
# __pycache__/
# venv/`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              테스트 작성
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# test_calculator.py
import pytest

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_add_strings():
    with pytest.raises(TypeError):
        add("2", 3)

# 실행
# pytest test_calculator.py`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 배포 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4. 실전 배포 전략
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Docker 컨테이너화
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]

# 빌드 & 실행
docker build -t myapp .
docker run -p 8000:8000 myapp`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              CI/CD 파이프라인
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest
      - name: Lint
        run: flake8 .`}
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

        <div className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
              <span>
                PEP 8 스타일 가이드를 준수하여 일관되고 가독성 높은 코드를 작성합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
              <span>
                가상 환경으로 프로젝트별 의존성을 분리하여 충돌을 방지합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
              <span>
                로깅, 환경 변수, 테스트로 프로덕션 레벨의 안정적인 코드를 작성합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
              <span>
                Docker와 CI/CD로 배포 자동화하여 개발 생산성을 극대화합니다
              </span>
            </li>
          </ul>
        </div>
      </section>

      {/* Final Message */}
      <section className="bg-gradient-to-r from-amber-100 to-yellow-100 dark:from-amber-900/30 dark:to-yellow-900/30 rounded-xl p-8 border-2 border-amber-300 dark:border-amber-700">
        <div className="flex items-center gap-3 mb-4">
          <Award className="w-8 h-8 text-amber-600 dark:text-amber-400" />
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
            축하합니다! 🎉
          </h3>
        </div>
        <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
          Python Programming 모듈의 모든 챕터를 완료하셨습니다!
          이제 여러분은 Python 기초부터 고급 기능, 그리고 프로덕션 배포까지 마스터한 프로페셔널 개발자입니다.
        </p>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mt-4">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
            다음 학습 경로 추천
          </h4>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">→</span>
              <span>웹 프레임워크 (Flask, FastAPI, Django)</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">→</span>
              <span>데이터 분석 (Pandas, NumPy, Matplotlib)</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">→</span>
              <span>머신러닝 (scikit-learn, TensorFlow, PyTorch)</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-amber-600 dark:text-amber-400">→</span>
              <span>자동화 (Selenium, BeautifulSoup, Scrapy)</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
