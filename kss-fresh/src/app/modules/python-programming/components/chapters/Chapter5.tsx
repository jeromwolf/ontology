'use client';

import { Box, CheckCircle2, Code2, Eye, Layers, Lightbulb, Play, Shield } from 'lucide-react';
import Link from 'next/link';

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 pb-24 space-y-16">
      {/* Introduction */}
      <section>
        <div className="flex items-center gap-3 mb-8">
          <Box className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            객체지향 프로그래밍 (OOP)
          </h2>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-8 mb-8">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
            객체지향 프로그래밍(OOP)은 현대 소프트웨어 개발의 핵심 패러다임입니다.
            데이터와 함수를 하나의 객체로 묶어 코드 재사용성과 유지보수성을 극대화합니다.
            클래스, 상속, 캡슐화, 다형성의 4대 원칙을 Python으로 완벽하게 구현해봅니다.
          </p>
        </div>
      </section>

      {/* Learning Objectives */}
      <section>
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          학습 목표
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-purple-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              1. 클래스와 객체의 개념 완전 이해
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              설계도(클래스)로 인스턴스(객체) 생성하기
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-pink-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              2. 상속과 다형성으로 코드 확장하기
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              부모 클래스 기능을 자식이 확장하고 오버라이딩
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-indigo-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              3. 캡슐화 원칙과 실전 적용
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Private 속성으로 데이터 보호 및 Getter/Setter
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-8 border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              4. 클래스 메서드와 정적 메서드 활용
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              @classmethod, @staticmethod 데코레이터 마스터
            </p>
          </div>
        </div>
      </section>

      {/* Section 1: 클래스와 객체 기초 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Box className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          1. 클래스와 객체 기초
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              클래스 정의와 인스턴스 생성
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              클래스는 객체를 만들기 위한 청사진(설계도)입니다.
              __init__ 메서드(생성자)로 객체 초기화 작업을 수행합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 기본 클래스 정의
class Person:
    # 클래스 변수 (모든 인스턴스가 공유)
    species = "Human"

    def __init__(self, name, age):
        # 인스턴스 변수 (각 객체마다 고유)
        self.name = name
        self.age = age

    def introduce(self):
        return f"안녕하세요, {self.name}입니다. {self.age}살입니다."

    def birthday(self):
        self.age += 1
        return f"생일 축하합니다! 이제 {self.age}살입니다."

# 인스턴스 생성
kelly = Person("Kelly", 25)
john = Person("John", 30)

# 메서드 호출
print(kelly.introduce())  # 안녕하세요, Kelly입니다. 25살입니다.
kelly.birthday()
print(kelly.age)  # 26

# 클래스 변수 접근
print(Person.species)  # Human
print(kelly.species)   # Human (모든 인스턴스가 공유)`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              특수 메서드 (Magic Methods)
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    # 문자열 표현 (print()시 호출)
    def __str__(self):
        return f"{self.title} by {self.author}"

    # 개발자용 표현 (repr()시 호출)
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.pages})"

    # 비교 연산자
    def __eq__(self, other):
        return self.pages == other.pages

    def __lt__(self, other):
        return self.pages < other.pages

# 사용 예제
book1 = Book("Python Guide", "Kelly", 300)
book2 = Book("AI Basics", "John", 450)

print(book1)          # Python Guide by Kelly
print(repr(book1))    # Book('Python Guide', 'Kelly', 300)
print(book1 == book2) # False
print(book1 < book2)  # True (300 < 450)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: 상속과 다형성 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Layers className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          2. 상속과 다형성
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              단일 상속 (Single Inheritance)
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              자식 클래스가 부모 클래스의 속성과 메서드를 물려받아 코드 재사용성을 높입니다.
              super()로 부모 클래스 메서드를 호출할 수 있습니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`# 부모 클래스
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Some sound"

    def info(self):
        return f"I am {self.name}"

# 자식 클래스 1: Dog
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 부모 __init__ 호출
        self.breed = breed

    # 메서드 오버라이딩
    def speak(self):
        return "멍멍!"

    # 새로운 메서드 추가
    def fetch(self):
        return f"{self.name}가 공을 가져옵니다"

# 자식 클래스 2: Cat
class Cat(Animal):
    def speak(self):
        return "야옹~"

    def climb(self):
        return f"{self.name}가 나무에 올라갑니다"

# 사용 예제
dog = Dog("바둑이", "진돗개")
cat = Cat("나비")

print(dog.speak())    # 멍멍!
print(dog.info())     # I am 바둑이
print(dog.fetch())    # 바둑이가 공을 가져옵니다
print(cat.speak())    # 야옹~

# 다형성 (Polymorphism)
animals = [dog, cat]
for animal in animals:
    print(f"{animal.name}: {animal.speak()}")
# 바둑이: 멍멍!
# 나비: 야옹~`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              다중 상속 (Multiple Inheritance)
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class Flyer:
    def fly(self):
        return "하늘을 납니다"

class Swimmer:
    def swim(self):
        return "물에서 헤엄칩니다"

# 다중 상속
class Duck(Animal, Flyer, Swimmer):
    def speak(self):
        return "꽥꽥!"

# 사용 예제
duck = Duck("도날드")
print(duck.speak())  # 꽥꽥!
print(duck.fly())    # 하늘을 납니다
print(duck.swim())   # 물에서 헤엄칩니다
print(duck.info())   # I am 도날드

# MRO (Method Resolution Order) 확인
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyer'>,
#  <class 'Swimmer'>, <class 'object'>)`}
              </pre>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1 flex-shrink-0" />
              <div>
                <h5 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Pro Tip: 추상 베이스 클래스 (ABC)
                </h5>
                <div className="bg-amber-100 dark:bg-amber-900/50 rounded p-3 font-mono text-sm">
                  <pre className="text-amber-900 dark:text-amber-200">
{`from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

# shape = Shape()  # 에러! 추상 클래스는 인스턴스화 불가
circle = Circle(5)
print(circle.area())  # 78.5`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 캡슐화 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Shield className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          3. 캡슐화와 접근 제어
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Private 속성과 메서드
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              언더스코어(__) 접두사로 private 속성을 만들어 외부 접근을 제한합니다.
              Property 데코레이터로 Getter/Setter를 구현합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # Private 속성

    # Getter
    @property
    def balance(self):
        return self.__balance

    # Setter
    @balance.setter
    def balance(self, amount):
        if amount < 0:
            raise ValueError("잔액은 0 이상이어야 합니다")
        self.__balance = amount

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("입금액은 양수여야 합니다")
        self.__balance += amount
        return f"{amount}원 입금 완료. 잔액: {self.__balance}원"

    def withdraw(self, amount):
        if amount > self.__balance:
            raise ValueError("잔액이 부족합니다")
        self.__balance -= amount
        return f"{amount}원 출금 완료. 잔액: {self.__balance}원"

    # Private 메서드
    def __log_transaction(self, type, amount):
        print(f"[LOG] {type}: {amount}원")

# 사용 예제
account = BankAccount("Kelly", 10000)
print(account.balance)     # 10000 (Getter 호출)
account.deposit(5000)
print(account.balance)     # 15000

# account.__balance = 0    # 에러! 직접 접근 불가 (private)
# account.balance = -1000  # 에러! Setter 검증 실패`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              프로퍼티 (Property) 활용
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("절대 영도 이하는 불가능합니다")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

# 사용 예제
temp = Temperature(25)
print(temp.celsius)     # 25
print(temp.fahrenheit)  # 77.0

temp.fahrenheit = 86    # Setter 호출
print(temp.celsius)     # 30.0`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 클래스 메서드와 정적 메서드 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
          <Code2 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          4. 클래스 메서드와 정적 메서드
        </h3>

        <div className="space-y-8">
          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              @classmethod: 클래스 레벨 메서드
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              첫 번째 인자로 cls(클래스 자체)를 받습니다.
              팩토리 메서드 패턴에 주로 사용됩니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # 클래스 메서드 (팩토리 메서드)
    @classmethod
    def from_string(cls, date_string):
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        from datetime import date
        today = date.today()
        return cls(today.year, today.month, today.day)

    def __str__(self):
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

# 사용 예제
date1 = Date(2025, 1, 10)         # 일반 생성자
date2 = Date.from_string("2025-12-25")  # 팩토리 메서드
date3 = Date.today()              # 오늘 날짜

print(date1)  # 2025-01-10
print(date2)  # 2025-12-25`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              @staticmethod: 정적 메서드
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              self나 cls를 받지 않는 독립적 함수입니다.
              클래스 네임스페이스 내에서 관련 유틸리티 함수를 그룹화할 때 사용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`class MathUtils:
    PI = 3.14159

    @staticmethod
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)

    @classmethod
    def circle_area(cls, radius):
        return cls.PI * radius ** 2

# 사용 예제
print(MathUtils.is_prime(17))      # True
print(MathUtils.factorial(5))      # 120
print(MathUtils.circle_area(10))   # 314.159

# 인스턴스 생성 없이도 사용 가능
utils = MathUtils()
print(utils.is_prime(20))          # False (권장하지 않음)`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-5">
            <h5 className="font-semibold text-blue-900 dark:text-blue-200 mb-3">
              메서드 타입 비교표
            </h5>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-blue-100 dark:bg-blue-900/50">
                  <tr>
                    <th className="p-3 text-left text-blue-900 dark:text-blue-200">타입</th>
                    <th className="p-3 text-left text-blue-900 dark:text-blue-200">첫 번째 인자</th>
                    <th className="p-3 text-left text-blue-900 dark:text-blue-200">사용 시기</th>
                  </tr>
                </thead>
                <tbody className="text-gray-700 dark:text-gray-300">
                  <tr className="border-t border-blue-200 dark:border-blue-800">
                    <td className="p-3 font-mono">instance method</td>
                    <td className="p-3 font-mono">self</td>
                    <td className="p-3">인스턴스 데이터 접근/수정</td>
                  </tr>
                  <tr className="border-t border-blue-200 dark:border-blue-800">
                    <td className="p-3 font-mono">@classmethod</td>
                    <td className="p-3 font-mono">cls</td>
                    <td className="p-3">팩토리 메서드, 클래스 변수 접근</td>
                  </tr>
                  <tr className="border-t border-blue-200 dark:border-blue-800">
                    <td className="p-3 font-mono">@staticmethod</td>
                    <td className="p-3">없음</td>
                    <td className="p-3">유틸리티 함수 (독립적)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: 실전 프로젝트 */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          5. 실전 프로젝트: 도서관 관리 시스템
        </h3>

        <div className="space-y-6">
          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              OOP 원칙을 모두 활용한 실전 프로젝트입니다.
              상속, 캡슐화, 다형성, 클래스 메서드를 종합적으로 적용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <pre className="text-gray-900 dark:text-white font-mono text-sm">
{`from datetime import datetime, timedelta

# 추상 베이스 클래스
class LibraryItem:
    _id_counter = 0

    def __init__(self, title, author):
        LibraryItem._id_counter += 1
        self.id = LibraryItem._id_counter
        self.title = title
        self.author = author
        self.borrowed = False
        self.due_date = None

    def borrow(self, days=14):
        if self.borrowed:
            raise ValueError(f"{self.title}는 이미 대출 중입니다")
        self.borrowed = True
        self.due_date = datetime.now() + timedelta(days=days)
        return f"{self.title} 대출 완료 (반납일: {self.due_date.date()})"

    def return_item(self):
        if not self.borrowed:
            raise ValueError(f"{self.title}는 대출 중이 아닙니다")
        self.borrowed = False
        self.due_date = None
        return f"{self.title} 반납 완료"

    def is_overdue(self):
        if not self.borrowed:
            return False
        return datetime.now() > self.due_date

# 자식 클래스: 책
class Book(LibraryItem):
    def __init__(self, title, author, isbn, pages):
        super().__init__(title, author)
        self.isbn = isbn
        self.pages = pages

    def __str__(self):
        status = "대출 중" if self.borrowed else "대출 가능"
        return f"[도서] {self.title} - {self.author} ({status})"

# 자식 클래스: 잡지
class Magazine(LibraryItem):
    def __init__(self, title, publisher, issue_number):
        super().__init__(title, publisher)
        self.issue_number = issue_number

    def borrow(self, days=7):  # 오버라이딩: 잡지는 7일만 대출
        return super().borrow(days)

    def __str__(self):
        status = "대출 중" if self.borrowed else "대출 가능"
        return f"[잡지] {self.title} {self.issue_number}호 ({status})"

# 도서관 시스템
class Library:
    def __init__(self, name):
        self.name = name
        self._items = []

    def add_item(self, item):
        self._items.append(item)
        return f"{item.title} 추가 완료"

    def find_by_title(self, title):
        return [item for item in self._items if title.lower() in item.title.lower()]

    def available_items(self):
        return [item for item in self._items if not item.borrowed]

    def overdue_items(self):
        return [item for item in self._items if item.is_overdue()]

    @property
    def total_items(self):
        return len(self._items)

    def __str__(self):
        return f"{self.name} 도서관 (총 {self.total_items}개 항목)"

# 실행 예제
library = Library("서울 중앙")

# 아이템 추가
book1 = Book("Python 마스터", "Kelly", "978-1234567890", 500)
book2 = Book("AI 기초", "John", "978-0987654321", 350)
magazine1 = Magazine("Tech Monthly", "TechPub", 125)

library.add_item(book1)
library.add_item(book2)
library.add_item(magazine1)

# 대출
print(book1.borrow())     # Python 마스터 대출 완료...
print(magazine1.borrow()) # Tech Monthly 125호 대출 완료...

# 조회
print(library)            # 서울 중앙 도서관 (총 3개 항목)
print(f"대출 가능: {len(library.available_items())}개")
print(f"연체: {len(library.overdue_items())}개")`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Simulator Link */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-5">
          <div className="flex items-start gap-3">
            <Eye className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-1 flex-shrink-0" />
            <div>
              <h5 className="font-semibold text-purple-900 dark:text-purple-200 mb-2">
                실습: OOP 클래스 다이어그램 생성기
              </h5>
              <p className="text-sm text-purple-800 dark:text-purple-300 mb-4">
                Python 코드를 입력하면 자동으로 UML 클래스 다이어그램을 생성합니다.
                상속 관계, 속성, 메서드를 시각적으로 확인하고 OOP 설계를 학습하세요.
              </p>
              <Link
                href="/modules/python-programming/simulators/oop-diagram-generator"
                className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                OOP 다이어그램 생성기 실행
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="border-t border-gray-200 dark:border-gray-700 pt-12 mt-8">
        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          핵심 요약
        </h3>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
              <span>
                클래스는 객체를 만드는 설계도이며, __init__로 초기화하고 self로 인스턴스 속성에 접근합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
              <span>
                상속으로 코드 재사용성을 높이고, 다형성으로 같은 인터페이스를 다르게 구현할 수 있습니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
              <span>
                캡슐화로 데이터를 보호하고, @property로 Getter/Setter 패턴을 구현합니다
              </span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
              <span>
                @classmethod는 팩토리 메서드에, @staticmethod는 유틸리티 함수에 사용합니다
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
          객체지향 프로그래밍을 마스터했다면, 이제 견고한 코드 작성을 위한 예외 처리를 학습할 차례입니다.
          Chapter 6에서 try-except, 커스텀 예외, 로깅 전략을 익히세요!
        </p>
        <Link
          href="/modules/python-programming/exception-handling"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          Chapter 6: 예외 처리 →
        </Link>
      </section>
    </div>
  );
}
