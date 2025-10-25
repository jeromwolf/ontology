'use client'

import React, { useState } from 'react'
import { FileText, Book, Code, Copy, Check, Download } from 'lucide-react'

type DocFormat = 'jsdoc' | 'sphinx' | 'javadoc'
type DocType = 'function' | 'readme' | 'api'

interface GeneratedDoc {
  format: DocFormat
  type: DocType
  documentation: string
  preview: string
}

export default function DocGenerator() {
  const [code, setCode] = useState('')
  const [docFormat, setDocFormat] = useState<DocFormat>('jsdoc')
  const [docType, setDocType] = useState<DocType>('function')
  const [result, setResult] = useState<GeneratedDoc | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [copied, setCopied] = useState(false)

  const exampleCodes = [
    {
      title: 'JavaScript 함수',
      format: 'jsdoc' as DocFormat,
      code: `function calculateTotal(items, taxRate = 0.1, discountCode = null) {
  const subtotal = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const discount = discountCode ? applyDiscount(subtotal, discountCode) : 0;
  const tax = (subtotal - discount) * taxRate;
  return subtotal - discount + tax;
}`
    },
    {
      title: 'Python 함수',
      format: 'sphinx' as DocFormat,
      code: `def process_user_data(user_id, include_history=False, max_items=100):
    user = fetch_user(user_id)
    if not user:
        raise UserNotFoundError(f"User {user_id} not found")

    data = {
        'profile': user.profile,
        'settings': user.settings
    }

    if include_history:
        data['history'] = user.get_history(limit=max_items)

    return data`
    },
    {
      title: 'Java 메서드',
      format: 'javadoc' as DocFormat,
      code: `public List<Product> searchProducts(String keyword,
                                    ProductCategory category,
                                    PriceRange priceRange,
                                    SortOrder sortOrder) {
    List<Product> results = productRepository
        .findByKeyword(keyword)
        .stream()
        .filter(p -> category == null || p.getCategory().equals(category))
        .filter(p -> priceRange == null || priceRange.contains(p.getPrice()))
        .sorted(sortOrder.getComparator())
        .collect(Collectors.toList());

    return results;
}`
    }
  ]

  const generateDocumentation = (sourceCode: string, format: DocFormat, type: DocType): GeneratedDoc => {
    let documentation = ''
    let preview = ''

    if (type === 'function') {
      if (format === 'jsdoc') {
        documentation = `/**
 * 주문 아이템들의 총액을 계산합니다.
 *
 * 이 함수는 아이템 가격의 합계를 계산하고, 할인 코드가 제공된 경우
 * 할인을 적용한 후 세금을 추가하여 최종 금액을 반환합니다.
 *
 * @param {Array<Object>} items - 주문 아이템 배열
 * @param {number} items[].price - 아이템 가격
 * @param {number} items[].quantity - 아이템 수량
 * @param {number} [taxRate=0.1] - 세율 (기본값: 0.1 = 10%)
 * @param {string|null} [discountCode=null] - 할인 코드 (선택사항)
 *
 * @returns {number} 세금과 할인이 적용된 최종 금액
 *
 * @throws {TypeError} items가 배열이 아닌 경우
 * @throws {RangeError} taxRate가 0보다 작거나 1보다 큰 경우
 *
 * @example
 * const items = [
 *   { price: 10, quantity: 2 },
 *   { price: 15, quantity: 1 }
 * ];
 * const total = calculateTotal(items, 0.1, 'SUMMER20');
 * console.log(total); // 31.5
 *
 * @see {@link applyDiscount} 할인 적용 함수
 * @since 1.0.0
 */
function calculateTotal(items, taxRate = 0.1, discountCode = null) {
  // ... implementation
}`

        preview = `# calculateTotal

주문 아이템들의 총액을 계산합니다.

## 설명
이 함수는 아이템 가격의 합계를 계산하고, 할인 코드가 제공된 경우 할인을 적용한 후 세금을 추가하여 최종 금액을 반환합니다.

## 매개변수
- **items** (Array<Object>) - 주문 아이템 배열
  - price (number) - 아이템 가격
  - quantity (number) - 아이템 수량
- **taxRate** (number, optional) - 세율 (기본값: 0.1)
- **discountCode** (string|null, optional) - 할인 코드

## 반환값
(number) 세금과 할인이 적용된 최종 금액

## 예외
- **TypeError** - items가 배열이 아닌 경우
- **RangeError** - taxRate가 유효하지 않은 경우

## 사용 예제
\`\`\`javascript
const items = [
  { price: 10, quantity: 2 },
  { price: 15, quantity: 1 }
];
const total = calculateTotal(items, 0.1, 'SUMMER20');
console.log(total); // 31.5
\`\`\``

      } else if (format === 'sphinx') {
        documentation = `def process_user_data(user_id, include_history=False, max_items=100):
    """사용자 데이터를 처리하고 반환합니다.

    이 함수는 사용자 ID를 기반으로 사용자 정보를 조회하고,
    선택적으로 사용자 히스토리를 포함하여 반환합니다.

    Parameters
    ----------
    user_id : int
        조회할 사용자의 ID
    include_history : bool, optional
        사용자 히스토리 포함 여부 (기본값: False)
    max_items : int, optional
        히스토리 최대 항목 수 (기본값: 100)

    Returns
    -------
    dict
        사용자 데이터를 포함하는 딕셔너리

        - profile (dict): 사용자 프로필 정보
        - settings (dict): 사용자 설정
        - history (list, optional): 사용자 히스토리 (include_history=True인 경우)

    Raises
    ------
    UserNotFoundError
        사용자를 찾을 수 없는 경우
    ValueError
        user_id가 유효하지 않은 경우
    PermissionError
        사용자 데이터 접근 권한이 없는 경우

    Examples
    --------
    기본 사용법:

    >>> data = process_user_data(12345)
    >>> print(data['profile']['name'])
    'John Doe'

    히스토리 포함:

    >>> data = process_user_data(12345, include_history=True, max_items=50)
    >>> len(data['history'])
    50

    See Also
    --------
    fetch_user : 사용자 조회 함수
    get_history : 히스토리 조회 함수

    Notes
    -----
    - 대량의 히스토리 조회 시 성능에 영향을 줄 수 있습니다.
    - max_items는 1~1000 범위 내에서 설정하는 것을 권장합니다.

    References
    ----------
    .. [1] User Data API Documentation
       https://docs.example.com/user-data-api
    """
    # ... implementation`

        preview = `# process_user_data

사용자 데이터를 처리하고 반환합니다.

## 설명
이 함수는 사용자 ID를 기반으로 사용자 정보를 조회하고, 선택적으로 사용자 히스토리를 포함하여 반환합니다.

## 매개변수
- **user_id** (int) - 조회할 사용자의 ID
- **include_history** (bool, optional) - 사용자 히스토리 포함 여부 (기본값: False)
- **max_items** (int, optional) - 히스토리 최대 항목 수 (기본값: 100)

## 반환값
**dict** - 사용자 데이터를 포함하는 딕셔너리
- profile (dict): 사용자 프로필 정보
- settings (dict): 사용자 설정
- history (list, optional): 사용자 히스토리

## 예외
- **UserNotFoundError** - 사용자를 찾을 수 없는 경우
- **ValueError** - user_id가 유효하지 않은 경우
- **PermissionError** - 접근 권한이 없는 경우`

      } else if (format === 'javadoc') {
        documentation = `/**
 * 주어진 검색 조건에 맞는 상품 목록을 검색합니다.
 *
 * <p>이 메서드는 키워드로 상품을 검색하고, 선택적으로 카테고리와 가격 범위로
 * 필터링한 후, 지정된 정렬 순서로 결과를 반환합니다.</p>
 *
 * <p>검색 과정:
 * <ol>
 *   <li>키워드로 초기 검색 수행</li>
 *   <li>카테고리 필터 적용 (지정된 경우)</li>
 *   <li>가격 범위 필터 적용 (지정된 경우)</li>
 *   <li>정렬 순서 적용</li>
 * </ol>
 * </p>
 *
 * @param keyword 검색할 키워드 (null이거나 빈 문자열일 수 없음)
 * @param category 상품 카테고리 필터 (null인 경우 모든 카테고리 포함)
 * @param priceRange 가격 범위 필터 (null인 경우 모든 가격 포함)
 * @param sortOrder 정렬 순서 (null일 수 없음)
 *
 * @return 검색 조건에 맞는 상품 목록 (비어있을 수 있음)
 *
 * @throws IllegalArgumentException keyword가 null이거나 빈 문자열인 경우
 * @throws IllegalArgumentException sortOrder가 null인 경우
 * @throws RepositoryException 데이터베이스 접근 중 오류가 발생한 경우
 *
 * @see Product
 * @see ProductCategory
 * @see PriceRange
 * @see SortOrder
 *
 * @since 2.0.0
 * @version 2.1.0
 *
 * @example
 * <pre>{@code
 * List<Product> results = searchProducts(
 *     "laptop",
 *     ProductCategory.ELECTRONICS,
 *     new PriceRange(500, 2000),
 *     SortOrder.PRICE_ASC
 * );
 *
 * results.forEach(p -> System.out.println(p.getName()));
 * }</pre>
 *
 * @apiNote 대량의 결과를 처리할 때는 페이지네이션 사용을 고려하세요.
 * @implNote Stream API를 사용하여 메모리 효율적으로 필터링합니다.
 */
public List<Product> searchProducts(String keyword,
                                   ProductCategory category,
                                   PriceRange priceRange,
                                   SortOrder sortOrder) {
    // ... implementation
}`

        preview = `# searchProducts

주어진 검색 조건에 맞는 상품 목록을 검색합니다.

## 설명
이 메서드는 키워드로 상품을 검색하고, 선택적으로 카테고리와 가격 범위로 필터링한 후, 지정된 정렬 순서로 결과를 반환합니다.

## 매개변수
- **keyword** (String) - 검색할 키워드
- **category** (ProductCategory) - 상품 카테고리 필터
- **priceRange** (PriceRange) - 가격 범위 필터
- **sortOrder** (SortOrder) - 정렬 순서

## 반환값
**List<Product>** - 검색 조건에 맞는 상품 목록

## 예외
- **IllegalArgumentException** - 필수 매개변수가 유효하지 않은 경우
- **RepositoryException** - 데이터베이스 오류

## 사용 예제
\`\`\`java
List<Product> results = searchProducts(
    "laptop",
    ProductCategory.ELECTRONICS,
    new PriceRange(500, 2000),
    SortOrder.PRICE_ASC
);
\`\`\``
      }
    } else if (type === 'readme') {
      documentation = `# 프로젝트 이름

간단한 한 줄 설명을 작성하세요.

## 📋 목차
- [소개](#소개)
- [주요 기능](#주요-기능)
- [설치 방법](#설치-방법)
- [사용법](#사용법)
- [API 문서](#api-문서)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

## 🎯 소개

이 프로젝트에 대한 자세한 설명을 작성하세요. 프로젝트의 목적, 해결하는 문제, 주요 특징 등을 포함합니다.

## ✨ 주요 기능

- **기능 1**: 첫 번째 주요 기능 설명
- **기능 2**: 두 번째 주요 기능 설명
- **기능 3**: 세 번째 주요 기능 설명

## 🚀 설치 방법

### 사전 요구사항
- Node.js 18.0 이상
- npm 또는 yarn

### 설치
\`\`\`bash
# 저장소 클론
git clone https://github.com/username/project.git

# 디렉토리 이동
cd project

# 의존성 설치
npm install
\`\`\`

## 💻 사용법

### 기본 사용법
\`\`\`javascript
import { calculateTotal } from './calculator';

const items = [
  { price: 10, quantity: 2 },
  { price: 15, quantity: 1 }
];

const total = calculateTotal(items);
console.log(total);
\`\`\`

### 고급 사용법
\`\`\`javascript
const total = calculateTotal(items, {
  taxRate: 0.1,
  discountCode: 'SUMMER20'
});
\`\`\`

## 📚 API 문서

자세한 API 문서는 [여기](https://docs.example.com)에서 확인하세요.

## 🤝 기여하기

기여를 환영합니다! 다음 단계를 따라주세요:

1. Fork the Project
2. Create your Feature Branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your Changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the Branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 \`LICENSE\` 파일을 참조하세요.

## 👥 저자

- **이름** - [GitHub](https://github.com/username)

## 🙏 감사의 글

- 프로젝트에 도움을 준 분들
- 참고한 리소스나 라이브러리`

      preview = documentation

    } else if (type === 'api') {
      documentation = `# API 문서

## 개요
이 API는 RESTful 방식으로 설계되었으며, JSON 형식으로 데이터를 주고받습니다.

## 기본 정보
- **Base URL**: \`https://api.example.com/v1\`
- **인증**: Bearer Token
- **Rate Limit**: 1000 requests/hour

## 인증

모든 API 요청에는 Authorization 헤더가 필요합니다:

\`\`\`http
Authorization: Bearer YOUR_API_TOKEN
\`\`\`

## 엔드포인트

### 사용자 조회
사용자 정보를 조회합니다.

**요청**
\`\`\`http
GET /users/{userId}
\`\`\`

**경로 매개변수**
- \`userId\` (integer, required) - 사용자 ID

**쿼리 매개변수**
- \`include_history\` (boolean, optional) - 히스토리 포함 여부 (기본값: false)
- \`max_items\` (integer, optional) - 최대 항목 수 (기본값: 100)

**응답 예시** (200 OK)
\`\`\`json
{
  "id": 12345,
  "profile": {
    "name": "John Doe",
    "email": "john@example.com"
  },
  "settings": {
    "theme": "dark",
    "language": "ko"
  },
  "history": [
    {
      "id": 1,
      "action": "login",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
\`\`\`

**오류 응답**
- \`404 Not Found\` - 사용자를 찾을 수 없음
\`\`\`json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID 12345 not found"
  }
}
\`\`\`

### 상품 검색
키워드로 상품을 검색합니다.

**요청**
\`\`\`http
GET /products/search
\`\`\`

**쿼리 매개변수**
- \`keyword\` (string, required) - 검색 키워드
- \`category\` (string, optional) - 카테고리 필터
- \`min_price\` (number, optional) - 최소 가격
- \`max_price\` (number, optional) - 최대 가격
- \`sort\` (string, optional) - 정렬 방식 (price_asc, price_desc, name)

**응답 예시** (200 OK)
\`\`\`json
{
  "total": 42,
  "page": 1,
  "per_page": 20,
  "results": [
    {
      "id": 101,
      "name": "Laptop Pro 15",
      "price": 1299.99,
      "category": "electronics"
    }
  ]
}
\`\`\`

## 오류 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 401 | 인증 실패 |
| 403 | 권한 없음 |
| 404 | 리소스를 찾을 수 없음 |
| 429 | Rate limit 초과 |
| 500 | 서버 오류 |

## Rate Limiting

- 시간당 1000 요청으로 제한
- 응답 헤더로 현재 사용량 확인 가능:
  - \`X-RateLimit-Limit\`: 시간당 최대 요청 수
  - \`X-RateLimit-Remaining\`: 남은 요청 수
  - \`X-RateLimit-Reset\`: 리셋 시간 (Unix timestamp)`

      preview = documentation
    }

    return {
      format,
      type,
      documentation,
      preview
    }
  }

  const handleGenerate = () => {
    if (code.trim().length === 0 && docType === 'function') return

    setIsGenerating(true)
    setTimeout(() => {
      const doc = generateDocumentation(code, docFormat, docType)
      setResult(doc)
      setIsGenerating(false)
    }, 1000)
  }

  const loadExample = (example: typeof exampleCodes[0]) => {
    setCode(example.code)
    setDocFormat(example.format)
    setDocType('function')
    setResult(null)
  }

  const handleCopy = () => {
    if (!result) return
    navigator.clipboard.writeText(result.documentation)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    if (!result) return

    const extension = docType === 'readme' ? 'md' : docFormat === 'jsdoc' ? 'js' : docFormat === 'sphinx' ? 'py' : 'java'
    const filename = `documentation.${extension}`

    const blob = new Blob([result.documentation], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              AI 문서 생성기
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            코드로부터 전문적인 문서를 자동으로 생성
          </p>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Doc Format */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">문서 형식:</h3>
            <div className="grid grid-cols-3 gap-3">
              {(['jsdoc', 'sphinx', 'javadoc'] as DocFormat[]).map((format) => (
                <button
                  key={format}
                  onClick={() => setDocFormat(format)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    docFormat === format
                      ? 'border-purple-600 bg-purple-50 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                      : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  {format.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Doc Type */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">문서 유형:</h3>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => setDocType('function')}
                className={`p-3 rounded-lg border-2 transition-all ${
                  docType === 'function'
                    ? 'border-purple-600 bg-purple-50 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Code className="w-5 h-5 mx-auto mb-1" />
                <div className="text-xs">함수</div>
              </button>
              <button
                onClick={() => setDocType('readme')}
                className={`p-3 rounded-lg border-2 transition-all ${
                  docType === 'readme'
                    ? 'border-purple-600 bg-purple-50 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                <FileText className="w-5 h-5 mx-auto mb-1" />
                <div className="text-xs">README</div>
              </button>
              <button
                onClick={() => setDocType('api')}
                className={`p-3 rounded-lg border-2 transition-all ${
                  docType === 'api'
                    ? 'border-purple-600 bg-purple-50 dark:bg-purple-900 text-purple-700 dark:text-purple-300'
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Book className="w-5 h-5 mx-auto mb-1" />
                <div className="text-xs">API</div>
              </button>
            </div>
          </div>
        </div>

        {/* Example Buttons */}
        {docType === 'function' && (
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">예제 코드:</h3>
            <div className="flex flex-wrap gap-3">
              {exampleCodes.map((example, index) => (
                <button
                  key={index}
                  onClick={() => loadExample(example)}
                  className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg border border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900 transition-colors text-sm"
                >
                  {example.title}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white">
                {docType === 'function' ? '함수 코드' : '프로젝트 정보'}
              </h2>
              <button
                onClick={handleGenerate}
                disabled={isGenerating || (docType === 'function' && code.trim().length === 0)}
                className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4" />
                    문서 생성
                  </>
                )}
              </button>
            </div>

            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder={
                docType === 'function'
                  ? '문서화할 함수 코드를 입력하세요...'
                  : docType === 'readme'
                  ? '프로젝트 이름, 설명, 주요 기능 등을 입력하세요...'
                  : 'API 엔드포인트, 매개변수 등을 입력하세요...'
              }
              className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-mono text-sm rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
              spellCheck={false}
            />
          </div>

          {/* Output */}
          <div>
            {result ? (
              <div className="space-y-6">
                {/* Generated Documentation */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold text-gray-800 dark:text-white">생성된 문서</h2>
                    <div className="flex gap-2">
                      <button
                        onClick={handleCopy}
                        className="p-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                        title="복사"
                      >
                        {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                      </button>
                      <button
                        onClick={handleDownload}
                        className="p-2 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
                        title="다운로드"
                      >
                        <Download className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  <div className="max-h-96 overflow-y-auto">
                    <pre className="p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 text-xs font-mono rounded-lg whitespace-pre-wrap">
                      {result.documentation}
                    </pre>
                  </div>
                </div>

                {/* Preview */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4">미리보기</h2>
                  <div className="prose dark:prose-invert max-w-none">
                    <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg text-sm max-h-96 overflow-y-auto">
                      <pre className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap font-sans">
                        {result.preview}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 h-full flex items-center justify-center">
                <div className="text-center text-gray-400 dark:text-gray-600">
                  <Book className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-lg">
                    {docType === 'function' ? '코드를 입력하고' : '정보를 입력하고'}
                    <br />
                    문서 생성 버튼을 클릭하세요
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
