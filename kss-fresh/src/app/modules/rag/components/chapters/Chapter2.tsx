'use client'

import ChunkingDemo from '../ChunkingDemo'

// Chapter 2: Document Processing
export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">문서 처리의 중요성</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG 시스템의 성능은 문서를 얼마나 효과적으로 처리하고 저장하는지에 크게 좌우됩니다.
          적절한 청킹 전략은 검색 정확도와 답변 품질에 직접적인 영향을 미칩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">청킹 전략 체험하기</h2>
        <ChunkingDemo />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">다양한 문서 형식 처리</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">구조화된 문서</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• PDF: PyPDF2, pdfplumber로 텍스트 추출</li>
              <li>• Word: python-docx로 단락별 처리</li>
              <li>• HTML: BeautifulSoup으로 태그 파싱</li>
              <li>• Markdown: 헤더 기준 자동 분할</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">비구조화된 문서</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 일반 텍스트: 문장/단락 기준 분할</li>
              <li>• 코드: AST 파싱으로 의미 단위 추출</li>
              <li>• 테이블: 행/열 구조 보존</li>
              <li>• 이미지: OCR + 캡션 생성</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}