'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, FileText, Scissors, Database, Code2 } from 'lucide-react'

export default function Chapter2Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/beginner"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          초급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <FileText size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 2: 문서 처리와 청킹</h1>
              <p className="text-emerald-100 text-lg">RAG의 핵심, 문서를 AI가 이해할 수 있게 만들기</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Document Processing */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <FileText className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 문서 처리의 중요성</h2>
              <p className="text-gray-600 dark:text-gray-400">다양한 형식의 문서를 일관된 형태로 변환</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">✅ 지원되는 문서 형식</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 bg-red-500 text-white rounded flex items-center justify-center text-xs font-bold">PDF</span>
                  <div>
                    <p className="text-sm font-medium text-blue-800 dark:text-blue-200">PDF 문서</p>
                    <p className="text-xs text-blue-600 dark:text-blue-300">PyPDF2, PDFMiner 활용</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 bg-blue-500 text-white rounded flex items-center justify-center text-xs font-bold">DOC</span>
                  <div>
                    <p className="text-sm font-medium text-blue-800 dark:text-blue-200">Word 문서</p>
                    <p className="text-xs text-blue-600 dark:text-blue-300">python-docx 라이브러리</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 bg-orange-500 text-white rounded flex items-center justify-center text-xs font-bold">HTML</span>
                  <div>
                    <p className="text-sm font-medium text-blue-800 dark:text-blue-200">웹 페이지</p>
                    <p className="text-xs text-blue-600 dark:text-blue-300">BeautifulSoup4 파싱</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 bg-gray-500 text-white rounded flex items-center justify-center text-xs font-bold">TXT</span>
                  <div>
                    <p className="text-sm font-medium text-blue-800 dark:text-blue-200">텍스트 파일</p>
                    <p className="text-xs text-blue-600 dark:text-blue-300">UTF-8 인코딩 지원</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">⚠️ 문서 처리의 도전과제</h3>
              <ul className="space-y-2 text-sm text-orange-700 dark:text-orange-300">
                <li>• <strong>레이아웃 보존:</strong> 표, 그림, 헤더/푸터 처리</li>
                <li>• <strong>인코딩 문제:</strong> 한글, 특수문자 깨짐 방지</li>
                <li>• <strong>메타데이터:</strong> 작성자, 생성일, 페이지 번호</li>
                <li>• <strong>품질 관리:</strong> OCR 오류, 불완전한 텍스트</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-emerald-50 dark:bg-emerald-900/20 p-6 rounded-xl border border-emerald-200 dark:border-emerald-700">
            <h3 className="font-bold text-emerald-800 dark:text-emerald-200 mb-3">💡 실무 팁</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-emerald-800 dark:text-emerald-200 mb-2">🔧 전처리 체크리스트:</p>
                <ul className="text-xs text-emerald-700 dark:text-emerald-300 space-y-1">
                  <li>✓ 불필요한 공백, 개행 제거</li>
                  <li>✓ 특수 문자 정규화</li>
                  <li>✓ 중복 내용 제거</li>
                </ul>
              </div>
              <div>
                <p className="text-sm font-medium text-emerald-800 dark:text-emerald-200 mb-2">📊 메타데이터 활용:</p>
                <ul className="text-xs text-emerald-700 dark:text-emerald-300 space-y-1">
                  <li>✓ 검색 필터링에 활용</li>
                  <li>✓ 권한 관리 기준</li>
                  <li>✓ 문서 출처 추적</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Chunking Strategies */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Scissors className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.2 청킹(Chunking) 전략</h2>
              <p className="text-gray-600 dark:text-gray-400">문서를 적절한 크기로 나누는 핵심 기술</p>
            </div>
          </div>

          <div className="space-y-6">
            {/* Fixed Size Chunking */}
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">1️⃣ 고정 크기 청킹</h3>
              
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">📏 설정 방법</h4>
                  <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <p><strong>청크 크기:</strong> 1000자</p>
                    <p><strong>중첩(Overlap):</strong> 200자</p>
                    <p><strong>분할 기준:</strong> 문장 단위</p>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">✅ 장점</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 구현이 간단</li>
                    <li>• 일관된 크기</li>
                    <li>• 처리 속도 빠름</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">❌ 단점</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 문맥 단절 가능</li>
                    <li>• 의미 단위 무시</li>
                    <li>• 품질 편차 발생</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Semantic Chunking */}
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">2️⃣ 의미적 청킹</h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3">🧠 작동 원리</h4>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <span className="w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">1</span>
                      <p className="text-sm text-green-700 dark:text-green-300">문단별로 임베딩 생성</p>
                    </div>
                    <div className="flex items-start gap-3">
                      <span className="w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">2</span>
                      <p className="text-sm text-green-700 dark:text-green-300">인접 문단 간 유사도 측정</p>
                    </div>
                    <div className="flex items-start gap-3">
                      <span className="w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">3</span>
                      <p className="text-sm text-green-700 dark:text-green-300">유사도 기준으로 묶거나 분할</p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3">⚡ 실제 사용예시</h4>
                  <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-sm">
                    <p className="text-gray-700 dark:text-gray-300 mb-2"><strong>청크 1:</strong> "회사 소개와 비전"</p>
                    <p className="text-gray-500 dark:text-gray-400 mb-3">→ 회사 관련 문단들이 자연스럽게 묶임</p>
                    
                    <p className="text-gray-700 dark:text-gray-300 mb-2"><strong>청크 2:</strong> "제품 및 서비스"</p>
                    <p className="text-gray-500 dark:text-gray-400">→ 제품 설명 문단들이 논리적으로 그룹화</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Overlap Strategy */}
            <div className="bg-amber-50 dark:bg-amber-900/20 p-6 rounded-xl border border-amber-200 dark:border-amber-700">
              <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-4">3️⃣ 중첩(Overlap) 전략</h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-3">🔄 중첩의 필요성</h4>
                  <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                    <div className="space-y-3">
                      <div className="border-l-4 border-red-500 pl-3">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">❌ 중첩 없을 때</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">문장이 중간에 끊어져 의미 손실</p>
                      </div>
                      <div className="border-l-4 border-green-500 pl-3">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">✅ 중첩 있을 때</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">문맥 연결성 유지, 검색 품질 향상</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-3">📊 최적 중첩 비율</h4>
                  <div className="space-y-3">
                    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">10-20%</span>
                        <span className="text-xs text-green-600 bg-green-100 dark:bg-green-900/20 px-2 py-1 rounded">권장</span>
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">일반적인 문서에 적합</p>
                    </div>
                    
                    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">30%</span>
                        <span className="text-xs text-amber-600 bg-amber-100 dark:bg-amber-900/20 px-2 py-1 rounded">고품질</span>
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">기술 문서, 법률 문서용</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Code Example */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
              <Code2 className="text-indigo-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.3 실습: 문서 처리 파이프라인</h2>
              <p className="text-gray-600 dark:text-gray-400">실제 코드로 문서 처리와 청킹 구현</p>
            </div>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 overflow-x-auto">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-gray-400 text-sm ml-2">document_processor.py</span>
            </div>
            
            <pre className="text-sm text-gray-300 leading-relaxed">
{`class DocumentProcessor:
    """문서 처리 및 청킹 파이프라인"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, file_path):
        """메인 문서 처리 함수"""
        
        # 1. 파일 형식 감지
        file_type = self.detect_file_type(file_path)
        
        # 2. 문서 파싱
        if file_type == 'pdf':
            text = self.parse_pdf(file_path)
        elif file_type == 'docx':
            text = self.parse_docx(file_path)
        elif file_type == 'html':
            text = self.parse_html(file_path)
        else:
            text = self.parse_text(file_path)
        
        # 3. 텍스트 전처리
        cleaned_text = self.clean_text(text)
        
        # 4. 청킹
        chunks = self.create_chunks(cleaned_text)
        
        return chunks
    
    def create_chunks(self, text):
        """고정 크기 + 중첩 청킹"""
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 청크 끝 위치 계산
            end = start + self.chunk_size
            
            # 문장 단위로 자르기 위해 마지막 마침표 찾기
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            
            # 청크 생성
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    'text': chunk,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_id': len(chunks)
                })
            
            # 다음 시작점 (중첩 고려)
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks

# 사용 예시
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.process_document("company_policy.pdf")

print(f"총 {len(chunks)}개 청크 생성")
for i, chunk in enumerate(chunks[:3]):
    print(f"\\nChunk {i+1}: {chunk['text'][:100]}...")
`}
            </pre>
          </div>
        </section>

        {/* Section 4: Best Practices */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실무 베스트 프랙티스</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4 flex items-center gap-2">
                <Database size={20} />
                청크 크기 최적화
              </h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li>• 짧은 문서: 500-800자</li>
                <li>• 긴 문서: 1000-1500자</li>
                <li>• 기술 문서: 1500-2000자</li>
                <li>• 대화형 문서: 300-500자</li>
              </ul>
            </div>
            
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4 flex items-center gap-2">
                <FileText size={20} />
                메타데이터 활용
              </h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li>• 문서 제목, 작성자 기록</li>
                <li>• 페이지 번호, 섹션 정보</li>
                <li>• 생성일, 수정일 추적</li>
                <li>• 태그, 카테고리 분류</li>
              </ul>
            </div>
            
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4 flex items-center gap-2">
                <Scissors size={20} />
                품질 관리
              </h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li>• 빈 청크 필터링</li>
                <li>• 중복 내용 제거</li>
                <li>• 최소 길이 기준 설정</li>
                <li>• 정기적인 품질 검증</li>
              </ul>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/beginner/chapter1"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: LLM의 한계점
          </Link>
          
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            초급 과정으로
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}