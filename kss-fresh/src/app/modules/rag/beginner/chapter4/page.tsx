'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Rocket, Code2, Package, CheckCircle2 } from 'lucide-react'

export default function Chapter4Page() {
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
              <Rocket size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 4: 첫 RAG 시스템 구축하기</h1>
              <p className="text-emerald-100 text-lg">30분 만에 작동하는 RAG 시스템을 만들어봅시다!</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Overview */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <Package className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.1 시스템 구성 요소</h2>
              <p className="text-gray-600 dark:text-gray-400">최소한의 RAG 시스템에 필요한 것들</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">📦 필수 패키지</h3>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg font-mono text-sm">
                <p className="text-gray-700 dark:text-gray-300">pip install langchain</p>
                <p className="text-gray-700 dark:text-gray-300">pip install openai</p>
                <p className="text-gray-700 dark:text-gray-300">pip install chromadb</p>
                <p className="text-gray-700 dark:text-gray-300">pip install pypdf</p>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">🔧 핵심 컴포넌트</h3>
              <ul className="space-y-2 text-green-700 dark:text-green-300">
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  <span>문서 로더 (Document Loader)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  <span>텍스트 분할기 (Text Splitter)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  <span>임베딩 모델 (Embeddings)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  <span>벡터 저장소 (Vector Store)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  <span>LLM (Language Model)</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Architecture Diagram */}
          <div className="mt-6 bg-gray-50 dark:bg-gray-900/50 p-6 rounded-xl">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4 text-center">RAG 파이프라인 아키텍처</h3>
            <div className="flex items-center justify-center gap-4 flex-wrap">
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-blue-500">
                <span className="text-sm font-medium">📄 문서</span>
              </div>
              <span className="text-2xl">→</span>
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-purple-500">
                <span className="text-sm font-medium">✂️ 청킹</span>
              </div>
              <span className="text-2xl">→</span>
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-green-500">
                <span className="text-sm font-medium">🔢 임베딩</span>
              </div>
              <span className="text-2xl">→</span>
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-amber-500">
                <span className="text-sm font-medium">💾 저장</span>
              </div>
              <span className="text-2xl">→</span>
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-red-500">
                <span className="text-sm font-medium">🔍 검색</span>
              </div>
              <span className="text-2xl">→</span>
              <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg border-2 border-indigo-500">
                <span className="text-sm font-medium">💬 생성</span>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Step by Step Implementation */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Code2 className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.2 단계별 구현</h2>
              <p className="text-gray-600 dark:text-gray-400">실제 코드로 RAG 시스템 구축하기</p>
            </div>
          </div>

          {/* Step 1: Setup */}
          <div className="mb-6">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Step 1: 환경 설정</h3>
            <div className="bg-gray-900 rounded-xl p-6">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{`import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key"`}</code>
              </pre>
            </div>
          </div>

          {/* Step 2: Document Loading */}
          <div className="mb-6">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Step 2: 문서 로딩 및 처리</h3>
            <div className="bg-gray-900 rounded-xl p-6">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{`# PDF 문서 로드
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

print(f"총 {len(chunks)}개의 청크가 생성되었습니다.")`}</code>
              </pre>
            </div>
          </div>

          {/* Step 3: Embeddings */}
          <div className="mb-6">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Step 3: 임베딩 및 벡터 저장</h3>
            <div className="bg-gray-900 rounded-xl p-6">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{`# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings()

# 벡터 데이터베이스 생성 및 저장
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("벡터 데이터베이스가 생성되었습니다!")`}</code>
              </pre>
            </div>
          </div>

          {/* Step 4: QA Chain */}
          <div className="mb-6">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Step 4: 질의응답 체인 생성</h3>
            <div className="bg-gray-900 rounded-xl p-6">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{`# LLM 초기화
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 검색기 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)`}</code>
              </pre>
            </div>
          </div>

          {/* Step 5: Using the System */}
          <div>
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-3">Step 5: 시스템 사용하기</h3>
            <div className="bg-gray-900 rounded-xl p-6">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{`# 질문하기
question = "이 문서의 핵심 내용은 무엇인가요?"
result = qa_chain({"query": question})

# 결과 출력
print("답변:", result["result"])
print("\\n참고한 문서:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 3: Complete Example */}
        <section className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8">
          <h2 className="text-2xl font-bold text-purple-800 dark:text-purple-200 mb-6">4.3 전체 코드 예제</h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">simple_rag.py - 복사해서 바로 실행 가능!</h3>
            <div className="bg-gray-900 rounded-xl p-6 max-h-96 overflow-y-auto">
              <pre className="text-sm text-gray-300">
                <code>{`#!/usr/bin/env python3
"""
Simple RAG System - 첫 번째 RAG 시스템
30분 만에 작동하는 RAG를 만들어보세요!
"""

import os
from typing import List
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class SimpleRAG:
    def __init__(self, persist_directory="./rag_db"):
        """RAG 시스템 초기화"""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_paths: List[str]):
        """다양한 문서 형식 로드"""
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                print(f"지원하지 않는 파일 형식: {file_path}")
                continue
            
            documents.extend(loader.load())
            print(f"✓ {file_path} 로드 완료")
        
        return documents
    
    def process_documents(self, documents):
        """문서를 청크로 분할"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"\\n📊 총 {len(chunks)}개의 청크가 생성되었습니다.")
        return chunks
    
    def create_vectorstore(self, chunks):
        """벡터 데이터베이스 생성"""
        print("\\n🔄 벡터 데이터베이스 생성 중...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.vectorstore.persist()
        print("✓ 벡터 데이터베이스 생성 완료!")
    
    def setup_qa_chain(self):
        """질의응답 체인 설정"""
        # 커스텀 프롬프트 템플릿
        template = """아래의 문맥을 사용하여 질문에 답변해주세요.
        
문맥: {context}

질문: {question}

답변할 때 다음 사항을 지켜주세요:
1. 문맥에 있는 정보만을 사용하여 답변하세요.
2. 문맥에 없는 내용은 추측하지 마세요.
3. 답변을 찾을 수 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

답변:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✓ QA 체인 설정 완료!")
    
    def build(self, file_paths: List[str]):
        """전체 RAG 시스템 구축"""
        print("🚀 RAG 시스템 구축을 시작합니다...\\n")
        
        # 1. 문서 로드
        documents = self.load_documents(file_paths)
        
        # 2. 문서 처리
        chunks = self.process_documents(documents)
        
        # 3. 벡터 DB 생성
        self.create_vectorstore(chunks)
        
        # 4. QA 체인 설정
        self.setup_qa_chain()
        
        print("\\n✅ RAG 시스템이 준비되었습니다!")
    
    def query(self, question: str):
        """질문에 대한 답변 생성"""
        if not self.qa_chain:
            print("❌ 먼저 build() 메서드를 실행하세요.")
            return None
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

def main():
    # 1. RAG 시스템 초기화
    rag = SimpleRAG()
    
    # 2. 문서 준비 (예시)
    file_paths = [
        "document1.pdf",
        "document2.txt",
    ]
    
    # 3. RAG 시스템 구축
    rag.build(file_paths)
    
    # 4. 대화형 질의응답
    print("\\n💬 질문을 입력하세요 (종료: 'quit')\\n")
    
    while True:
        question = input("질문> ")
        
        if question.lower() == 'quit':
            print("RAG 시스템을 종료합니다. 안녕히 가세요! 👋")
            break
        
        # 답변 생성
        result = rag.query(question)
        
        if result:
            print(f"\\n답변: {result['answer']}")
            print(f"\\n참고한 문서 ({len(result['sources'])}개):")
            for i, doc in enumerate(result['sources'], 1):
                print(f"{i}. {doc.page_content[:100]}...")
            print("-" * 50 + "\\n")

if __name__ == "__main__":
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경변수를 설정해주세요!")
        print("export OPENAI_API_KEY='your-api-key'")
    else:
        main()`}</code>
              </pre>
            </div>
          </div>

          {/* Usage Instructions */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl">
              <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-3">🚀 실행 방법</h4>
              <ol className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 font-bold">1.</span>
                  <span>API 키 설정: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">export OPENAI_API_KEY='sk-...'</code></span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 font-bold">2.</span>
                  <span>문서 준비: PDF 또는 TXT 파일</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 font-bold">3.</span>
                  <span>코드에서 파일 경로 수정</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600 font-bold">4.</span>
                  <span>실행: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">python simple_rag.py</code></span>
                </li>
              </ol>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl">
              <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-3">💡 확장 아이디어</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <span>웹 UI 추가 (Streamlit, Gradio)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <span>다양한 파일 형식 지원 (Word, HTML)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <span>대화 히스토리 관리</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <span>멀티모달 지원 (이미지, 표)</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Troubleshooting */}
        <section className="bg-amber-50 dark:bg-amber-900/20 rounded-2xl p-8">
          <h2 className="text-xl font-bold text-amber-800 dark:text-amber-200 mb-4">🔧 자주 발생하는 문제 해결</h2>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">1. ImportError: langchain 모듈을 찾을 수 없음</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                해결: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">pip install langchain openai chromadb pypdf</code>
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">2. OpenAI API 키 오류</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                해결: 환경변수 확인 <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">echo $OPENAI_API_KEY</code>
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">3. 메모리 부족 오류</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                해결: chunk_size를 500으로 줄이고, 문서를 작은 단위로 처리
              </p>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">🎉 축하합니다!</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold mb-3">✅ 완료한 내용</h3>
              <ul className="space-y-2 text-emerald-100">
                <li>• PDF/텍스트 문서 처리</li>
                <li>• 청킹 및 임베딩 생성</li>
                <li>• 벡터 데이터베이스 구축</li>
                <li>• 질의응답 시스템 구현</li>
                <li>• 대화형 인터페이스 제작</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-bold mb-3">🚀 다음 단계</h3>
              <ul className="space-y-2 text-emerald-100">
                <li>• 중급 과정에서 성능 최적화 배우기</li>
                <li>• 하이브리드 검색 구현하기</li>
                <li>• 프롬프트 엔지니어링 적용</li>
                <li>• 프로덕션 배포 준비</li>
                <li>• 평가 메트릭 구현</li>
              </ul>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/beginner/chapter3"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 청킹 전략
          </Link>
          
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            초급 과정 완료!
            <CheckCircle2 size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}