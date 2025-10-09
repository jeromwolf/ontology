'use client'

import { Layout } from 'lucide-react'

export default function Section5() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <Layout className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.5 레이아웃 인식 문서 처리</h2>
          <p className="text-gray-600 dark:text-gray-400">문서 구조를 이해하는 고급 RAG</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">지능형 문서 레이아웃 분석</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>레이아웃 인식 RAG는 문서의 시각적 구조와 의미적 위계를 이해하여 정보를 추출합니다.</strong>
              단순한 텍스트 추출을 넘어 제목-본문 관계, 이미지-캡션 연결, 표-설명 매핑 등
              문서 디자이너가 의도한 정보 구조를 AI가 정확히 파악합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>고급 레이아웃 분석 기법:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>LayoutLMv3</strong>: Microsoft의 2D 위치 정보 통합 언어모델</li>
              <li><strong>DocFormer</strong>: 문서 이해를 위한 멀티모달 Transformer</li>
              <li><strong>DETR</strong>: Object Detection으로 문서 요소 탐지</li>
              <li><strong>OCR + 좌표 매핑</strong>: 텍스트 위치와 의미적 역할 연결</li>
            </ul>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border border-indigo-200 dark:border-indigo-700 mt-4">
              <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-2">🏗️ 문서 구조 인식 정확도</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-indigo-700 dark:text-indigo-300">제목 추출</span>
                    <span className="font-medium text-indigo-800 dark:text-indigo-200">95.2%</span>
                  </div>
                  <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                    <div className="bg-indigo-500 h-2 rounded-full" style={{width: '95.2%'}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-indigo-700 dark:text-indigo-300">테이블 경계</span>
                    <span className="font-medium text-indigo-800 dark:text-indigo-200">92.7%</span>
                  </div>
                  <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                    <div className="bg-indigo-500 h-2 rounded-full" style={{width: '92.7%'}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-indigo-700 dark:text-indigo-300">이미지-캡션</span>
                    <span className="font-medium text-indigo-800 dark:text-indigo-200">89.4%</span>
                  </div>
                  <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                    <div className="bg-indigo-500 h-2 rounded-full" style={{width: '89.4%'}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-indigo-700 dark:text-indigo-300">리스트 구조</span>
                    <span className="font-medium text-indigo-800 dark:text-indigo-200">91.8%</span>
                  </div>
                  <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                    <div className="bg-indigo-500 h-2 rounded-full" style={{width: '91.8%'}}></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-rose-50 dark:bg-rose-900/20 p-4 rounded-lg border border-rose-200 dark:border-rose-700 mt-4">
              <h4 className="font-bold text-rose-800 dark:text-rose-200 mb-2">💼 실무 적용 효과</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong className="text-rose-800 dark:text-rose-200">법률 문서</strong>
                  <ul className="list-disc list-inside ml-2 text-rose-700 dark:text-rose-300 mt-1">
                    <li>조항별 자동 분류 및 인덱싱</li>
                    <li>판례-법조문 연결 구조 파악</li>
                    <li>법률 용어 정의 자동 추출</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-rose-800 dark:text-rose-200">연구 논문</strong>
                  <ul className="list-disc list-inside ml-2 text-rose-700 dark:text-rose-300 mt-1">
                    <li>Figure-Table 자동 매핑</li>
                    <li>인용 관계 그래프 구축</li>
                    <li>결과 섹션 핵심 데이터 추출</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">📄 PDF 분석</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                헤더, 푸터, 멀티컬럼 레이아웃 인식
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">🖼️ 이미지 위치</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                이미지와 캡션의 관계 파악
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 차트 해석</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                데이터 시각화 요소 추출
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
