'use client'

import { useState, useCallback, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCw, Download, FileText } from 'lucide-react'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import '@/styles/pdf-viewer.css'

interface PDFViewerProps {
  file: File | string
  className?: string
  onError?: (error: Error) => void
  onTextExtracted?: (text: string, pageIndex: number) => void
}

export default function PDFViewer({ 
  file, 
  className = '', 
  onError,
  onTextExtracted 
}: PDFViewerProps) {
  // Worker 설정
  useEffect(() => {
    if (typeof window !== 'undefined') {
      pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
    }
  }, []);
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [scale, setScale] = useState<number>(1.0)
  const [rotation, setRotation] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [extractedTexts, setExtractedTexts] = useState<Record<number, string>>({})

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setIsLoading(false)
    console.log(`PDF 로드 완료: ${numPages}페이지`)
  }, [])

  const onDocumentLoadError = useCallback((error: Error) => {
    console.error('PDF 로드 오류:', error)
    setIsLoading(false)
    onError?.(error)
  }, [onError])

  const extractTextFromPage = useCallback(async (pageNum: number) => {
    if (extractedTexts[pageNum]) return extractedTexts[pageNum]

    try {
      // react-pdf는 직접적인 텍스트 추출 API를 제공하지 않으므로
      // PDF.js를 직접 사용하여 텍스트 추출
      const loadingTask = pdfjs.getDocument(file)
      const pdf = await loadingTask.promise
      const page = await pdf.getPage(pageNum)
      const textContent = await page.getTextContent()
      
      const text = textContent.items
        .filter((item: any) => 'str' in item)
        .map((item: any) => item.str)
        .join(' ')

      setExtractedTexts(prev => ({ ...prev, [pageNum]: text }))
      onTextExtracted?.(text, pageNum - 1)
      
      return text
    } catch (error) {
      console.error(`페이지 ${pageNum} 텍스트 추출 오류:`, error)
      return ''
    }
  }, [file, extractedTexts, onTextExtracted])

  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1))
  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages))
  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0))
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5))
  const rotate = () => setRotation(prev => (prev + 90) % 360)

  const extractAllText = useCallback(async () => {
    const allTexts: string[] = []
    for (let i = 1; i <= numPages; i++) {
      const text = await extractTextFromPage(i)
      allTexts.push(text)
    }
    return allTexts.join('\n\n')
  }, [numPages, extractTextFromPage])

  return (
    <div className={`pdf-viewer ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <FileText className="text-red-600" size={20} />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            PDF 뷰어
          </span>
          {isLoading && (
            <span className="text-xs text-gray-500 ml-2">로딩 중...</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Page Navigation */}
          {numPages > 0 && (
            <>
              <button
                onClick={goToPrevPage}
                disabled={pageNumber <= 1}
                className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50"
              >
                <ChevronLeft size={16} />
              </button>
              <span className="text-sm px-2">
                {pageNumber} / {numPages}
              </span>
              <button
                onClick={goToNextPage}
                disabled={pageNumber >= numPages}
                className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50"
              >
                <ChevronRight size={16} />
              </button>
              <div className="h-4 border-l border-gray-300 mx-2" />
            </>
          )}

          {/* Zoom Controls */}
          <button
            onClick={zoomOut}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            title="축소"
          >
            <ZoomOut size={16} />
          </button>
          <span className="text-xs px-2 min-w-12 text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={zoomIn}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            title="확대"
          >
            <ZoomIn size={16} />
          </button>

          {/* Rotation */}
          <button
            onClick={rotate}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            title="회전"
          >
            <RotateCw size={16} />
          </button>

          {/* Extract Text */}
          <button
            onClick={() => extractTextFromPage(pageNumber)}
            className="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
            title="현재 페이지 텍스트 추출"
          >
            텍스트 추출
          </button>
        </div>
      </div>

      {/* PDF Document */}
      <div className="pdf-document-container overflow-auto bg-gray-100 dark:bg-gray-900 p-4">
        <div className="flex justify-center">
          <Document
            file={file}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div>
                <span className="ml-2 text-gray-600 dark:text-gray-400">PDF 로딩 중...</span>
              </div>
            }
            error={
              <div className="flex items-center justify-center p-8 text-red-600">
                <FileText size={24} className="mr-2" />
                PDF를 불러올 수 없습니다
              </div>
            }
          >
            {numPages > 0 && (
              <Page
                pageNumber={pageNumber}
                scale={scale}
                rotate={rotation}
                loading={
                  <div className="flex items-center justify-center p-4 bg-white border rounded">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600"></div>
                    <span className="ml-2">페이지 로딩 중...</span>
                  </div>
                }
                error={
                  <div className="flex items-center justify-center p-4 bg-red-50 border border-red-200 rounded text-red-600">
                    페이지를 불러올 수 없습니다
                  </div>
                }
                className="shadow-lg"
              />
            )}
          </Document>
        </div>
      </div>

      {/* Page Text Preview (for debugging) */}
      {extractedTexts[pageNumber] && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800 max-h-32 overflow-auto">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            페이지 {pageNumber} 추출된 텍스트:
          </h4>
          <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-3">
            {extractedTexts[pageNumber]}
          </p>
        </div>
      )}
    </div>
  )
}

// 텍스트 추출만을 위한 유틸리티 함수
export const extractTextFromPDF = async (file: File): Promise<string> => {
  try {
    // Worker가 제대로 설정되었는지 확인
    if (!pdfjs.GlobalWorkerOptions.workerSrc && typeof window !== 'undefined') {
      pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
    }

    // File을 ArrayBuffer로 변환
    const arrayBuffer = await file.arrayBuffer();
    
    // PDF 문서 로드
    const loadingTask = pdfjs.getDocument({
      data: arrayBuffer,
      useSystemFonts: true,
    });
    
    const pdf = await loadingTask.promise;
    const texts: string[] = [];
    
    console.log(`PDF 로드 성공: ${pdf.numPages}페이지`);

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      try {
        const page = await pdf.getPage(pageNum);
        const textContent = await page.getTextContent();
        
        const pageText = textContent.items
          .filter((item: any) => 'str' in item && item.str)
          .map((item: any) => item.str)
          .join(' ')
          .trim();
        
        if (pageText) {
          texts.push(`=== 페이지 ${pageNum} ===\n${pageText}`);
        }
      } catch (pageError) {
        console.error(`페이지 ${pageNum} 처리 오류:`, pageError);
        texts.push(`=== 페이지 ${pageNum} ===\n[텍스트 추출 실패]`);
      }
    }

    return texts.length > 0 ? texts.join('\n\n') : 'PDF에서 텍스트를 찾을 수 없습니다.';
  } catch (error: any) {
    console.error('PDF 텍스트 추출 오류:', error);
    
    // 더 자세한 에러 메시지 제공
    if (error.message?.includes('Failed to fetch')) {
      throw new Error('PDF.js Worker 로딩 실패. 네트워크 연결을 확인하거나 페이지를 새로고침해주세요.');
    } else if (error.message?.includes('Invalid PDF')) {
      throw new Error('유효하지 않은 PDF 파일입니다.');
    } else {
      throw new Error(`PDF 텍스트 추출 실패: ${error.message || error}`);
    }
  }
}