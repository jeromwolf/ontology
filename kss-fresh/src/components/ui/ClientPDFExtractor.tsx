'use client'

import { useEffect, useState } from 'react'

// 클라이언트 전용 PDF 텍스트 추출기
export function useClientPDFExtractor() {
  const [isReady, setIsReady] = useState(false)
  const [pdfjs, setPdfjs] = useState<any>(null)

  useEffect(() => {
    // 클라이언트 사이드에서만 실행
    if (typeof window !== 'undefined') {
      import('pdfjs-dist').then((pdfjsLib) => {
        // Worker 설정
        pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
        setPdfjs(pdfjsLib);
        setIsReady(true);
      }).catch(error => {
        console.error('PDF.js 로드 실패:', error);
      });
    }
  }, []);

  const extractText = async (file: File): Promise<string> => {
    if (!isReady || !pdfjs) {
      throw new Error('PDF.js가 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요.');
    }

    try {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
      const texts: string[] = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items
          .filter((item: any) => 'str' in item)
          .map((item: any) => item.str)
          .join(' ');
        
        texts.push(`=== 페이지 ${i} ===\n${pageText}`);
      }

      return texts.join('\n\n');
    } catch (error) {
      console.error('PDF 텍스트 추출 오류:', error);
      throw error;
    }
  };

  return { isReady, extractText };
}